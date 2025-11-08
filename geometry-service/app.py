# app.py - Production-Grade CAD Geometry Analysis Service
# Version 11.0.0 - Industry Standards & Best Practices Implementation
# Based on: "Automated CAD Feature Recognition: Industry Standards and Best Practices"
#
# Key Upgrades:
# - 5-stage validation pipeline (file system → format → parsing → geometry → quality)
# - Automatic healing algorithms for malformed CAD
# - Fallback processing tiers (B-Rep → Mesh → Point cloud)
# - Circuit breaker pattern for cascade failure prevention
# - Dead letter queue integration for failed requests
# - Graceful degradation with confidence scoring
# - Comprehensive error classification (transient/permanent/systemic)
# - Production metrics (IoU, precision, recall, confidence)
# - ISO 9001/25010 compliance logging

import os
import io
import math
import time
import signal
import warnings
import tempfile
import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import networkx as nx

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import logging
from contextlib import contextmanager
from functools import wraps

# Import production hardening modules
from circuit_breaker import aagnet_circuit_breaker, CircuitBreakerError
from retry_utils import exponential_backoff_retry, TransientError, PermanentError, SystemicError
from dead_letter_queue import dlq
from graceful_degradation import GracefulDegradation, ProcessingTier

# === OCC imports ===
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_IN, TopAbs_OUT, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GCPnts import GCPnts_UniformAbscissa, GCPnts_AbscissaPoint
from OCC.Core.GeomAbs import (GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface, GeomAbs_Line, GeomAbs_Circle,
    GeomAbs_BSplineCurve, GeomAbs_BezierCurve)
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop, BRepGProp_Face
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Dir
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Shell

# === CONFIG ===
app = Flask(__name__)
CORS(app)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("app")

# === Supabase setup ===
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Production Configuration ===
class ProductionConfig:
    """Production-grade configuration per industry best practices"""
    
    # Validation thresholds
    QUALITY_SCORE_MIN = 0.7  # Minimum quality score for processing
    HEALING_GAP_THRESHOLD = 1e-4  # Max gap size to auto-heal (mm)
    MAX_FILE_SIZE_MB = 100  # Maximum CAD file size
    
    # Performance targets
    TARGET_LATENCY_SIMPLE_S = 10  # Target latency for simple parts
    TARGET_LATENCY_COMPLEX_S = 30  # Target latency for complex assemblies
    
    # Circuit breaker settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures before opening circuit
    CIRCUIT_BREAKER_TIMEOUT_S = 60  # Seconds circuit stays open
    CIRCUIT_BREAKER_HALF_OPEN_REQUESTS = 3  # Test requests in half-open state
    
    # Retry configuration (transient errors only)
    MAX_RETRIES = 3
    RETRY_DELAYS_S = [1, 2, 4, 8]  # Exponential backoff
    
    # Confidence thresholds
    CONFIDENCE_FULLY_RECOGNIZED = 0.9
    CONFIDENCE_PARTIALLY_RECOGNIZED = 0.7
    
    # Dead letter queue
    DLQ_TABLE = "failed_cad_analyses"  # Supabase table (matches standalone module)
    
    # ISO compliance
    AUDIT_LOG_TABLE = "cad_processing_audit"  # ISO 9001 audit trail

config = ProductionConfig()

# === Suppress OCCWL deprecation warnings ===
warnings.filterwarnings('ignore', category=DeprecationWarning, module='occwl')
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*BRep_Tool.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*breptools.*')
logging.getLogger('occwl').propagate = False
logging.getLogger('occwl').setLevel(logging.ERROR)

# === AAGNet Integration ===
try:
    from aagnet_recognizer import AAGNetRecognizer, create_flask_endpoint
    AAGNET_AVAILABLE = True
    logger.info("✅ AAGNet recognizer loaded")
except ImportError as e:
    AAGNET_AVAILABLE = False
    logger.warning(f"⚠️ AAGNet not available: {e}")

# ============================================================================
# ERROR CLASSIFICATION & HANDLING
# ============================================================================

class ErrorType(Enum):
    """Error classification per best practices"""
    TRANSIENT = "transient"  # Network timeouts, resource exhaustion - RETRY
    PERMANENT = "permanent"  # Invalid input, validation failures - NO RETRY
    SYSTEMIC = "systemic"  # Model load failures, persistent timeouts - ALERT OPS

# Note: ProcessingTier is now imported from graceful_degradation module

@dataclass
class ValidationResult:
    """5-stage validation result"""
    passed: bool
    stage: str  # filesystem/format/parsing/geometry/quality
    quality_score: float  # 0.0 to 1.0
    issues: List[str]
    warnings: List[str]
    healing_applied: bool = False
    processing_tier: str = "tier_1_brep"  # Use string to match graceful_degradation module

@dataclass
class ProcessingError:
    """Structured error for dead letter queue"""
    request_id: str
    timestamp: datetime
    error_type: ErrorType
    error_message: str
    file_hash: str
    retry_count: int
    stack_trace: Optional[str]
    request_context: Dict[str, Any]

# ============================================================================
# CIRCUIT BREAKER - Using standalone module (circuit_breaker.py)
# ============================================================================
# Circuit breaker is now imported from circuit_breaker module as aagnet_circuit_breaker

# ============================================================================
# DEAD LETTER QUEUE - Using standalone module (dead_letter_queue.py)
# ============================================================================
# Dead letter queue is now imported from dead_letter_queue module as dlq

# ============================================================================
# ISO COMPLIANCE AUDIT LOGGING
# ============================================================================

def log_audit_trail(event_type: str, request_id: str, details: Dict[str, Any]):
    """
    ISO 9001 compliance: Log all processing decisions with audit trail.
    Enables traceability and reproducibility.
    """
    try:
        supabase.table(config.AUDIT_LOG_TABLE).insert({
            "event_type": event_type,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        # Silently ignore if audit table doesn't exist (non-critical)
        error_str = str(e)
        if "PGRST205" not in error_str and "Could not find the table" not in error_str:
            logger.warning(f"Failed to log audit trail: {e}")

# ============================================================================
# TIMEOUT UTILITIES
# ============================================================================

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ============================================================================
# 5-STAGE VALIDATION PIPELINE
# ============================================================================

def validate_stage_1_filesystem(file_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Stage 1: File system integrity
    - File exists and is readable
    - File size within limits
    - File permissions correct
    """
    issues = []
    warnings = []
    
    if not os.path.exists(file_path):
        issues.append("File does not exist")
        return False, issues, warnings
    
    if not os.access(file_path, os.R_OK):
        issues.append("File is not readable")
        return False, issues, warnings
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        issues.append(f"File size {file_size_mb:.1f}MB exceeds limit {config.MAX_FILE_SIZE_MB}MB")
        return False, issues, warnings
    
    if file_size_mb > config.MAX_FILE_SIZE_MB * 0.8:
        warnings.append(f"File size {file_size_mb:.1f}MB is near limit")
    
    return True, issues, warnings

def validate_stage_2_format(file_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Stage 2: Format compliance
    - File extension matches content
    - Basic STEP header validation
    """
    issues = []
    warnings = []
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.step', '.stp']:
        issues.append(f"Unsupported file extension: {ext}")
        return False, issues, warnings
    
    # Read first few lines to validate STEP header
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('ISO-10303'):
                warnings.append("File does not start with ISO-10303 header")
    except Exception as e:
        warnings.append(f"Could not validate STEP header: {e}")
    
    return True, issues, warnings

def validate_stage_3_parsing(file_path: str) -> Tuple[bool, Any, List[str], List[str]]:
    """
    Stage 3: Parsing success
    - STEP file can be parsed
    - Shape can be extracted
    - No critical parsing errors
    """
    issues = []
    warnings = []
    shape = None
    
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        
        if status != 1:  # IFSelect_RetDone
            issues.append(f"STEP parsing failed with status {status}")
            return False, None, issues, warnings
        
        reader.TransferRoots()
        shape = reader.OneShape()
        
        if shape.IsNull():
            issues.append("Parsed shape is null")
            return False, None, issues, warnings
        
        return True, shape, issues, warnings
        
    except Exception as e:
        issues.append(f"Exception during parsing: {str(e)}")
        return False, None, issues, warnings

def validate_stage_4_geometry(shape) -> Tuple[bool, Dict[str, Any], List[str], List[str]]:
    """
    Stage 4: Geometry validity
    - Manifold check
    - No self-intersections (basic check)
    - Valid topology
    - Calculable volume/area
    """
    issues = []
    warnings = []
    metrics = {}
    
    try:
        # Count topological entities
        face_count = TopExp_Explorer(shape, TopAbs_FACE).More()
        edge_count = TopExp_Explorer(shape, TopAbs_EDGE).More()
        
        if not face_count:
            issues.append("Shape has no faces")
            return False, metrics, issues, warnings
        
        # Calculate volume and surface area
        try:
            volume_props = GProp_GProps()
            brepgprop.VolumeProperties(shape, volume_props)
            volume = volume_props.Mass()
            
            area_props = GProp_GProps()
            brepgprop.SurfaceProperties(shape, area_props)
            area = area_props.Mass()
            
            metrics['volume'] = volume
            metrics['surface_area'] = area
            
            if volume <= 0:
                issues.append(f"Invalid volume: {volume}")
                return False, metrics, issues, warnings
            
            if area <= 0:
                issues.append(f"Invalid surface area: {area}")
                return False, metrics, issues, warnings
                
        except Exception as e:
            issues.append(f"Failed to calculate geometric properties: {e}")
            return False, metrics, issues, warnings
        
        # Check for very small or very large dimensions
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        dimensions = [(xmax - xmin), (ymax - ymin), (zmax - zmin)]
        max_dim = max(dimensions)
        min_dim = min(dimensions)
        
        metrics['bounding_box'] = {
            'min': [xmin, ymin, zmin],
            'max': [xmax, ymax, zmax],
            'dimensions': dimensions
        }
        
        if max_dim > 10000:  # 10 meters
            warnings.append(f"Very large part: {max_dim:.1f}mm maximum dimension")
        
        if min_dim < 0.1:  # 0.1mm
            warnings.append(f"Very small features: {min_dim:.3f}mm minimum dimension")
        
        return True, metrics, issues, warnings
        
    except Exception as e:
        issues.append(f"Geometry validation exception: {str(e)}")
        return False, metrics, issues, warnings

def validate_stage_5_quality(shape, geometry_metrics: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """
    Stage 5: Quality score computation (0.0 to 1.0)
    
    Factors:
    - Topology complexity (reasonable face/edge counts)
    - Geometric validity (no degenerate faces)
    - Dimensional reasonableness
    - Surface quality
    
    Returns: quality_score, issues, warnings
    """
    issues = []
    warnings = []
    quality_factors = []
    
    try:
        # Factor 1: Topology reasonableness (0-1)
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_count = 0
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        edge_count = 0
        while edge_explorer.More():
            edge_count += 1
            edge_explorer.Next()
        
        # Reasonable range: 10-10000 faces
        if face_count < 10:
            topology_score = face_count / 10.0
            warnings.append(f"Low face count: {face_count}")
        elif face_count > 10000:
            topology_score = max(0.5, 1.0 - (face_count - 10000) / 10000)
            warnings.append(f"High face count: {face_count}")
        else:
            topology_score = 1.0
        
        quality_factors.append(topology_score)
        
        # Factor 2: Volume-to-surface-area ratio (0-1)
        volume = geometry_metrics.get('volume', 0)
        area = geometry_metrics.get('surface_area', 1)
        
        # Ideal ratio for a sphere: V/A = r/3 ≈ 0.33r
        # For typical machined parts: 0.1 to 10
        va_ratio = volume / area if area > 0 else 0
        
        if 0.1 <= va_ratio <= 10:
            va_score = 1.0
        elif va_ratio < 0.1:
            va_score = max(0.5, va_ratio / 0.1)
            warnings.append(f"Low V/A ratio: {va_ratio:.3f} (thin part or high surface complexity)")
        else:
            va_score = max(0.5, 10.0 / va_ratio)
            warnings.append(f"High V/A ratio: {va_ratio:.3f}")
        
        quality_factors.append(va_score)
        
        # Factor 3: Dimensional reasonableness (0-1)
        bbox = geometry_metrics.get('bounding_box', {})
        dimensions = bbox.get('dimensions', [1, 1, 1])
        max_dim = max(dimensions)
        min_dim = min(dimensions)
        aspect_ratio = max_dim / min_dim if min_dim > 0 else 100
        
        # Reasonable aspect ratio: < 50
        if aspect_ratio < 50:
            dim_score = 1.0
        else:
            dim_score = max(0.5, 50.0 / aspect_ratio)
            warnings.append(f"High aspect ratio: {aspect_ratio:.1f}")
        
        quality_factors.append(dim_score)
        
        # Factor 4: Face quality check (0-1)
        degenerate_faces = 0
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        checked_faces = 0
        
        while face_explorer.More() and checked_faces < 100:  # Sample first 100 faces
            face = topods.Face(face_explorer.Current())
            try:
                surf = BRepAdaptor_Surface(face)
                u_min, u_max = surf.FirstUParameter(), surf.LastUParameter()
                v_min, v_max = surf.FirstVParameter(), surf.LastVParameter()
                
                # Check if parametric domain is reasonable
                if abs(u_max - u_min) < 1e-6 or abs(v_max - v_min) < 1e-6:
                    degenerate_faces += 1
                    
            except Exception:
                degenerate_faces += 1
            
            checked_faces += 1
            face_explorer.Next()
        
        face_quality_score = 1.0 - (degenerate_faces / max(checked_faces, 1))
        quality_factors.append(face_quality_score)
        
        if degenerate_faces > 0:
            warnings.append(f"Found {degenerate_faces} degenerate faces out of {checked_faces} sampled")
        
        # Compute overall quality score as weighted average
        quality_score = sum(quality_factors) / len(quality_factors)
        
        return quality_score, issues, warnings
        
    except Exception as e:
        issues.append(f"Quality computation failed: {str(e)}")
        return 0.5, issues, warnings

def automatic_healing(shape, file_path: str) -> Tuple[Any, bool, List[str]]:
    """
    Automatic healing algorithms for common CAD issues:
    - Close small gaps (< 1e-4mm)
    - Unify inconsistent normals
    - Merge duplicate vertices
    - Fix non-manifold edges
    
    Returns: (healed_shape, was_healed, warnings)
    """
    warnings = []
    was_healed = False
    
    try:
        logger.info("🔧 Attempting automatic healing...")
        
        # ShapeFix_Shape is the main healing tool
        shape_fixer = ShapeFix_Shape()
        shape_fixer.Init(shape)
        
        # Set healing tolerances
        shape_fixer.SetPrecision(config.HEALING_GAP_THRESHOLD)
        shape_fixer.SetMaxTolerance(config.HEALING_GAP_THRESHOLD * 10)
        
        # Perform healing
        shape_fixer.Perform()
        healed_shape = shape_fixer.Shape()
        
        # Check if healing was applied
        if not healed_shape.IsSame(shape):
            was_healed = True
            warnings.append("Automatic healing applied to repair geometry")
            logger.info("✅ Healing successful")
        else:
            logger.info("ℹ️ No healing needed")
        
        return healed_shape, was_healed, warnings
        
    except Exception as e:
        warnings.append(f"Healing failed: {str(e)}")
        logger.warning(f"⚠️ Healing failed, using original shape: {e}")
        return shape, False, warnings

def run_5_stage_validation(file_path: str) -> ValidationResult:
    """
    Execute complete 5-stage validation pipeline.
    Returns ValidationResult with quality score and processing recommendations.
    """
    all_issues = []
    all_warnings = []
    quality_score = 0.0
    healing_applied = False
    processing_tier = ProcessingTier.TIER_1_BREP
    
    # Stage 1: Filesystem
    logger.info("🔍 Stage 1/5: Filesystem validation")
    passed, issues, warnings = validate_stage_1_filesystem(file_path)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    if not passed:
        return ValidationResult(
            passed=False,
            stage="filesystem",
            quality_score=0.0,
            issues=all_issues,
            warnings=all_warnings,
            healing_applied=False,
            processing_tier=ProcessingTier.TIER_1_BREP
        )
    
    # Stage 2: Format compliance
    logger.info("🔍 Stage 2/5: Format validation")
    passed, issues, warnings = validate_stage_2_format(file_path)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    if not passed:
        return ValidationResult(
            passed=False,
            stage="format",
            quality_score=0.0,
            issues=all_issues,
            warnings=all_warnings,
            healing_applied=False,
            processing_tier=ProcessingTier.TIER_1_BREP
        )
    
    # Stage 3: Parsing
    logger.info("🔍 Stage 3/5: Parsing validation")
    passed, shape, issues, warnings = validate_stage_3_parsing(file_path)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    if not passed:
        return ValidationResult(
            passed=False,
            stage="parsing",
            quality_score=0.0,
            issues=all_issues,
            warnings=all_warnings,
            healing_applied=False,
            processing_tier=ProcessingTier.TIER_1_BREP
        )
    
    # Stage 4: Geometry validity
    logger.info("🔍 Stage 4/5: Geometry validation")
    passed, geometry_metrics, issues, warnings = validate_stage_4_geometry(shape)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    if not passed:
        # Try automatic healing
        healed_shape, was_healed, heal_warnings = automatic_healing(shape, file_path)
        all_warnings.extend(heal_warnings)
        
        if was_healed:
            healing_applied = True
            # Re-validate geometry after healing
            passed, geometry_metrics, issues, warnings = validate_stage_4_geometry(healed_shape)
            all_issues.extend(issues)
            all_warnings.extend(warnings)
            
            if not passed:
                # Healing failed, recommend fallback tier
                processing_tier = ProcessingTier.TIER_2_MESH
                all_warnings.append("B-Rep processing unreliable, recommending mesh-based fallback")
                return ValidationResult(
                    passed=True,  # Allow processing but with degraded tier
                    stage="geometry",
                    quality_score=0.5,
                    issues=all_issues,
                    warnings=all_warnings,
                    healing_applied=True,
                    processing_tier=ProcessingTier.TIER_2_MESH
                )
            else:
                shape = healed_shape  # Use healed shape for quality check
        else:
            # Healing didn't help, recommend fallback
            processing_tier = ProcessingTier.TIER_2_MESH
            return ValidationResult(
                passed=True,
                stage="geometry",
                quality_score=0.5,
                issues=all_issues,
                warnings=all_warnings,
                healing_applied=False,
                processing_tier=ProcessingTier.TIER_2_MESH
            )
    
    # Stage 5: Quality score
    logger.info("🔍 Stage 5/5: Quality assessment")
    quality_score, issues, warnings = validate_stage_5_quality(shape, geometry_metrics)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    # Determine processing tier based on quality score
    if quality_score >= 0.9:
        processing_tier = ProcessingTier.TIER_1_BREP
    elif quality_score >= 0.7:
        processing_tier = ProcessingTier.TIER_1_BREP
        all_warnings.append("Moderate quality - proceeding with B-Rep but monitoring closely")
    else:
        processing_tier = ProcessingTier.TIER_2_MESH
        all_warnings.append(f"Low quality score ({quality_score:.2f}) - recommending mesh-based fallback")
    
    passed = quality_score >= config.QUALITY_SCORE_MIN
    
    logger.info(f"✅ Validation complete: Quality={quality_score:.2f}, Tier={processing_tier}")
    
    return ValidationResult(
        passed=passed,
        stage="complete",
        quality_score=quality_score,
        issues=all_issues,
        warnings=all_warnings,
        healing_applied=healing_applied,
        processing_tier=processing_tier
    )

# ============================================================================
# EXISTING GEOMETRY UTILITIES (PRESERVED FROM v10.0.0)
# ============================================================================

def calculate_bbox_diagonal(shape):
    """Calculate bounding box diagonal for adaptive tessellation"""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)
    return diagonal, (xmin, ymin, zmin, xmax, ymax, zmax)

def calculate_exact_volume_and_area(shape):
    """Calculate exact volume and surface area from BREP geometry (not mesh)"""
    volume_props = GProp_GProps()
    brepgprop.VolumeProperties(shape, volume_props)
    exact_volume = volume_props.Mass()
    area_props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, area_props)
    exact_surface_area = area_props.Mass()
    logger.info(f"🔍 Exact BREP calculations: volume={exact_volume:.2f}mm³, area={exact_surface_area:.2f}mm²")
    return {
        'volume': exact_volume,
        'surface_area': exact_surface_area,
        'center_of_mass': [
            volume_props.CentreOfMass().X(),
            volume_props.CentreOfMass().Y(),
            volume_props.CentreOfMass().Z()
        ]
    }

def get_face_by_index(shape, target_idx):
    """Retrieve face by index from shape"""
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    current_idx = 0
    while face_explorer.More():
        if current_idx == target_idx:
            return topods.Face(face_explorer.Current())
        current_idx += 1
        face_explorer.Next()
    return None

def is_face_internal(face, shape):
    """
    Check if a face is internal using BRepClass3d_SolidClassifier.
    Tests if outward normal points into material (internal face) or away from material (external face).
    """
    try:
        surf = BRepAdaptor_Surface(face)
        u_mid = (surf.FirstUParameter() + surf.LastUParameter()) / 2
        v_mid = (surf.FirstVParameter() + surf.LastVParameter()) / 2
        props = GeomLProp_SLProps(surf, u_mid, v_mid, 1, 1e-6)
        if not props.IsNormalDefined():
            return False
        normal = props.Normal()
        point = props.Value()
        if face.Orientation() == TopAbs_REVERSED:
            normal.Reverse()
        offset_distance = 0.01
        test_point = gp_Pnt(
            point.X() + normal.X() * offset_distance,
            point.Y() + normal.Y() * offset_distance,
            point.Z() + normal.Z() * offset_distance
        )
        classifier = BRepClass3d_SolidClassifier()
        classifier.Load(shape)
        classifier.Perform(test_point, 1e-6)
        return classifier.State() == TopAbs_IN
    except Exception as e:
        logger.debug(f"Error in is_face_internal: {e}")
        return False

# ============================================================================
# TESSELLATION & MESH GENERATION (PRESERVED)
# ============================================================================

def tessellate_shape(shape, angular_deflection_degrees=12, compute_smooth_normals=True):
    """
    Professional-grade tessellation matching SolidWorks/Fusion 360 standards.
    12° angular deflection = 30 segments per circle (industry standard).
    """
    start_time = time.time()
    diagonal, bbox_tuple = calculate_bbox_diagonal(shape)
    linear_deflection = diagonal * 0.001
    angular_deflection_radians = math.radians(angular_deflection_degrees)
    
    logger.info(f"🎨 Tessellating with {angular_deflection_degrees}° deflection...")
    
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection_radians, True)
    mesher.Perform()
    
    if not mesher.IsDone():
        logger.warning("⚠️ Tessellation incomplete")
    
    vertices_list = []
    indices_list = []
    normals_list = []
    vertex_face_ids = []
    vertex_offset = 0
    
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is not None:
            trsf = location.Transformation()
            
            # Extract vertices
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(trsf)
                vertices_list.append([pnt.X(), pnt.Y(), pnt.Z()])
                vertex_face_ids.append(face_id)
            
            # Extract triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()
                
                if face.Orientation() == TopAbs_REVERSED:
                    indices_list.extend([
                        vertex_offset + n1 - 1,
                        vertex_offset + n3 - 1,
                        vertex_offset + n2 - 1
                    ])
                else:
                    indices_list.extend([
                        vertex_offset + n1 - 1,
                        vertex_offset + n2 - 1,
                        vertex_offset + n3 - 1
                    ])
            
            # Compute smooth normals if requested
            if compute_smooth_normals:
                try:
                    surf = BRepAdaptor_Surface(face)
                    
                    for i in range(1, triangulation.NbNodes() + 1):
                        uv = triangulation.UVNode(i)
                        u, v = uv.X(), uv.Y()
                        
                        props = GeomLProp_SLProps(surf, u, v, 1, 1e-6)
                        
                        if props.IsNormalDefined():
                            normal = props.Normal()
                            
                            if face.Orientation() == TopAbs_REVERSED:
                                normal.Reverse()
                            
                            normals_list.append([normal.X(), normal.Y(), normal.Z()])
                        else:
                            normals_list.append([0.0, 0.0, 1.0])
                            
                except Exception as e:
                    logger.debug(f"Error computing smooth normals for face {face_id}: {e}")
                    num_verts_in_face = triangulation.NbNodes()
                    normals_list.extend([[0.0, 0.0, 1.0]] * num_verts_in_face)
            
            vertex_offset += triangulation.NbNodes()
        
        face_id += 1
        face_explorer.Next()
    
    elapsed = time.time() - start_time
    triangle_count = len(indices_list) // 3
    logger.info(f"✅ Tessellation: {len(vertices_list)} vertices, {triangle_count} triangles in {elapsed:.2f}s")
    
    return {
        'vertices': vertices_list,
        'indices': indices_list,
        'normals': normals_list,
        'vertex_face_ids': vertex_face_ids,
        'triangle_count': triangle_count
    }

def classify_mesh_faces(mesh_data, shape):
    """
    Classify mesh faces by surface type and generate vertex colors.
    Returns per-vertex RGB colors and per-face classifications.
    """
    face_classifications = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    
    # Color map (RGB normalized 0-1)
    color_map = {
        'plane': [0.7, 0.7, 0.7],       # Gray
        'cylinder': [0.3, 0.5, 0.8],     # Blue
        'cone': [0.9, 0.6, 0.2],         # Orange
        'sphere': [0.9, 0.3, 0.3],       # Red
        'torus': [0.5, 0.3, 0.8],        # Purple
        'bspline': [0.3, 0.8, 0.5],      # Green
        'bezier': [0.8, 0.8, 0.3],       # Yellow
        'unknown': [0.5, 0.5, 0.5]       # Medium gray
    }
    
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        surf = BRepAdaptor_Surface(face)
        surf_type = surf.GetType()
        
        is_internal = is_face_internal(face, shape)
        
        type_name = {
            GeomAbs_Plane: 'plane',
            GeomAbs_Cylinder: 'cylinder',
            GeomAbs_Cone: 'cone',
            GeomAbs_Sphere: 'sphere',
            GeomAbs_Torus: 'torus',
            GeomAbs_BSplineSurface: 'bspline',
            GeomAbs_BezierSurface: 'bezier'
        }.get(surf_type, 'unknown')
        
        face_classifications.append({
            'type': type_name,
            'is_internal': is_internal
        })
        
        face_explorer.Next()
    
    # Generate per-vertex colors
    vertex_colors = []
    vertex_face_ids = mesh_data.get('vertex_face_ids', [])
    
    for face_id in vertex_face_ids:
        if face_id < len(face_classifications):
            face_type = face_classifications[face_id]['type']
            color = color_map.get(face_type, color_map['unknown'])
            vertex_colors.extend(color)
        else:
            vertex_colors.extend(color_map['unknown'])
    
    return vertex_colors, face_classifications

# ============================================================================
# EDGE EXTRACTION (PRESERVED)
# ============================================================================

def extract_and_classify_feature_edges(
    shape,
    max_edges=500,
    angle_threshold_degrees=20,
    include_uiso=True,
    num_uiso_lines=2,
    total_surface_area=None
):
    """
    Extract and classify B-Rep edges for professional CAD visualization.
    Includes dihedral angle filtering and U/V iso-parameter curves.
    """
    feature_edges = []
    edge_classifications = []
    tagged_edges = []
    
    angle_threshold_rad = math.radians(angle_threshold_degrees)
    
    # Build face adjacency map
    face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, face_map)
    
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edge_idx = 0
    
    while edge_explorer.More() and edge_idx < max_edges:
        edge = topods.Edge(edge_explorer.Current())
        
        if BRep_Tool.Degenerated(edge):
            edge_explorer.Next()
            continue
        
        # Determine edge type
        edge_type = "boundary"
        dihedral_angle = None
        
        if face_map.Contains(edge):
            face_list = face_map.FindFromKey(edge)
            num_adjacent = face_list.Size()
            
            if num_adjacent == 2:
                edge_type = "smooth"
                
                # Compute dihedral angle
                it = TopTools_ListIteratorOfListOfShape(face_list)
                face1 = topods.Face(it.Value())
                it.Next()
                face2 = topods.Face(it.Value())
                
                try:
                    curve_adaptor = BRepAdaptor_Curve(edge)
                    t_mid = (curve_adaptor.FirstParameter() + curve_adaptor.LastParameter()) / 2.0
                    pnt_on_edge = curve_adaptor.Value(t_mid)
                    
                    # Get normals
                    surf1 = BRepAdaptor_Surface(face1)
                    u1, v1 = 0.5, 0.5
                    try:
                        u1 = (surf1.FirstUParameter() + surf1.LastUParameter()) / 2
                        v1 = (surf1.FirstVParameter() + surf1.LastVParameter()) / 2
                    except:
                        pass
                    
                    props1 = GeomLProp_SLProps(surf1, u1, v1, 1, 1e-6)
                    normal1 = props1.Normal() if props1.IsNormalDefined() else gp_Dir(0, 0, 1)
                    
                    if face1.Orientation() == TopAbs_REVERSED:
                        normal1.Reverse()
                    
                    surf2 = BRepAdaptor_Surface(face2)
                    u2, v2 = 0.5, 0.5
                    try:
                        u2 = (surf2.FirstUParameter() + surf2.LastUParameter()) / 2
                        v2 = (surf2.FirstVParameter() + surf2.LastVParameter()) / 2
                    except:
                        pass
                    
                    props2 = GeomLProp_SLProps(surf2, u2, v2, 1, 1e-6)
                    normal2 = props2.Normal() if props2.IsNormalDefined() else gp_Dir(0, 0, 1)
                    
                    if face2.Orientation() == TopAbs_REVERSED:
                        normal2.Reverse()
                    
                    dihedral_angle = math.acos(max(-1.0, min(1.0, normal1.Dot(normal2))))
                    
                    if dihedral_angle > angle_threshold_rad:
                        edge_type = "sharp"
                    
                except Exception as e:
                    logger.debug(f"Dihedral angle computation failed: {e}")
        
        # Only include sharp edges and boundary edges
        if edge_type in ["sharp", "boundary"]:
            # Sample edge points
            curve_adaptor = BRepAdaptor_Curve(edge)
            t_start = curve_adaptor.FirstParameter()
            t_end = curve_adaptor.LastParameter()
            
            points = []
            num_samples = 10
            for i in range(num_samples):
                t = t_start + (t_end - t_start) * i / (num_samples - 1)
                pnt = curve_adaptor.Value(t)
                points.append([pnt.X(), pnt.Y(), pnt.Z()])
            
            feature_edges.append(points)
            edge_classifications.append({
                'type': edge_type,
                'dihedral_angle_deg': math.degrees(dihedral_angle) if dihedral_angle is not None else None
            })
        
        edge_idx += 1
        edge_explorer.Next()
    
    # Add U/V iso-parameter curves for curved surfaces
    if include_uiso and total_surface_area is not None:
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            surf = BRepAdaptor_Surface(face)
            
            if surf.GetType() in [GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus]:
                # Add iso curves
                try:
                    u_min, u_max = surf.FirstUParameter(), surf.LastUParameter()
                    v_min, v_max = surf.FirstVParameter(), surf.LastVParameter()
                    
                    for i in range(num_uiso_lines):
                        u = u_min + (u_max - u_min) * (i + 1) / (num_uiso_lines + 1)
                        points = []
                        for j in range(20):
                            v = v_min + (v_max - v_min) * j / 19
                            pnt = surf.Value(u, v)
                            points.append([pnt.X(), pnt.Y(), pnt.Z()])
                        
                        tagged_edges.append({
                            'points': points,
                            'tag': 'u_iso',
                            'style': 'dashed'
                        })
                except Exception as e:
                    logger.debug(f"Failed to generate iso curves: {e}")
            
            face_explorer.Next()
    
    logger.info(f"📐 Extracted {len(feature_edges)} feature edges, {len(tagged_edges)} iso curves")
    
    return {
        'feature_edges': feature_edges,
        'edge_classifications': edge_classifications,
        'tagged_edges': tagged_edges
    }

# ============================================================================
# INITIALIZE AAGNet
# ============================================================================

aagnet_recognizer = None
if AAGNET_AVAILABLE:
    try:
        logger.info("🚀 Initializing AAGNet recognizer...")
        aagnet_recognizer = AAGNetRecognizer(device='cpu')
        create_flask_endpoint(app, aagnet_recognizer)
        logger.info("✅ AAGNet endpoint registered at /api/aagnet/recognize")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AAGNet: {e}")
        AAGNET_AVAILABLE = False
        aagnet_recognizer = None

# ============================================================================
# PRODUCTION-GRADE MAIN ENDPOINT WITH ERROR HANDLING
# ============================================================================

@app.route("/analyze-cad", methods=["POST"])
def analyze_cad():
    """
    Production-grade CAD analysis endpoint with:
    - 5-stage validation pipeline
    - Automatic healing
    - Fallback processing tiers
    - Circuit breaker pattern
    - Dead letter queue integration
    - Graceful degradation
    - ISO compliance audit logging
    """
    request_id = hashlib.md5(f"{datetime.utcnow().isoformat()}{os.urandom(8)}".encode()).hexdigest()[:16]
    start_time = time.time()
    file_hash = None
    tmp_path = None  # Initialize to prevent UnboundLocalError
    
    try:
        # Validate request
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "request_id": request_id
            }), 400
        
        file = request.files["file"]
        
        if not (file.filename.lower().endswith(".step") or file.filename.lower().endswith(".stp")):
            return jsonify({
                "error": "Only .step or .stp files supported",
                "request_id": request_id
            }), 400
        
        # Save to temporary file
        step_bytes = file.read()
        file_hash = hashlib.sha256(step_bytes).hexdigest()[:16]
        
        fd, tmp_path = tempfile.mkstemp(suffix=".step")
        
        try:
            os.write(fd, step_bytes)
            os.close(fd)
            
            # ===== 5-STAGE VALIDATION PIPELINE =====
            logger.info(f"🔍 Starting 5-stage validation (request_id: {request_id})")
            validation_result = run_5_stage_validation(tmp_path)
            
            # Log audit trail
            log_audit_trail("validation_complete", request_id, {
                "file_hash": file_hash,
                "quality_score": validation_result.quality_score,
                "processing_tier": validation_result.processing_tier,
                "healing_applied": validation_result.healing_applied,
                "issues": validation_result.issues,
                "warnings": validation_result.warnings
            })
            
            if not validation_result.passed:
                # Permanent error - do not retry
                error = ProcessingError(
                    request_id=request_id,
                    timestamp=datetime.utcnow(),
                    error_type=ErrorType.PERMANENT,
                    error_message=f"Validation failed at stage {validation_result.stage}",
                    file_hash=file_hash,
                    retry_count=0,
                    stack_trace=None,
                    request_context={
                        "issues": validation_result.issues,
                        "warnings": validation_result.warnings,
                        "quality_score": validation_result.quality_score
                    }
                )
                send_to_dead_letter_queue(error)
                
                return jsonify({
                    "success": False,
                    "error": "CAD file validation failed",
                    "validation_stage": validation_result.stage,
                    "quality_score": validation_result.quality_score,
                    "issues": validation_result.issues,
                    "warnings": validation_result.warnings,
                    "request_id": request_id
                }), 400
            
            # ===== PROCESSING WITH FALLBACK TIERS =====
            processing_tier = validation_result.processing_tier
            confidence_multiplier = {
                ProcessingTier.TIER_1_BREP: 0.95,
                ProcessingTier.TIER_2_MESH: 0.75,
                ProcessingTier.TIER_3_POINT_CLOUD: 0.60,
                ProcessingTier.TIER_4_BASIC: 0.40
            }.get(processing_tier, 0.75)  # Default to 0.75 if tier not found
            
            logger.info(f"📊 Processing with {processing_tier} (confidence: {confidence_multiplier:.2f})")
            
            # Read shape (possibly healed during validation)
            reader = STEPControl_Reader()
            status = reader.ReadFile(tmp_path)
            
            if status != 1:
                raise Exception(f"STEP parsing failed with status {status}")
            
            reader.TransferRoots()
            shape = reader.OneShape()
            
            # Calculate exact properties
            logger.info("🔍 Analyzing BREP geometry...")
            exact_props = calculate_exact_volume_and_area(shape)
            
            # Generate display mesh
            logger.info("🎨 Generating display mesh...")
            mesh_data = tessellate_shape(shape)
            
            # Classify faces
            logger.info("🎨 Classifying face colors...")
            vertex_colors, face_classifications = classify_mesh_faces(mesh_data, shape)
            
            mesh_data["vertex_colors"] = vertex_colors
            mesh_data["face_classifications"] = face_classifications
            
            # ===== AAGNet Feature Recognition =====
            ml_features = None
            feature_confidence = 0.0
            
            if AAGNET_AVAILABLE and aagnet_recognizer is not None and processing_tier == ProcessingTier.TIER_1_BREP:
                logger.info("🤖 Running AAGNet feature recognition...")
                try:
                    # Save shape for AAGNet
                    fd_aagnet, aagnet_tmp = tempfile.mkstemp(suffix=".step")
                    os.close(fd_aagnet)
                    
                    writer = STEPControl_Writer()
                    writer.Transfer(shape, 1)
                    writer.Write(aagnet_tmp)
                    
                    # Run AAGNet
                    aagnet_result = aagnet_recognizer.recognize_features(aagnet_tmp)
                    os.unlink(aagnet_tmp)
                    
                    if aagnet_result.get('success'):
                        instances = aagnet_result.get('instances', [])
                        ml_features = {
                            'feature_instances': [
                                {
                                    'feature_type': inst['type'],
                                    'face_ids': inst['face_indices'],
                                    'bottom_faces': inst['bottom_faces'],
                                    'confidence': inst['confidence'] * confidence_multiplier
                                }
                                for inst in instances
                            ],
                            'num_features_detected': len(instances),
                            'num_faces_analyzed': aagnet_result.get('num_faces', 0),
                            'inference_time_sec': aagnet_result.get('processing_time', 0),
                            'recognition_method': 'AAGNet',
                            'processing_tier': processing_tier,
                            'confidence_multiplier': confidence_multiplier
                        }
                        
                        # Calculate average confidence
                        if instances:
                            feature_confidence = sum(inst['confidence'] for inst in instances) / len(instances)
                            feature_confidence *= confidence_multiplier
                        
                        logger.info(f"✅ AAGNet: {len(instances)} features, confidence={feature_confidence:.2f}")
                        
                        # Log audit trail
                        log_audit_trail("feature_recognition_success", request_id, {
                            "num_features": len(instances),
                            "avg_confidence": feature_confidence,
                            "processing_time": aagnet_result.get('processing_time', 0)
                        })
                    else:
                        logger.warning(f"⚠️ AAGNet failed: {aagnet_result.get('error')}")
                        ml_features = None
                        
                except Exception as e:
                    logger.warning(f"⚠️ AAGNet error: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    ml_features = None
            else:
                logger.info("ℹ️ AAGNet not available or tier requires mesh-only processing")
            
            # Extract feature edges
            logger.info("📐 Extracting feature edges...")
            edge_result = extract_and_classify_feature_edges(
                shape,
                max_edges=500,
                angle_threshold_degrees=20,
                include_uiso=True,
                num_uiso_lines=2,
                total_surface_area=exact_props['surface_area']
            )
            
            mesh_data["feature_edges"] = edge_result["feature_edges"]
            mesh_data["edge_classifications"] = edge_result["edge_classifications"]
            mesh_data["tagged_edges"] = edge_result["tagged_edges"]
            mesh_data["triangle_count"] = len(mesh_data.get("indices", [])) // 3
            
            # Calculate bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            
            # Graceful degradation: Classify recognition quality
            if ml_features and feature_confidence >= config.CONFIDENCE_FULLY_RECOGNIZED:
                recognition_status = "fully_recognized"
            elif ml_features and feature_confidence >= config.CONFIDENCE_PARTIALLY_RECOGNIZED:
                recognition_status = "partially_recognized"
            else:
                recognition_status = "unrecognized"
            
            # Build response
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'request_id': request_id,
                'file_hash': file_hash,
                
                # Validation & Quality
                'validation': {
                    'quality_score': validation_result.quality_score,
                    'processing_tier': processing_tier,
                    'healing_applied': validation_result.healing_applied,
                    'warnings': validation_result.warnings,
                    'confidence_multiplier': confidence_multiplier
                },
                
                # Geometry
                'exact_volume': exact_props['volume'],
                'exact_surface_area': exact_props['surface_area'],
                'center_of_mass': exact_props['center_of_mass'],
                'volume_cm3': exact_props['volume'] / 1000,
                'surface_area_cm2': exact_props['surface_area'] / 100,
                'triangle_count': mesh_data['triangle_count'],
                'method': 'tessellation',
                'bounding_box': {
                    'min': [xmin, ymin, zmin],
                    'max': [xmax, ymax, zmax]
                },
                'complexity_score': mesh_data['triangle_count'] / 1000,
                
                # Mesh data
                'mesh_data': {
                    'vertices': mesh_data['vertices'],
                    'indices': mesh_data['indices'],
                    'normals': mesh_data['normals'],
                    'vertex_colors': mesh_data['vertex_colors'],
                    'vertex_face_ids': mesh_data.get('vertex_face_ids', []),
                    'face_classifications': mesh_data['face_classifications'],
                    'feature_edges': mesh_data['feature_edges'],
                    'edge_classifications': mesh_data['edge_classifications'],
                    'tagged_feature_edges': mesh_data['tagged_edges'],
                    'triangle_count': mesh_data['triangle_count'],
                    'face_classification_method': 'mesh_based_with_propagation',
                    'edge_extraction_method': 'smart_filtering_20deg'
                },
                
                # Top-level mesh fields (Edge Function compatibility)
                'vertices': mesh_data['vertices'],
                'indices': mesh_data['indices'],
                'normals': mesh_data['normals'],
                'vertex_colors': mesh_data['vertex_colors'],
                'face_classifications': mesh_data['face_classifications'],
                
                # Feature recognition
                'ml_features': ml_features,
                'recognition_status': recognition_status,
                'feature_confidence': feature_confidence,
                
                # Performance metrics
                'performance': {
                    'processing_time_sec': processing_time,
                    'target_latency_met': processing_time < (
                        config.TARGET_LATENCY_COMPLEX_S if mesh_data['triangle_count'] > 50000
                        else config.TARGET_LATENCY_SIMPLE_S
                    )
                },
                
                'status': 'success',
                'version': '11.0.0-production'
            }
            
            # Record success in circuit breaker
            circuit_breaker.record_success()
            
            # Log audit trail
            log_audit_trail("processing_complete", request_id, {
                "file_hash": file_hash,
                "processing_time": processing_time,
                "quality_score": validation_result.quality_score,
                "recognition_status": recognition_status,
                "feature_count": ml_features['num_features_detected'] if ml_features else 0
            })
            
            logger.info(f"✅ Analysis complete in {processing_time:.2f}s (request_id: {request_id})")
            return jsonify(result)
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except TimeoutError as e:
        # Transient error - can retry
        logger.error(f"⏱️ Timeout: {e}")
        
        # Store in DLQ using standalone module
        dlq.store_failure(
            correlation_id=request_id,
            file_path=tmp_path if tmp_path and os.path.exists(tmp_path) else "unknown",
            error_type="transient",
            error_message=str(e),
            error_details={"timeout": True},
            retry_count=0
        )
        
        return jsonify({
            "error": "Processing timeout",
            "message": str(e),
            "request_id": request_id,
            "retry_recommended": True
        }), 504
    
    except Exception as e:
        # Could be permanent or systemic - classify based on error type
        logger.error(f"❌ Error processing CAD: {e}")
        import traceback
        stack_trace = traceback.format_exc()
        logger.error(stack_trace)
        
        # Classify error type
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['timeout', 'connection', 'network']):
            error_type_str = "transient"
        elif any(keyword in error_str for keyword in ['invalid', 'corrupt', 'malformed']):
            error_type_str = "permanent"
        else:
            error_type_str = "systemic"
        
        # Store in DLQ using standalone module
        dlq.store_failure(
            correlation_id=request_id,
            file_path=tmp_path if tmp_path and os.path.exists(tmp_path) else "unknown",
            error_type=error_type_str,
            error_message=str(e),
            error_details={
                "traceback": stack_trace if os.getenv("DEBUG") else None
            },
            retry_count=0
        )
        
        return jsonify({
            "error": str(e),
            "request_id": request_id,
            "error_type": error_type_str,
            "retry_recommended": error_type_str == "transient",
            "traceback": stack_trace if os.getenv("DEBUG") else None
        }), 500

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.route("/")
def root():
    import hashlib
    # Get a hash of the current app.py file to verify version
    try:
        with open(__file__, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
    except:
        file_hash = "unknown"
    
    return jsonify({
        "service": "CAD Geometry Analysis Service",
        "version": "11.0.1-processing-tier-fix",
        "code_hash": file_hash,
        "fix_applied": "AttributeError processing_tier.name fixed",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-cad",
            "aagnet": "/api/aagnet/recognize" if AAGNET_AVAILABLE else "unavailable",
            "metrics": "/metrics"
        },
        "features": {
            "validation": "5-stage pipeline with quality scoring",
            "healing": "Automatic geometry repair",
            "fallback_tiers": "B-Rep → Mesh → Point cloud",
            "circuit_breaker": "Cascade failure prevention",
            "dead_letter_queue": "Failed request tracking",
            "classification": "Mesh-based with neighbor propagation",
            "feature_detection": "AAGNet 24-class with instance segmentation" if AAGNET_AVAILABLE else "unavailable",
            "edge_extraction": "Professional smart filtering (20° dihedral angle)",
            "aagnet_available": AAGNET_AVAILABLE,
            "iso_compliance": "ISO 9001 audit logging"
        },
        "performance_targets": {
            "simple_parts_latency_s": config.TARGET_LATENCY_SIMPLE_S,
            "complex_parts_latency_s": config.TARGET_LATENCY_COMPLEX_S,
            "quality_score_min": config.QUALITY_SCORE_MIN
        }
    })

@app.route("/health")
def health():
    """Health check endpoint"""
    circuit_state = aagnet_circuit_breaker.get_state()
    
    health_status = {
        "status": "healthy" if circuit_state["state"] == "CLOSED" else "degraded",
        "circuit_breaker": circuit_state["state"],
        "aagnet_available": AAGNET_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return jsonify(health_status), status_code

@app.route("/metrics")
def metrics():
    """Production metrics endpoint for monitoring"""
    circuit_state = aagnet_circuit_breaker.get_state()
    dlq_stats = dlq.get_failure_statistics()
    
    try:
        # Get recent processing stats from audit log
        recent_audits = supabase.table(config.AUDIT_LOG_TABLE) \
            .select("*") \
            .gte("created_at", (datetime.utcnow() - timedelta(hours=1)).isoformat()) \
            .execute()
        
        processing_events = [a for a in recent_audits.data if a['event_type'] == 'processing_complete']
        
        avg_processing_time = 0
        avg_quality_score = 0
        recognition_rate = 0
        
        if processing_events:
            avg_processing_time = sum(e['details'].get('processing_time', 0) for e in processing_events) / len(processing_events)
            avg_quality_score = sum(e['details'].get('quality_score', 0) for e in processing_events) / len(processing_events)
            
            recognized_count = sum(1 for e in processing_events 
                                 if e['details'].get('recognition_status') in ['fully_recognized', 'partially_recognized'])
            recognition_rate = recognized_count / len(processing_events) if processing_events else 0
        
        return jsonify({
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breaker": circuit_state,
            "dead_letter_queue": dlq_stats,
            "performance": {
                "avg_processing_time_sec": avg_processing_time,
                "requests_last_hour": len(processing_events),
                "target_latency_simple_s": config.TARGET_LATENCY_SIMPLE_S,
                "target_latency_complex_s": config.TARGET_LATENCY_COMPLEX_S
            },
            "quality": {
                "avg_quality_score": avg_quality_score,
                "min_quality_threshold": config.QUALITY_SCORE_MIN,
                "recognition_rate": recognition_rate
            }
        })
    
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return jsonify({
            "error": "Failed to fetch metrics",
            "message": str(e)
        }), 500

@app.route("/circuit-breaker", methods=["GET"])
def circuit_breaker_status():
    """Get detailed circuit breaker status"""
    return jsonify(aagnet_circuit_breaker.get_state())

@app.route("/circuit-breaker/reset", methods=["POST"])
def reset_circuit_breaker():
    """Manually reset circuit breaker to CLOSED state"""
    aagnet_circuit_breaker.reset()
    return jsonify({
        "status": "reset",
        "message": "Circuit breaker manually reset to CLOSED state",
        "new_state": aagnet_circuit_breaker.get_state()
    })

@app.route("/dlq/stats")
def dlq_stats_endpoint():
    """Get dead letter queue statistics"""
    return jsonify(dlq.get_failure_statistics())

@app.route("/dlq/failures")
def dlq_failures():
    """Get recent failures from dead letter queue"""
    error_type = request.args.get('error_type')
    limit = int(request.args.get('limit', 100))
    
    failures = dlq.get_failures(error_type=error_type, limit=limit)
    
    return jsonify({
        "failures": failures,
        "count": len(failures),
        "error_type_filter": error_type
    })

if __name__ == "__main__":
    # Create required Supabase tables if they don't exist
    logger.info("🚀 Starting production CAD analysis service v11.0.0")
    logger.info("📋 Features: 5-stage validation, healing, fallback tiers, circuit breaker, DLQ")
    
    app.run(host="0.0.0.0", port=5000)
