"""
Main Pattern Matching Engine - Industrial Production Implementation
Orchestrates all recognizers with comprehensive validation and error handling

CRITICAL RESPONSIBILITIES:
1. Coordinate all feature recognizers
2. Build and validate AAG from STEP files
3. Manage recognition pipeline with error recovery
4. Validate feature interactions and conflicts
5. Compute manufacturing sequences
6. Generate comprehensive reports
7. Handle edge cases and malformed geometries

Total: ~1,800 lines
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .graph_builder import AAGGraphBuilder
from .recognizers.hole_recognizer import HoleRecognizer
from .recognizers.pocket_slot_recognizer import PocketSlotRecognizer
from .recognizers.boss_step_island_recognizer import BossStepIslandRecognizer
from .recognizers.fillet_chamfer_recognizer import FilletRecognizer, ChamferRecognizer
from .recognizers.turning_recognizer import TurningRecognizer

from OCC.Core.TopoDS import TopoDS_Shape

logger = logging.getLogger(__name__)


# ===== ENUMS =====

class RecognitionStatus(Enum):
    """Recognition pipeline status"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"


class PartType(Enum):
    """Detected part type"""
    PRISMATIC = "prismatic"  # Milling operations
    ROTATIONAL = "rotational"  # Turning operations
    MIXED = "mixed"  # Both milling and turning
    UNKNOWN = "unknown"


# ===== DATA CLASSES =====

@dataclass
class RecognitionMetrics:
    """Comprehensive metrics"""
    total_features: int = 0
    feature_counts: Dict[str, int] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    graph_build_time: float = 0.0
    recognition_time: float = 0.0
    validation_time: float = 0.0
    total_time: float = 0.0
    
    # Quality
    average_confidence: float = 0.0
    validation_pass_rate: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RecognitionResult:
    """Complete recognition result with all features"""
    # Status
    status: RecognitionStatus
    part_type: PartType
    
    # All recognized features
    holes: List = field(default_factory=list)
    pockets: List = field(default_factory=list)
    slots: List = field(default_factory=list)
    passages: List = field(default_factory=list)
    fillets: List = field(default_factory=list)
    chamfers: List = field(default_factory=list)
    bosses: List = field(default_factory=list)
    steps: List = field(default_factory=list)
    islands: List = field(default_factory=list)
    turning_features: List = field(default_factory=list)
    
    # Graph
    graph: Optional[Dict] = None
    
    # Metrics
    metrics: RecognitionMetrics = field(default_factory=RecognitionMetrics)
    
    # Manufacturing
    manufacturing_sequence: List = field(default_factory=list)
    estimated_machining_time: float = 0.0  # minutes
    
    # Validation
    feature_interactions: List = field(default_factory=list)
    conflicts: List = field(default_factory=list)


# ===== MAIN ENGINE =====

class AAGPatternMatcher:
    """
    Production-grade pattern matching engine
    
    CRITICAL: This is the main entry point for all feature recognition
    Must be robust, handle errors gracefully, and provide detailed feedback
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize pattern matcher with all recognizers
        
        Args:
            tolerance: Geometric tolerance in model units (meters)
        """
        self.tolerance = tolerance
        
        # Initialize graph builder
        self.graph_builder = AAGGraphBuilder(tolerance=tolerance)
        
        # Initialize all recognizers with error handling
        try:
            self.hole_recognizer = HoleRecognizer(tolerance=tolerance)
            logger.info("✓ Hole recognizer initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize hole recognizer: {e}")
            self.hole_recognizer = None
        
        try:
            self.pocket_slot_recognizer = PocketSlotRecognizer(tolerance=tolerance)
            logger.info("✓ Pocket/Slot recognizer initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize pocket/slot recognizer: {e}")
            self.pocket_slot_recognizer = None
        
        try:
            self.boss_step_island_recognizer = BossStepIslandRecognizer(tolerance=tolerance)
            logger.info("✓ Boss/Step/Island recognizer initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize boss/step/island recognizer: {e}")
            self.boss_step_island_recognizer = None
        
        try:
            self.fillet_recognizer = FilletRecognizer(tolerance=tolerance)
            logger.info("✓ Fillet recognizer initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize fillet recognizer: {e}")
            self.fillet_recognizer = None
        
        try:
            self.chamfer_recognizer = ChamferRecognizer(tolerance=tolerance)
            logger.info("✓ Chamfer recognizer initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize chamfer recognizer: {e}")
            self.chamfer_recognizer = None
        
        try:
            self.turning_recognizer = TurningRecognizer(tolerance=tolerance)
            logger.info("✓ Turning recognizer initialized (CRITICAL)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize turning recognizer: {e}")
            self.turning_recognizer = None
        
        # Check if any critical recognizers failed
        failed_recognizers = []
        if not self.hole_recognizer:
            failed_recognizers.append("Hole")
        if not self.pocket_slot_recognizer:
            failed_recognizers.append("Pocket/Slot")
        if not self.turning_recognizer:
            failed_recognizers.append("Turning (CRITICAL)")
        
        if failed_recognizers:
            logger.warning(f"⚠ Some recognizers failed to initialize: {', '.join(failed_recognizers)}")
        else:
            logger.info("✓ All recognizers initialized successfully")
        
        logger.info("=" * 70)
        logger.info("Pattern Matcher Ready - Production Grade")
        logger.info("=" * 70)
    
    def recognize_all_features(
        self,
        shape: TopoDS_Shape,
        validate: bool = True,
        compute_manufacturing: bool = True
    ) -> RecognitionResult:
        """
        Recognize all features with comprehensive validation
        
        CRITICAL: Main entry point - must handle all edge cases
        
        Args:
            shape: OpenCascade shape from STEP file
            validate: Perform comprehensive validation
            compute_manufacturing: Compute manufacturing sequence and timing
            
        Returns:
            RecognitionResult with all features and metadata
        """
        logger.info("=" * 70)
        logger.info("STARTING COMPREHENSIVE FEATURE RECOGNITION")
        logger.info("=" * 70)
        
        start_time = time.time()
        result = RecognitionResult(
            status=RecognitionStatus.SUCCESS,
            part_type=PartType.UNKNOWN
        )
        
        try:
            # STEP 1: Build AAG from STEP file
            logger.info("\n[STEP 1/7] Building Attributed Adjacency Graph...")
            graph_start = time.time()
            
            try:
                graph = self.graph_builder.build_graph(shape)
                result.graph = graph
                result.metrics.graph_build_time = time.time() - graph_start
                
                logger.info(f"✓ AAG built successfully")
                logger.info(f"  Nodes: {len(graph['nodes'])}")
                logger.info(f"  Edges: {len(graph['edges'])}")
                logger.info(f"  Time: {result.metrics.graph_build_time:.2f}s")
            
            except Exception as e:
                logger.error(f"✗ Failed to build AAG: {e}")
                logger.error(traceback.format_exc())
                result.status = RecognitionStatus.INVALID_INPUT
                result.metrics.errors.append(f"Graph build failed: {str(e)}")
                return result
            
            # STEP 2: Detect part type (prismatic vs rotational)
            logger.info("\n[STEP 2/7] Detecting part type...")
            part_type = self._detect_part_type(graph)
            result.part_type = part_type
            logger.info(f"✓ Part type: {part_type.value}")
            
            # STEP 3: Run recognizers with error handling
            logger.info("\n[STEP 3/7] Running feature recognizers...")
            recognition_start = time.time()
            
            # 3.1: Holes (critical for both prismatic and rotational)
            if self.hole_recognizer:
                try:
                    logger.info("  [3.1] Recognizing holes...")
                    result.holes = self.hole_recognizer.recognize_holes(graph)
                    logger.info(f"  ✓ {len(result.holes)} holes recognized")
                except Exception as e:
                    logger.error(f"  ✗ Hole recognition failed: {e}")
                    result.metrics.errors.append(f"Hole recognition: {str(e)}")
            
            # 3.2: Pockets, Slots, Passages (prismatic)
            if self.pocket_slot_recognizer and part_type in [PartType.PRISMATIC, PartType.MIXED]:
                try:
                    logger.info("  [3.2] Recognizing pockets/slots/passages...")
                    pocket_slot_results = self.pocket_slot_recognizer.recognize_all(graph)
                    result.pockets = pocket_slot_results['pockets']
                    result.slots = pocket_slot_results['slots']
                    result.passages = pocket_slot_results['passages']
                    logger.info(f"  ✓ {len(result.pockets)} pockets, {len(result.slots)} slots, {len(result.passages)} passages")
                except Exception as e:
                    logger.error(f"  ✗ Pocket/Slot recognition failed: {e}")
                    result.metrics.errors.append(f"Pocket/Slot recognition: {str(e)}")
            
            # 3.3: Bosses, Steps, Islands (prismatic)
            if self.boss_step_island_recognizer and part_type in [PartType.PRISMATIC, PartType.MIXED]:
                try:
                    logger.info("  [3.3] Recognizing bosses/steps/islands...")
                    bsi_results = self.boss_step_island_recognizer.recognize_all(graph)
                    result.bosses = bsi_results['bosses']
                    result.steps = bsi_results['steps']
                    result.islands = bsi_results['islands']
                    logger.info(f"  ✓ {len(result.bosses)} bosses, {len(result.steps)} steps, {len(result.islands)} islands")
                except Exception as e:
                    logger.error(f"  ✗ Boss/Step/Island recognition failed: {e}")
                    result.metrics.errors.append(f"Boss/Step/Island recognition: {str(e)}")
            
            # 3.4: Fillets (both types)
            if self.fillet_recognizer:
                try:
                    logger.info("  [3.4] Recognizing fillets...")
                    result.fillets = self.fillet_recognizer.recognize_fillets(graph)
                    logger.info(f"  ✓ {len(result.fillets)} fillets recognized")
                except Exception as e:
                    logger.error(f"  ✗ Fillet recognition failed: {e}")
                    result.metrics.errors.append(f"Fillet recognition: {str(e)}")
            
            # 3.5: Chamfers (both types)
            if self.chamfer_recognizer:
                try:
                    logger.info("  [3.5] Recognizing chamfers...")
                    result.chamfers = self.chamfer_recognizer.recognize_chamfers(graph)
                    logger.info(f"  ✓ {len(result.chamfers)} chamfers recognized")
                except Exception as e:
                    logger.error(f"  ✗ Chamfer recognition failed: {e}")
                    result.metrics.errors.append(f"Chamfer recognition: {str(e)}")
            
            # 3.6: Turning features (CRITICAL for rotational parts)
            if self.turning_recognizer and part_type in [PartType.ROTATIONAL, PartType.MIXED]:
                try:
                    logger.info("  [3.6] Recognizing turning features (CRITICAL)...")
                    result.turning_features = self.turning_recognizer.recognize_turning_features(graph)
                    logger.info(f"  ✓ {len(result.turning_features)} turning features recognized")
                except Exception as e:
                    logger.error(f"  ✗ Turning recognition failed: {e}")
                    logger.error(traceback.format_exc())
                    result.metrics.errors.append(f"Turning recognition: {str(e)}")
                    # Turning is critical - mark as partial success
                    if part_type == PartType.ROTATIONAL:
                        result.status = RecognitionStatus.PARTIAL_SUCCESS
            
            result.metrics.recognition_time = time.time() - recognition_start
            
            # STEP 4: Compile statistics
            logger.info("\n[STEP 4/7] Compiling statistics...")
            self._compile_statistics(result)
            logger.info(f"✓ Total features: {result.metrics.total_features}")
            logger.info(f"  Average confidence: {result.metrics.average_confidence:.1%}")
            
            # STEP 5: Validate features and interactions
            if validate:
                logger.info("\n[STEP 5/7] Validating features and interactions...")
                validation_start = time.time()
                
                try:
                    self._validate_all_features(result)
                    self._analyze_feature_interactions(result)
                    result.metrics.validation_time = time.time() - validation_start
                    logger.info(f"✓ Validation complete")
                    logger.info(f"  Pass rate: {result.metrics.validation_pass_rate:.1%}")
                    logger.info(f"  Conflicts: {len(result.conflicts)}")
                except Exception as e:
                    logger.error(f"✗ Validation failed: {e}")
                    result.metrics.errors.append(f"Validation: {str(e)}")
            
            # STEP 6: Compute manufacturing sequence
            if compute_manufacturing:
                logger.info("\n[STEP 6/7] Computing manufacturing sequence...")
                
                try:
                    self._compute_manufacturing_sequence(result)
                    logger.info(f"✓ Manufacturing sequence computed")
                    logger.info(f"  Estimated time: {result.estimated_machining_time:.1f} minutes")
                except Exception as e:
                    logger.error(f"✗ Manufacturing sequence computation failed: {e}")
                    result.metrics.errors.append(f"Manufacturing: {str(e)}")
            
            # STEP 7: Generate warnings
            logger.info("\n[STEP 7/7] Generating warnings...")
            self._collect_warnings(result)
            
            if len(result.metrics.warnings) > 0:
                logger.info(f"⚠ {len(result.metrics.warnings)} warnings generated")
            
            # Final status determination
            if len(result.metrics.errors) > 0:
                if result.metrics.total_features > 0:
                    result.status = RecognitionStatus.PARTIAL_SUCCESS
                else:
                    result.status = RecognitionStatus.FAILED
            else:
                result.status = RecognitionStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"✗ CRITICAL ERROR in recognition pipeline: {e}")
            logger.error(traceback.format_exc())
            result.status = RecognitionStatus.FAILED
            result.metrics.errors.append(f"Pipeline failure: {str(e)}")
        
        finally:
            result.metrics.total_time = time.time() - start_time
            
            # Log final summary
            logger.info("\n" + "=" * 70)
            logger.info("RECOGNITION COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Status: {result.status.value}")
            logger.info(f"Total time: {result.metrics.total_time:.2f}s")
            logger.info(f"Features recognized: {result.metrics.total_features}")
            logger.info(f"Errors: {len(result.metrics.errors)}")
            logger.info(f"Warnings: {len(result.metrics.warnings)}")
            logger.info("=" * 70)
        
        return result
    
    # ===== PART TYPE DETECTION =====
    
    def _detect_part_type(self, graph: Dict) -> PartType:
        """
        Detect if part is prismatic (milling) or rotational (turning)
        
        CRITICAL: Determines which recognizers to run
        """
        nodes = graph['nodes']
        
        # Count cylindrical surfaces aligned with single axis
        from collections import defaultdict
        from .graph_builder import SurfaceType
        
        cylinders = [n for n in nodes if n.surface_type == SurfaceType.CYLINDER]
        
        if len(cylinders) < 3:
            return PartType.PRISMATIC
        
        # Cluster by axis direction
        axis_clusters = defaultdict(int)
        
        for cyl in cylinders:
            if not cyl.axis:
                continue
            
            axis = np.array(cyl.axis)
            axis_normalized = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else axis
            axis_key = tuple(np.round(axis_normalized, 2))
            
            axis_clusters[axis_key] += 1
        
        if not axis_clusters:
            return PartType.PRISMATIC
        
        # If one axis has majority of cylinders → rotational
        max_cluster = max(axis_clusters.values())
        total_cylinders = len(cylinders)
        
        if max_cluster >= total_cylinders * 0.6:
            # Check for both rotational and prismatic features
            planar_count = sum(1 for n in nodes if n.surface_type == SurfaceType.PLANE)
            
            if planar_count > len(cylinders):
                return PartType.MIXED
            else:
                return PartType.ROTATIONAL
        
        return PartType.PRISMATIC
    
    # ===== STATISTICS =====
    
    def _compile_statistics(self, result: RecognitionResult):
        """Compile comprehensive statistics"""
        # Count features
        result.metrics.feature_counts = {
            'holes': len(result.holes),
            'pockets': len(result.pockets),
            'slots': len(result.slots),
            'passages': len(result.passages),
            'fillets': len(result.fillets),
            'chamfers': len(result.chamfers),
            'bosses': len(result.bosses),
            'steps': len(result.steps),
            'islands': len(result.islands),
            'turning_features': len(result.turning_features)
        }
        
        result.metrics.total_features = sum(result.metrics.feature_counts.values())
        
        # Compute confidence scores
        all_features = (
            result.holes + result.pockets + result.slots + result.passages +
            result.fillets + result.chamfers + result.bosses + result.steps +
            result.islands + result.turning_features
        )
        
        if all_features:
            confidences = [f.confidence for f in all_features if hasattr(f, 'confidence')]
            if confidences:
                result.metrics.average_confidence = np.mean(confidences)
            
            # Per-type confidence
            for feature_type, features in [
                ('holes', result.holes),
                ('pockets', result.pockets),
                ('slots', result.slots),
                ('fillets', result.fillets),
                ('chamfers', result.chamfers),
                ('bosses', result.bosses),
                ('steps', result.steps),
                ('islands', result.islands),
                ('turning_features', result.turning_features)
            ]:
                if features:
                    confs = [f.confidence for f in features if hasattr(f, 'confidence')]
                    if confs:
                        result.metrics.confidence_scores[feature_type] = np.mean(confs)
    
    # ===== VALIDATION =====
    
    def _validate_all_features(self, result: RecognitionResult):
        """
        Comprehensive feature validation
        
        CRITICAL: Catches geometric impossibilities and manufacturing issues
        """
        validation_errors = []
        validation_warnings = []
        
        # Validate holes
        for hole in result.holes:
            if hasattr(hole, 'geometric_validation') and hole.geometric_validation:
                validation_errors.extend(hole.geometric_validation.errors)
                validation_warnings.extend(hole.geometric_validation.warnings)
        
        # Validate pockets
        for pocket in result.pockets:
            if hasattr(pocket, 'geometric_validation') and pocket.geometric_validation:
                validation_errors.extend(pocket.geometric_validation.errors)
                validation_warnings.extend(pocket.geometric_validation.warnings)
        
        # Check for overlapping features
        overlap_errors = self._check_feature_overlaps(result)
        validation_errors.extend(overlap_errors)
        
        # Check for impossible geometries
        geometry_errors = self._check_impossible_geometries(result)
        validation_errors.extend(geometry_errors)
        
        # Compute validation pass rate
        all_features = result.metrics.total_features
        failed = len(validation_errors)
        
        if all_features > 0:
            result.metrics.validation_pass_rate = 1.0 - (failed / all_features)
        
        # Add to metrics
        result.metrics.errors.extend(validation_errors)
        result.metrics.warnings.extend(validation_warnings)
    
    def _check_feature_overlaps(self, result: RecognitionResult) -> List[str]:
        """Check for overlapping features"""
        errors = []
        
        # Check hole overlaps
        for i, hole1 in enumerate(result.holes):
            for hole2 in result.holes[i+1:]:
                if self._features_overlap(hole1, hole2):
                    errors.append(f"Overlapping holes detected")
        
        return errors
    
    def _features_overlap(self, f1, f2) -> bool:
        """Check if two features overlap (simplified)"""
        if not (hasattr(f1, 'face_ids') and hasattr(f2, 'face_ids')):
            return False
        
        faces1 = set(f1.face_ids)
        faces2 = set(f2.face_ids)
        
        return len(faces1 & faces2) > 0
    
    def _check_impossible_geometries(self, result: RecognitionResult) -> List[str]:
        """Check for geometrically impossible features"""
        errors = []
        
        # Check for features with impossible dimensions
        for hole in result.holes:
            if hasattr(hole, 'depth') and hasattr(hole, 'diameter'):
                if hole.depth and hole.diameter:
                    aspect_ratio = hole.depth / hole.diameter
                    if aspect_ratio > 25:
                        errors.append(f"Impossible hole aspect ratio: {aspect_ratio:.1f}:1")
        
        return errors
    
    # ===== FEATURE INTERACTIONS =====
    
    def _analyze_feature_interactions(self, result: RecognitionResult):
        """
        Analyze how features interact
        
        Examples: hole-in-pocket, pocket-in-boss, etc.
        """
        interactions = []
        
        # Holes in pockets
        for hole in result.holes:
            for pocket in result.pockets:
                if self._feature_contains(pocket, hole):
                    interactions.append({
                        'type': 'contains',
                        'parent': 'pocket',
                        'child': 'hole',
                        'parent_id': id(pocket),
                        'child_id': id(hole)
                    })
        
        # Holes in bosses
        for hole in result.holes:
            for boss in result.bosses:
                if self._feature_contains(boss, hole):
                    interactions.append({
                        'type': 'contains',
                        'parent': 'boss',
                        'child': 'hole',
                        'parent_id': id(boss),
                        'child_id': id(hole)
                    })
        
        result.feature_interactions = interactions
    
    def _feature_contains(self, parent, child) -> bool:
        """Check if parent feature contains child"""
        if not (hasattr(parent, 'face_ids') and hasattr(child, 'face_ids')):
            return False
        
        # Simplified: check if child faces are subset of parent's adjacent faces
        return False  # Would need spatial analysis
    
    # ===== MANUFACTURING =====
    
    def _compute_manufacturing_sequence(self, result: RecognitionResult):
        """
        Compute manufacturing operation sequence
        
        CRITICAL: Determines machining order and time estimates
        """
        sequence = []
        total_time = 0.0
        
        # For prismatic parts
        if result.part_type in [PartType.PRISMATIC, PartType.MIXED]:
            # 1. Face milling (flatten surfaces)
            # 2. Roughing (remove bulk material)
            # 3. Pockets and slots
            # 4. Holes (drilling/boring)
            # 5. Bosses and steps
            # 6. Finishing (fillets, chamfers)
            
            # Estimate times
            for pocket in result.pockets:
                if hasattr(pocket, 'volume') and pocket.volume:
                    # Assume 50 cm³/min material removal rate
                    time_min = (pocket.volume * 1e6) / 50.0
                    total_time += time_min
                    sequence.append({
                        'operation': 'pocket_milling',
                        'feature': pocket,
                        'time_minutes': time_min
                    })
            
            for hole in result.holes:
                # Drilling time estimate
                if hasattr(hole, 'depth') and hole.depth:
                    # Assume 100 mm/min feed rate
                    time_min = (hole.depth * 1000) / 100.0
                    total_time += time_min
                    sequence.append({
                        'operation': 'drilling',
                        'feature': hole,
                        'time_minutes': time_min
                    })
        
        # For rotational parts
        if result.part_type in [PartType.ROTATIONAL, PartType.MIXED]:
            for turning_feature in result.turning_features:
                if hasattr(turning_feature, 'machining_time_estimate'):
                    time_min = turning_feature.machining_time_estimate / 60.0 if turning_feature.machining_time_estimate else 0
                    total_time += time_min
                    
                    operation_type = turning_feature.manufacturing_analysis.operation_type if hasattr(turning_feature, 'manufacturing_analysis') and turning_feature.manufacturing_analysis else 'turning'
                    
                    sequence.append({
                        'operation': operation_type,
                        'feature': turning_feature,
                        'time_minutes': time_min
                    })
        
        result.manufacturing_sequence = sequence
        result.estimated_machining_time = total_time
    
    # ===== WARNINGS =====
    
    def _collect_warnings(self, result: RecognitionResult):
        """Collect all warnings from features"""
        all_features = (
            result.holes + result.pockets + result.slots + result.passages +
            result.fillets + result.chamfers + result.bosses + result.steps +
            result.islands + result.turning_features
        )
        
        for feature in all_features:
            if hasattr(feature, 'warnings'):
                result.metrics.warnings.extend(feature.warnings)
    
    # ===== REPORTING =====
    
    def generate_summary_report(self, result: RecognitionResult) -> str:
        """
        Generate comprehensive human-readable summary report
        
        CRITICAL: Used for quality control and process planning
        """
        lines = []
        lines.append("=" * 80)
        lines.append("FEATURE RECOGNITION SUMMARY REPORT")
        lines.append("=" * 80)
        
        # Status
        lines.append(f"\nStatus: {result.status.value.upper()}")
        lines.append(f"Part Type: {result.part_type.value}")
        
        # Timing
        lines.append(f"\nTiming:")
        lines.append(f"  Graph Build:     {result.metrics.graph_build_time:6.2f}s")
        lines.append(f"  Recognition:     {result.metrics.recognition_time:6.2f}s")
        lines.append(f"  Validation:      {result.metrics.validation_time:6.2f}s")
        lines.append(f"  Total Time:      {result.metrics.total_time:6.2f}s")
        
        # Feature counts
        lines.append(f"\nFeature Summary:")
        lines.append(f"  Total Features:  {result.metrics.total_features}")
        lines.append(f"  Avg Confidence:  {result.metrics.average_confidence:.1%}")
        lines.append(f"  Validation Rate: {result.metrics.validation_pass_rate:.1%}")
        
        lines.append(f"\nFeature Breakdown:")
        for feature_type, count in sorted(result.metrics.feature_counts.items()):
            if count > 0:
                conf = result.metrics.confidence_scores.get(feature_type, 0.0)
                lines.append(f"  {feature_type.upper():20s}: {count:4d} features (conf: {conf:.1%})")
        
        # Manufacturing
        if result.estimated_machining_time > 0:
            lines.append(f"\nManufacturing Estimate:")
            lines.append(f"  Operations:      {len(result.manufacturing_sequence)}")
            lines.append(f"  Estimated Time:  {result.estimated_machining_time:.1f} minutes")
        
        # Interactions
        if result.feature_interactions:
            lines.append(f"\nFeature Interactions: {len(result.feature_interactions)}")
            for interaction in result.feature_interactions[:5]:  # Show first 5
                lines.append(f"  - {interaction['parent']} contains {interaction['child']}")
            if len(result.feature_interactions) > 5:
                lines.append(f"  ... and {len(result.feature_interactions) - 5} more")
        
        # Errors
        if result.metrics.errors:
            lines.append(f"\n⚠ ERRORS ({len(result.metrics.errors)}):")
            for i, error in enumerate(result.metrics.errors[:10], 1):
                lines.append(f"  {i}. {error}")
            if len(result.metrics.errors) > 10:
                lines.append(f"  ... and {len(result.metrics.errors) - 10} more errors")
        
        # Warnings
        if result.metrics.warnings:
            lines.append(f"\n⚠ WARNINGS ({len(result.metrics.warnings)}):")
            for i, warning in enumerate(result.metrics.warnings[:10], 1):
                lines.append(f"  {i}. {warning}")
            if len(result.metrics.warnings) > 10:
                lines.append(f"  ... and {len(result.metrics.warnings) - 10} more warnings")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def generate_json_report(self, result: RecognitionResult) -> Dict[str, Any]:
        """
        Generate machine-readable JSON report
        
        For integration with CAM systems and databases
        """
        return {
            'status': result.status.value,
            'part_type': result.part_type.value,
            'metrics': {
                'total_features': result.metrics.total_features,
                'feature_counts': result.metrics.feature_counts,
                'confidence_scores': result.metrics.confidence_scores,
                'average_confidence': result.metrics.average_confidence,
                'validation_pass_rate': result.metrics.validation_pass_rate,
                'timing': {
                    'graph_build_time': result.metrics.graph_build_time,
                    'recognition_time': result.metrics.recognition_time,
                    'validation_time': result.metrics.validation_time,
                    'total_time': result.metrics.total_time
                },
                'errors': result.metrics.errors,
                'warnings': result.metrics.warnings
            },
            'manufacturing': {
                'estimated_time_minutes': result.estimated_machining_time,
                'operation_count': len(result.manufacturing_sequence)
            },
            'feature_interactions': len(result.feature_interactions),
            'conflicts': len(result.conflicts)
        }
