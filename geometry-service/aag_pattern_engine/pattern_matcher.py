"""
AAG Pattern Matcher - Analysis Situs Workflow
=============================================

CORRECTED VERSION - Fixed imports to match flat file structure

NEW WORKFLOW:
1. Volume decomposition (single removal volume)
2. Build AAG graph
3. Detect machining configurations (not split volumes!)
4. Run recognizers per configuration
5. Add tool accessibility
6. Merge results

OLD WORKFLOW (DEPRECATED):
1. Split removal into N volumes
2. Build AAG per volume
3. Run recognizers per volume
4. Merge
"""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

# CORRECTED IMPORTS - Fixed relative paths for package structure
from volume_decomposer import VolumeDecomposer  # Parent directory (geometry-service/)
from .graph_builder import AAGGraphBuilder, SurfaceType, Vexity, GraphNode, GraphEdge
from .machining_configuration_detector import MachiningConfigurationDetector, detect_machining_configurations
from .recognizers.hole_recognizer import HoleRecognizer
from .recognizers.pocket_recognizer import PocketRecognizer
from .recognizers.boss_step_island_recognizer import BossRecognizer
from .tool_accessibility_analyzer import ToolAccessibilityAnalyzer

# Optional legacy recognizers with standardized API adapters
# The adapters provide a unified interface for recognizers with different APIs
try:
    from .recognizers.recognizer_api_adapters import (
        StandardizedSlotRecognizer as SlotRecognizer,
        StandardizedFilletRecognizer as FilletRecognizer,
        StandardizedChamferRecognizer as ChamferRecognizer
    )
    HAS_SLOT_RECOGNIZER = True
    HAS_FILLET_RECOGNIZER = True
    # Logger not available at import time - will log during actual recognition
except ImportError as e:
    # Fallback to original recognizers (may have API mismatches)
    try:
        from .recognizers.slot_recognizer import SlotRecognizer
        HAS_SLOT_RECOGNIZER = True
    except ImportError:
        HAS_SLOT_RECOGNIZER = False
    try:
        from .recognizers.fillet_chamfer_recognizer import FilletRecognizer, ChamferRecognizer
        HAS_FILLET_RECOGNIZER = True
    except ImportError:
        HAS_FILLET_RECOGNIZER = False
    
try:
    from .recognizers.turning_recognizer import TurningRecognizer
    HAS_TURNING_RECOGNIZER = True
except ImportError:
    HAS_TURNING_RECOGNIZER = False

# DFM analysis and output standardization
try:
    import sys
    import os
    # Add parent directory to path for dfm_analyzer import
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from dfm_analyzer import DFMAnalyzer, generate_dfm_summary, format_dfm_warnings_for_output
    HAS_DFM_ANALYZER = True
except ImportError:
    HAS_DFM_ANALYZER = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠ DFM analyzer not available")

# Recognizer output utilities
try:
    from .recognizers.recognizer_utils import (
        standardize_features_list,
        convert_to_analysis_situs_format,
        merge_duplicate_face_ids
    )
    HAS_RECOGNIZER_UTILS = True
except ImportError:
    HAS_RECOGNIZER_UTILS = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠ Recognizer utilities not available - output format may not match Analysis Situs")

logger = logging.getLogger(__name__)


# ============================================================================
# ARCHITECTURAL NOTE: AAG Graph Object Structure
# ============================================================================
# AAGGraphBuilder.build() returns a dict: {'nodes': {}, 'adjacency': {}, 'statistics': {}}
# BUT recognizers and detectors expect the builder OBJECT itself (not the dict)
# because they access: builder.nodes, builder.adjacency, builder.get_adjacent_faces()
#
# Pattern in pattern_matcher.py:
# - Store both: aag_data = {'graph': dict_result, 'builder': builder_object}
# - Pass to detectors/recognizers: builder_object (has attributes/methods)
# - Use dict for: statistics, logging, serialization
# ============================================================================


# ============================================================================
# DATACLASS DEFINITIONS - Recognition API Types
# ============================================================================

class RecognitionStatus(Enum):
    """Feature recognition execution status"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"
    ERROR = "error"


class PartType(Enum):
    """Part manufacturing type classification"""
    PRISMATIC = "prismatic"           # CNC milling
    ROTATIONAL = "rotational"         # CNC turning/lathe
    HYBRID = "hybrid"                 # Turn-mill
    SHEET_METAL = "sheet_metal"       # Bending/forming
    FREEFORM = "freeform"             # 5-axis/complex
    UNKNOWN = "unknown"


@dataclass
class RecognitionMetrics:
    """Comprehensive recognition metrics"""
    total_features: int = 0
    feature_counts: Dict[str, int] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    average_confidence: float = 0.0  # ✅ ADDED: Average confidence across all features
    
    # Timing breakdown
    decomposition_time: float = 0.0      # Volume decomposition
    graph_build_time: float = 0.0        # AAG construction
    recognition_time: float = 0.0        # Pattern matching
    validation_time: float = 0.0         # Result validation
    total_time: float = 0.0              # End-to-end
    
    # Quality metrics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Graph statistics
    graph_nodes: int = 0
    graph_edges: int = 0


@dataclass
class RecognitionResult:
    """
    Complete feature recognition result
    
    This is the main return type for AAG pattern matching
    """
    # Status
    status: RecognitionStatus
    part_type: PartType
    
    # Recognized features (lists of feature dictionaries)
    holes: List = field(default_factory=list)
    pockets: List = field(default_factory=list)
    slots: List = field(default_factory=list)
    steps: List = field(default_factory=list)
    bosses: List = field(default_factory=list)
    fillets: List = field(default_factory=list)
    chamfers: List = field(default_factory=list)
    grooves: List = field(default_factory=list)
    threads: List = field(default_factory=list)
    islands: List = field(default_factory=list)
    turning_features: List = field(default_factory=list)  # ✅ ADDED: Turning features for lathe operations
    
    # Graph representation
    graph: Dict = field(default_factory=dict)
    
    # Metrics
    metrics: RecognitionMetrics = field(default_factory=RecognitionMetrics)
    
    # Metadata (includes volume decomposition info)
    metadata: Dict = field(default_factory=dict)
    
    # Manufacturing sequence (toolpath planning)
    manufacturing_sequence: List = field(default_factory=list)
    
    # Feature interactions
    feature_interactions: Dict = field(default_factory=dict)
    
    def get_all_features(self) -> List:
        """Return all features as a single list"""
        return (
            self.holes + self.pockets + self.slots + self.steps +
            self.bosses + self.fillets + self.chamfers + self.grooves +
            self.threads + self.islands + self.turning_features  # ✅ ADDED: Include turning features
        )


def _convert_graph_to_legacy_format(graph: Dict) -> Dict:
    """
    Convert new dict-based AAG graph to old typed format for legacy recognizers.
    
    Some recognizers expect nodes/edges as lists of dataclass objects,
    but the new system uses dict-based graphs for flexibility.
    """
    if not isinstance(graph, dict):
        return graph
        
    # If already has typed objects, return as-is
    if 'nodes' in graph and graph['nodes'] and isinstance(graph['nodes'][0], GraphNode):
        return graph
    
    # Convert dict nodes to GraphNode objects
    converted_nodes = []
    for node_dict in graph.get('nodes', []):
        node = GraphNode(
            face_id=node_dict.get('face_id', -1),
            surface_type=SurfaceType(node_dict.get('surface_type', 'unknown')),
            vexity=Vexity(node_dict.get('vexity', 'unknown')),
            area=node_dict.get('area', 0.0),
            normal=node_dict.get('normal', [0, 0, 1]),
            center=node_dict.get('center', [0, 0, 0]),
            radius=node_dict.get('radius'),
            axis=node_dict.get('axis'),
            angle_deg=node_dict.get('angle_deg')
        )
        converted_nodes.append(node)
    
    # Convert dict edges to GraphEdge objects
    converted_edges = []
    for edge_dict in graph.get('edges', []):
        edge = GraphEdge(
            source=edge_dict.get('source', -1),
            target=edge_dict.get('target', -1),
            edge_type=edge_dict.get('edge_type', 'unknown'),
            convexity=edge_dict.get('convexity', 'unknown'),
            angle=edge_dict.get('angle', 0.0)
        )
        converted_edges.append(edge)
    
    return {
        'nodes': converted_nodes,
        'edges': converted_edges,
        'adjacency': graph.get('adjacency', {})
    }


# ============================================================================
# AAG PATTERN MATCHER - Main Class
# ============================================================================

class AAGPatternMatcher:
    """
    Production-grade AAG pattern matching for machining features.
    
    Analysis Situs Compatible Workflow:
    1. Volume decomposition → Single removal volume
    2. AAG graph building
    3. Machining configuration detection
    4. Feature recognition per config
    5. Tool accessibility analysis
    6. Validation & output
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: Geometric tolerance for face adjacency
        """
        self.tolerance = tolerance
        self.part_shape = None
        self.part_type = "prismatic"
        
        # Volume decomposition
        self.volumes = []
        
        # AAG graphs (one per volume)
        self.graphs = []
        
        # Machining configurations
        self.configurations = []
        
        # Detected features
        self.features = []
        
        # Statistics
        self.statistics = {}
        
    def recognize_all_features(
        self,
        shape,
        validate: bool = True,
        compute_manufacturing: bool = False,
        use_volume_decomposition: bool = True
    ) -> RecognitionResult:
        """
        Main entry point for AAG-based feature recognition.
        
        Args:
            shape: TopoDS_Shape to analyze
            validate: Run validation checks
            compute_manufacturing: Generate manufacturing sequence
            use_volume_decomposition: Enable volume decomposition (Analysis Situs style)
            
        Returns:
            RecognitionResult with detected features
        """
        logger.info("=" * 70)
        logger.info("AAG PATTERN MATCHER - Starting Recognition")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        self.part_shape = shape
        
        try:
            # Quick part type classification
            self.part_type = self._quick_classify_part_type(shape)
            logger.info(f"Part type classified as: {self.part_type}")
            
            # STEP 1: Volume decomposition (if enabled)
            if use_volume_decomposition:
                decomp_result = self.run_recognition()
                
                if decomp_result['status'] == 'success':
                    # Convert dict result to RecognitionResult dataclass
                    return self._dict_to_recognition_result(decomp_result)
                else:
                    # Decomposition failed, return error result
                    return self._dict_to_recognition_result(decomp_result)
            else:
                # Legacy path: no volume decomposition
                logger.warning("Volume decomposition disabled - using legacy path")
                return self._recognize_without_decomposition(shape, validate, compute_manufacturing)
                
        except Exception as e:
            logger.error(f"Recognition failed: {e}", exc_info=True)
            
            return RecognitionResult(
                status=RecognitionStatus.ERROR,
                part_type=PartType.UNKNOWN,
                metrics=RecognitionMetrics(
                    errors=[str(e)],
                    total_time=time.time() - start_time
                )
            )
    
    def run_recognition(self) -> Dict:
        """
        Run complete AAG recognition pipeline with volume decomposition.
        
        Returns dict-based result for backward compatibility
        """
        start_time = time.time()
        
        try:
            # Step 1: Volume decomposition
            logger.info("\n[STEP 1/8] Decomposing into manufacturing volumes...")
            
            decomposer = VolumeDecomposer(tolerance=self.tolerance)
            self.volumes = decomposer.decompose(self.part_shape, self.part_type)
            
            if not self.volumes:
                logger.error("Volume decomposition returned no volumes!")
                return {
                    'status': 'error',
                    'error_message': 'Volume decomposition failed',
                    'features': [],
                    'statistics': {},
                    'warnings': []
                }
            
            logger.info(f"✓ Decomposition complete: {len(self.volumes)} volume(s)")
            
            # Step 2: Build AAG graphs (DUAL-GRAPH APPROACH)
            logger.info("\n[STEP 2/8] Building AAG for volume(s)...")
            
            self.graphs = []         # Removal volume graphs (for holes)
            self.part_graphs = []    # Part graphs (for pockets, slots, fillets)
            
            for i, vol_data in enumerate(self.volumes):
                removal_volume = vol_data['shape']
                
                # Build AAG on REMOVAL VOLUME (for holes)
                logger.info(f"  Building removal volume AAG for volume #{i+1}...")
                removal_builder = AAGGraphBuilder(removal_volume, tolerance=self.tolerance)
                removal_graph_result = removal_builder.build()
                
                if removal_graph_result and removal_graph_result.get('nodes'):
                    removal_aag_data = {
                        'graph': removal_graph_result,
                        'builder': removal_builder,
                        'volume_index': i,
                        'type': 'removal_volume',
                        'num_nodes': len(removal_graph_result.get('nodes', {})),
                        'num_edges': len(removal_builder.edges) if hasattr(removal_builder, 'edges') else 0
                    }
                    self.graphs.append(removal_aag_data)
                    logger.info(f"    ✓ Removal AAG built: {removal_aag_data['num_nodes']} nodes, {removal_aag_data['num_edges']} edges")
                else:
                    logger.warning(f"    ⚠ Removal AAG build failed for volume #{i+1}")
                
                # Build AAG on PART (for pockets, slots, fillets, chamfers)
                logger.info(f"  Building part AAG for volume #{i+1}...")
                part_builder = AAGGraphBuilder(self.part_shape, tolerance=self.tolerance)
                part_graph_result = part_builder.build()
                
                if part_graph_result and part_graph_result.get('nodes'):
                    part_aag_data = {
                        'graph': part_graph_result,
                        'builder': part_builder,
                        'volume_index': i,
                        'type': 'part',
                        'num_nodes': len(part_graph_result.get('nodes', {})),
                        'num_edges': len(part_builder.edges) if hasattr(part_builder, 'edges') else 0
                    }
                    self.part_graphs.append(part_aag_data)
                    logger.info(f"    ✓ Part AAG built: {part_aag_data['num_nodes']} nodes, {part_aag_data['num_edges']} edges")
                else:
                    logger.warning(f"    ⚠ Part AAG build failed for volume #{i+1}")
            
            if not self.graphs and not self.part_graphs:
                logger.error("No AAG graphs were built!")
                return {
                    'status': 'error',
                    'error_message': 'AAG graph building failed',
                    'features': [],
                    'statistics': {},
                    'warnings': []
                }
            
            logger.info(f"✓ Built {len(self.graphs)} removal volume graph(s) and {len(self.part_graphs)} part graph(s)")
            
            # Step 3: Detect machining configurations
            logger.info("\n[STEP 3/8] Detecting machining configurations...")
            
            self.configurations = []
            for i, aag_data in enumerate(self.graphs):
                logger.info(f"  Analyzing volume #{i+1}...")
                # Pass the builder object which has nodes/adjacency as attributes
                configs = detect_machining_configurations(aag_data['builder'])
                self.configurations.extend(configs)
                logger.info(f"    ✓ Found {len(configs)} machining config(s)")
            
            logger.info(f"✓ Total configurations: {len(self.configurations)}")
            
            # Step 4: Run recognizers
            logger.info("\n[STEP 4/8] Running feature recognizers...")
            
            self.features = []
            
            for config_idx, config in enumerate(self.configurations):
                # Convert dict to have expected fields
                # The detector returns: {'type': 'primary'/'secondary', 'axis': [...], ...}
                # We need to determine the actual machining config_type for recognizers:
                # - 2.5D_milling: Single-axis machining (most common for prismatic)
                # - 3axis_milling: Multi-axis but no undercuts
                # - 5axis_milling: Complex geometry with undercuts
                # - turning: Rotational features (lathe operations)
                if isinstance(config, dict):
                    # Determine config_type based on axis and geometry
                    # For now, assume all are 2.5D milling (most common for prismatic)
                    config_type = "2.5D_milling"
                    
                    # Could add logic here to detect 3axis, 5axis, or turning
                    # based on axis orientation, feature complexity, etc.
                else:
                    # If it's an object (shouldn't happen but be safe)
                    config_type = getattr(config, 'config_type', '2.5D_milling')
                
                logger.info(f"  Processing configuration #{config_idx+1} (type: {config_type})...")
                
                # Get graphs for this configuration (DUAL-GRAPH APPROACH)
                if not self.graphs or not self.part_graphs:
                    logger.warning("    ⚠ Missing required graphs, skipping configuration")
                    continue
                    
                removal_aag = self.graphs[0]      # Use for holes
                part_aag = self.part_graphs[0]    # Use for pockets/slots/fillets
                
                # Run appropriate recognizers based on config type
                if config_type in ["2.5D_milling", "3axis_milling", "5axis_milling"]:
                    
                    # === REMOVAL VOLUME RECOGNIZERS ===
                    
                    # CRITICAL FIX: Hole recognizer uses PART AAG (not removal volume)
                    # This fixes holes/fillets inversion by analyzing part topology
                    hole_rec = HoleRecognizer(part_aag['builder'])
                    holes = hole_rec.recognize()
                    self.features.extend(holes)
                    if holes:
                        logger.info(f"    ✓ HoleRecognizer: {len(holes)} holes detected")
                    
                    # === PART RECOGNIZERS ===
                    
                    # Pocket recognizer (uses part)
                    pocket_rec = PocketRecognizer(part_aag['builder'])
                    pockets = pocket_rec.recognize()
                    self.features.extend(pockets)
                    if pockets:
                        logger.info(f"    ✓ PocketRecognizer: {len(pockets)} pockets detected")
                    
                    # Slot recognizer (uses part)
                    if HAS_SLOT_RECOGNIZER:
                        try:
                            slot_rec = SlotRecognizer(part_aag['builder'])
                            slots = slot_rec.recognize()
                            self.features.extend(slots)
                            if slots:
                                logger.info(f"    ✓ SlotRecognizer: {len(slots)} slots detected")
                        except Exception as e:
                            logger.warning(f"    ⚠️ SlotRecognizer failed: {e}")
                    
                    # Boss recognizer (uses part - detects raised features)
                    boss_rec = BossRecognizer(part_aag['builder'])
                    bosses = boss_rec.recognize()
                    self.features.extend(bosses)
                    if bosses:
                        logger.info(f"    ✓ BossRecognizer: {len(bosses)} bosses detected")
                    
                    # Fillet/Chamfer recognizers (uses part)
                    if HAS_FILLET_RECOGNIZER:
                        try:
                            fillet_rec = FilletRecognizer(part_aag['builder'])
                            fillets = fillet_rec.recognize()
                            self.features.extend(fillets)
                            if fillets:
                                logger.info(f"    ✓ FilletRecognizer: {len(fillets)} fillets detected")
                        except Exception as e:
                            logger.warning(f"    ⚠️ FilletRecognizer failed: {e}")
                        
                        try:
                            chamfer_rec = ChamferRecognizer(part_aag['builder'])
                            chamfers = chamfer_rec.recognize()
                            self.features.extend(chamfers)
                            if chamfers:
                                logger.info(f"    ✓ ChamferRecognizer: {len(chamfers)} chamfers detected")
                        except Exception as e:
                            logger.warning(f"    ⚠️ ChamferRecognizer failed: {e}")
                
                elif config_type == "turning":
                    # Turning recognizer (if available) - uses different API
                    if HAS_TURNING_RECOGNIZER:
                        graph_dict = aag_data['graph']
                        
                        try:
                            turning_rec = TurningRecognizer(tolerance=self.tolerance)
                            turning_features = turning_rec.recognize_turning_features(graph_dict)
                            self.features.extend(turning_features)
                            if turning_features:
                                logger.info(f"    ✓ TurningRecognizer: {len(turning_features)} features detected")
                        except (AttributeError, KeyError) as e:
                            logger.warning(f"    ⚠️ TurningRecognizer failed (format mismatch): {e}")
            
            logger.info(f"✓ Feature detection complete: {len(self.features)} features")
            
            # Step 5: Tool accessibility analysis
            logger.info("\n[STEP 5/8] Analyzing tool accessibility...")
            
            # Use part graph for accessibility (features are on the part, not removal volume)
            if self.part_graphs:
                builder = self.part_graphs[0]['builder']
                
                analyzer = ToolAccessibilityAnalyzer(builder)
                self.features = analyzer.annotate_features_with_accessibility(self.features)
                logger.info("✓ Accessibility analysis complete")
            else:
                logger.warning("  ⚠ No part graph available for accessibility analysis")
            
            # Step 6: Standardize feature output format
            logger.info("\n[STEP 6/8] Standardizing feature output...")
            
            if HAS_RECOGNIZER_UTILS:
                # Standardize all features (adds faceIds camelCase format)
                self.features = standardize_features_list(self.features)
                
                # Merge any duplicate features
                original_count = len(self.features)
                self.features = merge_duplicate_face_ids(self.features)
                
                if len(self.features) < original_count:
                    logger.info(f"  ✓ Merged {original_count - len(self.features)} duplicate features")
                    
                logger.info(f"  ✓ Standardized {len(self.features)} features")
            else:
                logger.warning("  ⚠ Feature standardization skipped")
            
            # Step 7: DFM Analysis
            logger.info("\n[STEP 7/8] Running DFM analysis...")
            
            dfm_warnings = []
            if HAS_DFM_ANALYZER and self.graphs:
                try:
                    # Run DFM analysis on first graph (primary machining config)
                    builder = self.graphs[0]['builder']
                    dfm_analyzer = DFMAnalyzer(aag_graph=builder)
                    dfm_warnings = dfm_analyzer.analyze_features(self.features)
                    
                    if dfm_warnings:
                        logger.info(f"  ⚠️  Found {len(dfm_warnings)} DFM warnings:")
                        for warning in dfm_warnings[:5]:  # Log first 5
                            logger.info(f"    - Code {warning['code']}: {warning['message']}")
                        if len(dfm_warnings) > 5:
                            logger.info(f"    ... and {len(dfm_warnings) - 5} more")
                    else:
                        logger.info("  ✓ No DFM warnings")
                except Exception as e:
                    logger.error(f"  ✗ DFM analysis failed: {e}")
            else:
                logger.info("  ⚠ DFM analysis skipped (not available)")
            
            # Step 8: Compile statistics and warnings
            logger.info("\n[STEP 8/8] Compiling statistics...")
            
            self.statistics = self._compile_statistics()
            
            logger.info(f"✓ Average confidence: {self.statistics['avg_confidence']:.1f}%")
            
            # Generate system warnings (low confidence, missing data, etc.)
            system_warnings = self._generate_warnings()
            
            # Combine system warnings with DFM warnings
            all_warnings = system_warnings + dfm_warnings
            
            logger.info(f"⚠ {len(system_warnings)} system warnings, {len(dfm_warnings)} DFM warnings")
            
            # Done
            elapsed = time.time() - start_time
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("RECOGNITION COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Status: success")
            logger.info(f"Total time: {elapsed:.2f}s")
            logger.info(f"Features recognized: {len(self.features)}")
            logger.info(f"Warnings: {len(all_warnings)}")
            logger.info("=" * 70)
            
            return {
                'status': 'success',
                'features': self.features,
                'statistics': self.statistics,
                'warnings': all_warnings,  # Combined warnings
                'dfm_warnings': dfm_warnings,  # NEW: Separate DFM warnings
                'system_warnings': system_warnings,  # NEW: System warnings
                'elapsed_time': elapsed,
                'semantic_codes': format_dfm_warnings_for_output(dfm_warnings) if HAS_DFM_ANALYZER else {}  # NEW
            }
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}", exc_info=True)
            
            return {
                'status': 'error',
                'error_message': str(e),
                'features': [],
                'statistics': {},
                'warnings': [],
                'dfm_warnings': [],  # NEW
                'system_warnings': [],  # NEW
                'semantic_codes': {}  # NEW
            }
            
    
    def _dict_to_recognition_result(self, result_dict: Dict) -> RecognitionResult:
        """Convert dict-based result to RecognitionResult dataclass."""
        
        # Categorize features by type
        features = result_dict.get('features', [])
        holes = [f for f in features if 'hole' in f.get('type', '')]
        pockets = [f for f in features if 'pocket' in f.get('type', '')]
        slots = [f for f in features if 'slot' in f.get('type', '')]
        bosses = [f for f in features if 'boss' in f.get('type', '')]
        steps = [f for f in features if 'step' in f.get('type', '')]
        fillets = [f for f in features if 'fillet' in f.get('type', '')]
        chamfers = [f for f in features if 'chamfer' in f.get('type', '')]
        turning_features = [f for f in features if 'turning' in f.get('type', '')]  # ✅ ADDED
        
        # Compute average confidence across all features
        total_confidence = sum(f.get('confidence', 0.0) for f in features)
        avg_confidence = total_confidence / len(features) if features else 0.0  # ✅ ADDED
        
        # Create metrics
        stats = result_dict.get('statistics', {})
        metrics = RecognitionMetrics(
            total_features=stats.get('num_features', 0),
            feature_counts=stats.get('type_counts', {}),
            average_confidence=avg_confidence,  # ✅ ADDED
            recognition_time=result_dict.get('elapsed_time', 0.0),  # ✅ ADDED
            total_time=result_dict.get('elapsed_time', 0.0),
            graph_nodes=stats.get('num_aag_nodes', 0)
        )
        
        # Add confidence scores if available
        for feature in features:
            ftype = feature.get('type', 'unknown')
            conf = feature.get('confidence', 0.0)
            if ftype not in metrics.confidence_scores:
                metrics.confidence_scores[ftype] = conf
        
        # Determine status
        if result_dict.get('status') == 'success':
            status = RecognitionStatus.SUCCESS
        elif result_dict.get('status') == 'error':
            status = RecognitionStatus.ERROR
        else:
            status = RecognitionStatus.PARTIAL_SUCCESS
        
        # Add warnings/errors to metrics
        warnings = result_dict.get('warnings', [])
        for warning in warnings:
            if isinstance(warning, dict):
                metrics.warnings.append(warning.get('message', str(warning)))
            else:
                metrics.warnings.append(str(warning))
        
        if 'error_message' in result_dict:
            metrics.errors.append(result_dict['error_message'])
        
        return RecognitionResult(
            status=status,
            part_type=PartType.PRISMATIC if self.part_type == "prismatic" else PartType.ROTATIONAL,
            holes=holes,
            pockets=pockets,
            slots=slots,
            steps=steps,
            bosses=bosses,
            fillets=fillets,
            chamfers=chamfers,
            grooves=[],  # Not yet implemented
            threads=[],  # Not yet implemented
            islands=[],  # Not yet implemented
            turning_features=turning_features,  # ✅ ADDED
            graph={},  # Can add AAG graph data if needed
            metrics=metrics,
            metadata=stats,
            manufacturing_sequence=[],  # Can be computed later
            feature_interactions={}
        )
            
    def _compile_statistics(self) -> Dict:
        """Compile recognition statistics."""
        
        # Count features by type
        type_counts = {}
        for feature in self.features:
            ftype = feature.get('type', 'unknown')
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        
        # Compute average confidence
        total_conf = sum(f.get('confidence', 0.0) for f in self.features)
        avg_conf = (total_conf / len(self.features) * 100) if self.features else 0.0
        
        # AAG graph stats
        num_aag_nodes = sum(g['num_nodes'] for g in self.graphs)
        num_aag_edges = sum(g['num_edges'] for g in self.graphs)
        
        return {
            'num_features': len(self.features),
            'type_counts': type_counts,
            'avg_confidence': avg_conf,
            'num_aag_nodes': num_aag_nodes,
            'num_aag_edges': num_aag_edges,
            'num_volumes': len(self.volumes),
            'num_configurations': len(self.configurations)
        }
    
    def _generate_warnings(self) -> List[Dict]:
        """Generate warnings for detected issues."""
        warnings = []
        
        # Check for low confidence features
        for feature in self.features:
            conf = feature.get('confidence', 0.0)
            if conf < 0.5:
                warnings.append({
                    'type': 'low_confidence',
                    'message': f"Feature {feature.get('type', 'unknown')} has low confidence ({conf:.2f})",
                    'feature_id': feature.get('id', -1)
                })
        
        # Check for missing accessibility data
        for feature in self.features:
            if 'accessibility' not in feature:
                warnings.append({
                    'type': 'missing_accessibility',
                    'message': f"Feature {feature.get('type', 'unknown')} missing accessibility data",
                    'feature_id': feature.get('id', -1)
                })
        
        return warnings
    
    def _quick_classify_part_type(self, shape) -> str:
        """
        Quick heuristic part type classification.
        
        Returns: "prismatic" or "rotational"
        """
        # TODO: Implement more sophisticated classification
        # For now, default to prismatic
        return "prismatic"
    
    def _recognize_without_decomposition(self, shape, validate: bool, compute_manufacturing: bool) -> RecognitionResult:
        """Legacy path without volume decomposition"""
        logger.warning("Using legacy recognition without volume decomposition")
        
        return RecognitionResult(
            status=RecognitionStatus.ERROR,
            part_type=PartType.UNKNOWN,
            metrics=RecognitionMetrics(
                errors=["Volume decomposition disabled - legacy path not fully implemented"],
                total_time=0.0
            )
        )
