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

# Optional legacy recognizers (may not exist)
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

logger = logging.getLogger(__name__)


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
            center=node_dict.get('center', [0, 0, 0])
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
            
            # Step 2: Build AAG graphs
            logger.info("\n[STEP 2/8] Building AAG for volume(s)...")
            
            self.graphs = []
            for i, vol_data in enumerate(self.volumes):
                removal_volume = vol_data['shape']
                
                logger.info(f"  Building AAG for volume #{i+1}...")
                builder = AAGGraphBuilder(tolerance=self.tolerance)
                aag_data = builder.build_graph(removal_volume)
                
                if aag_data:
                    self.graphs.append(aag_data)
                    logger.info(f"    ✓ AAG built: {aag_data['num_nodes']} nodes, {aag_data['num_edges']} edges")
                else:
                    logger.warning(f"    ⚠ AAG build failed for volume #{i+1}")
            
            if not self.graphs:
                logger.error("No AAG graphs were built!")
                return {
                    'status': 'error',
                    'error_message': 'AAG graph building failed',
                    'features': [],
                    'statistics': {},
                    'warnings': []
                }
            
            logger.info(f"✓ Built {len(self.graphs)} AAG graph(s)")
            
            # Step 3: Detect machining configurations
            logger.info("\n[STEP 3/8] Detecting machining configurations...")
            
            self.configurations = []
            for i, aag_data in enumerate(self.graphs):
                logger.info(f"  Analyzing volume #{i+1}...")
                configs = detect_machining_configurations(aag_data['graph'])
                self.configurations.extend(configs)
                logger.info(f"    ✓ Found {len(configs)} machining config(s)")
            
            logger.info(f"✓ Total configurations: {len(self.configurations)}")
            
            # Step 4: Run recognizers
            logger.info("\n[STEP 4/8] Running feature recognizers...")
            
            self.features = []
            
            for config_idx, config in enumerate(self.configurations):
                logger.info(f"  Processing configuration #{config_idx+1} (type: {config.config_type})...")
                
                # Get the graph for this configuration
                # For now, use first graph (single volume approach)
                if not self.graphs:
                    continue
                    
                aag_data = self.graphs[0]
                graph = aag_data['graph']
                
                # Run appropriate recognizers based on config type
                if config.config_type in ["2.5D_milling", "3axis_milling", "5axis_milling"]:
                    # Hole recognizer
                    hole_rec = HoleRecognizer(tolerance=self.tolerance)
                    holes = hole_rec.recognize_holes(graph)
                    self.features.extend(holes)
                    if holes:
                        logger.info(f"    ✓ HoleRecognizer: {len(holes)} holes detected")
                    
                    # Pocket recognizer
                    pocket_rec = PocketRecognizer(tolerance=self.tolerance)
                    pockets = pocket_rec.recognize_pockets(graph)
                    self.features.extend(pockets)
                    if pockets:
                        logger.info(f"    ✓ PocketRecognizer: {len(pockets)} pockets detected")
                    
                    # Slot recognizer (if available)
                    if HAS_SLOT_RECOGNIZER:
                        slot_rec = SlotRecognizer(tolerance=self.tolerance)
                        slots = slot_rec.recognize_slots(graph)
                        self.features.extend(slots)
                        if slots:
                            logger.info(f"    ✓ SlotRecognizer: {len(slots)} slots detected")
                    
                    # Boss recognizer
                    boss_rec = BossRecognizer(tolerance=self.tolerance)
                    bosses = boss_rec.recognize_bosses(graph)
                    self.features.extend(bosses)
                    if bosses:
                        logger.info(f"    ✓ BossRecognizer: {len(bosses)} bosses detected")
                    
                    # Fillet/Chamfer recognizers (if available)
                    if HAS_FILLET_RECOGNIZER:
                        fillet_rec = FilletRecognizer(tolerance=self.tolerance)
                        fillets = fillet_rec.recognize_fillets(graph)
                        self.features.extend(fillets)
                        if fillets:
                            logger.info(f"    ✓ FilletRecognizer: {len(fillets)} fillets detected")
                        
                        chamfer_rec = ChamferRecognizer(tolerance=self.tolerance)
                        chamfers = chamfer_rec.recognize_chamfers(graph)
                        self.features.extend(chamfers)
                        if chamfers:
                            logger.info(f"    ✓ ChamferRecognizer: {len(chamfers)} chamfers detected")
                
                elif config.config_type == "turning":
                    # Turning recognizer (if available)
                    if HAS_TURNING_RECOGNIZER:
                        turning_rec = TurningRecognizer(tolerance=self.tolerance)
                        turning_features = turning_rec.recognize_turning_features(graph)
                        self.features.extend(turning_features)
                        if turning_features:
                            logger.info(f"    ✓ TurningRecognizer: {len(turning_features)} features detected")
            
            logger.info(f"✓ Feature detection complete: {len(self.features)} features")
            
            # Step 5: Tool accessibility analysis
            logger.info("\n[STEP 5/8] Analyzing tool accessibility...")
            
            if self.graphs:
                builder = self.graphs[0]['builder']
                
                analyzer = ToolAccessibilityAnalyzer(builder)
                self.features = analyzer.annotate_features_with_accessibility(self.features)
                
            logger.info("✓ Accessibility analysis complete")
            
            # Step 6: Compile statistics
            logger.info("\n[STEP 6/8] Compiling statistics...")
            
            self.statistics = self._compile_statistics()
            
            logger.info(f"✓ Average confidence: {self.statistics['avg_confidence']:.1f}%")
            
            # Step 7: Generate warnings
            logger.info("\n[STEP 7/8] Generating warnings...")
            
            warnings = self._generate_warnings()
            
            logger.info(f"⚠ {len(warnings)} warnings generated")
            
            # Done
            elapsed = time.time() - start_time
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("RECOGNITION COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Status: success")
            logger.info(f"Total time: {elapsed:.2f}s")
            logger.info(f"Features recognized: {len(self.features)}")
            logger.info(f"Warnings: {len(warnings)}")
            logger.info("=" * 70)
            
            return {
                'status': 'success',
                'features': self.features,
                'statistics': self.statistics,
                'warnings': warnings,
                'elapsed_time': elapsed
            }
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}", exc_info=True)
            
            return {
                'status': 'error',
                'error_message': str(e),
                'features': [],
                'statistics': {},
                'warnings': []
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
