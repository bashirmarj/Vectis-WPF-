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
            self.threads + self.islands
        )


def _convert_graph_to_legacy_format(graph: Dict) -> Dict:
    """
    Convert new dict-based AAG graph to old typed format for legacy recognizers.
    
    Args:
        graph: New format {nodes: {id: {attrs}}, adjacency: {id: [neighbors]}}
        
    Returns:
        Legacy format compatible with slot/fillet/turning recognizers
    """
    legacy_nodes = {}
    legacy_edges = []
    
    # Convert nodes
    for face_id, attrs in graph['nodes'].items():
        # Map string surface type to enum
        surface_type_str = attrs.get('surface_type', 'unknown')
        try:
            surface_type = SurfaceType(surface_type_str)
        except ValueError:
            surface_type = SurfaceType.UNKNOWN
        
        legacy_nodes[face_id] = GraphNode(
            face_id=face_id,
            surface_type=surface_type,
            area=attrs.get('area', 0.0),
            normal=tuple(attrs.get('normal', [0.0, 0.0, 1.0])),
            center=tuple(attrs.get('center')) if 'center' in attrs else None,
            radius=attrs.get('radius')
        )
    
    # Convert adjacency to edges
    seen_edges = set()
    for face_id, neighbors in graph['adjacency'].items():
        for neighbor in neighbors:
            neighbor_id = neighbor.get('face_id', neighbor.get('neighbor_id'))
            
            # Avoid duplicate edges (undirected graph)
            edge_key = tuple(sorted([face_id, neighbor_id]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            # Map vexity string to enum
            vexity_str = neighbor.get('vexity', 'smooth')
            try:
                vexity = Vexity(vexity_str)
            except ValueError:
                vexity = Vexity.SMOOTH
            
            legacy_edges.append(GraphEdge(
                face1=face_id,
                face2=neighbor_id,
                vexity=vexity,
                dihedral_deg=neighbor.get('dihedral_deg', 180.0)
            ))
    
    return {
        'nodes': legacy_nodes,
        'edges': legacy_edges,
        'adjacency': graph['adjacency']  # Keep original adjacency for compatibility
    }


class AAGPatternMatcher:
    """
    Orchestrates complete feature recognition pipeline.
    
    Process (Analysis Situs aligned):
    1. Decompose to removal volume (SINGLE volume)
    2. Build AAG graph
    3. Detect machining configurations (multiple setups on same solid)
    4. Run recognizers
    5. Add accessibility analysis
    6. Compile results
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize AAG Pattern Matcher.
        
        Args:
            tolerance: Geometric tolerance for recognition
        """
        self.tolerance = tolerance
        self.part_shape = None
        self.part_type = "prismatic"
        
        # Results storage
        self.volumes = []
        self.aag_graphs = []
        self.features = []
        self.statistics = {}
        
    def recognize_all_features(
        self,
        shape,
        validate: bool = True,
        compute_manufacturing: bool = True,
        use_volume_decomposition: bool = True
    ) -> RecognitionResult:
        """
        Recognize all manufacturing features in a CAD part.
        
        Args:
            shape: TopoDS_Shape of the part
            validate: Run validation checks on detected features
            compute_manufacturing: Compute manufacturing sequences
            use_volume_decomposition: Use volume decomposition approach
            
        Returns:
            RecognitionResult with all detected features organized by type
        """
        self.part_shape = shape
        
        # Run the recognition pipeline
        result_dict = self.run_recognition()
        
        # Convert dict result to RecognitionResult object
        return self._dict_to_recognition_result(result_dict)
    
    def run_recognition(self) -> Dict:
        """
        Run complete recognition pipeline.
        
        Returns:
            Result dict: {
                'status': 'success' or 'error',
                'features': [...],
                'statistics': {...},
                'warnings': [...]
            }
        """
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("STARTING COMPREHENSIVE FEATURE RECOGNITION")
        logger.info("=" * 70)
        
        try:
            # Step 1: Volume decomposition
            logger.info("\n[STEP 1/8] Decomposing into manufacturing volumes...")
            decomposer = VolumeDecomposer()
            self.volumes = decomposer.decompose(self.part_shape, self.part_type)
            
            if not self.volumes:
                raise RuntimeError("Volume decomposition failed")
                
            logger.info(f"✓ Found {len(self.volumes)} removal volume(s)")
            
            # Step 2: Build AAG graphs
            logger.info("\n[STEP 2/8] Building AAG for volume(s)...")
            
            for i, volume_data in enumerate(self.volumes):
                logger.info(f"  Building AAG for volume {i}...")
                
                # CORRECTED: AAGGraphBuilder takes shape in constructor
                builder = AAGGraphBuilder(volume_data['shape'])
                aag_result = builder.build()
                
                # Store both builder (for recognizers) and result data
                self.aag_graphs.append({
                    'volume_index': i,
                    'builder': builder,  # AAGGraphBuilder instance
                    'nodes': builder.nodes,
                    'adjacency': builder.adjacency,
                    'statistics': aag_result['statistics'],
                    'volume_hint': volume_data['hint']
                })
                
                logger.info(f"    ✓ {len(builder.nodes)} nodes, {len(builder.edges)} edges")
                
            logger.info(f"✓ AAG built for {len(self.aag_graphs)} volume(s)")
            
            # Step 3: Detect machining configurations
            logger.info("\n[STEP 3/8] Detecting machining configurations...")
            
            all_configurations = []
            
            for aag_data in self.aag_graphs:
                # CORRECTED: Pass builder instance to detector
                configs = detect_machining_configurations(aag_data['builder'])
                all_configurations.extend(configs)
                
            logger.info(f"✓ Detected {len(all_configurations)} machining configuration(s)")
            
            # Step 4: Run feature recognizers
            logger.info("\n[STEP 4/8] Running feature recognizers...")
            
            all_features = []
            
            for aag_data in self.aag_graphs:
                # CORRECTED: Use builder instance for recognizers
                builder = aag_data['builder']
                
                # Run new recognizers (expect AAGGraphBuilder instance)
                hole_recognizer = HoleRecognizer(builder)
                holes = hole_recognizer.recognize()
                
                pocket_recognizer = PocketRecognizer(builder)
                pockets = pocket_recognizer.recognize()
                
                boss_recognizer = BossRecognizer(builder)
                bosses = boss_recognizer.recognize()
                
                all_features.extend(holes)
                all_features.extend(pockets)
                all_features.extend(bosses)
                
                logger.info(f"  Volume {aag_data['volume_index']}: {len(holes)} holes, "
                           f"{len(pockets)} pockets, {len(bosses)} bosses")
                
                # Optional legacy recognizers
                slots = []
                fillets = []
                chamfers = []
                
                if HAS_SLOT_RECOGNIZER:
                    try:
                        # Convert graph to legacy format for old recognizers
                        legacy_graph = _convert_graph_to_legacy_format({
                            'nodes': aag_data['nodes'],
                            'adjacency': aag_data['adjacency']
                        })
                        
                        slot_recognizer = SlotRecognizer()
                        slots = slot_recognizer.recognize_slots(legacy_graph)
                        all_features.extend(slots)
                        logger.info(f"  + {len(slots)} slots")
                    except Exception as e:
                        logger.warning(f"  Slot recognition failed: {e}")
                
                if HAS_FILLET_RECOGNIZER:
                    try:
                        legacy_graph = _convert_graph_to_legacy_format({
                            'nodes': aag_data['nodes'],
                            'adjacency': aag_data['adjacency']
                        })
                        
                        fillet_recognizer = FilletRecognizer()
                        fillets = fillet_recognizer.recognize_fillets(legacy_graph)
                        
                        chamfer_recognizer = ChamferRecognizer()
                        chamfers = chamfer_recognizer.recognize_chamfers(legacy_graph)
                        
                        # Convert dataclass to dict if needed
                        if fillets and hasattr(fillets[0], '__dict__'):
                            fillets = [f.__dict__ for f in fillets]
                        if chamfers and hasattr(chamfers[0], '__dict__'):
                            chamfers = [f.__dict__ for f in chamfers]
                            
                        all_features.extend(fillets)
                        all_features.extend(chamfers)
                        logger.info(f"  + {len(fillets)} fillets, {len(chamfers)} chamfers")
                    except Exception as e:
                        logger.warning(f"  Fillet/chamfer recognition failed: {e}")
                
            self.features = all_features
            
            logger.info(f"✓ Total features recognized: {len(all_features)}")
            
            # Step 5: Add tool accessibility
            logger.info("\n[STEP 5/8] Analyzing tool accessibility...")
            
            for aag_data in self.aag_graphs:
                builder = aag_data['builder']
                
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
        
        # Create metrics
        stats = result_dict.get('statistics', {})
        metrics = RecognitionMetrics(
            total_features=stats.get('num_features', 0),
            feature_counts=stats.get('type_counts', {}),
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
            graph={},  # Can add AAG graph data if needed
            metrics=metrics,
            metadata=stats,
            manufacturing_sequence=[],  # Can be computed later
            feature_interactions={}
        )
            
    def _compile_statistics(self) -> Dict:
        """Compile recognition statistics."""
        # Type counts
        type_counts = {}
        for feature in self.features:
            ftype = feature.get('type', 'unknown')
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
            
        # Confidence stats
        confidences = [f.get('confidence', 0.5) for f in self.features]
        avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 0
        
        # Accessibility stats
        accessible = sum(1 for f in self.features 
                        if f.get('accessible_end_milling_axes') or f.get('accessible_side_milling_axes'))
        
        return {
            'num_features': len(self.features),
            'type_counts': type_counts,
            'avg_confidence': avg_confidence,
            'accessible_features': accessible,
            'inaccessible_features': len(self.features) - accessible,
            'num_volumes': len(self.volumes),
            'num_aag_nodes': sum(len(g['nodes']) for g in self.aag_graphs)
        }
        
    def _generate_warnings(self) -> List[Dict]:
        """Generate manufacturing warnings."""
        warnings = []
        
        # Check for inaccessible features
        for feature in self.features:
            if feature.get('manufacturing_warning') == 'inaccessible_feature':
                warnings.append({
                    'code': 'W001',
                    'severity': 'high',
                    'message': f"Feature {feature['type']} may be inaccessible for machining",
                    'feature_id': feature.get('face_ids', [None])[0] if feature.get('face_ids') else None
                })
                
        # Check for low confidence
        for feature in self.features:
            if feature.get('confidence', 1.0) < 0.5:
                warnings.append({
                    'code': 'W002',
                    'severity': 'medium',
                    'message': f"Low confidence recognition for {feature['type']}",
                    'feature_id': feature.get('face_ids', [None])[0] if feature.get('face_ids') else None
                })
                
        return warnings


def run_aag_recognition(part_shape, part_type: str = "prismatic"):
    """
    Convenience function for full AAG recognition.
    
    Args:
        part_shape: TopoDS_Shape
        part_type: "prismatic" or "rotational"
        
    Returns:
        Recognition result dict
    """
    matcher = AAGPatternMatcher()
    matcher.part_shape = part_shape
    matcher.part_type = part_type
    return matcher.run_recognition()
