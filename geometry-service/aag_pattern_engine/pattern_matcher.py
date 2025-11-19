"""
AAG Pattern Matcher - Analysis Situs Workflow
=============================================

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
from typing import List, Dict

# Import components with correct relative paths
from volume_decomposer import VolumeDecomposer
from .graph_builder import AAGGraphBuilder
from .machining_configuration_detector import MachiningConfigurationDetector
from .recognizers.hole_recognizer import HoleRecognizer
from .recognizers.pocket_recognizer import PocketRecognizer
from .recognizers.boss_step_island_recognizer import BossRecognizer
from .recognizers.slot_recognizer import SlotRecognizer
from .recognizers.fillet_chamfer_recognizer import FilletRecognizer, ChamferRecognizer
from .recognizers.turning_recognizer import TurningRecognizer
from .tool_accessibility_analyzer import ToolAccessibilityAnalyzer
from .graph_builder import SurfaceType, Vexity, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


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
            neighbor_id = neighbor['neighbor_id']
            
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
    
    def __init__(self, part_shape, part_type: str = "prismatic"):
        """
        Args:
            part_shape: TopoDS_Shape of part
            part_type: "prismatic" or "rotational"
        """
        self.part_shape = part_shape
        self.part_type = part_type
        
        # Results storage
        self.volumes = []
        self.aag_graphs = []
        self.features = []
        self.statistics = {}
        
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
                
                builder = AAGGraphBuilder()
                aag_result = builder.build(volume_data['shape'])
                
                self.aag_graphs.append({
                    'volume_index': i,
                    'nodes': aag_result['nodes'],
                    'adjacency': aag_result['adjacency'],
                    'statistics': aag_result['statistics'],
                    'volume_hint': volume_data['hint']
                })
                
                logger.info(f"    ✓ {len(aag_result['nodes'])} nodes, {len(aag_result['statistics']['num_edges'])} edges")
                
            logger.info(f"✓ AAG built for {len(self.aag_graphs)} volume(s)")
            
            # Step 3: Detect machining configurations
            logger.info("\n[STEP 3/8] Detecting machining configurations...")
            
            all_configurations = []
            
            for aag_data in self.aag_graphs:
                configs = detect_machining_configurations(aag_data)
                all_configurations.extend(configs)
                
            logger.info(f"✓ Detected {len(all_configurations)} machining configuration(s)")
            
            # Step 4: Run feature recognizers
            logger.info("\n[STEP 4/8] Running feature recognizers...")
            
            all_features = []
            
            for aag_data in self.aag_graphs:
                # Create simple graph object for new recognizers (holes, pockets, bosses)
                graph_obj = SimpleAAGGraph(
                    aag_data['nodes'],
                    aag_data['adjacency']
                )
                
                # Run new recognizers (dict-based API)
                hole_recognizer = HoleRecognizer(graph_obj)
                holes = hole_recognizer.recognize()
                
                pocket_recognizer = PocketRecognizer(graph_obj)
                pockets = pocket_recognizer.recognize()
                
                boss_recognizer = BossRecognizer(graph_obj)
                bosses = boss_recognizer.recognize()
                
                # Convert graph to legacy format for old recognizers
                legacy_graph = _convert_graph_to_legacy_format({
                    'nodes': aag_data['nodes'],
                    'adjacency': aag_data['adjacency']
                })
                
                # Run legacy recognizers (typed API)
                slot_recognizer = SlotRecognizer()
                slots = slot_recognizer.recognize_slots(legacy_graph)
                
                fillet_recognizer = FilletRecognizer()
                fillets = fillet_recognizer.recognize_fillets(legacy_graph)
                
                chamfer_recognizer = ChamferRecognizer()
                chamfers = chamfer_recognizer.recognize_chamfers(legacy_graph)
                
                # Note: TurningRecognizer needs different input (not AAG graph)
                # Skip for now - requires face loop extraction
                
                all_features.extend(holes)
                all_features.extend(pockets)
                all_features.extend(bosses)
                all_features.extend(slots)
                all_features.extend([f.__dict__ for f in fillets])  # Convert dataclass to dict
                all_features.extend([f.__dict__ for f in chamfers])  # Convert dataclass to dict
                
                logger.info(f"  Volume {aag_data['volume_index']}: {len(holes)} holes, {len(pockets)} pockets, "
                           f"{len(bosses)} bosses, {len(slots)} slots, {len(fillets)} fillets, {len(chamfers)} chamfers")
                
            self.features = all_features
            
            logger.info(f"✓ Total features recognized: {len(all_features)}")
            
            # Step 5: Add tool accessibility
            logger.info("\n[STEP 5/8] Analyzing tool accessibility...")
            
            for aag_data in self.aag_graphs:
                graph_obj = SimpleAAGGraph(
                    aag_data['nodes'],
                    aag_data['adjacency']
                )
                
                analyzer = ToolAccessibilityAnalyzer(graph_obj)
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
                    'feature_id': feature.get('face_ids', [None])[0]
                })
                
        # Check for low confidence
        for feature in self.features:
            if feature.get('confidence', 1.0) < 0.5:
                warnings.append({
                    'code': 'W002',
                    'severity': 'medium',
                    'message': f"Low confidence recognition for {feature['type']}",
                    'feature_id': feature.get('face_ids', [None])[0]
                })
                
        return warnings


class SimpleAAGGraph:
    """
    Simple wrapper to provide AAG graph interface to recognizers.
    """
    
    def __init__(self, nodes: Dict, adjacency: Dict):
        self.nodes = nodes
        self.adjacency = adjacency
        
    def get_adjacent_faces(self, face_id: int) -> List[int]:
        """Get adjacent face IDs."""
        return [n['face_id'] for n in self.adjacency.get(face_id, [])]


def run_aag_recognition(part_shape, part_type: str = "prismatic"):
    """
    Convenience function for full AAG recognition.
    
    Args:
        part_shape: TopoDS_Shape
        part_type: "prismatic" or "rotational"
        
    Returns:
        Recognition result dict
    """
    matcher = AAGPatternMatcher(part_shape, part_type)
    return matcher.run_recognition()
