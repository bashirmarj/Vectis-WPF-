"""Fillet Recognition Debug Logger
Enhanced logging patch to debug why 51 candidates → 0 fillets recognized.

Usage:
    python fillet_debug_logger.py <graph_json_path>
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Expected fillet faces from Analysis Situs blends.json
EXPECTED_FILLET_FACES = {39, 43, 44, 50, 53, 56, 57, 59, 74, 95, 96}

class FilletDebugAnalyzer:
    """Analyzes fillet recognition to find why candidates are rejected."""
    
    def __init__(self, graph: Dict):
        self.graph = graph
        self.nodes = graph.get('nodes', {})
        self.adjacency = graph.get('adjacency', {})
        
        # Statistics
        self.rejection_stats = {
            'not_blend_surface': 0,
            'insufficient_convex_edges': 0,
            'radius_out_of_range': 0,
            'area_too_large': 0,
            'not_in_adjacency': 0,
            'insufficient_connected_faces': 0,
            'classification_failed': 0
        }
    
    def analyze(self):
        """Run complete fillet recognition analysis with detailed logging."""
        logger.info("=" * 80)
        logger.info("🔍 FILLET RECOGNITION DEBUG ANALYSIS")
        logger.info("=" * 80)
        
        # Step 1: Graph structure
        self._log_graph_structure()
        
        # Step 2: Find blend candidates
        candidates = self._find_blend_candidates()
        
        # Step 3: Validate each candidate
        self._validate_candidates(candidates)
        
        # Step 4: Compare with expected
        self._compare_with_expected(candidates)
        
        # Step 5: Summary
        self._log_summary(candidates)
    
    def _log_graph_structure(self):
        """Log graph structure and composition."""
        logger.info("\n📊 GRAPH STRUCTURE:")
        logger.info(f"  Total faces: {len(self.nodes)}")
        logger.info(f"  Has adjacency map: {bool(self.adjacency)}")
        logger.info(f"  Adjacency entries: {len(self.adjacency)}")
        
        # Surface type distribution
        surface_types = {}
        for face_id, face_data in self.nodes.items():
            surf_type = face_data.get('surface_type', 'unknown')
            surface_types[surf_type] = surface_types.get(surf_type, 0) + 1
        
        logger.info("\n  Surface Type Distribution:")
        for surf_type, count in sorted(surface_types.items()):
            logger.info(f"    {surf_type}: {count}")
    
    def _find_blend_candidates(self) -> List[Dict]:
        """Find potential fillet faces (cylinders, tori, spheres, b-splines)."""
        logger.info("\n🔍 FINDING BLEND CANDIDATES:")
        
        blend_types = {'cylinder', 'torus', 'sphere', 'bspline'}
        candidates = []
        
        for face_id, face_data in self.nodes.items():
            surf_type = face_data.get('surface_type', '').lower()
            if surf_type in blend_types:
                candidates.append({
                    'id': face_id,
                    'type': surf_type,
                    'radius': face_data.get('radius'),
                    'area': face_data.get('area', 0.0),
                    'data': face_data
                })
        
        logger.info(f"  Found {len(candidates)} blend surface candidates")
        logger.info(f"    Cylinders: {sum(1 for c in candidates if c['type'] == 'cylinder')}")
        logger.info(f"    Tori: {sum(1 for c in candidates if c['type'] == 'torus')}")
        logger.info(f"    Spheres: {sum(1 for c in candidates if c['type'] == 'sphere')}")
        logger.info(f"    B-splines: {sum(1 for c in candidates if c['type'] == 'bspline')}")
        
        return candidates
    
    def _validate_candidates(self, candidates: List[Dict]):
        """Validate each candidate and log rejection reasons."""
        logger.info("\n🔬 VALIDATING CANDIDATES:")
        logger.info(f"  Total candidates to evaluate: {len(candidates)}\n")
        
        for i, candidate in enumerate(candidates, 1):
            face_id = candidate['id']
            logger.info(f"  [{i}/{len(candidates)}] Face {face_id}: {candidate['type']}")
            logger.info(f"       Radius: {candidate.get('radius', 'N/A')}")
            logger.info(f"       Area: {candidate['area']:.6f} m²")
            
            # Check 1: In adjacency?
            if face_id not in self.adjacency:
                logger.info(f"       ❌ REJECTED: Not in adjacency map")
                self.rejection_stats['not_in_adjacency'] += 1
                continue
            
            adjacent = self.adjacency[face_id]
            logger.info(f"       Adjacent faces: {len(adjacent)}")
            
            # Check 2: Convex edges (indicates fillet blending)
            convex_count = 0
            concave_count = 0
            smooth_count = 0
            
            for adj in adjacent:
                vexity = adj.get('vexity', 'smooth')
                if vexity == 'convex':
                    convex_count += 1
                elif vexity == 'concave':
                    concave_count += 1
                else:
                    smooth_count += 1
            
            logger.info(f"       Vexity: {convex_count} convex, {concave_count} concave, {smooth_count} smooth")
            
            if convex_count < 2:
                logger.info(f"       ❌ REJECTED: Only {convex_count} convex edges (need 2+)")
                self.rejection_stats['insufficient_convex_edges'] += 1
                continue
            
            # Check 3: Radius range
            min_radius = 0.0001  # 0.1mm
            max_radius = 0.100   # 100mm
            
            if candidate['radius']:
                radius_mm = candidate['radius'] * 1000
                logger.info(f"       Radius: {radius_mm:.3f}mm (range: 0.1-100mm)")
                
                if not (min_radius <= candidate['radius'] <= max_radius):
                    logger.info(f"       ❌ REJECTED: Radius out of range")
                    self.rejection_stats['radius_out_of_range'] += 1
                    continue
            
            # Check 4: Area (cylinders shouldn't be too large)
            if candidate['type'] == 'cylinder':
                area_cm2 = candidate['area'] * 10000
                logger.info(f"       Area: {area_cm2:.2f} cm² (max: 100 cm²)")
                
                if candidate['area'] > 0.01:  # > 100cm²
                    logger.info(f"       ❌ REJECTED: Area too large (likely shaft, not fillet)")
                    self.rejection_stats['area_too_large'] += 1
                    continue
            
            # Check 5: Connected faces
            connected_faces = [
                adj.get('face_id', adj.get('node_id'))
                for adj in adjacent
                if adj.get('vexity') == 'convex'
            ]
            connected_faces = [f for f in connected_faces if f is not None]
            
            logger.info(f"       Connected faces: {connected_faces}")
            
            if len(connected_faces) < 2:
                logger.info(f"       ❌ REJECTED: Only {len(connected_faces)} connected faces (need 2+)")
                self.rejection_stats['insufficient_connected_faces'] += 1
                continue
            
            # If we get here, candidate should be recognized
            logger.info(f"       ✅ PASSED: Should be recognized as fillet")
    
    def _compare_with_expected(self, candidates: List[Dict]):
        """Compare candidates with expected fillets from Analysis Situs."""
        logger.info("\n🎯 COMPARISON WITH EXPECTED FILLETS:")
        
        candidate_ids = {c['id'] for c in candidates}
        
        logger.info(f"  Expected fillet faces (from AS): {sorted(EXPECTED_FILLET_FACES)}")
        logger.info(f"  Candidates found: {sorted(candidate_ids)}")
        
        found_expected = candidate_ids & EXPECTED_FILLET_FACES
        missing_expected = EXPECTED_FILLET_FACES - candidate_ids
        extra_candidates = candidate_ids - EXPECTED_FILLET_FACES
        
        logger.info(f"\n  ✅ Found expected: {sorted(found_expected)} ({len(found_expected)}/11)")
        logger.info(f"  ❌ Missing expected: {sorted(missing_expected)} ({len(missing_expected)}/11)")
        logger.info(f"  ⚠️  Extra candidates: {len(extra_candidates)}")
        
        # Analyze missing fillets
        if missing_expected:
            logger.info(f"\n  🔍 WHY ARE THESE MISSING?")
            for face_id in sorted(missing_expected):
                if face_id in self.nodes:
                    face = self.nodes[face_id]
                    logger.info(f"    Face {face_id}:")
                    logger.info(f"      Type: {face.get('surface_type')}")
                    logger.info(f"      Radius: {face.get('radius')}")
                    logger.info(f"      Area: {face.get('area')}")
                    logger.info(f"      → Not a blend surface type!")
                else:
                    logger.info(f"    Face {face_id}: Not in graph!")
    
    def _log_summary(self, candidates: List[Dict]):
        """Log summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 REJECTION SUMMARY:")
        logger.info("=" * 80)
        
        total_rejected = sum(self.rejection_stats.values())
        logger.info(f"  Total candidates: {len(candidates)}")
        logger.info(f"  Total rejected: {total_rejected}")
        logger.info(f"  Should pass: {len(candidates) - total_rejected}")
        
        logger.info("\n  Rejection reasons:")
        for reason, count in sorted(self.rejection_stats.items()):
            if count > 0:
                percentage = (count / len(candidates) * 100) if candidates else 0
                logger.info(f"    {reason}: {count} ({percentage:.1f}%)")
        
        logger.info("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fillet_debug_logger.py <graph_json_path>")
        sys.exit(1)
    
    graph_path = Path(sys.argv[1])
    
    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}")
        sys.exit(1)
    
    # Load graph
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    
    # Run analysis
    analyzer = FilletDebugAnalyzer(graph)
    analyzer.analyze()


if __name__ == '__main__':
    main()
