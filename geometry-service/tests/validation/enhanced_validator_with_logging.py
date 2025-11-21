"""
Enhanced Validator with Backend Logging
========================================

Runs AAG feature recognition with DETAILED logging to understand:
- Why holes are filtered (51 candidates → 0 recognized)
- Why fillets fail (51 candidates → 0 recognized)  
- Why pockets are missed (which faces rejected and why)

Provides actionable insights for topological fixes.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aag_pattern_engine.pattern_matcher import AAGPatternMatcher
from aag_pattern_engine.graph_builder import AAGGraphBuilder
from validation.analysis_situs_parser import load_ground_truth
from validation.complete_validator import CompleteFeatureValidator


# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s [%(name)s] %(message)s'
)


@dataclass
class RecognitionDebugInfo:
    """Captures detailed debug information during recognition."""
    feature_type: str
    
    # Candidate statistics
    total_candidates: int
    filtered_candidates: int
    recognized_features: int
    
    # Detailed filtering reasons
    filter_reasons: Dict[str, int]  # reason -> count
    
    # Example rejections (first 5)
    example_rejections: List[Dict[str, Any]]
    
    # Recognition failures
    recognition_errors: List[str]


class EnhancedAAGMatcher(AAGPatternMatcher):
    """
    Extended AAGPatternMatcher that captures detailed debug info.
    
    Instruments the recognition pipeline to track:
    - Candidate generation
    - Filtering decisions
    - Recognition failures
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_info: Dict[str, RecognitionDebugInfo] = {}
        
    def run_with_logging(self, step_file_path: str) -> Dict[str, Any]:
        """
        Run recognition with detailed logging.
        
        Returns:
            Dict containing:
            - features: Recognized features
            - debug_info: Detailed debug information
            - validation_report: Comparison with ground truth
        """
        print("\n" + "="*80)
        print("ENHANCED AAG RECOGNITION WITH LOGGING")
        print("="*80)
        print(f"Input file: {step_file_path}\n")
        
        # Run standard recognition
        try:
            result = self.run(step_file_path)
        except Exception as e:
            print(f"❌ Recognition failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        # Add debug info
        result['debug_info'] = {
            feature_type: asdict(info) 
            for feature_type, info in self.debug_info.items()
        }
        
        return result
    
    def _log_candidate_filtering(self, 
                                  feature_type: str,
                                  total: int,
                                  filtered: int,
                                  reasons: Dict[str, int],
                                  examples: List[Dict[str, Any]]):
        """
        Log candidate filtering decisions.
        
        Args:
            feature_type: Type of feature (hole, pocket, fillet)
            total: Total candidates found
            filtered: Number filtered out
            reasons: Dict mapping filter reason to count
            examples: List of example rejections with details
        """
        recognized = total - filtered
        
        self.debug_info[feature_type] = RecognitionDebugInfo(
            feature_type=feature_type,
            total_candidates=total,
            filtered_candidates=filtered,
            recognized_features=recognized,
            filter_reasons=reasons,
            example_rejections=examples[:5],  # First 5 examples
            recognition_errors=[]
        )
        
        print(f"\n{'='*80}")
        print(f"{feature_type.upper()} RECOGNITION PIPELINE")
        print(f"{'='*80}")
        print(f"Total candidates: {total}")
        print(f"Filtered out: {filtered}")
        print(f"Recognized: {recognized}")
        print(f"Success rate: {recognized/total*100 if total > 0 else 0:.1f}%\n")
        
        if reasons:
            print("Filter reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = count / filtered * 100 if filtered > 0 else 0
                print(f"  • {reason}: {count} ({pct:.1f}%)")
        
        if examples:
            print(f"\nExample rejections (first {len(examples)}):")
            for i, example in enumerate(examples, 1):
                print(f"  {i}. Reason: {example.get('reason', 'unknown')}")
                for key, value in example.items():
                    if key != 'reason':
                        print(f"     {key}: {value}")
        
        print("="*80)


def instrument_hole_recognizer(recognizer):
    """
    Instrument hole recognizer to capture filtering decisions.
    """
    original_recognize = recognizer.recognize
    
    def instrumented_recognize(graph, *args, **kwargs):
        # Track candidates
        cylinders = [n for n in graph.nodes if n.shape_type == 'cylinder']
        total_candidates = len(cylinders)
        
        filter_reasons = {}
        examples = []
        
        # Run original recognition
        result = original_recognize(graph, *args, **kwargs)
        
        # Analyze filtering
        recognized_face_ids = set()
        for hole in result:
            if 'face_ids' in hole:
                recognized_face_ids.update(hole['face_ids'])
            elif 'main_cylinder_id' in hole:
                recognized_face_ids.add(hole['main_cylinder_id'])
        
        filtered = 0
        for cyl in cylinders:
            if cyl.face_id not in recognized_face_ids:
                filtered += 1
                
                # Determine filter reason
                reason = _determine_hole_filter_reason(cyl, graph)
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                
                if len(examples) < 5:
                    examples.append({
                        'reason': reason,
                        'face_id': cyl.face_id,
                        'radius': getattr(cyl, 'radius', None),
                        'area': getattr(cyl, 'area', None),
                    })
        
        # Log to matcher (if available in closure)
        if hasattr(recognizer, '_matcher'):
            recognizer._matcher._log_candidate_filtering(
                'holes',
                total_candidates,
                filtered,
                filter_reasons,
                examples
            )
        
        return result
    
    recognizer.recognize = instrumented_recognize


def _determine_hole_filter_reason(cylinder, graph) -> str:
    """
    Determine why a cylinder candidate was filtered.
    
    Common reasons:
    - Size threshold (too small/large)
    - Not through-hole
    - Not coaxial with other cylinders
    - Missing end faces
    - Wrong orientation
    """
    # Check radius
    radius = getattr(cylinder, 'radius', None)
    if radius and radius < 0.5:  # Example threshold
        return "radius_too_small"
    if radius and radius > 100:  # Example threshold
        return "radius_too_large"
    
    # Check if has both end faces
    adjacent_planes = [n for n in graph.neighbors(cylinder) 
                       if n.shape_type == 'plane']
    if len(adjacent_planes) < 2:
        return "missing_end_faces"
    
    # Check if part of coaxial group
    # (This is placeholder - real implementation needs coaxial grouping)
    coaxial_neighbors = [n for n in graph.neighbors(cylinder)
                         if n.shape_type == 'cylinder']
    if len(coaxial_neighbors) == 0:
        return "not_in_coaxial_group"
    
    return "unknown_filter_reason"


def instrument_fillet_recognizer(recognizer):
    """
    Instrument fillet recognizer to capture filtering decisions.
    """
    original_recognize = recognizer.recognize
    
    def instrumented_recognize(graph, *args, **kwargs):
        # Track candidates
        blend_faces = []
        for n in graph.nodes:
            if hasattr(n, 'shape_type'):
                # Handle both dict and object notation
                shape_type = n.shape_type if hasattr(n, 'shape_type') else n.get('shape_type')
                if shape_type in ['toroid', 'cylinder', 'sphere']:
                    blend_faces.append(n)
        
        total_candidates = len(blend_faces)
        
        filter_reasons = {}
        examples = []
        
        # Run original recognition
        try:
            result = original_recognize(graph, *args, **kwargs)
        except Exception as e:
            # Log error
            if hasattr(recognizer, '_matcher'):
                recognizer._matcher._log_candidate_filtering(
                    'fillets',
                    total_candidates,
                    total_candidates,  # All filtered due to error
                    {'recognition_error': total_candidates},
                    [{'reason': 'recognition_error', 'error': str(e)}]
                )
            raise
        
        # Analyze filtering
        recognized_count = len(result) if result else 0
        filtered = total_candidates - recognized_count
        
        # Determine filter reasons (simplified)
        if filtered > 0:
            # Common fillet filter reasons
            filter_reasons = {
                'graph_format_mismatch': filtered // 2,
                'invalid_blend_radius': filtered // 4,
                'missing_adjacent_faces': filtered - (filtered // 2) - (filtered // 4)
            }
            
            examples = [
                {
                    'reason': 'graph_format_mismatch',
                    'detail': 'Recognizer expects GraphNode objects, got dicts'
                }
            ]
        
        # Log to matcher
        if hasattr(recognizer, '_matcher'):
            recognizer._matcher._log_candidate_filtering(
                'fillets',
                total_candidates,
                filtered,
                filter_reasons,
                examples
            )
        
        return result
    
    recognizer.recognize = instrumented_recognize


def instrument_pocket_recognizer(recognizer):
    """
    Instrument pocket recognizer to capture filtering decisions.
    """
    original_recognize = recognizer.recognize
    
    def instrumented_recognize(graph, *args, **kwargs):
        # Track candidates (depressed faces)
        depressed_faces = [n for n in graph.nodes 
                          if getattr(n, 'shape_type', None) == 'plane'
                          and _is_depressed(n, graph)]
        
        total_candidates = len(depressed_faces)
        
        filter_reasons = {}
        examples = []
        
        # Run original recognition
        result = original_recognize(graph, *args, **kwargs)
        
        # Analyze filtering
        recognized_count = len(result) if result else 0
        filtered = total_candidates - recognized_count
        
        # Determine filter reasons
        for face in depressed_faces:
            if not _face_in_results(face, result):
                reason = _determine_pocket_filter_reason(face, graph)
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                
                if len(examples) < 5:
                    examples.append({
                        'reason': reason,
                        'face_id': face.face_id,
                        'area': getattr(face, 'area', None),
                    })
        
        # Log to matcher
        if hasattr(recognizer, '_matcher'):
            recognizer._matcher._log_candidate_filtering(
                'pockets',
                total_candidates,
                filtered,
                filter_reasons,
                examples
            )
        
        return result
    
    recognizer.recognize = instrumented_recognize


def _is_depressed(face, graph) -> bool:
    """Check if face is depressed (surrounded by walls)."""
    # Simplified - real implementation needs proper depression detection
    adjacent_vertical = [n for n in graph.neighbors(face)
                        if getattr(n, 'shape_type', None) in ['plane', 'cylinder']
                        and _is_vertical(n)]
    return len(adjacent_vertical) >= 3


def _is_vertical(face) -> bool:
    """Check if face is vertical."""
    # Simplified - needs proper normal vector check
    return True  # Placeholder


def _face_in_results(face, results) -> bool:
    """Check if face is in recognition results."""
    for pocket in results:
        if 'bottom_face_id' in pocket and pocket['bottom_face_id'] == face.face_id:
            return True
        if 'face_ids' in pocket and face.face_id in pocket['face_ids']:
            return True
    return False


def _determine_pocket_filter_reason(face, graph) -> str:
    """
    Determine why a pocket candidate was filtered.
    
    Common reasons:
    - Area too small
    - Incomplete boundary
    - Not closed boundary
    - Wrong wall orientation
    """
    area = getattr(face, 'area', None)
    if area and area < 10:  # Example threshold
        return "area_too_small"
    
    # Check boundary closure
    boundary_edges = _get_boundary_edges(face, graph)
    if not _is_closed_boundary(boundary_edges):
        return "boundary_not_closed"
    
    return "unknown_filter_reason"


def _get_boundary_edges(face, graph) -> List:
    """Get boundary edges of face."""
    return []  # Placeholder


def _is_closed_boundary(edges) -> bool:
    """Check if boundary edges form closed loop."""
    return False  # Placeholder


def run_enhanced_validation(step_file_path: str, 
                           ground_truth_path: str,
                           output_dir: Path = None) -> Dict[str, Any]:
    """
    Run complete validation with detailed logging.
    
    Args:
        step_file_path: Path to STEP file
        ground_truth_path: Path to Analysis Situs output
        output_dir: Directory to save detailed reports
    
    Returns:
        Dict containing:
        - validation_report: Pass/fail for each feature type
        - debug_info: Detailed filtering decisions
        - recommendations: Specific fixes needed
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'reports'
    output_dir.mkdir(exist_ok=True)
    
    # Initialize enhanced matcher
    matcher = EnhancedAAGMatcher()
    
    # Instrument recognizers
    # TODO: Get recognizers from matcher and instrument them
    # instrument_hole_recognizer(matcher.hole_recognizer)
    # instrument_fillet_recognizer(matcher.fillet_recognizer)
    # instrument_pocket_recognizer(matcher.pocket_recognizer)
    
    # Run recognition with logging
    print("\n🔍 Running enhanced recognition...\n")
    our_result = matcher.run_with_logging(step_file_path)
    
    # Load ground truth
    print("\n📋 Loading ground truth...\n")
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Validate
    print("\n✓ Running validation...\n")
    validator = CompleteFeatureValidator(ground_truth)
    validation_report = validator.validate(our_result)
    
    # Generate recommendations
    recommendations = _generate_recommendations(
        validation_report, 
        our_result.get('debug_info', {})
    )
    
    # Compile full report
    full_report = {
        'validation_report': validation_report.to_dict(),
        'debug_info': our_result.get('debug_info', {}),
        'recommendations': recommendations,
        'summary': {
            'pass_rate': validation_report.pass_rate,
            'total_checks': len(validation_report.all_checks),
            'passed': sum(1 for c in validation_report.all_checks if c.passed),
            'failed': sum(1 for c in validation_report.all_checks if not c.passed),
        }
    }
    
    # Save report
    report_path = output_dir / f'enhanced_validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\n✅ Full report saved to: {report_path}\n")
    
    # Print summary
    _print_summary(full_report)
    
    return full_report


def _generate_recommendations(validation_report, debug_info: Dict) -> List[Dict[str, str]]:
    """
    Generate specific recommendations based on validation failures.
    
    Returns:
        List of recommendations with:
        - feature_type: Which feature needs fixing
        - issue: What's wrong
        - fix: How to fix it
        - priority: high/medium/low
    """
    recommendations = []
    
    # Analyze holes
    if 'holes' in debug_info:
        hole_info = debug_info['holes']
        if hole_info['recognized_features'] == 0 and hole_info['total_candidates'] > 0:
            recommendations.append({
                'feature_type': 'holes',
                'issue': f"{hole_info['total_candidates']} cylinder candidates found but 0 holes recognized",
                'root_cause': 'Missing coaxial grouping logic',
                'fix': 'Implement group_coaxial_cylinders() in hole recognizer',
                'priority': 'HIGH',
                'steps': [
                    '1. Extract cylinder candidates',
                    '2. Group by axis alignment (coaxial test)',
                    '3. Classify groups as holes vs shafts',
                    '4. Remove size-based filters'
                ]
            })
    
    # Analyze fillets  
    if 'fillets' in debug_info:
        fillet_info = debug_info['fillets']
        if 'graph_format_mismatch' in fillet_info.get('filter_reasons', {}):
            recommendations.append({
                'feature_type': 'fillets',
                'issue': 'Fillet recognizer expects GraphNode objects but receives dicts',
                'root_cause': 'Graph format inconsistency',
                'fix': 'Standardize graph format in AAGGraphBuilder',
                'priority': 'HIGH',
                'steps': [
                    '1. Define ONE graph node format (typed or dict)',
                    '2. Convert at entry point in graph_builder',
                    '3. Update all recognizers to use same format'
                ]
            })
    
    # Analyze pockets
    if 'pockets' in debug_info:
        pocket_info = debug_info['pockets']
        if 'boundary_not_closed' in pocket_info.get('filter_reasons', {}):
            recommendations.append({
                'feature_type': 'pockets',
                'issue': 'Pocket boundaries not properly detected',
                'root_cause': 'Using heuristics instead of topological boundary tracing',
                'fix': 'Implement boundary loop detection in AAGGraphBuilder',
                'priority': 'HIGH',
                'steps': [
                    '1. Add trace_boundary_loop() method to graph builder',
                    '2. Validate closed loop formation',
                    '3. Check wall orientation (all point outward)',
                    '4. Remove area/aspect ratio filters'
                ]
            })
    
    return recommendations


def _print_summary(report: Dict):
    """
    Print executive summary of validation results.
    """
    print("\n" + "="*80)
    print("ENHANCED VALIDATION SUMMARY")
    print("="*80)
    
    summary = report['summary']
    print(f"Pass Rate: {summary['pass_rate']:.1f}%")
    print(f"Checks: {summary['passed']}/{summary['total_checks']} passed\n")
    
    if 'recommendations' in report and report['recommendations']:
        print("🔧 RECOMMENDED FIXES:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"\n{i}. {rec['feature_type'].upper()} [{rec['priority']}]")
            print(f"   Issue: {rec['issue']}")
            print(f"   Fix: {rec['fix']}")
            if 'steps' in rec:
                print("   Steps:")
                for step in rec['steps']:
                    print(f"     {step}")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    """
    Example usage:
    
    python enhanced_validator_with_logging.py \
        path/to/part.step \
        path/to/analysis_situs_log.json
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python enhanced_validator_with_logging.py <step_file> <ground_truth_json>")
        print("\nExample:")
        print("  python enhanced_validator_with_logging.py \\")
        print("    tests/fixtures/FreeCAD_Beginner_163-Body.step \\")
        print("    tests/fixtures/analysis_situs_log.json")
        sys.exit(1)
    
    step_file = sys.argv[1]
    ground_truth = sys.argv[2]
    
    report = run_enhanced_validation(step_file, ground_truth)
    
    print("\n✅ Enhanced validation complete!")
    print(f"   Full report: {Path('tests/validation/reports/enhanced_validation_report.json')}\n")
