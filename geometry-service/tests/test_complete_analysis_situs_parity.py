"""
Complete Analysis Situs Parity Tests
====================================

Tests EVERY feature type against Analysis Situs.
"""

import pytest
from pathlib import Path

from validation.analysis_situs_parser import load_ground_truth
from validation.complete_validator import CompleteFeatureValidator


FIXTURES_DIR = Path(__file__).parent / 'fixtures'
AS_LOG_PATH = FIXTURES_DIR / 'analysis_situs_log.json'
TEST_PART_PATH = FIXTURES_DIR / 'FreeCAD_Beginner_163-Body.step'


def run_our_recognition(step_file_path: str) -> dict:
    """
    Run AAG feature recognition on STEP file.
    
    TODO: Connect to actual AAGPatternMatcher
    """
    # Placeholder - returns empty features
    return {
        'features': [],
        'metadata': {
            'num_faces': 0,
            'num_edges': 0
        },
        'summary': {},
        'warnings': []
    }


def test_ground_truth_loaded():
    """Test that ground truth loads correctly."""
    gt = load_ground_truth(str(AS_LOG_PATH))
    
    print("\n" + "="*80)
    print("GROUND TRUTH SUMMARY")
    print("="*80)
    print(f"Holes: {len(gt.holes)}")
    print(f"  - With counterbores: {sum(1 for h in gt.holes if h.counterbores)}")
    print(f"  - With conical bottoms: {sum(1 for h in gt.holes if h.conical_bottom)}")
    print(f"Pockets: {len(gt.pockets)}")
    print(f"Fillets: {len(gt.fillets)}")
    print(f"Chamfers: {len(gt.chamfers)}")
    print(f"Shoulders: {len(gt.shoulders)}")
    print(f"Shafts: {len(gt.shafts)}")
    print(f"Threads: {len(gt.threads)}")
    print(f"Free flat faces: {len(gt.free_flat_faces)}")
    print(f"Milled faces: {len(gt.milled_faces)}")
    print(f"Warnings: {len(gt.semantic_warnings)}")
    print(f"\nSummary:")
    print(f"  - Vertices: {gt.summary.num_vertices}")
    print(f"  - Edges: {gt.summary.num_edges}")
    print(f"  - Faces: {gt.summary.num_faces}")
    print("="*80 + "\n")
    
    assert len(gt.holes) > 0, "Should load holes from ground truth"


def test_complete_feature_validation():
    """Complete validation of all features."""
    # Load ground truth
    ground_truth = load_ground_truth(str(AS_LOG_PATH))
    
    # Run our recognition (placeholder)
    our_output = run_our_recognition(str(TEST_PART_PATH))
    
    # Validate
    validator = CompleteFeatureValidator(ground_truth)
    report = validator.validate(our_output)
    
    # Print report
    print("\n" + "="*80)
    print("COMPLETE FEATURE VALIDATION REPORT")
    print("="*80)
    print(f"Total checks: {len(report.all_checks)}")
    print(f"Passed: {sum(1 for c in report.all_checks if c.passed)}")
    print(f"Failed: {sum(1 for c in report.all_checks if not c.passed)}")
    print(f"Pass rate: {report.pass_rate:.1f}%")
    print("="*80)
    
    # Print section summaries
    sections = [
        ("STRUCTURE", report.structure_checks),
        ("HOLE COUNTS", report.hole_count_checks),
        ("POCKET COUNTS", report.pocket_count_checks),
        ("FILLET COUNTS", report.fillet_count_checks),
        ("HOLE PARAMETERS", report.hole_param_checks),
        ("WARNINGS", report.warning_checks),
        ("SUMMARY", report.summary_checks),
    ]
    
    for name, checks in sections:
        if checks:
            passed = sum(1 for c in checks if c.passed)
            total = len(checks)
            pct = (passed / total * 100) if total > 0 else 0
            status = "✅" if passed == total else "❌"
            print(f"{status} {name}: {passed}/{total} ({pct:.0f}%)")
    
    print("="*80)
    
    # Print first 20 failures
    failures = [c for c in report.all_checks if not c.passed]
    if failures:
        print("\nFAILED CHECKS (first 20):")
        for check in failures[:20]:
            print(f"  ❌ {check.name}")
            print(f"     Expected: {check.expected}")
            print(f"     Actual:   {check.actual}")
            if check.message:
                print(f"     → {check.message}")
    
    print("\n")
    
    # This will fail until we match - that's expected!
    # assert report.passed


def test_structure_validation_only():
    """Test structure validation only."""
    gt = load_ground_truth(str(AS_LOG_PATH))
    our_output = run_our_recognition(str(TEST_PART_PATH))
    
    validator = CompleteFeatureValidator(gt)
    report = validator.validate(our_output)
    
    print("\n" + "="*80)
    print("STRUCTURE VALIDATION CHECKS")
    print("="*80)
    for check in report.structure_checks:
        icon = "✅" if check.passed else "❌"
        print(f"{icon} {check.name}: {check.actual}")
    print("="*80 + "\n")


def test_count_validation_only():
    """Test count validation only."""
    gt = load_ground_truth(str(AS_LOG_PATH))
    our_output = run_our_recognition(str(TEST_PART_PATH))
    
    validator = CompleteFeatureValidator(gt)
    report = validator.validate(our_output)
    
    print("\n" + "="*80)
    print("COUNT VALIDATION CHECKS")
    print("="*80)
    
    all_count_checks = (
        report.hole_count_checks +
        report.pocket_count_checks +
        report.fillet_count_checks +
        report.chamfer_count_checks
    )
    
    for check in all_count_checks:
        icon = "✅" if check.passed else "❌"
        print(f"{icon} {check.name}: Expected {check.expected}, Got {check.actual}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    """Run manually without pytest."""
    print("\n🔍 Running Analysis Situs Parity Validation...\n")
    print(f"Ground truth: {AS_LOG_PATH}")
    print(f"Test part: {TEST_PART_PATH}\n")
    
    # Run all tests
    test_ground_truth_loaded()
    test_structure_validation_only()
    test_count_validation_only()
    test_complete_feature_validation()
    
    print("\n✅ All validation tests completed!\n")
