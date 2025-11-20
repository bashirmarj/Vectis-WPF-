"""
Complete Feature Recognition Validator
======================================

Validates ALL features against Analysis Situs.
"""

from typing import Dict, List
from dataclasses import dataclass, field
from .analysis_situs_models import ASGroundTruth
from .validator import ValidationResult


@dataclass
class CompleteValidationReport:
    """Complete validation report for all features."""
    # Structure
    structure_checks: List[ValidationResult] = field(default_factory=list)
    
    # Feature counts
    hole_count_checks: List[ValidationResult] = field(default_factory=list)
    pocket_count_checks: List[ValidationResult] = field(default_factory=list)
    fillet_count_checks: List[ValidationResult] = field(default_factory=list)
    chamfer_count_checks: List[ValidationResult] = field(default_factory=list)
    shoulder_count_checks: List[ValidationResult] = field(default_factory=list)
    shaft_count_checks: List[ValidationResult] = field(default_factory=list)
    thread_count_checks: List[ValidationResult] = field(default_factory=list)
    
    # Face mappings
    face_mapping_checks: List[ValidationResult] = field(default_factory=list)
    
    # Parameters
    hole_param_checks: List[ValidationResult] = field(default_factory=list)
    pocket_param_checks: List[ValidationResult] = field(default_factory=list)
    fillet_param_checks: List[ValidationResult] = field(default_factory=list)
    
    # Warnings & DFM
    warning_checks: List[ValidationResult] = field(default_factory=list)
    
    # Summary
    summary_checks: List[ValidationResult] = field(default_factory=list)
    
    @property
    def all_checks(self) -> List[ValidationResult]:
        """Get all checks in one list."""
        return (
            self.structure_checks +
            self.hole_count_checks +
            self.pocket_count_checks +
            self.fillet_count_checks +
            self.chamfer_count_checks +
            self.shoulder_count_checks +
            self.shaft_count_checks +
            self.thread_count_checks +
            self.face_mapping_checks +
            self.hole_param_checks +
            self.pocket_param_checks +
            self.fillet_param_checks +
            self.warning_checks +
            self.summary_checks
        )
    
    @property
    def passed(self) -> bool:
        """True if all checks passed."""
        return all(check.passed for check in self.all_checks)
    
    @property
    def pass_rate(self) -> float:
        """Percentage of checks that passed."""
        if not self.all_checks:
            return 0.0
        passed_count = sum(1 for check in self.all_checks if check.passed)
        return (passed_count / len(self.all_checks)) * 100


class CompleteFeatureValidator:
    """Validates all features against Analysis Situs."""
    
    def __init__(self, ground_truth: ASGroundTruth):
        self.ground_truth = ground_truth
    
    def validate(self, our_output: Dict) -> CompleteValidationReport:
        """Run complete validation."""
        report = CompleteValidationReport()
        
        # Validate structure
        report.structure_checks = self.validate_structure(our_output)
        
        # Validate counts
        report.hole_count_checks = self.validate_hole_counts(our_output)
        report.pocket_count_checks = self.validate_pocket_counts(our_output)
        report.fillet_count_checks = self.validate_fillet_counts(our_output)
        report.chamfer_count_checks = self.validate_chamfer_counts(our_output)
        report.shoulder_count_checks = self.validate_shoulder_counts(our_output)
        report.shaft_count_checks = self.validate_shaft_counts(our_output)
        report.thread_count_checks = self.validate_thread_counts(our_output)
        
        # Validate parameters
        report.hole_param_checks = self.validate_hole_parameters(our_output)
        report.pocket_param_checks = self.validate_pocket_parameters(our_output)
        report.fillet_param_checks = self.validate_fillet_parameters(our_output)
        
        # Validate warnings
        report.warning_checks = self.validate_warnings(our_output)
        
        # Validate summary
        report.summary_checks = self.validate_summary(our_output)
        
        return report
    
    def validate_structure(self, our_output: Dict) -> List[ValidationResult]:
        """Validate JSON structure."""
        checks = []
        
        # Top-level structure
        checks.append(ValidationResult(
            name="has_features_array",
            passed='features' in our_output,
            expected=True,
            actual='features' in our_output
        ))
        
        checks.append(ValidationResult(
            name="has_metadata",
            passed='metadata' in our_output,
            expected=True,
            actual='metadata' in our_output
        ))
        
        # Check hole structure
        our_features = our_output.get('features', [])
        our_holes = [f for f in our_features if 'hole' in f.get('type', '').lower()]
        
        if our_holes:
            hole = our_holes[0]
            required_fields = ['faceIds', 'fullyRecognized', 'totalDepth', 'bores']
            
            for field in required_fields:
                checks.append(ValidationResult(
                    name=f"hole_has_{field}",
                    passed=field in hole,
                    expected=True,
                    actual=field in hole,
                    message=f"Holes should have '{field}' field"
                ))
            
            # Check bore structure
            if 'bores' in hole and hole['bores']:
                bore = hole['bores'][0]
                bore_fields = ['faceIds', 'diameter', 'depth']
                
                for field in bore_fields:
                    checks.append(ValidationResult(
                        name=f"bore_has_{field}",
                        passed=field in bore,
                        expected=True,
                        actual=field in bore
                    ))
        
        return checks
    
    def validate_hole_counts(self, our_output: Dict) -> List[ValidationResult]:
        """Validate hole feature counts."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_holes = [f for f in our_features if 'hole' in f.get('type', '').lower()]
        
        # Total hole count
        checks.append(ValidationResult(
            name="total_hole_count",
            passed=len(our_holes) == len(self.ground_truth.holes),
            expected=len(self.ground_truth.holes),
            actual=len(our_holes),
            message=f"Expected {len(self.ground_truth.holes)} holes"
        ))
        
        # Count holes with counterbores
        gt_cb_holes = sum(1 for h in self.ground_truth.holes if h.counterbores)
        our_cb_holes = sum(1 for h in our_holes 
                          if 'counterbore' in h.get('type', '').lower() 
                          or h.get('counterbores'))
        
        checks.append(ValidationResult(
            name="counterbore_hole_count",
            passed=our_cb_holes == gt_cb_holes,
            expected=gt_cb_holes,
            actual=our_cb_holes,
            message="Holes with counterbores"
        ))
        
        # Count holes with conical bottoms
        gt_conical = sum(1 for h in self.ground_truth.holes if h.conical_bottom)
        our_conical = sum(1 for h in our_holes if h.get('conicalBottom'))
        
        checks.append(ValidationResult(
            name="conical_bottom_count",
            passed=our_conical == gt_conical,
            expected=gt_conical,
            actual=our_conical,
            message="Holes with conical bottoms"
        ))
        
        return checks
    
    def validate_pocket_counts(self, our_output: Dict) -> List[ValidationResult]:
        """Validate pocket counts."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_pockets = [f for f in our_features if 'pocket' in f.get('type', '').lower()]
        
        checks.append(ValidationResult(
            name="pocket_count",
            passed=len(our_pockets) == len(self.ground_truth.pockets),
            expected=len(self.ground_truth.pockets),
            actual=len(our_pockets)
        ))
        
        return checks
    
    def validate_fillet_counts(self, our_output: Dict) -> List[ValidationResult]:
        """Validate fillet counts."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_fillets = [f for f in our_features if 'fillet' in f.get('type', '').lower()]
        
        checks.append(ValidationResult(
            name="fillet_count",
            passed=len(our_fillets) == len(self.ground_truth.fillets),
            expected=len(self.ground_truth.fillets),
            actual=len(our_fillets)
        ))
        
        # Separate convex/concave
        gt_convex = sum(1 for f in self.ground_truth.fillets if f.convex)
        gt_concave = len(self.ground_truth.fillets) - gt_convex
        
        our_convex = sum(1 for f in our_fillets if f.get('convex', False))
        our_concave = len(our_fillets) - our_convex
        
        checks.append(ValidationResult(
            name="convex_fillet_count",
            passed=our_convex == gt_convex,
            expected=gt_convex,
            actual=our_convex,
            message="External rounds (convex fillets)"
        ))
        
        checks.append(ValidationResult(
            name="concave_fillet_count",
            passed=our_concave == gt_concave,
            expected=gt_concave,
            actual=our_concave,
            message="Internal fillets (concave)"
        ))
        
        return checks
    
    def validate_chamfer_counts(self, our_output: Dict) -> List[ValidationResult]:
        """Validate chamfer counts."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_chamfers = [f for f in our_features if 'chamfer' in f.get('type', '').lower()]
        
        checks.append(ValidationResult(
            name="chamfer_count",
            passed=len(our_chamfers) == len(self.ground_truth.chamfers),
            expected=len(self.ground_truth.chamfers),
            actual=len(our_chamfers)
        ))
        
        return checks
    
    def validate_shoulder_counts(self, our_output: Dict) -> List[ValidationResult]:
        """Validate shoulder/step counts."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_shoulders = [f for f in our_features 
                        if f.get('type') in ['shoulder', 'step']]
        
        checks.append(ValidationResult(
            name="shoulder_count",
            passed=len(our_shoulders) == len(self.ground_truth.shoulders),
            expected=len(self.ground_truth.shoulders),
            actual=len(our_shoulders)
        ))
        
        return checks
    
    def validate_shaft_counts(self, our_output: Dict) -> List[ValidationResult]:
        """Validate shaft/boss counts."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_shafts = [f for f in our_features 
                     if f.get('type') in ['shaft', 'boss']]
        
        checks.append(ValidationResult(
            name="shaft_count",
            passed=len(our_shafts) == len(self.ground_truth.shafts),
            expected=len(self.ground_truth.shafts),
            actual=len(our_shafts)
        ))
        
        return checks
    
    def validate_thread_counts(self, our_output: Dict) -> List[ValidationResult]:
        """Validate thread counts."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_threads = [f for f in our_features if f.get('type') == 'thread']
        
        checks.append(ValidationResult(
            name="thread_count",
            passed=len(our_threads) == len(self.ground_truth.threads),
            expected=len(self.ground_truth.threads),
            actual=len(our_threads)
        ))
        
        return checks
    
    def validate_hole_parameters(self, our_output: Dict) -> List[ValidationResult]:
        """Validate hole geometric parameters."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_holes = [f for f in our_features if 'hole' in f.get('type', '').lower()]
        
        if our_holes and self.ground_truth.holes:
            # Compare first hole parameters
            our_hole = our_holes[0]
            gt_hole = self.ground_truth.holes[0]
            
            # Total depth
            if 'totalDepth' in our_hole:
                depth_match = abs(our_hole['totalDepth'] - gt_hole.total_depth) < 0.5
                checks.append(ValidationResult(
                    name="hole_depth_accuracy",
                    passed=depth_match,
                    expected=gt_hole.total_depth,
                    actual=our_hole['totalDepth'],
                    message="Total depth accuracy (±0.5mm)"
                ))
            
            # Bore diameters
            if our_hole.get('bores') and gt_hole.bores:
                our_bore = our_hole['bores'][0]
                gt_bore = gt_hole.bores[0]
                
                if 'diameter' in our_bore:
                    dia_match = abs(our_bore['diameter'] - gt_bore.diameter) < 0.5
                    checks.append(ValidationResult(
                        name="bore_diameter_accuracy",
                        passed=dia_match,
                        expected=gt_bore.diameter,
                        actual=our_bore['diameter'],
                        message="Bore diameter accuracy (±0.5mm)"
                    ))
        
        return checks
    
    def validate_pocket_parameters(self, our_output: Dict) -> List[ValidationResult]:
        """Validate pocket parameters."""
        checks = []
        
        # TODO: Implement pocket parameter validation
        
        return checks
    
    def validate_fillet_parameters(self, our_output: Dict) -> List[ValidationResult]:
        """Validate fillet radii."""
        checks = []
        
        our_features = our_output.get('features', [])
        our_fillets = [f for f in our_features if 'fillet' in f.get('type', '').lower()]
        
        if our_fillets and self.ground_truth.fillets:
            # Compare first fillet radius
            our_fillet = our_fillets[0]
            gt_fillet = self.ground_truth.fillets[0]
            
            if 'radius' in our_fillet:
                radius_match = abs(our_fillet['radius'] - gt_fillet.radius) < 0.1
                checks.append(ValidationResult(
                    name="fillet_radius_accuracy",
                    passed=radius_match,
                    expected=gt_fillet.radius,
                    actual=our_fillet['radius'],
                    message="Fillet radius accuracy (±0.1mm)"
                ))
        
        return checks
    
    def validate_warnings(self, our_output: Dict) -> List[ValidationResult]:
        """Validate DFM warnings."""
        checks = []
        
        our_warnings = our_output.get('warnings', [])
        
        checks.append(ValidationResult(
            name="warning_count",
            passed=len(our_warnings) == len(self.ground_truth.semantic_warnings),
            expected=len(self.ground_truth.semantic_warnings),
            actual=len(our_warnings),
            message="DFM warning count"
        ))
        
        # Check for specific warning codes
        if self.ground_truth.semantic_warnings:
            gt_codes = {w.code for w in self.ground_truth.semantic_warnings}
            our_codes = {w.get('code') for w in our_warnings if 'code' in w}
            
            for code in gt_codes:
                has_code = code in our_codes
                checks.append(ValidationResult(
                    name=f"has_warning_code_{code}",
                    passed=has_code,
                    expected=True,
                    actual=has_code,
                    message=f"Should detect warning code {code}"
                ))
        
        return checks
    
    def validate_summary(self, our_output: Dict) -> List[ValidationResult]:
        """Validate summary statistics."""
        checks = []
        
        our_summary = our_output.get('summary', {})
        gt_summary = self.ground_truth.summary
        
        # Face count
        if 'numFaces' in our_summary:
            checks.append(ValidationResult(
                name="face_count_match",
                passed=our_summary['numFaces'] == gt_summary.num_faces,
                expected=gt_summary.num_faces,
                actual=our_summary['numFaces']
            ))
        
        # Edge count
        if 'numEdges' in our_summary:
            checks.append(ValidationResult(
                name="edge_count_match",
                passed=our_summary['numEdges'] == gt_summary.num_edges,
                expected=gt_summary.num_edges,
                actual=our_summary['numEdges']
            ))
        
        return checks
