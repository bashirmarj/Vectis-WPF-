
import logging
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MOCK OCC MODULES (Required for imports)
sys.modules['OCC'] = MagicMock()
sys.modules['OCC.Core'] = MagicMock()
sys.modules['OCC.Core.TopoDS'] = MagicMock()
sys.modules['OCC.Core.TopExp'] = MagicMock()
sys.modules['OCC.Core.TopAbs'] = MagicMock()
sys.modules['OCC.Core.BRepAdaptor'] = MagicMock()
sys.modules['OCC.Core.GeomAbs'] = MagicMock()
sys.modules['OCC.Core.GProp'] = MagicMock()
sys.modules['OCC.Core.BRepGProp'] = MagicMock()
sys.modules['OCC.Core.BRep'] = MagicMock()
sys.modules['OCC.Core.TopTools'] = MagicMock()
sys.modules['OCC.Core.gp'] = MagicMock()
sys.modules['OCC.Core.BRepPrimAPI'] = MagicMock()
sys.modules['OCC.Core.BRepAlgoAPI'] = MagicMock()
sys.modules['OCC.Core.BRepBndLib'] = MagicMock()
sys.modules['OCC.Core.BRepBuilderAPI'] = MagicMock()
sys.modules['OCC.Core.Bnd'] = MagicMock()
sys.modules['OCC.Core.TColStd'] = MagicMock()
sys.modules['OCC.Core.TColgp'] = MagicMock()
sys.modules['OCC.Core.Poly'] = MagicMock()
sys.modules['OCC.Core.Precision'] = MagicMock()
sys.modules['OCC.Core.Interface'] = MagicMock()
sys.modules['OCC.Core.STEPControl'] = MagicMock()
sys.modules['OCC.Core.IFSelect'] = MagicMock()
sys.modules['OCC.Core.StlAPI'] = MagicMock()
sys.modules['OCC.Core.Quantity'] = MagicMock()
sys.modules['OCC.Core.Message'] = MagicMock()
sys.modules['OCC.Core.OSD'] = MagicMock()
sys.modules['OCC.Core.BRepMesh'] = MagicMock()
sys.modules['OCC.Core.BRepTools'] = MagicMock()
sys.modules['OCC.Core.TopLoc'] = MagicMock()
sys.modules['OCC.Core.BRepCheck'] = MagicMock()
sys.modules['OCC.Core.ShapeAnalysis'] = MagicMock()
sys.modules['OCC.Core.ShapeFix'] = MagicMock()
sys.modules['OCC.Core.BRepOffsetAPI'] = MagicMock()
sys.modules['OCC.Core.BRepFilletAPI'] = MagicMock()
sys.modules['OCC.Core.ChFi3d'] = MagicMock()
sys.modules['OCC.Core.Law'] = MagicMock()
sys.modules['OCC.Extend'] = MagicMock()
sys.modules['OCC.Extend.DataExchange'] = MagicMock()

from aag_pattern_engine.recognizers.hole_recognizer import HoleRecognizer

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class MockAAG:
    def __init__(self, nodes, adjacency):
        self.nodes = nodes
        self.adjacency = adjacency
        
    def get_adjacent_faces(self, face_id):
        return self.adjacency.get(face_id, [])

def test_hole_recognition_fixes():
    print("="*70)
    print("VERIFYING HOLE RECOGNITION FIXES")
    print("="*70)
    
    # 1. Test Coaxial Tolerance (Tightened)
    print("\nTest 1: Coaxial Tolerance")
    
    # Hole 1: Axis (0,0,1)
    h1 = {
        'face_id': 1,
        'axis': [0, 0, 1],
        'center': [0, 0, 0],
        'radius': 5.0,
        'area': 100.0,
        'type': 'through_hole',
        'depth': 10.0,
        'diameter': 10.0,
        'face_ids': [1],
        'bores': []
    }
    
    # Hole 2: Axis tilted by 5 degrees (should FAIL with new tolerance 0.999, passed with 0.99)
    # 5 deg tilt: cos(5) = 0.996
    # 0.996 < 0.999 -> Should NOT be coaxial
    angle_rad = 5.0 * np.pi / 180.0
    h2 = {
        'face_id': 2,
        'axis': [np.sin(angle_rad), 0, np.cos(angle_rad)],
        'center': [0, 0, 15],
        'radius': 5.0,
        'area': 100.0,
        'type': 'through_hole',
        'depth': 10.0,
        'diameter': 10.0,
        'face_ids': [2],
        'bores': []
    }
    
    # Hole 3: Axis tilted by 1 degree (should PASS if tolerance was loose, but 1 deg cos=0.9998)
    # Wait, 1 deg cos is 0.9998. 2.5 deg cos is 0.999.
    # So 5 deg (0.996) should FAIL.
    # 1 deg (0.9998) should PASS.
    
    rec = HoleRecognizer(MockAAG({}, {}))
    
    is_coaxial_5deg = rec._are_coaxial(h1, h2)
    print(f"  5 deg tilt (cos=0.996): Coaxial? {is_coaxial_5deg} (Expected: False)")
    
    if not is_coaxial_5deg:
        print("  [PASS] Coaxial tolerance tightened correctly")
    else:
        print("  [FAIL] Coaxial tolerance too loose")

    # 2. Test Countersink Detection (Relaxed)
    print("\nTest 2: Countersink Detection")
    
    # Hole: Center (0,0,0), Dia 10.0
    hole = {
        'center': [0, 0, 0],
        'axis': [0, 0, 1],
        'diameter': 10.0
    }
    
    # Cone: Center (0.05, 0, 10) - Offset by 0.05mm
    # Old tolerance: Dia/1000 = 0.01mm. 0.05 > 0.01 -> Failed
    # New tolerance: max(0.1, Dia*0.1) = 1.0mm. 0.05 < 1.0 -> Should Pass
    cone = {
        'center': [0.05, 0, 10],
        'axis': [0, 0, 1]
    }
    
    cones = [cone]
    has_cs = rec._find_countersink(hole, cones)
    print(f"  0.05mm offset (Dia 10mm): Found? {has_cs} (Expected: True)")
    
    if has_cs:
        print("  [PASS] Countersink tolerance relaxed correctly")
    else:
        print("  [FAIL] Countersink tolerance too tight")

    # 3. Test Conical Bottom Detection
    print("\nTest 3: Conical Bottom Detection")
    
    # Cylinder Face 10 connected to Cone Face 11
    nodes = {
        10: {'surface_type': 'cylinder', 'axis': [0,0,1], 'radius': 5.0},
        11: {'surface_type': 'cone', 'axis': [0,0,1], 'semi_angle': 59.0 * np.pi / 180.0} # 118 deg tip
    }
    adjacency = {
        10: [11],
        11: [10]
    }
    
    rec_bottom = HoleRecognizer(MockAAG(nodes, adjacency))
    has_bottom, b_type, b_angle = rec_bottom._analyze_bottom(10)
    
    print(f"  Bottom Analysis: Has={has_bottom}, Type={b_type}, Angle={b_angle}")
    
    if has_bottom and b_type == 'conical' and abs(b_angle - 118.0) < 0.1:
        print("  [PASS] Conical bottom detected correctly")
    else:
        print("  [FAIL] Conical bottom detection failed")

if __name__ == "__main__":
    test_hole_recognition_fixes()
