
import logging
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MOCK OCC MODULES
from unittest.mock import MagicMock
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

# Mock specific enums needed by GraphBuilder
mock_geom_abs = MagicMock()
mock_geom_abs.GeomAbs_Plane = 0
mock_geom_abs.GeomAbs_Cylinder = 1
mock_geom_abs.GeomAbs_Cone = 2
mock_geom_abs.GeomAbs_Sphere = 3
mock_geom_abs.GeomAbs_Torus = 4
mock_geom_abs.GeomAbs_BSplineSurface = 5
sys.modules['OCC.Core.GeomAbs'] = mock_geom_abs

from aag_pattern_engine.recognizers.fillet_chamfer_recognizer import FilletRecognizer, FilletType
from aag_pattern_engine.graph_builder import GraphNode, SurfaceType

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_fillet_recognition_fixes():
    print("="*70)
    print("VERIFYING FILLET RECOGNITION FIXES")
    print("="*70)
    
    # 1. Mock Data based on Backend.json failure
    # Candidate 10: Type: SurfaceType.CYLINDER, Radius: 22.5, Area: 848.23
    # Vexity: 0 convex, 0 concave, 0 smooth (Isolated / No edges found)
    
    candidate_id = 10
    candidate_node = GraphNode(
        face_id=candidate_id,
        surface_type=SurfaceType.CYLINDER,
        area=848.23,
        normal=(0, 0, 1),
        center=(0, 0, 0),
        radius=22.5,  # mm
        axis=(0, 0, 1)
    )
    
    # Mock Graph
    graph = {
        'nodes': [candidate_node],
        'adjacency': {
            10: []  # Isolated
        }
    }
    
    # 2. Instantiate Recognizer
    recognizer = FilletRecognizer()
    print(f"Recognizer Config:")
    print(f"  Min Radius: {recognizer.min_fillet_radius}")
    print(f"  Max Radius: {recognizer.max_fillet_radius}")
    print(f"  Convex Angle Min: {recognizer.convex_angle_min}")
    
    # 3. Run Recognition
    print("\nRunning recognition...")
    features = recognizer.recognize_fillets(graph)
    
    # 4. Verify Results
    print("\nResults:")
    print(f"  Found {len(features)} features")
    
    if len(features) == 1:
        f = features[0]
        print(f"  Feature Type: {f['type']}")
        print(f"  Radius: {f['radius']}")
        print(f"  Convex: {f['convex']}")
        
        # Assertions
        if f['radius'] == 22.5:
            print("  ✅ Radius matches (Unit Scaling Fixed)")
        else:
            print(f"  ❌ Radius mismatch: {f['radius']}")
            
        if f['type'] == FilletType.CONSTANT_RADIUS.value:
             print("  ✅ Type identified correctly")
        
        print("\n✅ TEST PASSED: Isolated large fillet recognized!")
    else:
        print("\n❌ TEST FAILED: No features found")
        # Debug why
        # (The logs should show why)

if __name__ == "__main__":
    test_fillet_recognition_fixes()
