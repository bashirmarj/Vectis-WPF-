
import unittest
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock OCC and other dependencies
from unittest.mock import MagicMock
sys.modules['OCC'] = MagicMock()
sys.modules['OCC.Core'] = MagicMock()
sys.modules['OCC.Core.BRepPrimAPI'] = MagicMock()
sys.modules['OCC.Core.BRepAlgoAPI'] = MagicMock()
sys.modules['OCC.Core.BRepBndLib'] = MagicMock()
sys.modules['OCC.Core.Bnd'] = MagicMock()
sys.modules['OCC.Core.gp'] = MagicMock()
sys.modules['OCC.Core.BRepBuilderAPI'] = MagicMock()
sys.modules['OCC.Core.TopExp'] = MagicMock()
sys.modules['OCC.Core.TopAbs'] = MagicMock()
sys.modules['OCC.Core.GProp'] = MagicMock()
sys.modules['OCC.Core.BRepGProp'] = MagicMock()
sys.modules['OCC.Core.TopoDS'] = MagicMock()
sys.modules['OCC.Core.BRepAdaptor'] = MagicMock()
sys.modules['OCC.Core.GeomAbs'] = MagicMock()

from volume_decomposer import VolumeDecomposer
from lump_classifier import LumpClassifier

class TestVolumeDecompositionPipeline(unittest.TestCase):
    
    def test_pipeline_flow(self):
        """Test the flow from decomposition to classification."""
        
        # Mock Decomposer
        decomposer = VolumeDecomposer()
        # Mock _decompose_prismatic to return a list of lumps directly (simulating the new logic)
        decomposer._decompose_prismatic = MagicMock(return_value=[
            {'shape': MagicMock(), 'hint': 'unknown', 'volume_mm3': 100.0}
        ])
        
        results = decomposer.decompose(MagicMock(), part_type="prismatic")
        self.assertEqual(len(results), 1)
        
        # Mock Classifier
        classifier = LumpClassifier()
        # Mock _analyze_surfaces to return a "hole-like" signature
        classifier._analyze_surfaces = MagicMock(return_value={
            'plane': 0, 'cylinder': 1, 'cone': 0, 'other': 0,
            'cylinders': [{'radius': 5.0, 'axis': MagicMock()}],
            'planes': []
        })
        # Mock axis object
        axis_mock = MagicMock()
        axis_mock.X.return_value = 0
        axis_mock.Y.return_value = 0
        axis_mock.Z.return_value = 1
        classifier._analyze_surfaces.return_value['cylinders'][0]['axis'] = axis_mock
        
        # Mock GProp for boundary analysis
        # We need to mock GProp_GProps and its CentreOfMass method
        # Since _analyze_boundaries creates a NEW GProp_GProps instance, we need to mock the class
        
        # This is tricky with MagicMock on sys.modules. 
        # Easier way: Mock _analyze_boundaries directly since we are testing flow, not OCC details
        classifier._analyze_boundaries = MagicMock(return_value={
            'top': True, 'bottom': True, 'sides': 0, 'count': 2
        })
        
        # Mock stock bbox
        stock_bbox = {
            'xmin': -10, 'xmax': 10,
            'ymin': -10, 'ymax': 10,
            'zmin': 0, 'zmax': 20
        }
        
        classification = classifier.classify(results[0]['shape'], stock_bbox)
        
        self.assertEqual(classification['type'], 'hole')
        self.assertEqual(classification['subtype'], 'simple_hole_through')
        self.assertEqual(classification['diameter'], 10.0)
        print("Pipeline flow verified")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
