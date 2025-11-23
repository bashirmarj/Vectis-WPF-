
import unittest
import sys
import os
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock OCC and other dependencies
from unittest.mock import MagicMock
sys.modules['OCC'] = MagicMock()
sys.modules['OCC.Core'] = MagicMock()
sys.modules['OCC.Core.BRepPrimAPI'] = MagicMock()
sys.modules['OCC.Core.TopoDS'] = MagicMock()
sys.modules['OCC.Core.TopAbs'] = MagicMock()
sys.modules['OCC.Core.BRepAdaptor'] = MagicMock()
sys.modules['OCC.Core.BRepLProp'] = MagicMock()
sys.modules['OCC.Core.GProp'] = MagicMock()
sys.modules['OCC.Core.BRepGProp'] = MagicMock()
sys.modules['OCC.Core.gp'] = MagicMock()
sys.modules['OCC.Core.BRep'] = MagicMock()
sys.modules['OCC.Core.TopExp'] = MagicMock()
sys.modules['OCC.Core.GeomAbs'] = MagicMock()
sys.modules['OCC.Core.BRepBuilderAPI'] = MagicMock()
sys.modules['OCC.Core.TopLoc'] = MagicMock()
sys.modules['OCC.Core.TopTools'] = MagicMock()
sys.modules['OCC.Core.GeomLProp'] = MagicMock()
sys.modules['OCC.Core.BRepMesh'] = MagicMock()
sys.modules['volume_decomposer'] = MagicMock()

from aag_pattern_engine.recognizers.recognizer_utils import merge_split_faces
from aag_pattern_engine.recognizers.fillet_chamfer_recognizer import FilletChamferRecognizer
from aag_pattern_engine.recognizers.hole_recognizer import HoleRecognizer
from aag_pattern_engine.recognizers.pocket_recognizer import PocketRecognizer
from aag_pattern_engine.recognizers.boss_step_island_recognizer import BossRecognizer

class MockAAG:
    def __init__(self, nodes, adjacency):
        self.nodes = nodes
        self.adjacency = adjacency
        
    def get_adjacent_faces(self, face_id):
        return [n['face_id'] for n in self.adjacency.get(face_id, [])]

class TestFeatureRecognitionFixes(unittest.TestCase):
    
    def test_merge_split_faces(self):
        """Test merging of split cylinder faces."""
        # Mock graph with two split cylinder halves
        nodes = {
            1: {'surface_type': 'cylinder', 'radius': 10.0, 'axis': [0,0,1], 'center': [0,0,0]},
            2: {'surface_type': 'cylinder', 'radius': 10.0, 'axis': [0,0,1], 'center': [0,0,0]},
            3: {'surface_type': 'plane'}
        }
        # They are adjacent
        adjacency = {
            1: [{'face_id': 2}, {'face_id': 3}],
            2: [{'face_id': 1}, {'face_id': 3}],
            3: [{'face_id': 1}, {'face_id': 2}]
        }
        
        aag = MockAAG(nodes, adjacency)
        
        # Feature only detected on face 1
        features = [{'type': 'hole', 'face_ids': [1], 'confidence': 0.9}]
        
        merged = merge_split_faces(features, aag)
        
        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0]['face_ids']), {1, 2})
        print("✓ Split face merging verified")

    def test_pocket_logical_bottom(self):
        """Test pocket recognition with split bottom faces."""
        # Mock graph with split bottom (faces 1, 2) and walls (3, 4, 5, 6)
        nodes = {
            1: {'surface_type': 'plane', 'normal': [0,0,1], 'center': [0,0,0], 'area': 10.0},
            2: {'surface_type': 'plane', 'normal': [0,0,1], 'center': [0,0,0], 'area': 10.0},
            3: {'surface_type': 'plane', 'normal': [1,0,0], 'center': [5,0,5]}, # Wall
            4: {'surface_type': 'plane', 'normal': [-1,0,0], 'center': [-5,0,5]}, # Wall
        }
        adjacency = {
            1: [{'face_id': 2}, {'face_id': 3}],
            2: [{'face_id': 1}, {'face_id': 4}],
            3: [{'face_id': 1}],
            4: [{'face_id': 2}]
        }
        
        aag = MockAAG(nodes, adjacency)
        recognizer = PocketRecognizer(aag)
        
        # Mock helper methods to avoid complex geometry checks
        recognizer._is_horizontal = lambda f: True
        recognizer._is_vertical_wall = lambda b, w: True
        recognizer._validate_concave_walls = lambda b, w: True
        recognizer._compute_depth = lambda b, w: 10.0
        recognizer._classify_pocket_type = lambda b, w, d: 'rectangular_pocket'
        
        pockets = recognizer.recognize()
        
        self.assertEqual(len(pockets), 1)
        self.assertEqual(set(pockets[0]['bottom_faces']), {1, 2})
        print("✓ Logical pocket bottom grouping verified")

if __name__ == '__main__':
    unittest.main()
