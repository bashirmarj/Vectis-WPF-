# New file: aag_pattern_engine/recognizers/rib_web_recognizer.py

"""
Rib and Web Feature Recognizer
Recognizes structural reinforcement features
"""

@dataclass
class RibWebFeature:
    """Rib or web feature"""
    type: str  # 'rib', 'web', 'gusset'
    face_ids: List[int]
    
    thickness: float
    height: float
    length: float
    
    connects_faces: List[int]  # Faces being connected
    
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


class RibWebRecognizer:
    """
    Recognize ribs and webs (thin structural features)
    """
    
    def recognize_ribs_webs(self, graph: Dict) -> List[RibWebFeature]:
        """Recognize all ribs and webs"""
        nodes = graph['nodes']
        adjacency = self._build_adjacency_map(graph)
        
        ribs_webs = []
        
        # Find thin vertical planar faces
        for node in nodes:
            if node.surface_type != SurfaceType.PLANE:
                continue
            
            # Check if vertical
            normal = np.array(node.normal)
            up = np.array([0, 0, 1])
            if abs(np.dot(normal, up)) > 0.2:
                continue  # Not vertical
            
            # Check if thin (high aspect ratio)
            # Estimate dimensions from area
            area = node.area
            
            # Ribs are thin: estimate thickness from adjacent faces
            thickness = self._estimate_thickness(node, adjacency, nodes)
            
            if thickness > 0.010:  # > 10mm = not a rib
                continue
            
            # Find connected faces
            connected = self._find_connected_faces(node, adjacency, nodes)
            
            if len(connected) >= 2:
                # This is a rib connecting two surfaces
                height = self._estimate_height(node)
                length = area / height if height > 0 else 0
                
                ribs_webs.append(RibWebFeature(
                    type='rib',
                    face_ids=[node.id],
                    thickness=thickness,
                    height=height,
                    length=length,
                    connects_faces=connected,
                    confidence=0.87
                ))
        
        return ribs_webs
    
    def _estimate_thickness(self, face: GraphNode, adjacency: Dict, nodes: List[GraphNode]) -> float:
        """Estimate rib thickness"""
        # Find parallel opposite face
        adjacent = adjacency[face.id]
        normal = np.array(face.normal)
        
        for adj in adjacent:
            adj_node = nodes[adj['node_id']]
            if adj_node.surface_type == SurfaceType.PLANE:
                adj_normal = np.array(adj_node.normal)
                # Check if parallel and opposite
                dot = np.dot(normal, adj_normal)
                if dot < -0.9:
                    # Calculate distance
                    center1 = np.array(face.centroid)
                    center2 = np.array(adj_node.centroid)
                    thickness = np.linalg.norm(center1 - center2)
                    return thickness
        
        # Estimate from area (assume rectangular)
        return np.sqrt(face.area) * 0.1
    
    def _estimate_height(self, face: GraphNode) -> float:
        """Estimate rib height (vertical extent)"""
        # Simplified: assume height is dominant dimension
        return np.sqrt(face.area)
    
    def _find_connected_faces(self, rib: GraphNode, adjacency: Dict, nodes: List[GraphNode]) -> List[int]:
        """Find faces that rib connects"""
        adjacent = adjacency[rib.id]
        connected = []
        
        for adj in adjacent:
            adj_node = nodes[adj['node_id']]
            # Large planar faces are likely what rib connects
            if adj_node.surface_type == SurfaceType.PLANE and adj_node.area > rib.area * 2:
                connected.append(adj_node.id)
        
        return connected
