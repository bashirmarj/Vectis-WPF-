    def _is_hole_not_fillet(
        self,
        candidate: GraphNode,
        adjacency: Dict
    ) -> bool:
        """
        Distinguish hole cylinder from fillet cylinder using topology.
        
        Returns True if this is a HOLE (reject as fillet), False otherwise.
        """
        candidate_id = candidate.id
        
        if candidate_id not in adjacency:
            return False
        
        adjacent_list = adjacency[candidate_id]
        
        # Count neighbor types
        planar_count = 0
        for adj_edge in adjacent_list:
            neighbor_type = adj_edge.get('surface_type', '')
            if 'plane' in str(neighbor_type).lower():
                planar_count += 1
        
        # HOLE CRITERION: Adjacent to planar caps
        if planar_count >= 1:
            return True  # It's a hole
        
        # RADIUS CHECK: Too large for a fillet
        if candidate.radius and candidate.radius > (self.max_fillet_radius / 1000.0):
            return True  # Too large, must be a hole
        
        return False  # Might be a fillet
    
