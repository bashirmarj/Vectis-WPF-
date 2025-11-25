"""
Patch fillet_chamfer_recognizer.py to add topology-based hole filtering
"""

# Read the file
with open('aag_pattern_engine/recognizers/fillet_chamfer_recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and modify max_fillet_radius (line ~207)
for i, line in enumerate(lines):
    if 'self.max_fillet_radius = 100.0' in line:
        lines[i] = '        self.max_fillet_radius = 3.0     # 3mm (PRODUCTION: Strict limit to exclude hole cylinders)\r\n'
        print(f"OK Updated max_fillet_radius at line {i+1}")
        break

# Find insertion point for _is_hole_not_fillet() (before _is_fillet_surface, line ~392)
for i, line in enumerate(lines):
    if 'def _is_fillet_surface(' in line and i > 300:
        # Insert the new method before this line
        new_method = '''    def _is_hole_not_fillet(
        self,
        candidate: GraphNode,
        adjacency: Dict
    ) -> bool:
        """Distinguish hole cylinder from fillet cylinder using topology."""
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
            return True
        
        # RADIUS CHECK: Too large for a fillet
        if candidate.radius and candidate.radius > (self.max_fillet_radius / 1000.0):
            return True
        
        return False
    
'''
        lines.insert(i, new_method)
        print(f"OK Inserted _is_hole_not_fillet() at line {i+1}")
        break

# Find and call _is_hole_not_fillet() in _is_fillet_surface (after blend check, before "# Validate radius")
for i, line in enumerate(lines):
    if '# Validate radius' in line and 'candidate.radius' in lines[i+1] and i > 450:
        # Insert the filter call before this comment
        filter_call = '''        # PRODUCTION FIX: Check if this is a hole cylinder (not a fillet)
        if self._is_hole_not_fillet(candidate, adjacency):
            return False
        
'''
        lines.insert(i, filter_call)
        print(f"OK Inserted hole filter call at line {i+1}")
        break

# Find and remove isolated cylinder fallback (lines ~478-488, in the blend count check)
in_isolated_block = False
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if '# CRITICAL FIX: Handle isolated cylindrical blends' in line:
        in_isolated_block = True
        start_idx = i 
    elif in_isolated_block and 'return False' in line and 'Radius out of range' in lines[i-1]:
        end_idx = i + 2  # Include the return and blank line
        break

if start_idx and end_idx:
    # Replace with simpler logic
    replacement = '''             # PRODUCTION FIX: Removed "isolated cylindrical blend" fallback
            # Reason: Isolated cylinders are typically holes, not fillets
            # True fillets MUST blend at least 2 faces

'''
    del lines[start_idx:end_idx]
    lines.insert(start_idx, replacement)
    print(f"OK Removed isolated cylinder fallback (lines {start_idx+1}-{end_idx+1})")

# Write back
with open('aag_pattern_engine/recognizers/fillet_chamfer_recognizer.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\nDONE Patch complete!")
