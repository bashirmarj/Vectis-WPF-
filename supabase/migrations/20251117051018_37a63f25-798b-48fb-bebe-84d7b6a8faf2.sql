-- Add face_mapping column to cad_meshes table
-- This maps BREP face IDs to mesh triangle indices for feature highlighting
ALTER TABLE cad_meshes 
ADD COLUMN IF NOT EXISTS face_mapping jsonb;

-- Add comment for documentation
COMMENT ON COLUMN cad_meshes.face_mapping IS 'Maps BREP face IDs to triangle indices: {face_id: {triangle_indices: number[], triangle_range: [start, end]}}';