-- Add edge_classifications column to cad_meshes table
ALTER TABLE cad_meshes 
ADD COLUMN IF NOT EXISTS edge_classifications JSONB;