-- Add geometric_features column to cad_meshes table
ALTER TABLE cad_meshes 
ADD COLUMN IF NOT EXISTS geometric_features JSONB DEFAULT NULL;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_cad_meshes_geometric_features 
ON cad_meshes USING gin(geometric_features);