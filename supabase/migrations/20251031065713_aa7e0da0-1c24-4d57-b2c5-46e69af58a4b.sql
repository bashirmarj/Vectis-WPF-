-- Add tagged_feature_edges column to cad_meshes for direct feature_id lookup
ALTER TABLE cad_meshes ADD COLUMN IF NOT EXISTS tagged_feature_edges jsonb DEFAULT '[]'::jsonb;