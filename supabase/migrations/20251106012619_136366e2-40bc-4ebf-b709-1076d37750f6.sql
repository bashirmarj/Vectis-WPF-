-- Migration: Add AAGNet multi-task feature recognition support
-- Date: 2025-01-05
-- Description: Extends cad_meshes table to store AAGNet instance segmentation and extended attributes

-- Add new columns to cad_meshes table for AAGNet output
ALTER TABLE cad_meshes 
ADD COLUMN IF NOT EXISTS instance_features JSONB,
ADD COLUMN IF NOT EXISTS extended_attributes JSONB,
ADD COLUMN IF NOT EXISTS semantic_labels INTEGER[],
ADD COLUMN IF NOT EXISTS recognition_method VARCHAR(50) DEFAULT 'AAGNet',
ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER;

-- Add indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_cad_meshes_instance_features 
ON cad_meshes USING gin(instance_features);

CREATE INDEX IF NOT EXISTS idx_cad_meshes_recognition_method 
ON cad_meshes(recognition_method);

-- Add comments for documentation
COMMENT ON COLUMN cad_meshes.instance_features IS 'AAGNet instance segmentation results: {instances: [{type, face_indices, bottom_faces, confidence}]}';
COMMENT ON COLUMN cad_meshes.extended_attributes IS 'Extended geometric attributes: {face_attributes: [[]], edge_attributes: [[]]}';
COMMENT ON COLUMN cad_meshes.semantic_labels IS 'Per-face semantic segmentation labels (0-24, where 24 is stock)';
COMMENT ON COLUMN cad_meshes.recognition_method IS 'Feature recognition method used: AAGNet, UV-Net, or rule-based';
COMMENT ON COLUMN cad_meshes.processing_time_ms IS 'Time taken for feature recognition in milliseconds';

-- Create new table for feature instance details (optional - for detailed queries)
CREATE TABLE IF NOT EXISTS manufacturing_feature_instances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  mesh_id UUID REFERENCES cad_meshes(id) ON DELETE CASCADE,
  feature_type VARCHAR(100) NOT NULL,
  face_indices INTEGER[] NOT NULL,
  bottom_face_indices INTEGER[],
  confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for feature instance queries
CREATE INDEX IF NOT EXISTS idx_feature_instances_mesh_id 
ON manufacturing_feature_instances(mesh_id);

CREATE INDEX IF NOT EXISTS idx_feature_instances_feature_type 
ON manufacturing_feature_instances(feature_type);

CREATE INDEX IF NOT EXISTS idx_feature_instances_confidence 
ON manufacturing_feature_instances(confidence);

-- Add comments
COMMENT ON TABLE manufacturing_feature_instances IS 'Individual machining feature instances detected by AAGNet';
COMMENT ON COLUMN manufacturing_feature_instances.feature_type IS 'Machining feature class: chamfer, through_hole, blind_hole, pocket, slot, step, etc.';
COMMENT ON COLUMN manufacturing_feature_instances.face_indices IS 'Array of face indices belonging to this feature instance';
COMMENT ON COLUMN manufacturing_feature_instances.bottom_face_indices IS 'Array of bottom face indices for this feature (if applicable)';
COMMENT ON COLUMN manufacturing_feature_instances.confidence IS 'Model confidence score (0.0 to 1.0)';

-- Add trigger for updated_at
CREATE TRIGGER update_feature_instances_updated_at
BEFORE UPDATE ON manufacturing_feature_instances
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Enable RLS on manufacturing_feature_instances
ALTER TABLE manufacturing_feature_instances ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for manufacturing_feature_instances
CREATE POLICY "Users can view all feature instances"
ON manufacturing_feature_instances
FOR SELECT
USING (true);

CREATE POLICY "Service role can insert feature instances"
ON manufacturing_feature_instances
FOR INSERT
WITH CHECK (true);

CREATE POLICY "Service role can update feature instances"
ON manufacturing_feature_instances
FOR UPDATE
USING (true);

CREATE POLICY "Service role can delete feature instances"
ON manufacturing_feature_instances
FOR DELETE
USING (true);