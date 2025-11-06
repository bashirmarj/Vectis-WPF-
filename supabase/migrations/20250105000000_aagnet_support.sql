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
COMMENT ON COLUMN manufacturing_feature_instances.face_indices IS 'Array of face indices that constitute this feature instance';
COMMENT ON COLUMN manufacturing_feature_instances.bottom_face_indices IS 'Array of bottom face indices for this feature';
COMMENT ON COLUMN manufacturing_feature_instances.confidence IS 'Confidence score from neural network (0-1)';

-- Create function to automatically populate feature instances table
CREATE OR REPLACE FUNCTION populate_feature_instances()
RETURNS TRIGGER AS $$
BEGIN
  -- Only populate if instance_features is not null
  IF NEW.instance_features IS NOT NULL AND NEW.instance_features->'instances' IS NOT NULL THEN
    -- Delete existing instances for this mesh
    DELETE FROM manufacturing_feature_instances WHERE mesh_id = NEW.id;
    
    -- Insert new instances
    INSERT INTO manufacturing_feature_instances (
      mesh_id,
      feature_type,
      face_indices,
      bottom_face_indices,
      confidence
    )
    SELECT
      NEW.id,
      instance->>'type',
      ARRAY(SELECT jsonb_array_elements_text(instance->'face_indices')::INTEGER),
      ARRAY(SELECT jsonb_array_elements_text(instance->'bottom_faces')::INTEGER),
      (instance->>'confidence')::FLOAT
    FROM jsonb_array_elements(NEW.instance_features->'instances') AS instance;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS populate_feature_instances_trigger ON cad_meshes;
CREATE TRIGGER populate_feature_instances_trigger
  AFTER INSERT OR UPDATE OF instance_features ON cad_meshes
  FOR EACH ROW
  EXECUTE FUNCTION populate_feature_instances();

-- Add RLS policies (if RLS is enabled)
ALTER TABLE manufacturing_feature_instances ENABLE ROW LEVEL SECURITY;

-- Policy: Allow authenticated users to read their own feature instances
CREATE POLICY "Users can view their own feature instances"
  ON manufacturing_feature_instances
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM cad_meshes
      WHERE cad_meshes.id = manufacturing_feature_instances.mesh_id
      -- Add your user/organization ownership check here
    )
  );

-- Policy: Service role can do anything
CREATE POLICY "Service role has full access to feature instances"
  ON manufacturing_feature_instances
  FOR ALL
  USING (auth.role() = 'service_role');

-- Example data structure for instance_features JSONB:
-- {
--   "instances": [
--     {
--       "type": "through_hole",
--       "face_indices": [0, 1, 2],
--       "bottom_faces": [],
--       "confidence": 0.95
--     },
--     {
--       "type": "rectangular_pocket",
--       "face_indices": [3, 4, 5, 6, 7],
--       "bottom_faces": [7],
--       "confidence": 0.88
--     }
--   ],
--   "num_instances": 2,
--   "num_faces": 50
-- }

-- Example data structure for extended_attributes JSONB:
-- {
--   "face_attributes": [
--     [1.0, 0.0, 0.0, 0.0, 0.0, 0.523, 0.0, 0.1, 0.2, 0.3],  -- Face 0: plane, area, centroid
--     [0.0, 1.0, 0.0, 0.0, 0.0, 0.785, 0.0, 0.5, 0.6, 0.7]   -- Face 1: cylinder, area, centroid
--   ],
--   "edge_attributes": [
--     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.314],  -- Edge convexity + length
--     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.628]
--   ]
-- }

-- Grant permissions
GRANT ALL ON manufacturing_feature_instances TO authenticated;
GRANT ALL ON manufacturing_feature_instances TO service_role;

-- Create view for easier querying
CREATE OR REPLACE VIEW v_feature_summary AS
SELECT 
  m.id AS mesh_id,
  m.triangle_count,
  m.recognition_method,
  m.processing_time_ms,
  COUNT(f.id) AS total_features,
  AVG(f.confidence) AS avg_confidence,
  COUNT(CASE WHEN f.confidence >= 0.9 THEN 1 END) AS high_confidence_features,
  COUNT(CASE WHEN f.confidence >= 0.7 AND f.confidence < 0.9 THEN 1 END) AS medium_confidence_features,
  COUNT(CASE WHEN f.confidence < 0.7 THEN 1 END) AS low_confidence_features,
  jsonb_agg(
    jsonb_build_object(
      'type', f.feature_type,
      'confidence', f.confidence,
      'num_faces', array_length(f.face_indices, 1)
    ) ORDER BY f.confidence DESC
  ) AS features
FROM cad_meshes m
LEFT JOIN manufacturing_feature_instances f ON f.mesh_id = m.id
GROUP BY m.id, m.triangle_count, m.recognition_method, m.processing_time_ms;

-- Grant view permissions
GRANT SELECT ON v_feature_summary TO authenticated;
GRANT SELECT ON v_feature_summary TO service_role;

COMMENT ON VIEW v_feature_summary IS 'Summary view of feature recognition results per mesh';
