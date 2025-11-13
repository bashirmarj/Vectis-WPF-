-- Migration: Remove AAGNet support and upgrade to full prismatic taxonomy
-- Date: 2025-11-12
-- Description: Removes AAGNet-specific columns, drops manufacturing_feature_instances table,
--              and prepares for comprehensive 60+ prismatic feature taxonomy

-- Drop manufacturing_feature_instances table (AAGNet-specific)
DROP TABLE IF EXISTS manufacturing_feature_instances CASCADE;

-- Remove AAGNet-specific columns from cad_meshes
ALTER TABLE cad_meshes 
DROP COLUMN IF EXISTS instance_features,
DROP COLUMN IF EXISTS extended_attributes,
DROP COLUMN IF EXISTS semantic_labels,
DROP COLUMN IF EXISTS processing_time_ms;

-- Drop old check constraint
ALTER TABLE cad_meshes
DROP CONSTRAINT IF EXISTS cad_meshes_recognition_method_check;

-- Add new check constraint with additional allowed values
ALTER TABLE cad_meshes
ADD CONSTRAINT cad_meshes_recognition_method_check 
CHECK (recognition_method IN (
  'AAGNet', 
  'rule_based', 
  'hybrid', 
  'rule_based_enhanced', 
  'edge_based_taxonomy',
  'crash_free',
  'production'
));

-- Update recognition_method column default
ALTER TABLE cad_meshes 
ALTER COLUMN recognition_method SET DEFAULT 'rule_based_enhanced';

-- Update existing records to new method
UPDATE cad_meshes 
SET recognition_method = 'rule_based_enhanced' 
WHERE recognition_method = 'AAGNet' OR recognition_method IS NULL;

-- Add comment explaining the new system
COMMENT ON COLUMN cad_meshes.recognition_method IS 'Feature recognition method: rule_based_enhanced (60+ prismatic features), edge_based_taxonomy, crash_free, or production';

-- Expand part_features table to support comprehensive taxonomy
ALTER TABLE part_features
ADD COLUMN IF NOT EXISTS feature_hierarchy VARCHAR(100),
ADD COLUMN IF NOT EXISTS parent_feature_id UUID REFERENCES part_features(id) ON DELETE CASCADE,
ADD COLUMN IF NOT EXISTS boundary_condition VARCHAR(50),
ADD COLUMN IF NOT EXISTS profile_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS tolerance_class VARCHAR(50),
ADD COLUMN IF NOT EXISTS surface_finish VARCHAR(50);

-- Add indexes for new columns
CREATE INDEX IF NOT EXISTS idx_part_features_hierarchy 
ON part_features(feature_hierarchy);

CREATE INDEX IF NOT EXISTS idx_part_features_parent 
ON part_features(parent_feature_id);

CREATE INDEX IF NOT EXISTS idx_part_features_boundary 
ON part_features(boundary_condition);

-- Add comments
COMMENT ON COLUMN part_features.feature_hierarchy IS 'Feature classification: hole, pocket, slot, step, boss, fillet, chamfer, etc.';
COMMENT ON COLUMN part_features.parent_feature_id IS 'Parent feature for compound features (e.g., counterbore -> pilot hole)';
COMMENT ON COLUMN part_features.boundary_condition IS 'Boundary type: closed, open, through, blind, etc.';
COMMENT ON COLUMN part_features.profile_type IS 'Profile shape: rectangular, circular, irregular, etc.';
COMMENT ON COLUMN part_features.tolerance_class IS 'Tolerance specification per ISO or ASME Y14.5';
COMMENT ON COLUMN part_features.surface_finish IS 'Surface finish requirement (Ra value or class)';