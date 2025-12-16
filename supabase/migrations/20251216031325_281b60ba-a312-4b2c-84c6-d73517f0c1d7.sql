-- Add analysis columns to project_parts table for CAD analysis data
ALTER TABLE project_parts
ADD COLUMN IF NOT EXISTS analysis_data jsonb,
ADD COLUMN IF NOT EXISTS volume_cm3 numeric,
ADD COLUMN IF NOT EXISTS surface_area_cm2 numeric,
ADD COLUMN IF NOT EXISTS complexity_score integer;