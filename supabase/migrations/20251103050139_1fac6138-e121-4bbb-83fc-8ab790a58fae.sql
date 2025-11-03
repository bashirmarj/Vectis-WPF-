-- Add 'cancelled' to the allowed status values for training_jobs
-- First, let's check what constraint exists
DO $$ 
BEGIN
  -- Drop the existing constraint if it exists
  IF EXISTS (
    SELECT 1 FROM pg_constraint 
    WHERE conname = 'training_jobs_status_check'
  ) THEN
    ALTER TABLE public.training_jobs DROP CONSTRAINT training_jobs_status_check;
  END IF;
  
  -- Add new constraint with 'cancelled' included
  ALTER TABLE public.training_jobs 
  ADD CONSTRAINT training_jobs_status_check 
  CHECK (status IN ('pending', 'pending_local', 'running', 'completed', 'failed', 'cancelled'));
END $$;