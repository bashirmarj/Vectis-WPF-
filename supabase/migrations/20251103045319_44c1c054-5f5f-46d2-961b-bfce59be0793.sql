-- Add cancelled status and tracking columns to training_jobs
ALTER TABLE public.training_jobs 
ADD COLUMN IF NOT EXISTS cancelled_at timestamp with time zone,
ADD COLUMN IF NOT EXISTS cancelled_by uuid REFERENCES public.profiles(id);