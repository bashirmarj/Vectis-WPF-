-- Add dataset_id column to training_jobs table
ALTER TABLE public.training_jobs 
ADD COLUMN dataset_id uuid REFERENCES public.training_datasets(id);

-- Create index for better query performance
CREATE INDEX idx_training_jobs_dataset_id ON public.training_jobs(dataset_id);