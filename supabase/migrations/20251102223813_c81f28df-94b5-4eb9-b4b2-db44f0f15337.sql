-- Create storage buckets for ML training
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES 
  ('training-datasets', 'training-datasets', false, 52428800, ARRAY['application/octet-stream', 'application/json']::text[]),
  ('trained-models', 'trained-models', false, 524288000, ARRAY['application/octet-stream']::text[]);

-- Create storage policies for training datasets (admin only)
CREATE POLICY "Admin can upload training datasets"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'training-datasets' AND
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

CREATE POLICY "Admin can view training datasets"
ON storage.objects
FOR SELECT
TO authenticated
USING (
  bucket_id = 'training-datasets' AND
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

CREATE POLICY "Admin can delete training datasets"
ON storage.objects
FOR DELETE
TO authenticated
USING (
  bucket_id = 'training-datasets' AND
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

-- Create storage policies for trained models (admin only)
CREATE POLICY "Admin can upload trained models"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'trained-models' AND
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

CREATE POLICY "Admin can view trained models"
ON storage.objects
FOR SELECT
TO authenticated
USING (
  bucket_id = 'trained-models' AND
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

CREATE POLICY "Service role can access trained models"
ON storage.objects
FOR SELECT
TO service_role
USING (bucket_id = 'trained-models');

-- Create training jobs table
CREATE TABLE IF NOT EXISTS public.training_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  started_by UUID REFERENCES auth.users(id),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'downloading', 'processing', 'training', 'completed', 'failed')),
  dataset_name TEXT NOT NULL,
  dataset_path TEXT,
  model_version TEXT,
  model_path TEXT,
  train_samples INTEGER,
  val_samples INTEGER,
  test_samples INTEGER,
  epochs INTEGER DEFAULT 100,
  batch_size INTEGER DEFAULT 16,
  learning_rate DOUBLE PRECISION DEFAULT 0.001,
  best_val_accuracy DOUBLE PRECISION,
  best_val_iou DOUBLE PRECISION,
  final_train_loss DOUBLE PRECISION,
  training_time_seconds INTEGER,
  error_message TEXT,
  logs JSONB DEFAULT '[]'::jsonb,
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Enable RLS on training_jobs
ALTER TABLE public.training_jobs ENABLE ROW LEVEL SECURITY;

-- Admin can view all training jobs
CREATE POLICY "Admin can view all training jobs"
ON public.training_jobs
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

-- Admin can create training jobs
CREATE POLICY "Admin can create training jobs"
ON public.training_jobs
FOR INSERT
TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

-- Admin can update training jobs
CREATE POLICY "Admin can update training jobs"
ON public.training_jobs
FOR UPDATE
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

-- Create trigger for updated_at
CREATE TRIGGER update_training_jobs_updated_at
BEFORE UPDATE ON public.training_jobs
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create dataset metadata table
CREATE TABLE IF NOT EXISTS public.training_datasets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  name TEXT UNIQUE NOT NULL,
  description TEXT,
  source_url TEXT,
  storage_path TEXT,
  num_files INTEGER,
  num_classes INTEGER,
  file_format TEXT,
  download_status TEXT DEFAULT 'pending' CHECK (download_status IN ('pending', 'downloading', 'completed', 'failed')),
  download_progress INTEGER DEFAULT 0,
  total_size_bytes BIGINT,
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Enable RLS on training_datasets
ALTER TABLE public.training_datasets ENABLE ROW LEVEL SECURITY;

-- Admin can manage datasets
CREATE POLICY "Admin can view training datasets"
ON public.training_datasets
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

CREATE POLICY "Admin can create training datasets"
ON public.training_datasets
FOR INSERT
TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

CREATE POLICY "Admin can update training datasets"
ON public.training_datasets
FOR UPDATE
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
);

-- Insert MFCAD dataset metadata
INSERT INTO public.training_datasets (name, description, source_url, num_classes, file_format, metadata)
VALUES (
  'MFCAD',
  'Machining Feature CAD Dataset - B-Rep graphs with 16 feature classes for machining feature recognition',
  'https://github.com/hducg/UV-Net',
  16,
  'DGL Binary',
  '{
    "classes": [
      "rectangular_through_slot",
      "triangular_through_slot", 
      "rectangular_passage",
      "triangular_passage",
      "6sides_passage",
      "rectangular_through_step",
      "2sides_through_step",
      "slanted_through_step",
      "rectangular_blind_step",
      "triangular_blind_step",
      "rectangular_blind_slot",
      "rectangular_pocket",
      "triangular_pocket",
      "6sides_pocket",
      "chamfer",
      "stock"
    ],
    "paper": "Graph representation of 3d cad models for machining feature recognition with deep learning (IDETC-CIE 2020)"
  }'::jsonb
)
ON CONFLICT (name) DO NOTHING;