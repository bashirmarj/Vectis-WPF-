-- Create ground truth features table for labeled training data
CREATE TABLE public.ground_truth_features (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  file_name TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  features JSONB NOT NULL DEFAULT '{}'::jsonb,
  feature_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
  source TEXT NOT NULL DEFAULT 'manual',
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  created_by UUID REFERENCES public.profiles(id),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create index on file_hash for quick lookups
CREATE INDEX idx_ground_truth_file_hash ON public.ground_truth_features(file_hash);

-- Enable RLS
ALTER TABLE public.ground_truth_features ENABLE ROW LEVEL SECURITY;

-- RLS Policies for ground_truth_features
CREATE POLICY "Admins can view all ground truth features"
  ON public.ground_truth_features
  FOR SELECT
  USING (has_role(auth.uid(), 'admin'::app_role));

CREATE POLICY "Admins can insert ground truth features"
  ON public.ground_truth_features
  FOR INSERT
  WITH CHECK (has_role(auth.uid(), 'admin'::app_role));

CREATE POLICY "Admins can update ground truth features"
  ON public.ground_truth_features
  FOR UPDATE
  USING (has_role(auth.uid(), 'admin'::app_role));

CREATE POLICY "Admins can delete ground truth features"
  ON public.ground_truth_features
  FOR DELETE
  USING (has_role(auth.uid(), 'admin'::app_role));

-- Create feature validation table for tracking model performance
CREATE TABLE public.feature_validations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  quotation_id UUID REFERENCES public.quotation_submissions(id) ON DELETE CASCADE,
  line_item_id UUID REFERENCES public.quote_line_items(id) ON DELETE CASCADE,
  file_name TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  model_version TEXT NOT NULL DEFAULT 'uvnet-v1',
  model_type TEXT NOT NULL DEFAULT 'rule-based',
  detected_features JSONB NOT NULL DEFAULT '{}'::jsonb,
  ground_truth_id UUID REFERENCES public.ground_truth_features(id) ON DELETE SET NULL,
  metrics JSONB DEFAULT '{}'::jsonb,
  validation_status TEXT NOT NULL DEFAULT 'pending',
  confidence_scores JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  validated_at TIMESTAMP WITH TIME ZONE,
  validated_by UUID REFERENCES public.profiles(id),
  notes TEXT
);

-- Create indexes for efficient queries
CREATE INDEX idx_validations_quotation ON public.feature_validations(quotation_id);
CREATE INDEX idx_validations_file_hash ON public.feature_validations(file_hash);
CREATE INDEX idx_validations_status ON public.feature_validations(validation_status);
CREATE INDEX idx_validations_model_version ON public.feature_validations(model_version);

-- Enable RLS
ALTER TABLE public.feature_validations ENABLE ROW LEVEL SECURITY;

-- RLS Policies for feature_validations
CREATE POLICY "Admins can view all validations"
  ON public.feature_validations
  FOR SELECT
  USING (has_role(auth.uid(), 'admin'::app_role));

CREATE POLICY "Admins can insert validations"
  ON public.feature_validations
  FOR INSERT
  WITH CHECK (has_role(auth.uid(), 'admin'::app_role));

CREATE POLICY "Admins can update validations"
  ON public.feature_validations
  FOR UPDATE
  USING (has_role(auth.uid(), 'admin'::app_role));

CREATE POLICY "Admins can delete validations"
  ON public.feature_validations
  FOR DELETE
  USING (has_role(auth.uid(), 'admin'::app_role));

-- Create model performance tracking table
CREATE TABLE public.model_performance (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  model_version TEXT NOT NULL,
  model_type TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  total_files INTEGER NOT NULL DEFAULT 0,
  overall_accuracy NUMERIC,
  feature_type_metrics JSONB DEFAULT '{}'::jsonb,
  confusion_matrix JSONB DEFAULT '{}'::jsonb,
  average_inference_time_ms NUMERIC,
  trained_at TIMESTAMP WITH TIME ZONE,
  evaluated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  notes TEXT,
  UNIQUE(model_version, dataset_name)
);

-- Enable RLS
ALTER TABLE public.model_performance ENABLE ROW LEVEL SECURITY;

-- RLS Policies for model_performance
CREATE POLICY "Anyone can view model performance"
  ON public.model_performance
  FOR SELECT
  USING (true);

CREATE POLICY "Admins can manage model performance"
  ON public.model_performance
  FOR ALL
  USING (has_role(auth.uid(), 'admin'::app_role));

-- Add trigger for updated_at on ground_truth_features
CREATE TRIGGER update_ground_truth_updated_at
  BEFORE UPDATE ON public.ground_truth_features
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();