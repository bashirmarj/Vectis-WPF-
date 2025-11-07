-- Create failed_cad_analyses table for Dead Letter Queue
CREATE TABLE IF NOT EXISTS public.failed_cad_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  correlation_id TEXT NOT NULL,
  file_path TEXT NOT NULL,
  error_type TEXT NOT NULL CHECK (error_type IN ('transient', 'permanent', 'systemic')),
  error_message TEXT NOT NULL,
  error_details JSONB DEFAULT '{}'::jsonb,
  retry_count INTEGER DEFAULT 0,
  traceback TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_failed_cad_analyses_correlation_id ON public.failed_cad_analyses(correlation_id);
CREATE INDEX IF NOT EXISTS idx_failed_cad_analyses_error_type ON public.failed_cad_analyses(error_type);
CREATE INDEX IF NOT EXISTS idx_failed_cad_analyses_created_at ON public.failed_cad_analyses(created_at DESC);

-- Enable RLS
ALTER TABLE public.failed_cad_analyses ENABLE ROW LEVEL SECURITY;

-- Service role has full access
CREATE POLICY "Service role has full access to failed_cad_analyses"
ON public.failed_cad_analyses
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Authenticated users can read failures (for debugging)
CREATE POLICY "Authenticated users can view failed_cad_analyses"
ON public.failed_cad_analyses
FOR SELECT
TO authenticated
USING (true);