-- Feature recognition analytics table for monitoring and continuous improvement
CREATE TABLE IF NOT EXISTS feature_recognition_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  -- Correlation
  correlation_id TEXT NOT NULL,
  part_id UUID REFERENCES quotation_submissions(id),
  
  -- Performance metrics
  num_features_detected INTEGER,
  num_features_validated INTEGER DEFAULT 0,
  num_duplicates_removed INTEGER DEFAULT 0,
  processing_time_sec DECIMAL(8,2),
  avg_confidence DECIMAL(3,2),
  
  -- Detection details
  recognition_method TEXT,
  feature_summary JSONB,
  
  -- Customer feedback
  customer_feedback TEXT,
  was_accurate BOOLEAN,
  accuracy_rating INTEGER CHECK (accuracy_rating BETWEEN 1 AND 5),
  
  -- Debugging
  backend_error TEXT,
  crashed BOOLEAN DEFAULT false
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_feature_logs_created ON feature_recognition_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feature_logs_part ON feature_recognition_logs(part_id);
CREATE INDEX IF NOT EXISTS idx_feature_logs_accuracy ON feature_recognition_logs(was_accurate);
CREATE INDEX IF NOT EXISTS idx_feature_logs_method ON feature_recognition_logs(recognition_method);

-- Enable RLS
ALTER TABLE feature_recognition_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
CREATE POLICY "Service role full access"
  ON feature_recognition_logs
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Policy: Public can read all logs (for analytics dashboard)
CREATE POLICY "Public can view logs"
  ON feature_recognition_logs
  FOR SELECT
  TO anon, authenticated
  USING (true);

-- Policy: Public can update logs (for customer feedback)
CREATE POLICY "Public can update logs"
  ON feature_recognition_logs
  FOR UPDATE
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);