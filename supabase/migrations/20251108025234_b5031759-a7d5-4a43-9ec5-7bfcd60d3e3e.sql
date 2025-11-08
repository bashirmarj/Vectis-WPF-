-- Migration: Create cad_processing_audit table for ISO 9001 compliance
-- Purpose: Audit trail for all CAD processing decisions (traceability & reproducibility)
-- Created: 2025-11-07

-- Drop table if exists (for clean reinstall)
DROP TABLE IF EXISTS public.cad_processing_audit CASCADE;

-- Create audit table
CREATE TABLE public.cad_processing_audit (
    id BIGSERIAL PRIMARY KEY,
    
    -- Identification
    event_type TEXT NOT NULL,                    -- validation_complete, healing_applied, processing_complete, circuit_breaker_opened
    request_id TEXT NOT NULL,                    -- Correlation ID for request tracking
    
    -- Timing
    timestamp TIMESTAMPTZ NOT NULL,              -- Event timestamp
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Event details (JSONB for flexibility)
    details JSONB DEFAULT '{}'::jsonb,
    
    -- Indexing for performance
    CONSTRAINT event_type_check CHECK (event_type IN (
        'validation_complete',
        'healing_applied',
        'processing_complete',
        'circuit_breaker_opened',
        'circuit_breaker_closed',
        'error_occurred'
    ))
);

-- Create indexes for common queries
CREATE INDEX idx_cad_processing_audit_request_id ON public.cad_processing_audit(request_id);
CREATE INDEX idx_cad_processing_audit_event_type ON public.cad_processing_audit(event_type);
CREATE INDEX idx_cad_processing_audit_created_at ON public.cad_processing_audit(created_at DESC);
CREATE INDEX idx_cad_processing_audit_timestamp ON public.cad_processing_audit(timestamp DESC);

-- Create composite index for common time-range + event-type queries
CREATE INDEX idx_cad_processing_audit_created_event ON public.cad_processing_audit(created_at DESC, event_type);

-- Add comments for documentation
COMMENT ON TABLE public.cad_processing_audit IS 'ISO 9001 compliance audit trail for CAD processing decisions';
COMMENT ON COLUMN public.cad_processing_audit.event_type IS 'Type of processing event (validation_complete, healing_applied, etc.)';
COMMENT ON COLUMN public.cad_processing_audit.request_id IS 'Correlation ID linking related events in the same request';
COMMENT ON COLUMN public.cad_processing_audit.timestamp IS 'Timestamp when event occurred';
COMMENT ON COLUMN public.cad_processing_audit.details IS 'Event-specific details stored as JSONB (file_hash, quality_score, feature_count, etc.)';

-- Enable Row Level Security (RLS)
ALTER TABLE public.cad_processing_audit ENABLE ROW LEVEL SECURITY;

-- Create policy: Allow service role to insert audit logs
CREATE POLICY "Allow service role to insert audit logs"
ON public.cad_processing_audit
FOR INSERT
TO service_role
WITH CHECK (true);

-- Create policy: Allow service role to read audit logs
CREATE POLICY "Allow service role to read audit logs"
ON public.cad_processing_audit
FOR SELECT
TO service_role
USING (true);

-- Grant permissions
GRANT INSERT, SELECT ON public.cad_processing_audit TO service_role;
GRANT USAGE ON SEQUENCE public.cad_processing_audit_id_seq TO service_role;

-- Create function to automatically set timestamp if not provided
CREATE OR REPLACE FUNCTION public.set_audit_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.timestamp IS NULL THEN
        NEW.timestamp := NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to auto-set timestamp
CREATE TRIGGER set_audit_timestamp_trigger
    BEFORE INSERT ON public.cad_processing_audit
    FOR EACH ROW
    EXECUTE FUNCTION public.set_audit_timestamp();

-- Create view for recent processing events (last 24 hours)
CREATE OR REPLACE VIEW public.recent_processing_events AS
SELECT 
    id,
    event_type,
    request_id,
    timestamp,
    created_at,
    details,
    details->>'file_hash' AS file_hash,
    (details->>'processing_time')::numeric AS processing_time_sec,
    (details->>'quality_score')::numeric AS quality_score,
    details->>'recognition_status' AS recognition_status,
    (details->>'feature_count')::integer AS feature_count
FROM public.cad_processing_audit
WHERE created_at >= NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

-- Grant permissions on view
GRANT SELECT ON public.recent_processing_events TO service_role;

-- Insert initial test record to verify table is working
INSERT INTO public.cad_processing_audit (
    event_type,
    request_id,
    timestamp,
    details
) VALUES (
    'validation_complete',
    'migration_test',
    NOW(),
    jsonb_build_object(
        'test', true,
        'migration_version', '20251107_cad_processing_audit',
        'description', 'Initial migration test record'
    )
);

-- Create retention policy function (optional - auto-delete old logs after 90 days)
CREATE OR REPLACE FUNCTION public.cleanup_old_audit_logs()
RETURNS void AS $$
BEGIN
    DELETE FROM public.cad_processing_audit
    WHERE created_at < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION public.cleanup_old_audit_logs() TO service_role;