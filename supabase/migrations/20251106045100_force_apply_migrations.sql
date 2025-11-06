-- Force trigger migration application
-- This ensures all previous migrations (including bucket creation) are applied

-- Simple query to force migration pipeline execution
SELECT 'Migrations should be applied now' as status;

-- Also ensure RLS is enabled on storage.objects (sometimes needed)
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;
