-- Create cad-files storage bucket and policies
-- This fixes the "Bucket not found" error when uploading STEP files

-- 1) Create the cad-files bucket if it doesn't exist
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
SELECT 
    'cad-files',
    'cad-files', 
    true,
    52428800, -- 50MB limit
    ARRAY['application/step', 'application/stp', 'application/iges', 'application/igs', 'text/plain', 'application/octet-stream']
WHERE NOT EXISTS (
    SELECT 1 FROM storage.buckets WHERE id = 'cad-files'
);

-- 2) Allow public read access to all objects in cad-files bucket
CREATE POLICY IF NOT EXISTS "Public read access for cad-files" 
ON storage.objects 
FOR SELECT 
TO public 
USING (bucket_id = 'cad-files');

-- 3) Allow authenticated users to upload files to uploads/ folder
CREATE POLICY IF NOT EXISTS "Authenticated upload to cad-files/uploads" 
ON storage.objects 
FOR INSERT 
TO authenticated 
WITH CHECK (
    bucket_id = 'cad-files' 
    AND (storage.foldername(name))[1] = 'uploads'
    AND auth.role() = 'authenticated'
);

-- 4) Allow authenticated users to update their own uploaded files
CREATE POLICY IF NOT EXISTS "Users can update own cad-files objects" 
ON storage.objects 
FOR UPDATE 
TO authenticated 
USING (
    bucket_id = 'cad-files' 
    AND owner = auth.uid()
) 
WITH CHECK (
    bucket_id = 'cad-files' 
    AND owner = auth.uid()
);

-- 5) Allow authenticated users to delete their own uploaded files
CREATE POLICY IF NOT EXISTS "Users can delete own cad-files objects" 
ON storage.objects 
FOR DELETE 
TO authenticated 
USING (
    bucket_id = 'cad-files' 
    AND owner = auth.uid()
);

-- Optional: Create trained-models bucket for ML model storage
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
SELECT 
    'trained-models',
    'trained-models', 
    false, -- Private bucket
    104857600, -- 100MB limit for model files
    ARRAY['application/octet-stream', 'application/x-pytorch', 'application/x-tensorflow']
WHERE NOT EXISTS (
    SELECT 1 FROM storage.buckets WHERE id = 'trained-models'
);

-- Allow service role to read/write trained-models (for your training scripts)
CREATE POLICY IF NOT EXISTS "Service role access to trained-models" 
ON storage.objects 
FOR ALL 
TO service_role 
USING (bucket_id = 'trained-models') 
WITH CHECK (bucket_id = 'trained-models');
