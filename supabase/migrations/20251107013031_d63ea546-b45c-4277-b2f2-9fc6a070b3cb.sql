-- Create cad-files storage bucket for CAD file uploads
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'cad-files',
  'cad-files',
  true,
  52428800, -- 50MB limit
  ARRAY['application/step', 'application/stp', 'application/iges', 'application/igs', 'text/plain', 'application/octet-stream']
)
ON CONFLICT (id) DO NOTHING;

-- Allow public read access to cad-files
CREATE POLICY "Public can read cad-files"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'cad-files');

-- Allow authenticated users to upload to cad-files
CREATE POLICY "Authenticated users can upload to cad-files"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'cad-files');

-- Allow users to update their own files in cad-files
CREATE POLICY "Users can update their own files in cad-files"
ON storage.objects
FOR UPDATE
TO authenticated
USING (bucket_id = 'cad-files' AND auth.uid()::text = (storage.foldername(name))[1])
WITH CHECK (bucket_id = 'cad-files' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow users to delete their own files in cad-files
CREATE POLICY "Users can delete their own files in cad-files"
ON storage.objects
FOR DELETE
TO authenticated
USING (bucket_id = 'cad-files' AND auth.uid()::text = (storage.foldername(name))[1]);