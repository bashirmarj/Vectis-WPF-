-- Add RLS policies for trained-models storage bucket
-- Allow authenticated users to upload files
CREATE POLICY "Authenticated users can upload to trained-models"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'trained-models');

-- Allow authenticated users to read files
CREATE POLICY "Authenticated users can read trained-models"
ON storage.objects
FOR SELECT
TO authenticated
USING (bucket_id = 'trained-models');

-- Allow users to update their own files
CREATE POLICY "Users can update their own files in trained-models"
ON storage.objects
FOR UPDATE
TO authenticated
USING (bucket_id = 'trained-models' AND auth.uid()::text = (storage.foldername(name))[1])
WITH CHECK (bucket_id = 'trained-models' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow users to delete their own files
CREATE POLICY "Users can delete their own files in trained-models"
ON storage.objects
FOR DELETE
TO authenticated
USING (bucket_id = 'trained-models' AND auth.uid()::text = (storage.foldername(name))[1]);