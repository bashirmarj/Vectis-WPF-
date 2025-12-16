-- Create part-thumbnails bucket (public for easy access)
INSERT INTO storage.buckets (id, name, public)
VALUES ('part-thumbnails', 'part-thumbnails', true)
ON CONFLICT (id) DO NOTHING;

-- Allow authenticated users to upload thumbnails to their own folder
CREATE POLICY "Users can upload their own thumbnails"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'part-thumbnails' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow public read access to thumbnails
CREATE POLICY "Thumbnails are publicly accessible"
ON storage.objects FOR SELECT
TO public
USING (bucket_id = 'part-thumbnails');

-- Allow users to update their own thumbnails
CREATE POLICY "Users can update their own thumbnails"
ON storage.objects FOR UPDATE
TO authenticated
USING (bucket_id = 'part-thumbnails' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow users to delete their own thumbnails
CREATE POLICY "Users can delete their own thumbnails"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'part-thumbnails' AND auth.uid()::text = (storage.foldername(name))[1]);