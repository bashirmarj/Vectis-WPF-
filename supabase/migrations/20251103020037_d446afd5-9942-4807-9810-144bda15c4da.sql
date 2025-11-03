-- Allow admins to update (overwrite) training dataset files
CREATE POLICY "Admin can update training datasets"
ON storage.objects
FOR UPDATE
TO authenticated
USING (
  bucket_id = 'training-datasets' 
  AND EXISTS (
    SELECT 1 FROM user_roles 
    WHERE user_roles.user_id = auth.uid() 
    AND user_roles.role = 'admin'::app_role
  )
)
WITH CHECK (
  bucket_id = 'training-datasets' 
  AND EXISTS (
    SELECT 1 FROM user_roles 
    WHERE user_roles.user_id = auth.uid() 
    AND user_roles.role = 'admin'::app_role
  )
);