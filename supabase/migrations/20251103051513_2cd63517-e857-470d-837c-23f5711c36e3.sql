-- Add DELETE policy for admin users on training_jobs table
CREATE POLICY "Admin can delete training jobs"
ON public.training_jobs
FOR DELETE
TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.user_roles
    WHERE user_id = auth.uid()
      AND role = 'admin'::app_role
  )
);