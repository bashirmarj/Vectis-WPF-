-- Make cad-files bucket private to protect customer CAD files
UPDATE storage.buckets 
SET public = false 
WHERE id = 'cad-files';