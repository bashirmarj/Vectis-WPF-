-- Update cad-files bucket to allow correct MIME types for STEP/IGES files
UPDATE storage.buckets 
SET allowed_mime_types = ARRAY[
  'model/step',
  'model/step+xml',
  'model/step+zip',
  'model/iges',
  'application/step',
  'application/stp',
  'application/iges',
  'application/igs',
  'application/vnd.step',
  'text/plain',
  'application/octet-stream'
]
WHERE id = 'cad-files';