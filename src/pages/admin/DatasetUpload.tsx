import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { Upload, FileIcon, Loader2, ArrowLeft } from 'lucide-react';
import { Progress } from '@/components/ui/progress';

const DatasetUpload = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [graphFiles, setGraphFiles] = useState<File[]>([]);
  const [labelFiles, setLabelFiles] = useState<File[]>([]);
  const [splitFile, setSplitFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleGraphFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files).filter(f => f.name.endsWith('.bin'));
      setGraphFiles(prev => [...prev, ...files]);
    }
  };

  const handleLabelFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files).filter(f => f.name.endsWith('.json') && f.name.includes('_ids'));
      setLabelFiles(prev => [...prev, ...files]);
    }
  };

  const handleSplitFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0] && e.target.files[0].name === 'split.json') {
      setSplitFile(e.target.files[0]);
    }
  };

  const uploadFiles = async () => {
    if (!user) return;

    setUploading(true);
    setUploadProgress(0);

    try {
      const totalFiles = graphFiles.length + labelFiles.length + (splitFile ? 1 : 0);
      let uploadedCount = 0;

      // Upload graph files
      for (const file of graphFiles) {
        const filePath = `MFCAD/graph/${file.name}`;
        const { error } = await supabase.storage
          .from('training-datasets')
          .upload(filePath, file, { upsert: true });

        if (error) throw error;
        uploadedCount++;
        setUploadProgress((uploadedCount / totalFiles) * 100);
      }

      // Upload label files
      for (const file of labelFiles) {
        const filePath = `MFCAD/labels/${file.name}`;
        const { error } = await supabase.storage
          .from('training-datasets')
          .upload(filePath, file, { upsert: true });

        if (error) throw error;
        uploadedCount++;
        setUploadProgress((uploadedCount / totalFiles) * 100);
      }

      // Upload split file
      if (splitFile) {
        const filePath = `MFCAD/split/${splitFile.name}`;
        const { error } = await supabase.storage
          .from('training-datasets')
          .upload(filePath, splitFile, { upsert: true });

        if (error) throw error;
        uploadedCount++;
        setUploadProgress((uploadedCount / totalFiles) * 100);
      }

      // Update dataset status
      await supabase
        .from('training_datasets')
        .update({ 
          download_status: 'completed',
          num_files: graphFiles.length,
          metadata: { uploaded_at: new Date().toISOString() }
        })
        .eq('name', 'MFCAD');

      toast({
        title: "Upload Complete",
        description: `Successfully uploaded ${totalFiles} files to training-datasets bucket.`,
      });

      setGraphFiles([]);
      setLabelFiles([]);
      setSplitFile(null);
    } catch (error: any) {
      toast({
        title: "Upload Failed",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <>
      <Navigation />
      <main className="min-h-screen bg-background py-12">
        <div className="container mx-auto px-4 max-w-4xl">
          <Button
            variant="ghost"
            onClick={() => navigate('/admin')}
            className="mb-6"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Admin
          </Button>

          <Card>
            <CardHeader>
              <CardTitle>Upload Training Dataset</CardTitle>
              <CardDescription>
                Upload MFCAD dataset files for UV-Net training
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Graph Files (.bin) */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Graph Files (.bin)</label>
                <div className="border-2 border-dashed rounded-lg p-6 hover:border-primary/50 transition-colors">
                  <input
                    type="file"
                    multiple
                    accept=".bin"
                    onChange={handleGraphFiles}
                    className="hidden"
                    id="graph-files"
                  />
                  <label htmlFor="graph-files" className="cursor-pointer flex flex-col items-center gap-2">
                    <Upload className="h-8 w-8 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      Click to upload graph files (.bin)
                    </span>
                  </label>
                </div>
                {graphFiles.length > 0 && (
                  <div className="text-sm text-muted-foreground">
                    {graphFiles.length} graph files selected
                  </div>
                )}
              </div>

              {/* Label Files (*_ids.json) */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Label Files (*_ids.json)</label>
                <div className="border-2 border-dashed rounded-lg p-6 hover:border-primary/50 transition-colors">
                  <input
                    type="file"
                    multiple
                    accept=".json"
                    onChange={handleLabelFiles}
                    className="hidden"
                    id="label-files"
                  />
                  <label htmlFor="label-files" className="cursor-pointer flex flex-col items-center gap-2">
                    <FileIcon className="h-8 w-8 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      Click to upload label files (*_ids.json)
                    </span>
                  </label>
                </div>
                {labelFiles.length > 0 && (
                  <div className="text-sm text-muted-foreground">
                    {labelFiles.length} label files selected
                  </div>
                )}
              </div>

              {/* Split File (split.json) */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Split File (split.json)</label>
                <div className="border-2 border-dashed rounded-lg p-6 hover:border-primary/50 transition-colors">
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleSplitFile}
                    className="hidden"
                    id="split-file"
                  />
                  <label htmlFor="split-file" className="cursor-pointer flex flex-col items-center gap-2">
                    <FileIcon className="h-8 w-8 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      Click to upload split.json
                    </span>
                  </label>
                </div>
                {splitFile && (
                  <div className="text-sm text-muted-foreground">
                    {splitFile.name} selected
                  </div>
                )}
              </div>

              {uploading && (
                <div className="space-y-2">
                  <Progress value={uploadProgress} />
                  <p className="text-sm text-muted-foreground text-center">
                    Uploading... {Math.round(uploadProgress)}%
                  </p>
                </div>
              )}

              <Button
                onClick={uploadFiles}
                disabled={uploading || (graphFiles.length === 0 && labelFiles.length === 0 && !splitFile)}
                className="w-full"
              >
                {uploading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Upload Files
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      </main>
      <Footer />
    </>
  );
};

export default DatasetUpload;
