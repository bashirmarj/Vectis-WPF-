import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { Upload, FileIcon, Loader2, ArrowLeft, AlertCircle, CheckCircle, Pause, Play } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';

const DatasetUpload = () => {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [isAdmin, setIsAdmin] = useState(false);
  const [checkingRole, setCheckingRole] = useState(true);
  
  const [graphFiles, setGraphFiles] = useState<File[]>([]);
  const [labelFiles, setLabelFiles] = useState<File[]>([]);
  const [splitFile, setSplitFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentFile, setCurrentFile] = useState<string>('');
  const [batchSize, setBatchSize] = useState<number>(8);
  const [failedUploads, setFailedUploads] = useState<{file: File, error: string, attempts?: number}[]>([]);
  const [uploadSpeed, setUploadSpeed] = useState<number>(0);
  const [startTime, setStartTime] = useState<number>(0);
  const [isPaused, setIsPaused] = useState(false);
  const [currentBatch, setCurrentBatch] = useState<number>(0);
  const [totalBatches, setTotalBatches] = useState<number>(0);
  const [fileType, setFileType] = useState<string>('');

  // Check if user has admin role
  useEffect(() => {
    const checkAdminRole = async () => {
      if (!user) {
        setCheckingRole(false);
        return;
      }

      try {
        const { data, error } = await supabase
          .from('user_roles')
          .select('role')
          .eq('user_id', user.id)
          .eq('role', 'admin')
          .maybeSingle();

        if (error) {
          console.error('Error checking role:', error);
          setIsAdmin(false);
        } else {
          setIsAdmin(!!data);
        }
      } catch (error) {
        console.error('Error checking admin role:', error);
        setIsAdmin(false);
      } finally {
        setCheckingRole(false);
      }
    };

    checkAdminRole();
  }, [user]);

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

  // Upload single file with retry logic
  const uploadWithRetry = async (file: File, folder: string, maxRetries = 3): Promise<{file: File, success: boolean, error?: string, attempts: number}> => {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const { error } = await supabase.storage
          .from('training-datasets')
          .upload(`MFCAD/${folder}/${file.name}`, file, { upsert: true });
        
        if (error) {
          if (attempt === maxRetries) {
            return { file, success: false, error: error.message, attempts: attempt };
          }
          // Exponential backoff: 500ms, 1s, 2s
          await new Promise(r => setTimeout(r, 500 * Math.pow(2, attempt - 1)));
          continue;
        }
        
        return { file, success: true, attempts: attempt };
      } catch (error: any) {
        if (attempt === maxRetries) {
          return { file, success: false, error: error.message || 'Network error', attempts: attempt };
        }
        // Exponential backoff
        await new Promise(r => setTimeout(r, 500 * Math.pow(2, attempt - 1)));
      }
    }
    return { file, success: false, error: 'Max retries exceeded', attempts: maxRetries };
  };

  const uploadInBatches = async (files: File[], folder: string, batchSize: number, totalFiles: number, uploadedSoFar: number) => {
    const failed: {file: File, error: string, attempts?: number}[] = [];
    let uploadedCount = uploadedSoFar;
    const totalBatchCount = Math.ceil(files.length / batchSize);

    for (let i = 0; i < files.length; i += batchSize) {
      // Check if paused
      while (isPaused) {
        await new Promise(r => setTimeout(r, 100));
      }

      const batch = files.slice(i, i + batchSize);
      const batchNum = Math.floor(i / batchSize) + 1;
      
      setCurrentBatch(batchNum);
      setTotalBatches(totalBatchCount);
      setFileType(folder);
      setCurrentFile(`Batch ${batchNum}/${totalBatchCount}: ${folder} (${batch.length} files)`);

      const results = await Promise.all(
        batch.map(file => uploadWithRetry(file, folder))
      );

      // Track failures
      results.forEach(result => {
        if (!result.success) {
          failed.push({ 
            file: result.file, 
            error: result.error || 'Unknown error',
            attempts: result.attempts 
          });
        }
      });

      uploadedCount += batch.length;
      const overallProgress = (uploadedCount / totalFiles) * 100;
      setUploadProgress(overallProgress);

      // Calculate upload speed
      const elapsedSeconds = (Date.now() - startTime) / 1000;
      const speed = uploadedCount / elapsedSeconds;
      setUploadSpeed(speed);

      // Update failed uploads in real-time
      setFailedUploads(failed);

      // Add delay between batches to prevent rate limiting
      if (i + batchSize < files.length) {
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    }

    return { failed, uploadedCount };
  };

  const uploadFiles = async () => {
    console.log('Upload button clicked');
    console.log('User:', user?.id);
    console.log('Is Admin:', isAdmin);
    console.log('Files selected:', { 
      graphFiles: graphFiles.length, 
      labelFiles: labelFiles.length, 
      splitFile: splitFile?.name 
    });

    if (!user) {
      toast({
        title: "Authentication Required",
        description: "Please log in to upload files.",
        variant: "destructive",
      });
      return;
    }

    if (!isAdmin) {
      toast({
        title: "Admin Access Required",
        description: "You must have admin privileges to upload datasets.",
        variant: "destructive",
      });
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setCurrentFile('');
    setFailedUploads([]);
    setUploadSpeed(0);
    setStartTime(Date.now());
    setIsPaused(false);
    setCurrentBatch(0);
    setTotalBatches(0);
    setFileType('');

    try {
      const totalFiles = graphFiles.length + labelFiles.length + (splitFile ? 1 : 0);
      const allFailed: {file: File, error: string}[] = [];
      let totalUploaded = 0;

      // Upload graph files in batches
      if (graphFiles.length > 0) {
        const { failed, uploadedCount } = await uploadInBatches(graphFiles, 'graph', batchSize, totalFiles, totalUploaded);
        allFailed.push(...failed);
        totalUploaded = uploadedCount;
      }

      // Upload label files in batches
      if (labelFiles.length > 0) {
        const { failed, uploadedCount } = await uploadInBatches(labelFiles, 'labels', batchSize, totalFiles, totalUploaded);
        allFailed.push(...failed);
        totalUploaded = uploadedCount;
      }

      // Upload split file (single file, no batching needed)
      if (splitFile) {
        setCurrentFile(`Uploading split: ${splitFile.name}`);
        console.log(`Uploading split file: ${splitFile.name}`);
        
        const filePath = `MFCAD/split/${splitFile.name}`;
        const { error } = await supabase.storage
          .from('training-datasets')
          .upload(filePath, splitFile, { upsert: true });

        if (error) {
          console.error(`Error uploading ${splitFile.name}:`, error);
          allFailed.push({ file: splitFile, error: error.message });
        }
        
        totalUploaded++;
        setUploadProgress(100);
      }

      setFailedUploads(allFailed);

      // Update dataset status
      await supabase
        .from('training_datasets')
        .update({ 
          download_status: 'completed',
          num_files: graphFiles.length,
          metadata: { uploaded_at: new Date().toISOString() }
        })
        .eq('name', 'MFCAD');

      const successCount = totalFiles - allFailed.length;
      const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);

      if (allFailed.length === 0) {
        toast({
          title: "Upload Complete",
          description: `Successfully uploaded ${successCount} files in ${elapsedTime}s (${Math.round(uploadSpeed)} files/sec)`,
        });
        setGraphFiles([]);
        setLabelFiles([]);
        setSplitFile(null);
      } else {
        toast({
          title: "Upload Completed with Errors",
          description: `${successCount} succeeded, ${allFailed.length} failed. Check console for details.`,
          variant: "destructive",
        });
        console.error('Failed uploads:', allFailed);
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      toast({
        title: "Upload Failed",
        description: error.message || "An unknown error occurred during upload.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
      setCurrentFile('');
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
              {/* Auth Status */}
              {authLoading || checkingRole ? (
                <Alert>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <AlertDescription>
                    Checking authentication...
                  </AlertDescription>
                </Alert>
              ) : !user ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    You must be logged in to upload datasets. Please <Button variant="link" className="p-0 h-auto" onClick={() => navigate('/auth')}>sign in</Button>.
                  </AlertDescription>
                </Alert>
              ) : !isAdmin ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Admin privileges required. Current user: {user.email}
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    Authenticated as admin: {user.email}
                  </AlertDescription>
                </Alert>
              )}

              {/* Batch Size Control */}
              {user && isAdmin && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    Concurrent Uploads: {batchSize}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    step="1"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    disabled={uploading}
                    className="w-full accent-primary"
                  />
                  <p className="text-xs text-muted-foreground">
                    Recommended: 5-10 for stability. Higher values may cause network errors.
                  </p>
                </div>
              )}

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
                <div className="space-y-3">
                  <Progress value={uploadProgress} />
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-center">
                      {isPaused ? '⏸️ Paused' : currentFile || 'Uploading...'}
                    </p>
                    <div className="text-xs text-muted-foreground text-center space-y-1">
                      <div>{uploadProgress.toFixed(1)}% complete</div>
                      {uploadSpeed > 0 && (
                        <>
                          <div>Speed: {uploadSpeed.toFixed(1)} files/sec</div>
                          <div>
                            Elapsed: {Math.floor((Date.now() - startTime) / 1000)}s
                            {uploadProgress > 0 && (() => {
                              const totalFiles = graphFiles.length + labelFiles.length + (splitFile ? 1 : 0);
                              const remaining = totalFiles * (1 - uploadProgress / 100);
                              const timeRemaining = remaining / uploadSpeed;
                              return ` • Remaining: ~${Math.floor(timeRemaining / 60)}m ${Math.floor(timeRemaining % 60)}s`;
                            })()}
                          </div>
                        </>
                      )}
                      {fileType && <div>Current: {fileType} files</div>}
                      {totalBatches > 0 && <div>Batch: {currentBatch}/{totalBatches}</div>}
                    </div>
                  </div>
                  {failedUploads.length > 0 && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        {failedUploads.length} upload(s) failed (will retry automatically)
                      </AlertDescription>
                    </Alert>
                  )}
                  <Button
                    variant="outline"
                    onClick={() => setIsPaused(!isPaused)}
                    className="w-full"
                  >
                    {isPaused ? (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Resume Upload
                      </>
                    ) : (
                      <>
                        <Pause className="mr-2 h-4 w-4" />
                        Pause Upload
                      </>
                    )}
                  </Button>
                </div>
              )}

              <Button
                onClick={uploadFiles}
                disabled={uploading || !user || !isAdmin || (graphFiles.length === 0 && labelFiles.length === 0 && !splitFile)}
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
