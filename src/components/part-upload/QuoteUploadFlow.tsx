import { useState, useCallback } from 'react';
import { ModeSelector } from './ModeSelector';
import { FileUploadScreen } from './FileUploadScreen';
import { QuoteConfigForm } from './QuoteConfigForm';
import { toast } from 'sonner';
import { supabase } from '@/integrations/supabase/client';

type FlowStep = 'mode-selection' | 'upload' | 'configure' | 'project-select';
type FlowMode = 'quick' | 'project';

interface FileWithAnalysis {
  file: File;
  isAnalyzing: boolean;
  analysis?: {
    meshData: any;
    volume?: number;
    surfaceArea?: number;
    boundingBox?: { width: number; height: number; depth: number };
  };
  uploadProgress?: number;
  analysisProgress?: number;
  uploadSpeed?: number;
  analysisStatus?: string;
}

export function QuoteUploadFlow() {
  const [step, setStep] = useState<FlowStep>('mode-selection');
  const [mode, setMode] = useState<FlowMode | null>(null);
  const [files, setFiles] = useState<FileWithAnalysis[]>([]);

  // Analyze CAD file using backend geometry-service via analyze-cad edge function
  const analyzeFile = useCallback(async (file: File, index: number) => {
    setFiles(prev => prev.map((f, i) => 
      i === index ? { ...f, isAnalyzing: true, analysisStatus: 'Uploading file...', analysisProgress: 10 } : f
    ));

    try {
      // 1. Upload to temporary storage
      const tempPath = `temp/${crypto.randomUUID()}/${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from('cad-files')
        .upload(tempPath, file);

      if (uploadError) {
        throw new Error(`Upload failed: ${uploadError.message}`);
      }

      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Creating signed URL...', analysisProgress: 25 } : f
      ));

      // 2. Get signed URL
      const { data: signedUrlData, error: signedUrlError } = await supabase.storage
        .from('cad-files')
        .createSignedUrl(tempPath, 3600);

      if (signedUrlError || !signedUrlData?.signedUrl) {
        throw new Error('Failed to create signed URL');
      }

      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Analyzing geometry...', analysisProgress: 40 } : f
      ));

      // 3. Call analyze-cad edge function (routes to geometry-service on Render.com)
      const { data, error } = await supabase.functions.invoke('analyze-cad', {
        body: {
          fileName: file.name,
          fileUrl: signedUrlData.signedUrl,
        }
      });

      if (error) {
        throw new Error(`Analysis failed: ${error.message}`);
      }

      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Processing mesh data...', analysisProgress: 80 } : f
      ));

      // 4. Fetch mesh data if stored separately (large meshes)
      let meshData = data.mesh_data;
      if (data.mesh_storage_url && !meshData) {
        const meshResponse = await fetch(data.mesh_storage_url);
        if (meshResponse.ok) {
          meshData = await meshResponse.json();
        }
      }

      // 5. Extract bounding box from response
      const boundingBox = {
        width: data.part_width_cm || 0,
        height: data.part_height_cm || 0,
        depth: data.part_depth_cm || 0,
      };

      setFiles(prev => prev.map((f, i) => 
        i === index ? {
          ...f,
          isAnalyzing: false,
          analysisStatus: 'Complete',
          analysisProgress: 100,
          analysis: {
            meshData,
            boundingBox,
            volume: data.volume_cm3,
            surfaceArea: data.surface_area_cm2,
          }
        } : f
      ));

      // Clean up temp file
      supabase.storage.from('cad-files').remove([tempPath]).catch(console.error);

    } catch (error) {
      console.error('Analysis error:', error);
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, isAnalyzing: false, analysis: undefined, analysisStatus: 'Failed' } : f
      ));
      toast.error(error instanceof Error ? error.message : 'Analysis failed');
    }
  }, []);

  // Handle mode selection
  const handleModeSelect = useCallback((selectedMode: 'quick' | 'project') => {
    setMode(selectedMode);
    if (selectedMode === 'quick') {
      setStep('upload');
    } else {
      setStep('project-select');
    }
  }, []);

  // Handle file selection
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (!selectedFiles) return;

    const newFiles: FileWithAnalysis[] = Array.from(selectedFiles).map(file => ({
      file,
      isAnalyzing: false,
    }));

    // In quick mode, only allow single file
    if (mode === 'quick') {
      setFiles(newFiles.slice(0, 1));
      analyzeFile(newFiles[0].file, 0);
    } else {
      setFiles(prev => [...prev, ...newFiles]);
      newFiles.forEach((_, idx) => analyzeFile(newFiles[idx].file, files.length + idx));
    }

    // Reset input
    e.target.value = '';
  }, [mode, files.length, analyzeFile]);

  // Handle file drop
  const handleFileDrop = useCallback((droppedFiles: FileList) => {
    const newFiles: FileWithAnalysis[] = Array.from(droppedFiles).map(file => ({
      file,
      isAnalyzing: false,
    }));

    if (mode === 'quick') {
      setFiles(newFiles.slice(0, 1));
      analyzeFile(newFiles[0].file, 0);
    } else {
      const startIndex = files.length;
      setFiles(prev => [...prev, ...newFiles]);
      newFiles.forEach((f, idx) => analyzeFile(f.file, startIndex + idx));
    }
  }, [mode, files.length, analyzeFile]);

  // Handle file removal
  const handleRemoveFile = useCallback((index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  // Handle retry
  const handleRetryFile = useCallback((index: number) => {
    analyzeFile(files[index].file, index);
  }, [files, analyzeFile]);

  // Continue to configure step
  const handleContinueToConfig = useCallback(() => {
    if (files.length === 0) {
      toast.error('Please upload a file first');
      return;
    }
    setStep('configure');
  }, [files]);

  // Go back to previous step
  const handleBack = useCallback(() => {
    if (step === 'configure') {
      setStep('upload');
    } else if (step === 'upload' || step === 'project-select') {
      setStep('mode-selection');
      setMode(null);
      setFiles([]);
    }
  }, [step]);

  // Render based on current step
  if (step === 'mode-selection') {
    return <ModeSelector onSelectMode={handleModeSelect} />;
  }

  if (step === 'project-select') {
    // Project batch mode - redirect back with message for now
    return (
      <div className="max-w-5xl mx-auto space-y-4">
        <button
          onClick={handleBack}
          className="text-white/70 hover:text-white text-sm flex items-center gap-2"
        >
          ← Back to options
        </button>
        <div className="backdrop-blur-sm border border-white/20 rounded-sm p-8 text-center" style={{ backgroundColor: "rgba(60, 60, 60, 0.75)" }}>
          <h3 className="text-xl font-semibold text-white mb-2">Project Batch Upload</h3>
          <p className="text-gray-400 mb-4">
            To upload multiple parts to a project, please log in to your dashboard.
          </p>
          <a href="/dashboard" className="text-primary hover:underline">
            Go to Dashboard →
          </a>
        </div>
      </div>
    );
  }

  if (step === 'upload') {
    return (
      <div className="space-y-4">
        <button
          onClick={handleBack}
          className="text-white/70 hover:text-white text-sm flex items-center gap-2"
        >
          ← Back to options
        </button>
        <FileUploadScreen
          files={files}
          onFileSelect={handleFileSelect}
          onFileDrop={handleFileDrop}
          onRemoveFile={handleRemoveFile}
          onRetryFile={handleRetryFile}
          onContinue={handleContinueToConfig}
          isAnalyzing={files.some(f => f.isAnalyzing)}
          singleFileMode={mode === 'quick'}
          continueButtonText="Continue to Configure"
        />
      </div>
    );
  }

  if (step === 'configure' && files.length > 0) {
    return (
      <QuoteConfigForm
        file={files[0].file}
        meshData={files[0].analysis?.meshData}
        boundingBox={files[0].analysis?.boundingBox}
        onBack={handleBack}
        onSubmitSuccess={() => {
          setStep('mode-selection');
          setMode(null);
          setFiles([]);
        }}
      />
    );
  }

  return null;
}
