import { useState, useCallback } from 'react';
import { ModeSelector } from './ModeSelector';
import { FileUploadScreen } from './FileUploadScreen';
import { QuoteConfigForm } from './QuoteConfigForm';
import { toast } from 'sonner';
import { parseCADFile } from '@/lib/geometryAnalyzer';

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

  // Analyze CAD file using existing parseCADFile function
  const analyzeFile = useCallback(async (file: File, index: number) => {
    setFiles(prev => prev.map((f, i) => 
      i === index ? { ...f, isAnalyzing: true, analysisStatus: 'Parsing geometry...', analysisProgress: 30 } : f
    ));

    try {
      const result = await parseCADFile(file);

      if (!result) {
        throw new Error('Failed to parse CAD file');
      }

      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Processing mesh...', analysisProgress: 70 } : f
      ));

      // Convert triangles to mesh data format
      const vertices: number[] = [];
      const indices: number[] = [];
      const normals: number[] = [];

      if (result.triangles) {
        result.triangles.forEach((tri, triIndex) => {
          const baseIndex = triIndex * 3;
          
          // Add vertices
          vertices.push(tri.vertices[0].x, tri.vertices[0].y, tri.vertices[0].z);
          vertices.push(tri.vertices[1].x, tri.vertices[1].y, tri.vertices[1].z);
          vertices.push(tri.vertices[2].x, tri.vertices[2].y, tri.vertices[2].z);
          
          // Add indices
          indices.push(baseIndex, baseIndex + 1, baseIndex + 2);
          
          // Use triangle normal
          const normal = tri.normal || { x: 0, y: 1, z: 0 };
          normals.push(normal.x, normal.y, normal.z);
          normals.push(normal.x, normal.y, normal.z);
          normals.push(normal.x, normal.y, normal.z);
        });
      }

      const meshData = {
        vertices,
        indices,
        normals,
        triangle_count: result.triangle_count,
      };

      const boundingBox = {
        width: result.part_width_cm,
        height: result.part_height_cm,
        depth: result.part_depth_cm,
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
            volume: result.volume_cm3,
            surfaceArea: result.surface_area_cm2,
          }
        } : f
      ));

    } catch (error) {
      console.error('Analysis error:', error);
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, isAnalyzing: false, analysis: undefined, analysisStatus: 'Failed' } : f
      ));
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
