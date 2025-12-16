import { useState, useCallback, useRef } from 'react';
import { ModeSelector } from './ModeSelector';
import { FileUploadScreen } from './FileUploadScreen';
import { QuoteConfigForm } from './QuoteConfigForm';
import { toast } from 'sonner';
import * as occtimportjs from 'occt-import-js';

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
  const occtInitialized = useRef(false);
  const occtRef = useRef<any>(null);

  // Initialize OCCT
  const initOCCT = useCallback(async () => {
    if (occtInitialized.current && occtRef.current) return occtRef.current;
    
    try {
      // Handle both default and namespace export
      const initFunction = (occtimportjs as any).default || occtimportjs;
      const occt = await initFunction({
        locateFile: (path: string) => {
          if (path.endsWith('.wasm')) {
            return `/node_modules/occt-import-js/dist/${path}`;
          }
          return path;
        }
      });
      occtRef.current = occt;
      occtInitialized.current = true;
      return occt;
    } catch (error) {
      console.error('Failed to initialize OCCT:', error);
      throw error;
    }
  }, []);

  // Analyze CAD file
  const analyzeFile = useCallback(async (file: File, index: number) => {
    setFiles(prev => prev.map((f, i) => 
      i === index ? { ...f, isAnalyzing: true, analysisStatus: 'Initializing...', analysisProgress: 0 } : f
    ));

    try {
      // Initialize OCCT
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Loading geometry engine...', analysisProgress: 10 } : f
      ));
      const occt = await initOCCT();

      // Read file
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Reading file...', analysisProgress: 30 } : f
      ));
      const arrayBuffer = await file.arrayBuffer();
      const fileBuffer = new Uint8Array(arrayBuffer);

      // Parse geometry
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Parsing geometry...', analysisProgress: 50 } : f
      ));
      const result = occt.ReadStepFile(fileBuffer, null);

      if (!result.success || !result.meshes || result.meshes.length === 0) {
        throw new Error('Failed to parse CAD file');
      }

      // Process mesh data
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, analysisStatus: 'Processing mesh...', analysisProgress: 70 } : f
      ));

      // Combine all meshes into a single mesh
      let totalVertices: number[] = [];
      let totalIndices: number[] = [];
      let totalNormals: number[] = [];
      let vertexOffset = 0;

      for (const mesh of result.meshes) {
        if (mesh.attributes?.position?.array) {
          totalVertices.push(...Array.from(mesh.attributes.position.array as Float32Array));
        }
        if (mesh.attributes?.normal?.array) {
          totalNormals.push(...Array.from(mesh.attributes.normal.array as Float32Array));
        }
        if (mesh.index?.array) {
          const indexArray = Array.from(mesh.index.array as Uint32Array);
          const indices = indexArray.map((idx) => idx + vertexOffset);
          totalIndices.push(...indices);
        }
        vertexOffset += (mesh.attributes?.position?.array?.length || 0) / 3;
      }

      const meshData = {
        vertices: totalVertices,
        indices: totalIndices,
        normals: totalNormals,
        triangle_count: totalIndices.length / 3,
      };

      // Calculate bounding box
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

      for (let i = 0; i < totalVertices.length; i += 3) {
        minX = Math.min(minX, totalVertices[i]);
        minY = Math.min(minY, totalVertices[i + 1]);
        minZ = Math.min(minZ, totalVertices[i + 2]);
        maxX = Math.max(maxX, totalVertices[i]);
        maxY = Math.max(maxY, totalVertices[i + 1]);
        maxZ = Math.max(maxZ, totalVertices[i + 2]);
      }

      const boundingBox = {
        width: maxX - minX,
        height: maxY - minY,
        depth: maxZ - minZ,
      };

      // Set analysis complete
      setFiles(prev => prev.map((f, i) => 
        i === index ? {
          ...f,
          isAnalyzing: false,
          analysisStatus: 'Complete',
          analysisProgress: 100,
          analysis: {
            meshData,
            boundingBox,
          }
        } : f
      ));

    } catch (error) {
      console.error('Analysis error:', error);
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, isAnalyzing: false, analysis: undefined, analysisStatus: 'Failed' } : f
      ));
    }
  }, [initOCCT]);

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
