import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { useNavigate } from "react-router-dom";
import { FileUploadScreen } from "./part-upload/FileUploadScreen";
import PartConfigScreen from "./part-upload/PartConfigScreen";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Lock } from "lucide-react";

interface FileWithQuantity {
  file: File;
  quantity: number;
  material?: string;
  process?: string;
  meshData?: {
    vertices: number[];
    indices: number[];
    normals: number[];
    vertex_colors?: string[]; // ✅ Added for topology colors
    triangle_count: number;
    face_types?: string[];
    feature_edges?: number[][][];
  };
  analysis?: {
    volume_cm3?: number;
    surface_area_cm2?: number;
    complexity_score?: number;
    confidence?: number;
    method?: string;
    triangle_count?: number;
    geometric_features?: any;
    recommended_processes?: string[];
    routing_reasoning?: string[];
    machining_summary?: any[];
  };
  quote?: any;
  isAnalyzing?: boolean;
  uploadProgress?: number; // 0-100 for storage upload phase
  analysisProgress?: number; // 0-100 for analysis phase
  uploadSpeed?: number; // bytes per second
  analysisStatus?: string; // Status message for analysis phase
}

export const PartUploadForm = () => {
  const [currentScreen, setCurrentScreen] = useState<"upload" | "configure">("upload");
  const [files, setFiles] = useState<FileWithQuantity[]>([]);
  const [materials, setMaterials] = useState<string[]>([]);
  const [processes, setProcesses] = useState<string[]>([]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [uploading, setUploading] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  // Load materials & processes from Supabase
  useEffect(() => {
    const fetchMaterials = async () => {
      const { data } = await supabase
        .from("material_costs")
        .select("material_name")
        .eq("is_active", true)
        .order("material_name");
      if (data) setMaterials(data.map((m) => m.material_name));
    };

    const fetchProcesses = async () => {
      const { data } = await supabase
        .from("manufacturing_processes")
        .select("name")
        .eq("is_active", true)
        .order("name");
      if (data) setProcesses(data.map((p) => p.name));
    };

    fetchMaterials();
    fetchProcesses();
  }, []);

  // ✅ FIXED: Analyze file using Supabase Edge Function with progress tracking
  const analyzeFile = async (fileWithQty: FileWithQuantity, index: number) => {
    try {
      setFiles((prev) => prev.map((f, i) => (i === index ? { 
        ...f, 
        isAnalyzing: true, 
        uploadProgress: 0,
        analysisProgress: 0,
        analysisStatus: 'Preparing upload...'
      } : f)));

      console.log(`📤 Sending ${fileWithQty.file.name} to edge function (may take 30-60s on first use)...`);

      // Step 1: Upload to Supabase Storage with progress tracking using XMLHttpRequest
      const fileExt = fileWithQty.file.name.split('.').pop()?.toLowerCase() || 'step';
      const fileName = `${Date.now()}_${fileWithQty.file.name}`;
      const filePath = `uploads/${fileName}`;

      // Get the upload URL from Supabase
      const { data: uploadUrlData, error: uploadUrlError } = await supabase.storage
        .from('cad-files')
        .createSignedUploadUrl(filePath);

      if (uploadUrlError) throw new Error(`Failed to create upload URL: ${uploadUrlError.message}`);

      // Upload with progress tracking using XMLHttpRequest
      const uploadStartTime = Date.now();
      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        let lastLoaded = 0;
        let lastTime = uploadStartTime;

        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded / event.total) * 100);
            const currentTime = Date.now();
            const timeDiff = (currentTime - lastTime) / 1000; // seconds
            const bytesDiff = event.loaded - lastLoaded;
            const speed = timeDiff > 0 ? bytesDiff / timeDiff : 0;

            setFiles((prev) => prev.map((f, i) => (i === index ? { 
              ...f, 
              uploadProgress: progress,
              uploadSpeed: speed,
              analysisStatus: `Uploading... ${progress}%`
            } : f)));

            lastLoaded = event.loaded;
            lastTime = currentTime;
          }
        });

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            setFiles((prev) => prev.map((f, i) => (i === index ? { 
              ...f, 
              uploadProgress: 100,
              analysisStatus: 'Upload complete, preparing analysis...'
            } : f)));
            resolve();
          } else {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        });

        xhr.addEventListener('error', () => reject(new Error('Upload failed')));
        xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));

        xhr.open('PUT', uploadUrlData.signedUrl);
        xhr.setRequestHeader('Content-Type', 'model/step');
        xhr.send(fileWithQty.file);
      });

      // Step 2: Create signed URL for analysis
      const { data: urlData, error: urlError } = await supabase.storage
        .from('cad-files')
        .createSignedUrl(filePath, 3600); // 1 hour expiry

      if (urlError) {
        throw new Error(`Failed to create signed URL: ${urlError.message}`);
      }

      // Step 3: Call edge function with simulated progress updates
      // Simulate analysis progress updates (estimated based on typical processing time)
      const progressInterval = setInterval(() => {
        setFiles((prev) => prev.map((f, i) => {
          if (i !== index) return f;
          const currentProgress = f.analysisProgress || 0;
          if (currentProgress >= 95) return f; // Cap at 95% until complete
          
          // Increment progress gradually
          const newProgress = Math.min(95, currentProgress + 5);
          const status = 
            newProgress < 30 ? 'Processing geometry...' :
            newProgress < 60 ? 'Extracting features...' :
            newProgress < 80 ? 'Running ML analysis...' :
            'Finalizing results...';
          
          return { ...f, analysisProgress: newProgress, analysisStatus: status };
        }));
      }, 1500); // Update every 1.5 seconds

      setFiles((prev) => prev.map((f, i) => (i === index ? { 
        ...f, 
        analysisProgress: 0,
        analysisStatus: 'Processing geometry...'
      } : f)));

      const { data: result, error } = await supabase.functions.invoke("analyze-cad", {
        body: {
          fileUrl: urlData.signedUrl,
          fileName: fileWithQty.file.name,
          fileSize: fileWithQty.file.size,
          fileType: fileExt,
          material: fileWithQty.material,
        },
      });

      clearInterval(progressInterval);

      if (error) {
        throw new Error(error.message || "Edge function error");
      }

      console.log("✅ Edge function response:", result);
      console.log("📊 Available keys in response:", Object.keys(result));
      console.log("🏭 Manufacturing features:", result.manufacturing_features);
      console.log("📋 Feature summary:", result.feature_summary);

      // ✅ Extract mesh data directly from response (no meshId)
      const meshData = result.mesh_data || result.meshData || {};

      // Add vertex_colors to meshData if available
      if (result.vertex_colors || meshData.vertex_colors) {
        meshData.vertex_colors = result.vertex_colors || meshData.vertex_colors;
      }

      console.log("🎨 Mesh data received:", {
        hasVertexColors: !!meshData.vertex_colors,
        vertexColorCount: meshData.vertex_colors?.length,
        triangleCount: result.triangle_count || meshData.triangle_count,
        hasVertices: !!meshData.vertices,
        hasIndices: !!meshData.indices,
        hasNormals: !!meshData.normals,
        hasVertexFaceIds: !!meshData.vertex_face_ids,
        vertexFaceIdsCount: meshData.vertex_face_ids?.length,
        hasTaggedEdges: !!meshData.tagged_edges,
        taggedEdgesCount: meshData.tagged_edges?.length || 0,
        hasEdgeClassifications: !!meshData.edge_classifications,
        edgeClassificationsCount: meshData.edge_classifications?.length || 0,
      });

      // Extract analysis data
      const analysis = {
        volume_cm3: result.volume_cm3,
        surface_area_cm2: result.surface_area_cm2,
        complexity_score: result.complexity_score,
        confidence: result.confidence,
        method: result.method,
        triangle_count: result.triangle_count,
        geometric_features: result.geometric_features,
        recommended_processes: result.recommended_processes,
        routing_reasoning: result.routing_reasoning,
        machining_summary: result.machining_summary,
      };

      console.log("💾 Storing analysis data:", {
        hasAnalysis: !!analysis,
        hasMeshData: !!meshData,
      });

      // ✅ Store meshData and analysis directly (no meshId caching)
      setFiles((prev) =>
        prev.map((f, i) =>
          i === index
            ? {
                ...f,
                meshData, // ✅ Direct mesh data for 3D viewer
                analysis, // For FeatureTree and analysis display
                isAnalyzing: false,
                uploadProgress: 100,
                analysisProgress: 100,
                analysisStatus: 'Complete',
              }
            : f,
        ),
      );

      console.log("✅ File updated with fresh mesh data (no caching)");

      toast({
        title: "✅ CAD Analysis Complete",
        description: `${fileWithQty.file.name} analyzed successfully`,
      });
    } catch (error: any) {
      console.error("❌ Error analyzing file:", error);
      setFiles((prev) => prev.map((f, i) => (i === index ? { 
        ...f, 
        isAnalyzing: false,
        uploadProgress: 0,
        analysisProgress: 0,
        analysisStatus: 'Failed',
      } : f)));
      toast({
        title: "Analysis Failed",
        description: error.message || "Backend may be waking up - please try again",
        variant: "destructive",
      });
    }
  };

  // Validate CAD file by MIME type and extension
  const validateCADFile = (file: File): boolean => {
    const validMimeTypes = [
      'model/step',
      'model/step+xml', 
      'model/iges',
      'application/step',
      'application/stp',
      'application/iges',
      'application/igs',
      'text/plain',
      'application/octet-stream'
    ];
    
    const validExtensions = ['.step', '.stp', '.iges', '.igs'];
    const hasValidExtension = validExtensions.some(ext => 
      file.name.toLowerCase().endsWith(ext)
    );
    
    // Accept if either MIME type or extension is valid
    return validMimeTypes.includes(file.type) || hasValidExtension;
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    const invalidFiles = selectedFiles.filter(file => !validateCADFile(file));

    if (invalidFiles.length > 0) {
      toast({
        title: "Invalid file type",
        description: `Only STEP (.step, .stp) or IGES (.iges, .igs) files are supported. Invalid files: ${invalidFiles.map(f => f.name).join(', ')}`,
        variant: "destructive",
      });
      return;
    }

    const filesWithQuantity = selectedFiles.map((file) => ({
      file,
      quantity: 1,
    }));

    const newFiles = [...files, ...filesWithQuantity];
    setFiles(newFiles);

    // Analyze files sequentially
    for (let idx = 0; idx < filesWithQuantity.length; idx++) {
      const fileIndex = files.length + idx;
      await analyzeFile(filesWithQuantity[idx], fileIndex);
    }
  };

  const handleRemoveFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    if (selectedFileIndex >= files.length - 1) {
      setSelectedFileIndex(Math.max(0, files.length - 2));
    }
  };

  const handleRetryAnalysis = async (index: number) => {
    const file = files[index];
    if (!file) return;
    
    toast({
      title: "🔄 Retrying Analysis",
      description: `Re-analyzing ${file.file.name}...`,
    });
    
    await analyzeFile(file, index);
  };

  const handleContinue = () => {
    // Check if all files have been analyzed
    const unanalyzedFiles = files.filter((f) => !f.analysis && !f.isAnalyzing);
    const failedFiles = files.filter((f) => !f.analysis && !f.isAnalyzing);
    
    if (unanalyzedFiles.length > 0) {
      const fileList = unanalyzedFiles.map(f => f.file.name).join(', ');
      toast({
        title: "⚠️ Files Not Analyzed",
        description: `${unanalyzedFiles.length} file(s) failed analysis: ${fileList}. Please retry or remove them to continue.`,
        variant: "destructive",
      });
      return;
    }

    console.log(
      "📋 Continuing with files:",
      files.map((f) => ({
        name: f.file.name,
        hasMeshData: !!f.meshData,
        hasAnalysis: !!f.analysis,
      })),
    );

    setCurrentScreen("configure");
  };

  const handleBack = () => setCurrentScreen("upload");

  const handleUpdateFile = (index: number, updates: Partial<FileWithQuantity>) => {
    setFiles((prev) => prev.map((f, i) => (i === index ? { ...f, ...updates } : f)));
  };

  const handleSubmit = async (formData: any) => {
    console.log("📤 Submitting quote request:", formData);
    setUploading(true);

    try {
      // Convert files to base64 for transmission
      const filesWithContent = await Promise.all(
        files.map(async (f) => {
          const reader = new FileReader();
          return new Promise<any>((resolve) => {
            reader.onload = () => {
              const base64 = (reader.result as string).split(',')[1];
              resolve({
                name: f.file.name,
                content: base64,
                size: f.file.size,
                quantity: f.quantity,
              });
            };
            reader.readAsDataURL(f.file);
          });
        })
      );

      const { data: submission, error } = await supabase.functions.invoke(
        'send-quotation-request',
        {
          body: {
            name: formData.name,
            company: formData.company || '',
            email: formData.email,
            phone: formData.phone,
            shippingAddress: formData.shippingAddress,
            message: formData.message || '',
            files: filesWithContent,
            drawingFiles: [],
          }
        }
      );

      if (error) throw error;

      toast({
        title: "✅ Quote Submitted",
        description: `Quote request ${submission?.quote_number || ''} submitted successfully!`,
      });
      
      // Reset form and return to upload screen
      setFiles([]);
      setCurrentScreen("upload");
      
    } catch (error: any) {
      console.error("Submission error:", error);
      toast({
        title: "❌ Submission Failed",
        description: error.message || "Failed to submit quote request",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const isAnalyzing = files.some((f) => f.isAnalyzing);

  // Show loading state while checking auth
  if (loading) {
    return (
      <Card>
        <CardContent className="py-12">
          <div className="text-center">
            <p className="text-muted-foreground">Loading...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Show auth required message if not logged in
  if (!user) {
    return (
      <Card>
        <CardContent className="py-12">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
              <Lock className="h-8 w-8 text-primary" />
            </div>
            <h3 className="text-xl font-semibold">Authentication Required</h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              Please sign in or create an account to upload CAD files and request a quote.
            </p>
            <div className="flex gap-4 justify-center">
              <Button 
                onClick={() => navigate('/auth?returnTo=/services/custom-parts')}
                size="lg"
              >
                Sign In
              </Button>
              <Button 
                onClick={() => navigate('/auth?returnTo=/services/custom-parts')}
                variant="outline"
                size="lg"
              >
                Create Account
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (currentScreen === "upload") {
    return (
      <FileUploadScreen
        files={files}
        onFileSelect={handleFileSelect}
        onRemoveFile={handleRemoveFile}
        onRetryFile={handleRetryAnalysis}
        onContinue={handleContinue}
        isAnalyzing={isAnalyzing}
      />
    );
  }

  return (
    <PartConfigScreen
      files={files}
      materials={materials}
      processes={processes}
      onBack={handleBack}
      onSubmit={handleSubmit}
      onUpdateFile={handleUpdateFile}
      selectedFileIndex={selectedFileIndex}
      onSelectFile={setSelectedFileIndex}
      isSubmitting={uploading}
    />
  );
};
