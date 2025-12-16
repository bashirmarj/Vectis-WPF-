import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { useNavigate } from "react-router-dom";
import { FileUploadScreen } from "./part-upload/FileUploadScreen";
import { ProjectSelector } from "./part-upload/ProjectSelector";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Lock } from "lucide-react";
import { useCustomerProjects } from "@/hooks/useCustomerProjects";
import { generateThumbnail } from "@/lib/thumbnailGenerator";

interface FileWithQuantity {
  file: File;
  quantity: number;
  material?: string;
  process?: string;
  meshData?: {
    vertices: number[];
    indices: number[];
    normals: number[];
    vertex_colors?: string[];
    triangle_count: number;
    face_types?: string[];
    feature_edges?: number[][][];
    face_mapping?: Record<number, { triangle_indices: number[]; triangle_range: [number, number] }>;
    vertex_face_ids?: number[];
    tagged_edges?: Array<{
      feature_id: number;
      start: [number, number, number];
      end: [number, number, number];
      type: "arc" | "circle" | "line";
      iso_type?: string;
      diameter?: number;
      radius?: number;
      length?: number;
    }>;
    edge_classifications?: any[];
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
  uploadProgress?: number;
  analysisProgress?: number;
  uploadSpeed?: number;
  analysisStatus?: string;
  filePath?: string; // Storage path for the uploaded file
}

export const PartUploadForm = () => {
  const [currentScreen, setCurrentScreen] = useState<"project" | "upload" | "configure">("project");
  const [files, setFiles] = useState<FileWithQuantity[]>([]);
  const [materials, setMaterials] = useState<string[]>([]);
  const [processes, setProcesses] = useState<string[]>([]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const { toast } = useToast();
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  
  const { projects, loading: projectsLoading, createProject } = useCustomerProjects();

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

  // Analyze file using Supabase Edge Function with progress tracking
  const analyzeFile = async (fileWithQty: FileWithQuantity, index: number) => {
    try {
      setFiles((prev) => prev.map((f, i) => (i === index ? { 
        ...f, 
        isAnalyzing: true, 
        uploadProgress: 0,
        analysisProgress: 0,
        analysisStatus: 'Preparing upload...'
      } : f)));

      console.log(`📤 Sending ${fileWithQty.file.name} to edge function...`);

      // Step 1: Upload to Supabase Storage with progress tracking
      const fileExt = fileWithQty.file.name.split('.').pop()?.toLowerCase() || 'step';
      const fileName = `${Date.now()}_${fileWithQty.file.name}`;
      const filePath = `uploads/${user?.id}/${selectedProjectId}/${fileName}`;

      const { data: uploadUrlData, error: uploadUrlError } = await supabase.storage
        .from('cad-files')
        .createSignedUploadUrl(filePath);

      if (uploadUrlError) throw new Error(`Failed to create upload URL: ${uploadUrlError.message}`);

      // Upload with progress tracking
      const uploadStartTime = Date.now();
      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        let lastLoaded = 0;
        let lastTime = uploadStartTime;

        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded / event.total) * 100);
            const currentTime = Date.now();
            const timeDiff = (currentTime - lastTime) / 1000;
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
              analysisStatus: 'Upload complete, preparing analysis...',
              filePath: filePath
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

      // Store file path
      setFiles((prev) => prev.map((f, i) => (i === index ? { ...f, filePath } : f)));

      // Step 2: Create signed URL for analysis
      const { data: urlData, error: urlError } = await supabase.storage
        .from('cad-files')
        .createSignedUrl(filePath, 3600);

      if (urlError) {
        throw new Error(`Failed to create signed URL: ${urlError.message}`);
      }

      // Step 3: Call edge function with progress updates
      const progressInterval = setInterval(() => {
        setFiles((prev) => prev.map((f, i) => {
          if (i !== index) return f;
          const currentProgress = f.analysisProgress || 0;
          if (currentProgress >= 95) return f;
          
          const newProgress = Math.min(95, currentProgress + 5);
          const status = 
            newProgress < 30 ? 'Processing geometry...' :
            newProgress < 60 ? 'Extracting features...' :
            newProgress < 80 ? 'Running ML analysis...' :
            'Finalizing results...';
          
          return { ...f, analysisProgress: newProgress, analysisStatus: status };
        }));
      }, 1500);

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
      
      // Extract mesh data with fallback checks
      let meshData = null;
      
      if (result.mesh_data && (result.mesh_data.vertices || result.mesh_data.vertex_array)) {
        meshData = {
          vertices: result.mesh_data.vertices || result.mesh_data.vertex_array,
          indices: result.mesh_data.indices || result.mesh_data.index_array || result.mesh_data.faces,
          normals: result.mesh_data.normals || result.mesh_data.normal_array,
          vertex_colors: result.mesh_data.vertex_colors || result.mesh_data.colors,
          tagged_edges: result.mesh_data.tagged_edges,
          edge_classifications: result.mesh_data.edge_classifications,
          triangle_count: result.mesh_data.triangle_count || (result.mesh_data.indices?.length / 3),
          face_mapping: result.mesh_data.face_mapping,
          vertex_face_ids: result.mesh_data.vertex_face_ids,
        };
      } else if (result.vertices || result.vertex_array) {
        meshData = {
          vertices: result.vertices || result.vertex_array,
          indices: result.indices || result.index_array || result.faces,
          normals: result.normals || result.normal_array,
          vertex_colors: result.vertex_colors || result.colors,
          tagged_edges: result.tagged_edges,
          edge_classifications: result.edge_classifications,
          triangle_count: result.triangle_count || (result.indices?.length / 3),
          face_mapping: result.face_mapping,
          vertex_face_ids: result.vertex_face_ids,
        };
      } else if (result.geometry?.vertices) {
        meshData = {
          vertices: result.geometry.vertices,
          indices: result.geometry.indices || result.geometry.faces,
          normals: result.geometry.normals,
          vertex_colors: result.geometry.vertex_colors,
          triangle_count: result.geometry.triangle_count,
          face_mapping: result.geometry.face_mapping,
          vertex_face_ids: result.geometry.vertex_face_ids,
        };
      }

      // Fetch mesh from URL if meshData is missing vertices
      if (result.mesh_url && meshData && !meshData.vertices) {
        try {
          const meshResponse = await fetch(result.mesh_url);
          if (meshResponse.ok) {
            meshData = await meshResponse.json();
          }
        } catch (meshError) {
          console.error("Error fetching mesh data:", meshError);
        }
      }

      if (meshData && (result.vertex_colors || meshData.vertex_colors)) {
        meshData.vertex_colors = result.vertex_colors || meshData.vertex_colors;
      }

      // Extract analysis data
      const analysis = {
        volume_cm3: result.volume_cm3,
        surface_area_cm2: result.surface_area_cm2,
        complexity_score: result.complexity_score,
        confidence: result.confidence,
        method: result.method,
        triangle_count: result.triangle_count,
        geometric_features: null,
        recommended_processes: result.recommended_processes,
        routing_reasoning: result.routing_reasoning,
        machining_summary: result.machining_summary,
      };

      setFiles((prev) =>
        prev.map((f, i) =>
          i === index
            ? {
                ...f,
                meshData,
                analysis,
                isAnalyzing: false,
                uploadProgress: 100,
                analysisProgress: 100,
                analysisStatus: 'Complete',
              }
            : f,
        ),
      );

      toast({
        title: "✅ CAD Analysis Complete",
        description: `${fileWithQty.file.name} analyzed successfully`,
      });
    } catch (error: any) {
      console.error("Error analyzing file:", error);
      setFiles((prev) => prev.map((f, i) => (i === index ? { 
        ...f, 
        isAnalyzing: false,
        uploadProgress: 0,
        analysisProgress: 0,
        analysisStatus: 'Preview unavailable',
      } : f)));
      toast({
        title: "Preview Unavailable",
        description: "Failed to load your step file. We will still receive your request.",
      });
    }
  };

  // Validate CAD file
  const validateCADFile = (file: File): boolean => {
    const validExtensions = ['.step', '.stp', '.iges', '.igs'];
    const hasValidExtension = validExtensions.some(ext => 
      file.name.toLowerCase().endsWith(ext)
    );
    return hasValidExtension;
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    const invalidFiles = selectedFiles.filter(file => !validateCADFile(file));

    if (invalidFiles.length > 0) {
      toast({
        title: "Invalid file type",
        description: `Only STEP (.step, .stp) or IGES (.iges, .igs) files are supported.`,
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

    for (let idx = 0; idx < filesWithQuantity.length; idx++) {
      const fileIndex = files.length + idx;
      await analyzeFile(filesWithQuantity[idx], fileIndex);
    }
  };

  const handleFileDrop = async (droppedFiles: FileList) => {
    const selectedFiles = Array.from(droppedFiles);
    if (selectedFiles.length === 0) return;

    const validFiles = selectedFiles.filter(file => validateCADFile(file));
    const invalidFiles = selectedFiles.filter(file => !validateCADFile(file));

    if (invalidFiles.length > 0) {
      toast({
        title: "Invalid file type",
        description: `Only STEP (.step, .stp) or IGES (.iges, .igs) files are supported.`,
        variant: "destructive",
      });
    }

    if (validFiles.length === 0) return;

    const filesWithQuantity = validFiles.map((file) => ({
      file,
      quantity: 1,
    }));

    const newFiles = [...files, ...filesWithQuantity];
    setFiles(newFiles);

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

  const handleContinueFromProject = () => {
    if (!selectedProjectId) {
      toast({
        title: "Select a project",
        description: "Please select or create a project before uploading files.",
        variant: "destructive",
      });
      return;
    }
    setCurrentScreen("upload");
  };

  const handleContinueFromUpload = async () => {
    if (!user || !selectedProjectId) return;
    
    const failedFiles = files.filter((f) => !f.analysis && !f.isAnalyzing);
    
    if (failedFiles.length > 0) {
      toast({
        title: "Some files couldn't be previewed",
        description: "We will still process your request.",
      });
    }

    // Save parts directly and redirect to dashboard
    console.log("📤 Saving parts to project:", selectedProjectId);
    setUploading(true);

    try {
      // Save each file as a project part
      for (const fileData of files) {
        // First, save mesh data to cad_meshes if available
        let meshId = null;
        let thumbnailUrl = null;

        if (fileData.meshData && fileData.meshData.vertices?.length > 0) {
          // Save mesh to database
          const { data: meshRecord, error: meshError } = await supabase
            .from('cad_meshes')
            .insert({
              file_name: fileData.file.name,
              file_hash: `${fileData.file.name}_${fileData.file.size}_${Date.now()}`,
              vertices: fileData.meshData.vertices,
              indices: fileData.meshData.indices || [],
              normals: fileData.meshData.normals || [],
              triangle_count: fileData.meshData.triangle_count || 0,
              vertex_colors: fileData.meshData.vertex_colors || [],
              feature_edges: fileData.meshData.feature_edges || [],
              face_mapping: fileData.meshData.face_mapping || {},
              vertex_face_ids: fileData.meshData.vertex_face_ids || [],
              tagged_feature_edges: fileData.meshData.tagged_edges || [],
              edge_classifications: fileData.meshData.edge_classifications || [],
              geometric_features: fileData.analysis?.geometric_features || {},
            })
            .select('id')
            .single();

          if (!meshError && meshRecord) {
            meshId = meshRecord.id;
          } else {
            console.warn('Failed to save mesh:', meshError);
          }

          // Generate and upload thumbnail
          try {
            console.log('🖼️ Generating thumbnail for:', fileData.file.name);
            const thumbnailBlob = await generateThumbnail({
              vertices: fileData.meshData.vertices,
              indices: fileData.meshData.indices || [],
              normals: fileData.meshData.normals || [],
            });

            const thumbnailPath = `${user.id}/${selectedProjectId}/${Date.now()}_${fileData.file.name.replace(/\.[^/.]+$/, '')}_thumb.png`;
            
            const { error: uploadError } = await supabase.storage
              .from('part-thumbnails')
              .upload(thumbnailPath, thumbnailBlob, {
                contentType: 'image/png',
                upsert: true,
              });

            if (!uploadError) {
              const { data: publicUrlData } = supabase.storage
                .from('part-thumbnails')
                .getPublicUrl(thumbnailPath);
              
              thumbnailUrl = publicUrlData.publicUrl;
              console.log('✅ Thumbnail uploaded:', thumbnailUrl);
            } else {
              console.warn('Failed to upload thumbnail:', uploadError);
            }
          } catch (thumbError) {
            console.warn('Failed to generate thumbnail:', thumbError);
          }
        }

        // Create project part
        const { error: partError } = await supabase
          .from('project_parts')
          .insert({
            project_id: selectedProjectId,
            user_id: user.id,
            file_name: fileData.file.name,
            file_path: fileData.filePath || '',
            mesh_id: meshId,
            thumbnail_url: thumbnailUrl,
            processing_method: fileData.process || 'CNC-Milling',
            material: fileData.material || 'Aluminum 6061',
            quantity: fileData.quantity,
            analysis_data: fileData.meshData ? { meshData: fileData.meshData } : null,
            volume_cm3: fileData.analysis?.volume_cm3 || null,
            surface_area_cm2: fileData.analysis?.surface_area_cm2 || null,
            complexity_score: fileData.analysis?.complexity_score || null,
            status: 'draft',
          });

        if (partError) {
          console.error('Failed to create part:', partError);
          throw partError;
        }
      }

      toast({
        title: "✅ Parts Saved",
        description: `${files.length} part(s) added to your project.`,
      });
      
      // Redirect to dashboard with project selected
      navigate(`/dashboard?project=${selectedProjectId}`);
      
    } catch (error: any) {
      console.error("Submission error:", error);
      toast({
        title: "❌ Failed to Save Parts",
        description: error.message || "Failed to save parts to project",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const handleBack = () => {
    if (currentScreen === "upload") {
      setCurrentScreen("project");
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

  // Project Selection Screen
  if (currentScreen === "project") {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="backdrop-blur-sm border border-white/20 rounded-sm overflow-hidden shadow-[0_0_30px_rgba(255,255,255,0.15)]" style={{ backgroundColor: "rgba(60, 60, 60, 0.75)" }}>
          <div className="p-6 pb-2">
            <h2 className="text-2xl font-bold text-white">Step 1: Select Project</h2>
            <p className="text-gray-400 mt-1">
              Choose an existing project or create a new one to organize your parts
            </p>
          </div>
          <div className="p-6 pt-4 space-y-6">
            <ProjectSelector
              projects={projects}
              selectedProjectId={selectedProjectId}
              onSelectProject={setSelectedProjectId}
              onCreateProject={createProject}
              loading={projectsLoading}
            />
            <div className="flex justify-end pt-4">
              <Button
                size="lg"
                onClick={handleContinueFromProject}
                disabled={!selectedProjectId}
                className="bg-primary hover:bg-primary/90 text-white"
              >
                Continue to Upload
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (currentScreen === "upload") {
    return (
      <div className="space-y-4">
        {/* Back button */}
        <div className="max-w-5xl mx-auto">
          <Button
            variant="ghost"
            onClick={handleBack}
            className="text-gray-400 hover:text-white"
          >
            ← Back to Project Selection
          </Button>
        </div>
        <FileUploadScreen
          files={files}
          onFileSelect={handleFileSelect}
          onFileDrop={handleFileDrop}
          onRemoveFile={handleRemoveFile}
          onRetryFile={handleRetryAnalysis}
          onContinue={handleContinueFromUpload}
          isAnalyzing={isAnalyzing}
          isSaving={uploading}
        />
      </div>
    );
  }

  // Default: redirect to project selection
  return null;
};
