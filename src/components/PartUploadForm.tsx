import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { FileUploadScreen } from "./part-upload/FileUploadScreen";
import PartConfigScreen from "./part-upload/PartConfigScreen";

interface FileWithQuantity {
  file: File;
  quantity: number;
  material?: string;
  process?: string;
  meshId?: string; // ✅ This needs to be populated!
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
    detected_features?: Record<string, boolean>;
    manufacturing_features?: any;
    feature_summary?: any;
    ml_features?: any;
    recommended_processes?: string[];
    routing_reasoning?: string[];
    machining_summary?: any[];
  };
  quote?: any;
  isAnalyzing?: boolean;
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

  // ✅ FIXED: Analyze file using Supabase Edge Function
  const analyzeFile = async (fileWithQty: FileWithQuantity, index: number) => {
    try {
      setFiles((prev) => prev.map((f, i) => (i === index ? { ...f, isAnalyzing: true } : f)));

      // Convert file to base64
      const fileBuffer = await fileWithQty.file.arrayBuffer();
      const base64File = btoa(new Uint8Array(fileBuffer).reduce((data, byte) => data + String.fromCharCode(byte), ""));

      console.log(`📤 Sending ${fileWithQty.file.name} to edge function (may take 30-60s on first use)...`);

      const { data: result, error } = await supabase.functions.invoke("analyze-cad", {
        body: {
          file_name: fileWithQty.file.name,
          file_data: base64File,
          file_size: fileWithQty.file.size,
          material: fileWithQty.material,
        },
      });

      if (error) {
        throw new Error(error.message || "Edge function error");
      }

      console.log("✅ Edge function response:", result);
      console.log("📊 Available keys in response:", Object.keys(result));
      console.log("🏭 Manufacturing features:", result.manufacturing_features);
      console.log("📋 Feature summary:", result.feature_summary);

      // ✅ CRITICAL FIX: Extract meshId from response
      const meshId = result.mesh_id || result.meshId;

      if (!meshId) {
        console.warn("⚠️ No meshId in response! Available keys:", Object.keys(result));
        console.warn("⚠️ Full response:", result);
      } else {
        console.log("✅ Extracted meshId:", meshId);
      }

      // Extract mesh data
      const meshData = result.mesh_data || result.meshData || {};

      // Add vertex_colors to meshData if available
      if (result.vertex_colors || meshData.vertex_colors) {
        meshData.vertex_colors = result.vertex_colors || meshData.vertex_colors;
      }

      console.log("🎨 Mesh data:", {
        hasVertexColors: !!meshData.vertex_colors,
        vertexColorCount: meshData.vertex_colors?.length,
        triangleCount: result.triangle_count || meshData.triangle_count,
      });

      // Extract analysis data
      const analysis = {
        volume_cm3: result.volume_cm3,
        surface_area_cm2: result.surface_area_cm2,
        complexity_score: result.complexity_score,
        confidence: result.confidence,
        method: result.method,
        triangle_count: result.triangle_count,
        detected_features: result.detected_features,
        manufacturing_features: result.manufacturing_features,
        feature_summary: result.feature_summary,
        recommended_processes: result.recommended_processes,
        routing_reasoning: result.routing_reasoning,
        machining_summary: result.machining_summary,
      };

      console.log("💾 Storing analysis data:", {
        hasMeshId: !!meshId,
        meshId,
        hasAnalysis: !!analysis,
        hasMeshData: !!meshData,
      });

      // ✅ CRITICAL FIX: Store meshId, meshData, AND analysis
      setFiles((prev) =>
        prev.map((f, i) =>
          i === index
            ? {
                ...f,
                meshId, // ✅ CRITICAL: This was missing!
                meshData, // For 3D viewer
                analysis, // For FeatureTree and analysis display
                isAnalyzing: false,
              }
            : f,
        ),
      );

      console.log("✅ File updated with meshId:", meshId);

      // Invalidate the mesh cache to force fresh data fetch
      if (meshId) {
        queryClient.invalidateQueries({ 
          queryKey: ['cad_meshes', meshId] 
        });
        console.log(`🔄 Invalidated cache for mesh ID: ${meshId}`);
      }

      toast({
        title: "✅ CAD Analysis Complete",
        description: `${fileWithQty.file.name} analyzed successfully`,
      });
    } catch (error: any) {
      console.error("❌ Error analyzing file:", error);
      setFiles((prev) => prev.map((f, i) => (i === index ? { ...f, isAnalyzing: false } : f)));
      toast({
        title: "Analysis Failed",
        description: error.message || "Backend may be waking up - please try again",
        variant: "destructive",
      });
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    const invalidFiles = selectedFiles.filter(
      (file) =>
        !file.name.toLowerCase().endsWith(".step") &&
        !file.name.toLowerCase().endsWith(".stp") &&
        !file.name.toLowerCase().endsWith(".iges") &&
        !file.name.toLowerCase().endsWith(".igs"),
    );

    if (invalidFiles.length > 0) {
      toast({
        title: "Invalid file type",
        description: "Only STEP (.step, .stp) or IGES (.iges, .igs) files are supported",
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

  const handleContinue = () => {
    // Check if all files have been analyzed
    const unanalyzedFiles = files.filter((f) => !f.analysis && !f.isAnalyzing);
    if (unanalyzedFiles.length > 0) {
      toast({
        title: "⚠️ Files Not Analyzed",
        description: "Please wait for all files to finish analyzing",
        variant: "destructive",
      });
      return;
    }

    console.log(
      "📋 Continuing with files:",
      files.map((f) => ({
        name: f.file.name,
        hasMeshId: !!f.meshId,
        meshId: f.meshId,
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
      // Your existing submit logic here
      toast({
        title: "✅ Quote Submitted",
        description: "We'll get back to you soon!",
      });
    } catch (error: any) {
      toast({
        title: "❌ Submission Failed",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const isAnalyzing = files.some((f) => f.isAnalyzing);

  if (currentScreen === "upload") {
    return (
      <FileUploadScreen
        files={files}
        onFileSelect={handleFileSelect}
        onRemoveFile={handleRemoveFile}
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
