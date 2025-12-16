import { useState, useEffect } from 'react';
import { X, Maximize2, Minimize2, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { supabase } from '@/integrations/supabase/client';
import { CADViewer } from '@/components/CADViewer';
import { ProjectPart } from '@/hooks/useProjectParts';

interface PartViewerDialogProps {
  part: ProjectPart | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  triangle_count: number;
  vertex_colors?: string[];
  feature_edges?: number[][][];
  face_mapping?: Record<number, { triangle_indices: number[]; triangle_range: [number, number] }>;
  vertex_face_ids?: number[];
  tagged_feature_edges?: any[];
  edge_classifications?: any[];
  geometric_features?: any;
}

export function PartViewerDialog({ part, open, onOpenChange }: PartViewerDialogProps) {
  const [meshData, setMeshData] = useState<MeshData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!open || !part) {
      setMeshData(null);
      setError(null);
      return;
    }

    const fetchMeshData = async () => {
      // First try to load from mesh_id if available
      if (part.mesh_id) {
        setLoading(true);
        setError(null);

        try {
          const { data, error: fetchError } = await supabase
            .from('cad_meshes')
            .select('*')
            .eq('id', part.mesh_id)
            .maybeSingle();

          if (fetchError) throw fetchError;

          if (data) {
            setMeshData({
              vertices: data.vertices,
              indices: data.indices,
              normals: data.normals,
              triangle_count: data.triangle_count,
              vertex_colors: data.vertex_colors || undefined,
              feature_edges: data.feature_edges as any || undefined,
              face_mapping: data.face_mapping as any || undefined,
              vertex_face_ids: data.vertex_face_ids as any || undefined,
              tagged_feature_edges: data.tagged_feature_edges as any || undefined,
              edge_classifications: data.edge_classifications as any || undefined,
              geometric_features: data.geometric_features as any || undefined,
            });
            setLoading(false);
            return;
          }
        } catch (err) {
          console.error('Error fetching mesh from cad_meshes:', err);
        }
      }

      // Fall back to analysis_data if mesh_id not available
      if (part.analysis_data && typeof part.analysis_data === 'object') {
        const analysisData = part.analysis_data as any;
        if (analysisData.meshData) {
          setMeshData(analysisData.meshData);
          setLoading(false);
          return;
        }
      }

      // If no mesh data available, show error
      setError('No 3D preview available for this part');
      setLoading(false);
    };

    fetchMeshData();
  }, [open, part]);

  if (!part) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent 
        className={`${isFullscreen ? 'max-w-[95vw] h-[95vh]' : 'max-w-4xl h-[80vh]'} p-0 flex flex-col`}
      >
        <DialogHeader className="p-4 border-b flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="text-lg">{part.file_name}</DialogTitle>
              <p className="text-sm text-muted-foreground mt-1">
                {part.processing_method} • {part.material}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsFullscreen(!isFullscreen)}
              >
                {isFullscreen ? (
                  <Minimize2 className="h-4 w-4" />
                ) : (
                  <Maximize2 className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </DialogHeader>

        <div className="flex-1 relative">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center bg-background">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">Loading 3D model...</p>
              </div>
            </div>
          ) : error ? (
            <div className="absolute inset-0 flex items-center justify-center bg-background">
              <div className="text-center">
                <p className="text-muted-foreground">{error}</p>
                <p className="text-sm text-muted-foreground mt-2">
                  The part was uploaded but 3D preview is not available.
                </p>
              </div>
            </div>
          ) : meshData ? (
            <CADViewer
              meshData={meshData}
              fileName={part.file_name}
            />
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}
