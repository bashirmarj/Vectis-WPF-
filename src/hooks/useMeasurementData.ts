import { useState, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/hooks/use-toast';

export interface TaggedFeatureEdge {
  feature_id: number;
  start: [number, number, number];
  end: [number, number, number];
  type: 'line' | 'circle' | 'arc' | 'ellipse' | 'spline';
  diameter?: number;
  radius?: number;
  length?: number;
  center?: [number, number, number];
}

export interface MeasurementFace {
  face_id: number;
  surface_type: string;
  center: [number, number, number] | null;
  normal: [number, number, number] | null;
  area_mm2: number;
  radius?: number;
  axis?: [number, number, number];
}

export interface MeasurementEdge {
  edge_id: number;
  edge_type: string;
  start_point: [number, number, number];
  end_point: [number, number, number];
  length_mm: number;
  radius?: number;
  diameter?: number;
  center?: [number, number, number];
  adjacent_faces: number[];
}

export interface MeasurementData {
  faces: MeasurementFace[];
  edges: MeasurementEdge[];
  edge_to_face_map: Record<number, number[]>;
  face_adjacency: Record<number, number[]>;
}

export interface MeasurementMetadata {
  bounding_box: {
    min: [number, number, number];
    max: [number, number, number];
    dimensions: [number, number, number];
  };
  volume_mm3: number;
  surface_area_mm2: number;
  face_count: number;
  edge_count: number;
}

export interface MeasurementAnalysisResult {
  success: boolean;
  correlation_id: string;
  processing_time_ms: number;
  mesh_data: {
    vertices: number[];
    indices: number[];
    normals: number[];
    face_mapping?: Record<string, any>;
    vertex_face_ids?: number[];
  } | null;
  measurement_data: MeasurementData | null;
  tagged_edges: TaggedFeatureEdge[];
  metadata: MeasurementMetadata | null;
}

interface UseMeasurementDataReturn {
  isLoading: boolean;
  error: string | null;
  measurementResult: MeasurementAnalysisResult | null;
  taggedEdges: TaggedFeatureEdge[];
  fetchMeasurementData: (file: File) => Promise<MeasurementAnalysisResult | null>;
  fetchMeasurementDataFromUrl: (fileUrl: string, fileName: string) => Promise<MeasurementAnalysisResult | null>;
  clearData: () => void;
}

export function useMeasurementData(): UseMeasurementDataReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [measurementResult, setMeasurementResult] = useState<MeasurementAnalysisResult | null>(null);
  const [taggedEdges, setTaggedEdges] = useState<TaggedFeatureEdge[]>([]);

  const fetchMeasurementData = useCallback(async (file: File): Promise<MeasurementAnalysisResult | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const { data, error: invokeError } = await supabase.functions.invoke('analyze-measurement', {
        body: formData,
      });

      if (invokeError) {
        throw new Error(invokeError.message || 'Failed to analyze file');
      }

      if (!data?.success) {
        throw new Error(data?.error || 'Analysis failed');
      }

      const result = data as MeasurementAnalysisResult;
      setMeasurementResult(result);
      setTaggedEdges(result.tagged_edges || []);

      console.log('📏 Measurement data loaded:', {
        taggedEdgesCount: result.tagged_edges?.length || 0,
        facesCount: result.measurement_data?.faces?.length || 0,
        edgesCount: result.measurement_data?.edges?.length || 0,
      });

      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch measurement data';
      setError(message);
      toast({
        title: 'Measurement Analysis Failed',
        description: message,
        variant: 'destructive',
      });
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchMeasurementDataFromUrl = useCallback(async (
    fileUrl: string, 
    fileName: string
  ): Promise<MeasurementAnalysisResult | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const { data, error: invokeError } = await supabase.functions.invoke('analyze-measurement', {
        body: JSON.stringify({ fileUrl, fileName }),
      });

      if (invokeError) {
        throw new Error(invokeError.message || 'Failed to analyze file');
      }

      if (!data?.success) {
        throw new Error(data?.error || 'Analysis failed');
      }

      const result = data as MeasurementAnalysisResult;
      setMeasurementResult(result);
      setTaggedEdges(result.tagged_edges || []);

      console.log('📏 Measurement data loaded from URL:', {
        taggedEdgesCount: result.tagged_edges?.length || 0,
        facesCount: result.measurement_data?.faces?.length || 0,
        edgesCount: result.measurement_data?.edges?.length || 0,
      });

      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch measurement data';
      setError(message);
      toast({
        title: 'Measurement Analysis Failed',
        description: message,
        variant: 'destructive',
      });
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearData = useCallback(() => {
    setMeasurementResult(null);
    setTaggedEdges([]);
    setError(null);
  }, []);

  return {
    isLoading,
    error,
    measurementResult,
    taggedEdges,
    fetchMeasurementData,
    fetchMeasurementDataFromUrl,
    clearData,
  };
}
