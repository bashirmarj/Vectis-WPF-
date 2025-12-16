import { useState, useEffect } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';

export type PartStatus = 'draft' | 'confirmation_required' | 'quoted' | 'order_preparation' | 'completed';

export interface ProjectPart {
  id: string;
  project_id: string;
  user_id: string;
  file_name: string;
  file_path: string;
  thumbnail_url: string | null;
  part_number: string | null;
  processing_method: string | null;
  material: string | null;
  surface_treatment: string | null;
  status: PartStatus;
  quantity: number;
  unit_price: number | null;
  subtotal: number | null;
  mesh_id: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
  // New analysis columns
  analysis_data: Record<string, any> | null;
  volume_cm3: number | null;
  surface_area_cm2: number | null;
  complexity_score: number | null;
}

export function useProjectParts(projectId?: string) {
  const [parts, setParts] = useState<ProjectPart[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();
  const { toast } = useToast();

  const fetchParts = async () => {
    if (!user) {
      setParts([]);
      setLoading(false);
      return;
    }

    try {
      let query = supabase
        .from('project_parts')
        .select('*')
        .order('created_at', { ascending: false });

      if (projectId) {
        query = query.eq('project_id', projectId);
      }

      const { data, error } = await query;

      if (error) throw error;
      setParts((data as ProjectPart[]) || []);
    } catch (error: any) {
      toast({
        title: 'Error fetching parts',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const createPart = async (partData: {
    project_id: string;
    file_name: string;
    file_path: string;
    thumbnail_url?: string;
    part_number?: string;
    processing_method?: string;
    material?: string;
    surface_treatment?: string;
    quantity?: number;
    mesh_id?: string;
    analysis_data?: Record<string, any>;
    volume_cm3?: number;
    surface_area_cm2?: number;
    complexity_score?: number;
  }) => {
    if (!user) return null;

    try {
      const { data, error } = await supabase
        .from('project_parts')
        .insert({
          ...partData,
          user_id: user.id,
        })
        .select()
        .single();

      if (error) throw error;

      setParts((prev) => [data as ProjectPart, ...prev]);
      toast({
        title: 'Part added',
        description: `"${partData.file_name}" has been added successfully.`,
      });
      return data as ProjectPart;
    } catch (error: any) {
      toast({
        title: 'Error adding part',
        description: error.message,
        variant: 'destructive',
      });
      return null;
    }
  };

  const updatePart = async (id: string, updates: Partial<ProjectPart>) => {
    try {
      const { data, error } = await supabase
        .from('project_parts')
        .update(updates)
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;

      setParts((prev) => prev.map((p) => (p.id === id ? (data as ProjectPart) : p)));
      return data as ProjectPart;
    } catch (error: any) {
      toast({
        title: 'Error updating part',
        description: error.message,
        variant: 'destructive',
      });
      return null;
    }
  };

  const deletePart = async (id: string) => {
    try {
      const { error } = await supabase
        .from('project_parts')
        .delete()
        .eq('id', id);

      if (error) throw error;

      setParts((prev) => prev.filter((p) => p.id !== id));
      toast({
        title: 'Part deleted',
        description: 'The part has been deleted successfully.',
      });
      return true;
    } catch (error: any) {
      toast({
        title: 'Error deleting part',
        description: error.message,
        variant: 'destructive',
      });
      return false;
    }
  };

  const deleteParts = async (ids: string[]) => {
    try {
      const { error } = await supabase
        .from('project_parts')
        .delete()
        .in('id', ids);

      if (error) throw error;

      setParts((prev) => prev.filter((p) => !ids.includes(p.id)));
      toast({
        title: 'Parts deleted',
        description: `${ids.length} parts have been deleted successfully.`,
      });
      return true;
    } catch (error: any) {
      toast({
        title: 'Error deleting parts',
        description: error.message,
        variant: 'destructive',
      });
      return false;
    }
  };

  useEffect(() => {
    fetchParts();
  }, [user, projectId]);

  return {
    parts,
    loading,
    createPart,
    updatePart,
    deletePart,
    deleteParts,
    refetch: fetchParts,
  };
}
