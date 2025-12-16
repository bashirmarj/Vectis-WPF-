import { useState, useEffect } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';

export interface CustomerProject {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  is_shared: boolean;
  parent_folder_id: string | null;
  created_at: string;
  updated_at: string;
}

export function useCustomerProjects() {
  const [projects, setProjects] = useState<CustomerProject[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();
  const { toast } = useToast();

  const fetchProjects = async () => {
    if (!user) {
      setProjects([]);
      setLoading(false);
      return;
    }

    try {
      const { data, error } = await supabase
        .from('customer_projects')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      setProjects(data || []);
    } catch (error: any) {
      toast({
        title: 'Error fetching projects',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const createProject = async (name: string, description?: string, parentFolderId?: string) => {
    if (!user) return null;

    try {
      const { data, error } = await supabase
        .from('customer_projects')
        .insert({
          user_id: user.id,
          name,
          description: description || null,
          parent_folder_id: parentFolderId || null,
        })
        .select()
        .single();

      if (error) throw error;

      setProjects((prev) => [data, ...prev]);
      toast({
        title: 'Project created',
        description: `"${name}" has been created successfully.`,
      });
      return data;
    } catch (error: any) {
      toast({
        title: 'Error creating project',
        description: error.message,
        variant: 'destructive',
      });
      return null;
    }
  };

  const updateProject = async (id: string, updates: Partial<Pick<CustomerProject, 'name' | 'description' | 'is_shared'>>) => {
    try {
      const { data, error } = await supabase
        .from('customer_projects')
        .update(updates)
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;

      setProjects((prev) => prev.map((p) => (p.id === id ? data : p)));
      return data;
    } catch (error: any) {
      toast({
        title: 'Error updating project',
        description: error.message,
        variant: 'destructive',
      });
      return null;
    }
  };

  const deleteProject = async (id: string) => {
    try {
      const { error } = await supabase
        .from('customer_projects')
        .delete()
        .eq('id', id);

      if (error) throw error;

      setProjects((prev) => prev.filter((p) => p.id !== id));
      toast({
        title: 'Project deleted',
        description: 'The project has been deleted successfully.',
      });
      return true;
    } catch (error: any) {
      toast({
        title: 'Error deleting project',
        description: error.message,
        variant: 'destructive',
      });
      return false;
    }
  };

  useEffect(() => {
    fetchProjects();
  }, [user]);

  return {
    projects,
    loading,
    createProject,
    updateProject,
    deleteProject,
    refetch: fetchProjects,
  };
}
