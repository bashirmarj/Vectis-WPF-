import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Upload, X } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useCustomerProjects } from '@/hooks/useCustomerProjects';
import { useProjectParts, ProjectPart } from '@/hooks/useProjectParts';
import { ProjectSidebar } from '@/components/dashboard/ProjectSidebar';
import { PartsTable } from '@/components/dashboard/PartsTable';
import { ReportPanel } from '@/components/dashboard/ReportPanel';
import Navigation from '@/components/Navigation';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

export default function CustomerDashboard() {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { toast } = useToast();
  
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState(false);

  const { 
    projects, 
    loading: projectsLoading, 
    createProject, 
    updateProject, 
    deleteProject 
  } = useCustomerProjects();

  const { 
    parts, 
    loading: partsLoading, 
    createPart, 
    updatePart, 
    deleteParts,
    refetch: refetchParts 
  } = useProjectParts(selectedProjectId || undefined);

  // Handle project selection from URL params
  useEffect(() => {
    const projectId = searchParams.get('project');
    if (projectId && projects.length > 0) {
      const projectExists = projects.some(p => p.id === projectId);
      if (projectExists) {
        setSelectedProjectId(projectId);
      }
    }
  }, [searchParams, projects]);

  useEffect(() => {
    if (!authLoading && !user) {
      navigate('/auth');
    }
  }, [user, authLoading, navigate]);

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0 || !user) return;

    // Ensure we have a project selected or create one
    let projectId = selectedProjectId;
    if (!projectId) {
      const newProject = await createProject('New Project');
      if (!newProject) {
        toast({
          title: 'Error',
          description: 'Failed to create project for uploads',
          variant: 'destructive',
        });
        return;
      }
      projectId = newProject.id;
      setSelectedProjectId(projectId);
    }

    setUploading(true);

    for (const file of Array.from(files)) {
      try {
        // Upload file to storage
        const fileExt = file.name.split('.').pop();
        const filePath = `${user.id}/${projectId}/${Date.now()}-${file.name}`;
        
        const { error: uploadError } = await supabase.storage
          .from('cad-files')
          .upload(filePath, file);

        if (uploadError) throw uploadError;

        // Create part record
        await createPart({
          project_id: projectId,
          file_name: file.name,
          file_path: filePath,
          processing_method: 'CNC-Milling',
          material: 'Aluminum 6061',
        });
      } catch (error: any) {
        toast({
          title: `Error uploading ${file.name}`,
          description: error.message,
          variant: 'destructive',
        });
      }
    }

    setUploading(false);
    setIsUploadModalOpen(false);
    refetchParts();
  };

  // Redirect to services page for full upload experience
  const handleNewQuotation = () => {
    navigate('/services/custom-parts');
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navigation />
      
      <div className="flex-1 flex">
        {/* Left Sidebar - Projects */}
        <ProjectSidebar
          projects={projects}
          selectedProjectId={selectedProjectId}
          onSelectProject={setSelectedProjectId}
          onCreateProject={createProject}
          onDeleteProject={deleteProject}
          onUpdateProject={updateProject}
        />

        {/* Main Content - Parts Table */}
        <PartsTable
          parts={parts}
          projects={projects}
          loading={partsLoading || projectsLoading}
          onUpdatePart={updatePart}
          onDeleteParts={deleteParts}
          onUploadClick={handleNewQuotation}
        />

        {/* Right Panel - Report */}
        <ReportPanel
          parts={parts}
          selectedIds={selectedIds}
        />
      </div>

      {/* Upload Modal */}
      <Dialog open={isUploadModalOpen} onOpenChange={setIsUploadModalOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Upload CAD Files</DialogTitle>
          </DialogHeader>
          <div
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary/50 transition-colors"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFileUpload(e.dataTransfer.files);
            }}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <input
              id="file-input"
              type="file"
              multiple
              accept=".step,.stp,.iges,.igs,.stl"
              className="hidden"
              onChange={(e) => handleFileUpload(e.target.files)}
            />
            {uploading ? (
              <div className="flex flex-col items-center gap-2">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
                <p className="text-sm text-muted-foreground">Uploading files...</p>
              </div>
            ) : (
              <>
                <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-sm font-medium">
                  Drag & drop files here or click to browse
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Supported formats: STEP, IGES, STL
                </p>
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
