import { useState } from 'react';
import { Plus, FolderOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { CustomerProject } from '@/hooks/useCustomerProjects';

interface ProjectSelectorProps {
  projects: CustomerProject[];
  selectedProjectId: string | null;
  onSelectProject: (projectId: string) => void;
  onCreateProject: (name: string) => Promise<CustomerProject | null>;
  loading?: boolean;
}

export function ProjectSelector({
  projects,
  selectedProjectId,
  onSelectProject,
  onCreateProject,
  loading,
}: ProjectSelectorProps) {
  const [isCreating, setIsCreating] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [creating, setCreating] = useState(false);

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;
    
    setCreating(true);
    const newProject = await onCreateProject(newProjectName.trim());
    setCreating(false);
    
    if (newProject) {
      onSelectProject(newProject.id);
      setNewProjectName('');
      setIsCreating(false);
    }
  };

  const selectedProject = projects.find((p) => p.id === selectedProjectId);

  if (loading) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-sm border border-gray-200 bg-white">
        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary" />
        <span className="text-gray-600">Loading projects...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <FolderOpen className="h-5 w-5 text-primary" />
        <h3 className="font-semibold text-gray-900">Select or Create a Project</h3>
      </div>

      {isCreating ? (
        <div className="flex gap-2">
          <Input
            placeholder="Enter project name..."
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            className="flex-1 bg-white border-gray-300 text-gray-900 placeholder:text-gray-500"
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleCreateProject();
              if (e.key === 'Escape') {
                setIsCreating(false);
                setNewProjectName('');
              }
            }}
            autoFocus
          />
          <Button
            onClick={handleCreateProject}
            disabled={!newProjectName.trim() || creating}
            className="bg-primary hover:bg-primary/90"
          >
            {creating ? 'Creating...' : 'Create'}
          </Button>
          <Button
            variant="outline"
            onClick={() => {
              setIsCreating(false);
              setNewProjectName('');
            }}
            className="border-gray-300 text-gray-700 hover:bg-gray-100"
          >
            Cancel
          </Button>
        </div>
      ) : (
        <div className="flex gap-2">
          <Select
            value={selectedProjectId || ''}
            onValueChange={onSelectProject}
          >
            <SelectTrigger className="flex-1 bg-white border-gray-300 text-gray-900">
              <SelectValue placeholder="Select a project..." />
            </SelectTrigger>
            <SelectContent>
              {projects.length === 0 ? (
                <div className="p-2 text-sm text-muted-foreground text-center">
                  No projects yet. Create your first project.
                </div>
              ) : (
                projects.map((project) => (
                  <SelectItem key={project.id} value={project.id}>
                    {project.name}
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            onClick={() => setIsCreating(true)}
            className="gap-2 border-gray-300 text-gray-700 hover:bg-gray-100"
          >
            <Plus className="h-4 w-4" />
            New Project
          </Button>
        </div>
      )}

      {selectedProject && (
        <div className="p-3 rounded-sm border border-primary/30 bg-primary/5">
          <p className="text-sm text-gray-700">
            Files will be uploaded to: <span className="font-medium text-gray-900">{selectedProject.name}</span>
          </p>
        </div>
      )}
    </div>
  );
}
