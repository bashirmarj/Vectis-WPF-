import { useState } from 'react';
import { FolderPlus, FolderOpen, Folder, ChevronRight, ChevronDown, Trash2, Edit2, Share2, MoreVertical } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { CustomerProject } from '@/hooks/useCustomerProjects';
import { cn } from '@/lib/utils';

interface ProjectSidebarProps {
  projects: CustomerProject[];
  selectedProjectId: string | null;
  onSelectProject: (projectId: string | null) => void;
  onCreateProject: (name: string) => Promise<CustomerProject | null>;
  onDeleteProject: (id: string) => Promise<boolean>;
  onUpdateProject: (id: string, updates: Partial<CustomerProject>) => Promise<CustomerProject | null>;
}

export function ProjectSidebar({
  projects,
  selectedProjectId,
  onSelectProject,
  onCreateProject,
  onDeleteProject,
  onUpdateProject,
}: ProjectSidebarProps) {
  const [isCreating, setIsCreating] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    shared: true,
    myProjects: true,
  });
  const [editingProject, setEditingProject] = useState<CustomerProject | null>(null);
  const [deletingProject, setDeletingProject] = useState<CustomerProject | null>(null);
  const [editName, setEditName] = useState('');

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;
    const project = await onCreateProject(newProjectName.trim());
    if (project) {
      setNewProjectName('');
      setIsCreating(false);
      onSelectProject(project.id);
    }
  };

  const handleEditProject = async () => {
    if (!editingProject || !editName.trim()) return;
    await onUpdateProject(editingProject.id, { name: editName.trim() });
    setEditingProject(null);
    setEditName('');
  };

  const handleDeleteProject = async () => {
    if (!deletingProject) return;
    const success = await onDeleteProject(deletingProject.id);
    if (success && selectedProjectId === deletingProject.id) {
      onSelectProject(null);
    }
    setDeletingProject(null);
  };

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const sharedProjects = projects.filter((p) => p.is_shared);
  const myProjects = projects.filter((p) => !p.is_shared);

  const renderProjectItem = (project: CustomerProject) => (
    <div
      key={project.id}
      className={cn(
        'group flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer transition-colors',
        selectedProjectId === project.id
          ? 'bg-primary/10 text-primary'
          : 'hover:bg-muted text-muted-foreground hover:text-foreground'
      )}
      onClick={() => onSelectProject(project.id)}
    >
      {selectedProjectId === project.id ? (
        <FolderOpen className="h-4 w-4 flex-shrink-0" />
      ) : (
        <Folder className="h-4 w-4 flex-shrink-0" />
      )}
      <span className="flex-1 truncate text-sm">{project.name}</span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <MoreVertical className="h-3 w-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              setEditingProject(project);
              setEditName(project.name);
            }}
          >
            <Edit2 className="h-4 w-4 mr-2" />
            Rename
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              onUpdateProject(project.id, { is_shared: !project.is_shared });
            }}
          >
            <Share2 className="h-4 w-4 mr-2" />
            {project.is_shared ? 'Unshare' : 'Share'}
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              setDeletingProject(project);
            }}
            className="text-destructive focus:text-destructive"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );

  return (
    <div className="w-64 border-r bg-card flex flex-col h-full">
      <div className="p-4 border-b">
        <Button
          onClick={() => setIsCreating(true)}
          className="w-full justify-start"
          variant="outline"
        >
          <FolderPlus className="h-4 w-4 mr-2" />
          Create New Project
        </Button>
      </div>

      <ScrollArea className="flex-1 p-2">
        {/* Shared Projects Section */}
        <div className="mb-4">
          <button
            onClick={() => toggleSection('shared')}
            className="flex items-center gap-1 w-full px-2 py-1 text-xs font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground"
          >
            {expandedSections.shared ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            Shared Projects
          </button>
          {expandedSections.shared && (
            <div className="mt-1 space-y-0.5">
              {sharedProjects.length === 0 ? (
                <p className="px-3 py-2 text-xs text-muted-foreground italic">No shared projects</p>
              ) : (
                sharedProjects.map(renderProjectItem)
              )}
            </div>
          )}
        </div>

        {/* My Projects Section */}
        <div>
          <button
            onClick={() => toggleSection('myProjects')}
            className="flex items-center gap-1 w-full px-2 py-1 text-xs font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground"
          >
            {expandedSections.myProjects ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            My Projects
          </button>
          {expandedSections.myProjects && (
            <div className="mt-1 space-y-0.5">
              {myProjects.length === 0 ? (
                <p className="px-3 py-2 text-xs text-muted-foreground italic">No projects yet</p>
              ) : (
                myProjects.map(renderProjectItem)
              )}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* All Projects Button */}
      <div className="p-2 border-t">
        <Button
          variant={selectedProjectId === null ? 'secondary' : 'ghost'}
          className="w-full justify-start"
          onClick={() => onSelectProject(null)}
        >
          <Folder className="h-4 w-4 mr-2" />
          All Parts
        </Button>
      </div>

      {/* Create Project Dialog */}
      <Dialog open={isCreating} onOpenChange={setIsCreating}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Project</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Project name"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreating(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateProject}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Project Dialog */}
      <Dialog open={!!editingProject} onOpenChange={() => setEditingProject(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename Project</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Project name"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleEditProject()}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingProject(null)}>
              Cancel
            </Button>
            <Button onClick={handleEditProject}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deletingProject} onOpenChange={() => setDeletingProject(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{deletingProject?.name}"? This will also delete all parts
              in this project. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteProject} className="bg-destructive text-destructive-foreground">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
