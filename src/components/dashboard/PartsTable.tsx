import { useState } from 'react';
import { Plus, Settings, Move, Copy, Trash2, Share2, Upload, FileBox, Minus, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ProjectPart, PartStatus } from '@/hooks/useProjectParts';
import { CustomerProject } from '@/hooks/useCustomerProjects';
import { cn } from '@/lib/utils';

interface PartsTableProps {
  parts: ProjectPart[];
  projects: CustomerProject[];
  loading: boolean;
  onUpdatePart: (id: string, updates: Partial<ProjectPart>) => Promise<ProjectPart | null>;
  onDeleteParts: (ids: string[]) => Promise<boolean>;
  onUploadClick: () => void;
}

const statusConfig: Record<PartStatus, { label: string; className: string }> = {
  draft: { label: 'Draft', className: 'bg-muted text-muted-foreground' },
  confirmation_required: { label: 'Confirmation Required', className: 'bg-yellow-500/20 text-yellow-600' },
  quoted: { label: 'Quoted', className: 'bg-blue-500/20 text-blue-600' },
  order_preparation: { label: 'Order Preparation', className: 'bg-purple-500/20 text-purple-600' },
  completed: { label: 'Completed', className: 'bg-green-500/20 text-green-600' },
};

const processingMethods = [
  'CNC-Milling',
  'CNC-Turning',
  'CNC-Milling/Turning',
  'Wire EDM',
  'Sheet Metal',
  'Die Casting',
];

const materials = [
  'Aluminum 6061',
  'Aluminum 7075',
  'Stainless Steel 304',
  'Stainless Steel 316',
  'Steel 1045',
  'Brass',
  'Copper',
  'Titanium',
  'ABS',
  'POM',
  'Nylon',
];

type FilterTab = 'all' | 'auto' | 'manual' | 'unable';

export function PartsTable({
  parts,
  projects,
  loading,
  onUpdatePart,
  onDeleteParts,
  onUploadClick,
}: PartsTableProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<FilterTab>('all');

  const filteredParts = parts.filter((part) => {
    if (activeTab === 'all') return true;
    if (activeTab === 'auto') return part.status === 'quoted' || part.status === 'completed';
    if (activeTab === 'manual') return part.status === 'confirmation_required';
    if (activeTab === 'unable') return part.status === 'draft';
    return true;
  });

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredParts.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredParts.map((p) => p.id)));
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return;
    const success = await onDeleteParts(Array.from(selectedIds));
    if (success) {
      setSelectedIds(new Set());
    }
  };

  const handleQuantityChange = (part: ProjectPart, delta: number) => {
    const newQuantity = Math.max(1, part.quantity + delta);
    onUpdatePart(part.id, { quantity: newQuantity });
  };

  const getProjectName = (projectId: string) => {
    const project = projects.find((p) => p.id === projectId);
    return project?.name || 'Unknown';
  };

  const tabs: { key: FilterTab; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'auto', label: 'Auto quotation' },
    { key: 'manual', label: 'Manual quotation' },
    { key: 'unable', label: 'Unable to quote' },
  ];

  return (
    <div className="flex-1 flex flex-col bg-background">
      {/* Toolbar */}
      <div className="p-4 border-b flex items-center gap-2 flex-wrap">
        <Button onClick={onUploadClick} className="gap-2">
          <Plus className="h-4 w-4" />
          New Quotation
        </Button>
        <Button variant="outline" size="icon">
          <Settings className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" disabled={selectedIds.size === 0}>
          <Move className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" disabled={selectedIds.size === 0}>
          <Copy className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          disabled={selectedIds.size === 0}
          onClick={handleDeleteSelected}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" disabled={selectedIds.size === 0}>
          <Share2 className="h-4 w-4" />
        </Button>
      </div>

      {/* Filter Tabs */}
      <div className="px-4 border-b">
        <div className="flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                'px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px',
                activeTab === tab.key
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
          </div>
        ) : filteredParts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
            <FileBox className="h-16 w-16 mb-4 opacity-50" />
            <p className="text-lg font-medium">No parts yet</p>
            <p className="text-sm">Upload CAD files to get started</p>
            <Button onClick={onUploadClick} className="mt-4 gap-2">
              <Upload className="h-4 w-4" />
              Upload Parts
            </Button>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">
                  <Checkbox
                    checked={selectedIds.size === filteredParts.length && filteredParts.length > 0}
                    onCheckedChange={toggleSelectAll}
                  />
                </TableHead>
                <TableHead className="w-16">Preview</TableHead>
                <TableHead>Project</TableHead>
                <TableHead>Part Name</TableHead>
                <TableHead>Processing</TableHead>
                <TableHead>Material</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="w-28">Quantity</TableHead>
                <TableHead className="text-right">Subtotal</TableHead>
                <TableHead className="w-24">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredParts.map((part) => (
                <TableRow key={part.id} className={cn(selectedIds.has(part.id) && 'bg-muted/50')}>
                  <TableCell>
                    <Checkbox
                      checked={selectedIds.has(part.id)}
                      onCheckedChange={() => toggleSelect(part.id)}
                    />
                  </TableCell>
                  <TableCell>
                    <div className="w-12 h-12 bg-muted rounded flex items-center justify-center">
                      {part.thumbnail_url ? (
                        <img
                          src={part.thumbnail_url}
                          alt={part.file_name}
                          className="w-full h-full object-cover rounded"
                        />
                      ) : (
                        <FileBox className="h-6 w-6 text-muted-foreground" />
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">{getProjectName(part.project_id)}</TableCell>
                  <TableCell>
                    <div>
                      <p className="font-medium truncate max-w-[200px]">{part.file_name}</p>
                      {part.part_number && (
                        <p className="text-xs text-muted-foreground">{part.part_number}</p>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Select
                      value={part.processing_method || ''}
                      onValueChange={(value) => onUpdatePart(part.id, { processing_method: value })}
                    >
                      <SelectTrigger className="w-[150px] h-8">
                        <SelectValue placeholder="Select..." />
                      </SelectTrigger>
                      <SelectContent>
                        {processingMethods.map((method) => (
                          <SelectItem key={method} value={method}>
                            {method}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </TableCell>
                  <TableCell>
                    <Select
                      value={part.material || ''}
                      onValueChange={(value) => onUpdatePart(part.id, { material: value })}
                    >
                      <SelectTrigger className="w-[150px] h-8">
                        <SelectValue placeholder="Select..." />
                      </SelectTrigger>
                      <SelectContent>
                        {materials.map((material) => (
                          <SelectItem key={material} value={material}>
                            {material}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </TableCell>
                  <TableCell>
                    <Badge className={cn('font-normal', statusConfig[part.status].className)}>
                      {statusConfig[part.status].label}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => handleQuantityChange(part, -1)}
                        disabled={part.quantity <= 1}
                      >
                        <Minus className="h-3 w-3" />
                      </Button>
                      <Input
                        type="number"
                        value={part.quantity}
                        onChange={(e) =>
                          onUpdatePart(part.id, { quantity: Math.max(1, parseInt(e.target.value) || 1) })
                        }
                        className="w-14 h-7 text-center"
                        min={1}
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => handleQuantityChange(part, 1)}
                      >
                        <Plus className="h-3 w-3" />
                      </Button>
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-medium">
                    {part.subtotal ? `$${part.subtotal.toFixed(2)}` : '-'}
                  </TableCell>
                  <TableCell>
                    <Button variant="outline" size="sm" className="gap-1">
                      Next
                      <ChevronRight className="h-3 w-3" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}
