import { useEffect, useState } from 'react';
import * as XLSX from 'xlsx';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, useSortable, horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';
import { Loader2, Save, Plus, Trash2, Settings, X, Download, Upload, Edit } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';

interface ManufacturingProcess {
  id: string;
  name: string;
  base_rate_per_hour: number;
  setup_cost: number;
  complexity_multiplier: number;
  is_active: boolean;
  feed_rate_mm_per_min?: number;
  spindle_speed_rpm?: number;
  depth_of_cut_mm?: number;
  tool_change_time_minutes?: number;
  rapid_feed_rate_mm_per_min?: number;
}

interface CrossSection {
  width: number;
  thickness: number;
  cost_per_inch: number;
  weight_per_foot?: number;
  weight_per_bar?: number;
  shape?: 'rectangular' | 'circular'; // 'rectangular' is default for backward compatibility
}

interface SheetConfiguration {
  width: number;
  height: number;
  thickness: number;
  cost_per_sheet: number;
  unit: 'cm' | 'inch';
}

interface MaterialCategory {
  id: string;
  name: string;
  description: string | null;
  display_order: number;
}

interface MaterialCost {
  id: string;
  material_name: string;
  cost_per_cubic_cm: number;
  cost_per_square_cm: number;
  density: number | null;
  finish_options: any;
  is_active: boolean;
  pricing_method?: string;
  cross_sections?: CrossSection[];
  sheet_configurations?: SheetConfiguration[];
  default_nesting_efficiency?: number;
  category_id?: string | null;
  price_per_lb?: number;
  machinability_rating?: number;
  hardness_brinell?: number;
  cutting_speed_m_per_min?: number;
}

interface SurfaceTreatment {
  id: string;
  name: string;
  category: 'heat_treatment' | 'post_process' | 'surface_treatment';
  cost_per_cm2: number;
  description: string | null;
  is_active: boolean;
}

interface MaterialProcessParameter {
  id: string;
  material_id: string;
  process_id: string;
  spindle_speed_rpm: number | null;
  feed_rate_mm_per_min: number | null;
  depth_of_cut_mm: number | null;
  cutting_speed_m_per_min: number | null;
  material_removal_rate_adjustment: number;
  tool_wear_multiplier: number;
  surface_finish_factor: number;
  setup_time_multiplier: number;
  cycle_time_multiplier: number;
}

const PricingSettings = () => {
  // Helper function to convert decimal to fraction
  const decimalToFraction = (decimal: number): string => {
    if (decimal === 0) return '0';
    
    // Common fractions used in metalworking with tolerance for floating point
    const fractions: Array<[number, string]> = [
      [0.0625, '1/16'],
      [0.09375, '3/32'],
      [0.125, '1/8'],
      [0.1875, '3/16'],
      [0.25, '1/4'],
      [0.3125, '5/16'],
      [0.375, '3/8'],
      [0.4375, '7/16'],
      [0.5, '1/2'],
      [0.5625, '9/16'],
      [0.625, '5/8'],
      [0.6875, '11/16'],
      [0.75, '3/4'],
      [0.8125, '13/16'],
      [0.875, '7/8'],
      [0.9375, '15/16'],
      [1, '1'],
      [1.125, '1-1/8'],
      [1.25, '1-1/4'],
      [1.5, '1-1/2'],
      [1.75, '1-3/4'],
      [2, '2'],
      [2.5, '2-1/2'],
      [3, '3'],
      [4, '4'],
      [4.5, '4-1/2'],
      [5, '5'],
      [6, '6'],
    ];
    
    // Find closest match with small tolerance
    const tolerance = 0.0001;
    for (const [value, fraction] of fractions) {
      if (Math.abs(decimal - value) < tolerance) {
        return fraction;
      }
    }
    
    // Fallback to decimal with 4 decimal places
    return decimal.toFixed(4);
  };

  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [processes, setProcesses] = useState<ManufacturingProcess[]>([]);
  const [categories, setCategories] = useState<MaterialCategory[]>([]);
  const [materials, setMaterials] = useState<MaterialCost[]>([]);
  const [treatments, setTreatments] = useState<SurfaceTreatment[]>([]);
  const [materialProcessParams, setMaterialProcessParams] = useState<MaterialProcessParameter[]>([]);
  const [selectedMaterials, setSelectedMaterials] = useState<Record<string, string>>({});
  const [localParamsState, setLocalParamsState] = useState<Record<string, Partial<MaterialProcessParameter>>>({});
  const [activeTab, setActiveTab] = useState<string>('');
  const [selectedCrossSections, setSelectedCrossSections] = useState<Record<string, number>>({});
  const [editingTreatmentId, setEditingTreatmentId] = useState<string | null>(null);
  const [treatmentFormData, setTreatmentFormData] = useState<{
    name: string;
    category: 'heat_treatment' | 'post_process' | 'surface_treatment';
    cost_per_cm2: number;
    description: string;
    is_active: boolean;
  }>({
    name: "",
    category: "surface_treatment",
    cost_per_cm2: 0,
    description: "",
    is_active: true
  });

  useEffect(() => {
    const checkAdminAndFetchData = async () => {
      if (authLoading) return;

      if (!user) {
        navigate('/auth');
        return;
      }

      const { data: roles, error: roleError } = await supabase
        .from('user_roles')
        .select('role')
        .eq('user_id', user.id)
        .eq('role', 'admin')
        .single();

      if (roleError || !roles) {
        toast({
          title: 'Access Denied',
          description: 'You do not have permission to access this page.',
          variant: 'destructive',
        });
        navigate('/');
        return;
      }

      await fetchPricingData();
    };

    checkAdminAndFetchData();
  }, [user, authLoading, navigate, toast]);

  const fetchPricingData = async () => {
    setLoading(true);
    try {
      // Fetch each table individually with specific error handling
      const processesResponse = await supabase.from('manufacturing_processes').select('*').order('name');
      if (processesResponse.error) {
        console.error('Failed to load processes:', processesResponse.error);
        throw new Error('Failed to load manufacturing processes');
      }

      const materialsResponse = await supabase.from('material_costs').select('*').order('material_name');
      if (materialsResponse.error) {
        console.error('Failed to load materials:', materialsResponse.error);
        throw new Error('Failed to load material costs');
      }

      const categoriesResponse = await supabase.from('material_categories').select('*').order('display_order');
      if (categoriesResponse.error) {
        console.error('Failed to load categories:', categoriesResponse.error);
        throw new Error('Failed to load material categories');
      }

      const treatmentsResponse = await supabase.from('surface_treatments').select('*').order('category', { ascending: true }).order('name', { ascending: true });
      if (treatmentsResponse.error) {
        console.error('Failed to load treatments:', treatmentsResponse.error);
        throw new Error('Failed to load surface treatments');
      }

      const materialProcessParamsResponse = await supabase.from('material_process_parameters').select('*');
      if (materialProcessParamsResponse.error) {
        console.error('Failed to load material-process parameters:', materialProcessParamsResponse.error);
        throw new Error('Failed to load material-process parameters');
      }

      setProcesses(processesResponse.data || []);
      setCategories(categoriesResponse.data || []);
      setTreatments((treatmentsResponse.data as SurfaceTreatment[]) || []);
      setMaterialProcessParams((materialProcessParamsResponse.data as MaterialProcessParameter[]) || []);
      
      // Parse finish_options and cross_sections from JSONB to array
      const parsedMaterials = (materialsResponse.data || []).map(m => ({
        ...m,
        finish_options: Array.isArray(m.finish_options) ? m.finish_options : [],
        pricing_method: m.pricing_method || 'weight',
        cross_sections: Array.isArray(m.cross_sections) 
          ? (m.cross_sections as unknown as CrossSection[])
          : [],
        sheet_configurations: Array.isArray(m.sheet_configurations)
          ? (m.sheet_configurations as unknown as SheetConfiguration[])
          : [],
        default_nesting_efficiency: m.default_nesting_efficiency || 0.75,
        category_id: m.category_id || null
      }));
      setMaterials(parsedMaterials);
      
      toast({
        title: 'Success',
        description: `Loaded ${materialProcessParamsResponse.data?.length || 0} material-process parameters`,
      });
    } catch (error: any) {
      console.error('Pricing data fetch error:', error);
      toast({
        title: 'Error',
        description: error.message || 'Failed to load pricing data',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const updateProcess = (id: string, field: keyof ManufacturingProcess, value: any) => {
    setProcesses(prev =>
      prev.map(p => (p.id === id ? { ...p, [field]: value } : p))
    );
  };

  const updateMaterial = (id: string, field: keyof MaterialCost, value: any) => {
    setMaterials(prev =>
      prev.map(m => (m.id === id ? { ...m, [field]: value } : m))
    );
  };

  const saveProcesses = async () => {
    setSaving(true);
    try {
    const updates = processes.map(p => ({
      id: p.id,
      name: p.name,
      base_rate_per_hour: p.base_rate_per_hour,
      setup_cost: p.setup_cost,
      complexity_multiplier: p.complexity_multiplier,
      is_active: p.is_active,
      feed_rate_mm_per_min: p.feed_rate_mm_per_min ?? 500,
      spindle_speed_rpm: p.spindle_speed_rpm ?? 3000,
      depth_of_cut_mm: p.depth_of_cut_mm ?? 2.0,
      tool_change_time_minutes: p.tool_change_time_minutes ?? 2.0,
      rapid_feed_rate_mm_per_min: p.rapid_feed_rate_mm_per_min ?? 5000,
    }));

      for (const update of updates) {
        const { error } = await supabase
          .from('manufacturing_processes')
          .update(update)
          .eq('id', update.id);

        if (error) throw error;
      }

      toast({
        title: 'Success',
        description: 'Manufacturing processes updated successfully',
      });
    } catch (error: any) {
      toast({
        title: 'Error',
        description: 'Failed to update processes',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const saveMaterials = async () => {
    setSaving(true);
    try {
      const updates = materials.map(m => ({
        id: m.id,
        material_name: m.material_name,
        cost_per_cubic_cm: m.cost_per_cubic_cm,
        cost_per_square_cm: m.cost_per_square_cm,
        density: m.density,
        finish_options: m.finish_options,
        is_active: m.is_active,
        pricing_method: m.pricing_method || 'weight',
        cross_sections: (m.cross_sections || []) as any,
        sheet_configurations: (m.sheet_configurations || []) as any,
        default_nesting_efficiency: m.default_nesting_efficiency || 0.75,
        category_id: m.category_id || null,
        price_per_lb: m.price_per_lb || null,
      }));

      for (const update of updates) {
        const { error } = await supabase
          .from('material_costs')
          .update(update)
          .eq('id', update.id);

        if (error) throw error;
      }

      toast({
        title: 'Success',
        description: 'Material costs updated successfully',
      });
    } catch (error: any) {
      toast({
        title: 'Error',
        description: 'Failed to update materials',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const addNewProcess = async () => {
    try {
      const { data, error } = await supabase
        .from('manufacturing_processes')
        .insert({
          name: 'New Process',
          base_rate_per_hour: 50.0,
          setup_cost: 100.0,
          complexity_multiplier: 1.0,
          is_active: true,
        })
        .select()
        .single();

      if (error) throw error;
      if (data) {
        setProcesses(prev => [...prev, data]);
        toast({
          title: 'Success',
          description: 'New process added',
        });
      }
    } catch (error: any) {
      toast({
        title: 'Error',
        description: 'Failed to add process',
        variant: 'destructive',
      });
    }
  };

  const addNewMaterial = async () => {
    // Determine category_id based on active tab
    const defaultCategoryId = activeTab === 'uncategorized' ? null : (activeTab || null);
    
    // Generate unique material name
    let newMaterialName = 'New Material';
    let counter = 1;
    while (materials.some(m => m.material_name === newMaterialName)) {
      counter++;
      newMaterialName = `New Material ${counter}`;
    }
    
    try {
      const { data, error } = await supabase
        .from('material_costs')
        .insert({
          material_name: newMaterialName,
          cost_per_cubic_cm: 0.1,
          cost_per_square_cm: 0.01,
          density: 1.0,
          finish_options: ['As-machined'],
          is_active: true,
          pricing_method: 'weight',
          cross_sections: [],
          category_id: defaultCategoryId,
        })
        .select()
        .single();

      if (error) throw error;
      if (data) {
        setMaterials(prev => [...prev, {
          ...data,
          finish_options: Array.isArray(data.finish_options) ? data.finish_options : [],
          pricing_method: data.pricing_method || 'weight',
          cross_sections: Array.isArray(data.cross_sections) 
            ? (data.cross_sections as unknown as CrossSection[])
            : [],
          sheet_configurations: Array.isArray(data.sheet_configurations)
            ? (data.sheet_configurations as unknown as SheetConfiguration[])
            : [],
          default_nesting_efficiency: data.default_nesting_efficiency || 0.75,
          category_id: data.category_id || null
        }]);
        toast({
          title: 'Success',
          description: 'New material added',
        });
      }
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to add material',
        variant: 'destructive',
      });
    }
  };


  const addNewCategory = async () => {
    const categoryName = prompt('Enter new category name:');
    if (!categoryName?.trim()) return;

    // Check if category already exists
    const existingCategory = categories.find(
      cat => cat.name.toLowerCase() === categoryName.trim().toLowerCase()
    );
    
    if (existingCategory) {
      toast({
        title: 'Category Already Exists',
        description: `A category named "${categoryName}" already exists.`,
        variant: 'destructive',
      });
      return;
    }

    try {
      const { data, error } = await supabase
        .from('material_categories')
        .insert({
          name: categoryName.trim(),
          description: null,
          display_order: categories.length + 1,
        })
        .select()
        .single();

      if (error) {
        // Handle duplicate key error specifically
        if (error.code === '23505') {
          toast({
            title: 'Category Already Exists',
            description: `A category named "${categoryName}" already exists.`,
            variant: 'destructive',
          });
        } else {
          throw error;
        }
        return;
      }
      
      if (data) {
        setCategories(prev => [...prev, data].sort((a, b) => a.display_order - b.display_order));
        toast({
          title: 'Success',
          description: 'New category added',
        });
      }
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to add category',
        variant: 'destructive',
      });
    }
  };

  const addCrossSection = (materialId: string) => {
    setMaterials(prev =>
      prev.map(m => m.id === materialId 
        ? { 
            ...m, 
            cross_sections: [...(m.cross_sections || []), { width: 0, thickness: 0, cost_per_inch: 0, weight_per_foot: 0, weight_per_bar: 0, shape: 'rectangular' }] 
          }
        : m
      )
    );
  };

  const handleExcelUpload = async (e: React.ChangeEvent<HTMLInputElement>, materialId: string, material: MaterialCost) => {
    const file = e.target.files?.[0];
    if (!file) return;

    toast({
      title: "Processing Excel file...",
      description: "Extracting cross-section data from the spreadsheet",
    });

    try {
      const data = await file.arrayBuffer();
      const workbook = XLSX.read(data);
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: null }) as any[][];

      const crossSections: CrossSection[] = [];
      const pricePerLb = material.price_per_lb || 1.0;
      
      // Detect if this is a round bar file based on filename
      const isRoundBar = file.name.toLowerCase().includes('round');

      // Helper to parse values including fractions
      const parseValue = (val: any): number => {
        if (val === null || val === undefined || val === '') return 0;
        if (typeof val === 'number') return val;
        if (typeof val === 'string') {
          const trimmed = val.trim();
          if (!trimmed) return 0;
          // Handle fractions like "1/16", "3/32"
          if (trimmed.includes('/') && !trimmed.includes(' ')) {
            const parts = trimmed.split('/');
            return parseFloat(parts[0]) / parseFloat(parts[1]);
          }
          // Handle mixed numbers like "1 1/8"
          if (trimmed.includes(' ') && trimmed.includes('/')) {
            const parts = trimmed.split(' ');
            const whole = parseFloat(parts[0]);
            const fracParts = parts[1].split('/');
            return whole + (parseFloat(fracParts[0]) / parseFloat(fracParts[1]));
          }
          return parseFloat(trimmed) || 0;
        }
        return 0;
      };

      let currentThickness = 0;

      // Skip header row, start from row 1
      for (let i = 1; i < jsonData.length; i++) {
        const row = jsonData[i];
        if (!row || row.length === 0) continue;

        const col0 = parseValue(row[0]); // First column value
        const col1 = parseValue(row[1]); // Second column (weight per foot or empty)
        const col2 = parseValue(row[2]); // Third column (weight per bar or empty)

        if (isRoundBar) {
          // For round bars: first column is diameter, second is weight per foot
          if (col0 > 0 && col1 > 0) {
            const diameter = col0;
            const weightPerFoot = col1;
            const weightPerBar = col2 > 0 ? col2 : weightPerFoot * 20; // Round bars often come in 20ft lengths
            
            crossSections.push({
              width: diameter,
              thickness: diameter, // For circular, thickness = diameter
              weight_per_foot: weightPerFoot,
              weight_per_bar: weightPerBar,
              cost_per_inch: (weightPerFoot / 12) * pricePerLb,
              shape: 'circular'
            });
          }
        } else {
          // For flat bars (rectangular): detect thickness header and width rows
          // Detect thickness header: first column has value, but columns 1 and 2 are empty/zero
          if (col0 > 0 && col1 === 0 && col2 === 0) {
            currentThickness = col0;
            console.log(`Found thickness header: ${currentThickness}`);
            continue;
          }

          // Data row: has thickness set, first column is width, and has weight data
          if (currentThickness > 0 && col0 > 0 && col1 > 0) {
            const width = col0;
            const weightPerFoot = col1;
            const weightPerBar = col2 > 0 ? col2 : weightPerFoot * 12;
            
            crossSections.push({
              thickness: currentThickness,
              width,
              weight_per_foot: weightPerFoot,
              weight_per_bar: weightPerBar,
              cost_per_inch: (weightPerFoot / 12) * pricePerLb,
              shape: 'rectangular'
            });
          }
        }
      }

      if (crossSections.length > 0) {
        setMaterials(prev =>
          prev.map(m => {
            if (m.id === materialId) {
              // Combine existing and new cross-sections (shapes already set)
              const combined = [...(m.cross_sections || []), ...crossSections];
              
              // Sort by thickness first (ascending), then by width (ascending)
              const sorted = combined.sort((a, b) => {
                if (a.thickness !== b.thickness) {
                  return a.thickness - b.thickness;
                }
                return a.width - b.width;
              });
              
              return { ...m, cross_sections: sorted };
            }
            return m;
          })
        );
        
        const shapeType = isRoundBar ? 'circular (round bar)' : 'rectangular (flat bar)';
        toast({
          title: "Success!",
          description: `Added ${crossSections.length} ${shapeType} cross-sections.`,
        });
      } else {
        toast({
          title: "No data found",
          description: "Could not extract cross-section data. Check console for parsing details.",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('Error processing Excel file:', error);
      toast({
        title: "Error",
        description: "Failed to process the Excel file. Make sure it's a valid .xlsx or .xls file",
        variant: "destructive"
      });
    }
    
    // Reset the input
    e.target.value = '';
  };

  const handleMaterialsExcelImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    toast({
      title: "Processing Excel file...",
      description: "Importing materials from spreadsheet",
    });

    try {
      const data = await file.arrayBuffer();
      const workbook = XLSX.read(data);
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json(worksheet) as any[];

      const importedMaterials: MaterialCost[] = [];

      for (const row of jsonData) {
        const materialName = row['Material Name'] || row['material_name'] || row['Name'];
        if (!materialName) continue;

        // Parse pricing method
        let pricingMethod = (row['Pricing Method'] || row['pricing_method'] || 'weight').toLowerCase();
        if (!['weight', 'linear_inch', 'sheet'].includes(pricingMethod)) {
          pricingMethod = 'weight';
        }

        // Parse category
        let categoryId = null;
        const categoryName = row['Category'] || row['category'];
        if (categoryName) {
          const category = categories.find(c => c.name.toLowerCase() === categoryName.toLowerCase());
          categoryId = category?.id || null;
        }

        // Parse finish options
        let finishOptions = ['As-machined'];
        if (row['Finish Options'] || row['finish_options']) {
          const finishStr = String(row['Finish Options'] || row['finish_options']);
          finishOptions = finishStr.split(',').map(f => f.trim()).filter(Boolean);
        }

        // Insert material
        const { data: newMaterial, error } = await supabase
          .from('material_costs')
          .insert({
            material_name: materialName,
            cost_per_cubic_cm: parseFloat(row['Cost Per Cubic CM'] || row['cost_per_cubic_cm'] || 0.1),
            cost_per_square_cm: parseFloat(row['Cost Per Square CM'] || row['cost_per_square_cm'] || 0.01),
            density: row['Density'] || row['density'] ? parseFloat(row['Density'] || row['density']) : null,
            price_per_lb: row['Price Per LB'] || row['price_per_lb'] ? parseFloat(row['Price Per LB'] || row['price_per_lb']) : null,
            finish_options: finishOptions,
            is_active: true,
            pricing_method: pricingMethod,
            cross_sections: [],
            sheet_configurations: [],
            default_nesting_efficiency: parseFloat(row['Nesting Efficiency'] || row['nesting_efficiency'] || 0.75),
            category_id: categoryId,
          })
          .select()
          .single();

        if (error) {
          console.error('Error importing material:', materialName, error);
          continue;
        }

        if (newMaterial) {
          importedMaterials.push({
            ...newMaterial,
            finish_options: Array.isArray(newMaterial.finish_options) ? newMaterial.finish_options : [],
            pricing_method: newMaterial.pricing_method || 'weight',
            cross_sections: [],
            sheet_configurations: [],
            default_nesting_efficiency: newMaterial.default_nesting_efficiency || 0.75,
            category_id: newMaterial.category_id || null
          });
        }
      }

      if (importedMaterials.length > 0) {
        setMaterials(prev => [...prev, ...importedMaterials]);
        toast({
          title: "Success!",
          description: `Imported ${importedMaterials.length} material(s) from Excel file`,
        });
      } else {
        toast({
          title: "No materials imported",
          description: "No valid material data found in the Excel file",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('Error processing Excel file:', error);
      toast({
        title: "Error",
        description: "Failed to process the Excel file",
        variant: "destructive"
      });
    }

    // Reset the input
    e.target.value = '';
  };

  const handleTableUpload = async (e: React.ChangeEvent<HTMLInputElement>, materialId: string, material: MaterialCost) => {
    const file = e.target.files?.[0];
    if (!file) return;

    toast({
      title: "Processing table...",
      description: "Analyzing the uploaded image to extract cross-section data",
    });

    try {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const base64Image = event.target?.result as string;
        
        const { data, error } = await supabase.functions.invoke('parse-material-table', {
          body: { 
            image: base64Image,
            price_per_lb: material.price_per_lb || 1.0 
          }
        });

        if (error) throw error;

        if (data.cross_sections && data.cross_sections.length > 0) {
          setMaterials(prev =>
            prev.map(m => m.id === materialId
              ? { ...m, cross_sections: [...(m.cross_sections || []), ...data.cross_sections] }
              : m
            )
          );
          
          toast({
            title: "Success!",
            description: `Imported ${data.cross_sections.length} cross-sections from the table`,
          });
        } else {
          toast({
            title: "No data found",
            description: "Could not extract cross-section data from the image",
            variant: "destructive"
          });
        }
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error('Error processing table:', error);
      toast({
        title: "Error",
        description: "Failed to process the table image",
        variant: "destructive"
      });
    }
    
    // Reset the input
    e.target.value = '';
  };

  const importEMJData = (materialId: string) => {
    const EMJ_COLD_ROLLED_STRIP: Record<string, Array<{ width: number; weight_per_foot: number; weight_per_bar: number }>> = {
      '0.0625': [ // 1/16"
        { width: 0.25, weight_per_foot: 0.0532, weight_per_bar: 0.6381 },
        { width: 0.375, weight_per_foot: 0.0798, weight_per_bar: 0.9572 },
        { width: 0.5, weight_per_foot: 0.1064, weight_per_bar: 1.276 },
        { width: 0.625, weight_per_foot: 0.1329, weight_per_bar: 1.595 },
        { width: 0.75, weight_per_foot: 0.1595, weight_per_bar: 1.914 },
        { width: 0.875, weight_per_foot: 0.1861, weight_per_bar: 2.233 },
        { width: 1.0, weight_per_foot: 0.2127, weight_per_bar: 2.552 },
        { width: 1.125, weight_per_foot: 0.2393, weight_per_bar: 2.871 },
        { width: 1.25, weight_per_foot: 0.2659, weight_per_bar: 3.191 },
        { width: 1.5, weight_per_foot: 0.3191, weight_per_bar: 3.829 },
        { width: 1.75, weight_per_foot: 0.3722, weight_per_bar: 4.467 },
        { width: 2.0, weight_per_foot: 0.4254, weight_per_bar: 5.105 },
        { width: 2.5, weight_per_foot: 0.5318, weight_per_bar: 6.381 },
        { width: 3.0, weight_per_foot: 0.6381, weight_per_bar: 7.657 }
      ],
      '0.09375': [ // 3/32"
        { width: 0.375, weight_per_foot: 0.1196, weight_per_bar: 1.436 },
        { width: 0.5, weight_per_foot: 0.1595, weight_per_bar: 1.914 },
        { width: 0.625, weight_per_foot: 0.1994, weight_per_bar: 2.393 },
        { width: 0.75, weight_per_foot: 0.2393, weight_per_bar: 2.871 },
        { width: 0.875, weight_per_foot: 0.2792, weight_per_bar: 3.350 },
        { width: 1.0, weight_per_foot: 0.3191, weight_per_bar: 3.829 },
        { width: 1.125, weight_per_foot: 0.3589, weight_per_bar: 4.307 },
        { width: 1.25, weight_per_foot: 0.3988, weight_per_bar: 4.786 },
        { width: 1.5, weight_per_foot: 0.4786, weight_per_bar: 5.743 },
        { width: 1.75, weight_per_foot: 0.5583, weight_per_bar: 6.700 }
      ],
      '0.125': [ // 1/8"
        { width: 0.1875, weight_per_foot: 0.0798, weight_per_bar: 0.9572 },
        { width: 0.25, weight_per_foot: 0.1064, weight_per_bar: 1.276 },
        { width: 0.375, weight_per_foot: 0.1595, weight_per_bar: 1.914 },
        { width: 0.5, weight_per_foot: 0.2127, weight_per_bar: 2.552 },
        { width: 0.625, weight_per_foot: 0.2659, weight_per_bar: 3.191 },
        { width: 0.75, weight_per_foot: 0.3191, weight_per_bar: 3.829 },
        { width: 0.875, weight_per_foot: 0.3722, weight_per_bar: 4.467 },
        { width: 1.0, weight_per_foot: 0.4254, weight_per_bar: 5.105 },
        { width: 1.125, weight_per_foot: 0.4786, weight_per_bar: 5.743 },
        { width: 1.25, weight_per_foot: 0.5318, weight_per_bar: 6.381 },
        { width: 1.5, weight_per_foot: 0.6381, weight_per_bar: 7.657 },
        { width: 1.75, weight_per_foot: 0.7445, weight_per_bar: 8.933 },
        { width: 2.0, weight_per_foot: 0.8508, weight_per_bar: 10.21 },
        { width: 2.5, weight_per_foot: 1.064, weight_per_bar: 12.76 },
        { width: 3.0, weight_per_foot: 1.276, weight_per_bar: 15.31 },
        { width: 4.0, weight_per_foot: 1.702, weight_per_bar: 20.42 },
        { width: 4.5, weight_per_foot: 1.914, weight_per_bar: 22.97 },
        { width: 5.0, weight_per_foot: 2.127, weight_per_bar: 25.52 },
        { width: 6.0, weight_per_foot: 2.552, weight_per_bar: 30.63 }
      ]
    };

    const material = materials.find(m => m.id === materialId);
    if (!material) return;

    const pricePerLb = material.price_per_lb || 1.0;
    const allCrossSections: CrossSection[] = [];

    Object.entries(EMJ_COLD_ROLLED_STRIP).forEach(([thickness, sizes]) => {
      const thicknessValue = parseFloat(thickness);
      
      sizes.forEach(size => {
        allCrossSections.push({
          width: size.width,
          thickness: thicknessValue,
          weight_per_foot: size.weight_per_foot,
          weight_per_bar: size.weight_per_bar,
          cost_per_inch: (size.weight_per_foot / 12) * pricePerLb
        });
      });
    });

    setMaterials(prev =>
      prev.map(m => m.id === materialId
        ? { ...m, cross_sections: allCrossSections }
        : m
      )
    );

    toast({
      title: 'Success',
      description: `Imported ${allCrossSections.length} EMJ standard cross-sections`,
    });
  };

  const removeCrossSection = (materialId: string, index: number) => {
    setMaterials(prev =>
      prev.map(m => m.id === materialId
        ? { 
            ...m, 
            cross_sections: (m.cross_sections || []).filter((_, i) => i !== index) 
          }
        : m
      )
    );
  };

  const updateCrossSection = (materialId: string, index: number, field: keyof CrossSection, value: number | string) => {
    setMaterials(prev =>
      prev.map(m => {
        if (m.id === materialId && m.cross_sections) {
          const updated = [...m.cross_sections];
          const currentSection = updated[index];
          
          // If changing to circular, set thickness to match width (diameter)
          if (field === 'shape' && value === 'circular') {
            updated[index] = { ...currentSection, shape: 'circular', thickness: currentSection.width };
          } 
          // If already circular and updating width (diameter), also update thickness
          else if (field === 'width' && currentSection.shape === 'circular') {
            updated[index] = { ...currentSection, width: value as number, thickness: value as number };
          }
          // If changing from circular to rectangular, keep current values
          else if (field === 'shape' && value === 'rectangular') {
            updated[index] = { ...currentSection, shape: 'rectangular' };
          }
          else {
            updated[index] = { ...currentSection, [field]: value };
          }
          
          return { ...m, cross_sections: updated };
        }
        return m;
      })
    );
  };

  const addSheetConfiguration = (materialId: string) => {
    setMaterials(prev =>
      prev.map(m => m.id === materialId
        ? {
            ...m,
            sheet_configurations: [
              ...(m.sheet_configurations || []),
              { width: 48, height: 96, thickness: 0.125, cost_per_sheet: 100, unit: 'inch' as const }
            ]
          }
        : m
      )
    );
  };

  const removeSheetConfiguration = (materialId: string, index: number) => {
    setMaterials(prev =>
      prev.map(m => m.id === materialId
        ? {
            ...m,
            sheet_configurations: (m.sheet_configurations || []).filter((_, i) => i !== index)
          }
        : m
      )
    );
  };

  const updateSheetConfiguration = (materialId: string, index: number, field: keyof SheetConfiguration, value: number | string) => {
    setMaterials(prev =>
      prev.map(m => m.id === materialId
        ? {
            ...m,
            sheet_configurations: (m.sheet_configurations || []).map((sc, i) =>
              i === index ? { ...sc, [field]: value } : sc
            )
          }
        : m
      )
    );
  };

  const deleteProcess = async (id: string) => {
    try {
      const { error } = await supabase
        .from('manufacturing_processes')
        .delete()
        .eq('id', id);

      if (error) throw error;
      setProcesses(prev => prev.filter(p => p.id !== id));
      toast({
        title: 'Success',
        description: 'Process deleted',
      });
    } catch (error: any) {
      toast({
        title: 'Error',
        description: 'Failed to delete process',
        variant: 'destructive',
      });
    }
  };

  const deleteMaterial = async (id: string) => {
    try {
      const { error } = await supabase
        .from('material_costs')
        .delete()
        .eq('id', id);

      if (error) throw error;
      setMaterials(prev => prev.filter(m => m.id !== id));
      toast({
        title: 'Success',
        description: 'Material deleted',
      });
    } catch (error: any) {
      toast({
        title: 'Error',
        description: 'Failed to delete material',
        variant: 'destructive',
      });
    }
  };

  // Surface Treatment handlers
  const handleTreatmentSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      if (editingTreatmentId) {
        const { error } = await supabase
          .from('surface_treatments')
          .update(treatmentFormData)
          .eq('id', editingTreatmentId);

        if (error) throw error;
        
        toast({
          title: "Treatment updated",
          description: "Surface treatment has been updated successfully.",
        });
      } else {
        const { error } = await supabase
          .from('surface_treatments')
          .insert([treatmentFormData]);

        if (error) throw error;
        
        toast({
          title: "Treatment created",
          description: "New surface treatment has been added.",
        });
      }

      resetTreatmentForm();
      await fetchPricingData();
    } catch (error: any) {
      toast({
        title: "Error saving treatment",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  const handleEditTreatment = (treatment: SurfaceTreatment) => {
    setEditingTreatmentId(treatment.id);
    setTreatmentFormData({
      name: treatment.name,
      category: treatment.category,
      cost_per_cm2: treatment.cost_per_cm2,
      description: treatment.description || "",
      is_active: treatment.is_active
    });
  };

  const handleDeleteTreatment = async (id: string) => {
    if (!confirm("Are you sure you want to delete this treatment?")) return;

    try {
      const { error } = await supabase
        .from('surface_treatments')
        .delete()
        .eq('id', id);

      if (error) throw error;

      toast({
        title: "Treatment deleted",
        description: "Surface treatment has been removed.",
      });

      await fetchPricingData();
    } catch (error: any) {
      toast({
        title: "Error deleting treatment",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  const resetTreatmentForm = () => {
    setEditingTreatmentId(null);
    setTreatmentFormData({
      name: "",
      category: "surface_treatment",
      cost_per_cm2: 0,
      description: "",
      is_active: true
    });
  };

  const getMaterialProcessParam = (processId: string, materialId: string) => {
    return materialProcessParams.find(
      p => p.process_id === processId && p.material_id === materialId
    );
  };

  const saveMaterialProcessParams = async (processId: string, materialId: string, params: Partial<MaterialProcessParameter>) => {
    const existing = getMaterialProcessParam(processId, materialId);
    
    if (existing) {
      const { error } = await supabase
        .from("material_process_parameters")
        .update(params)
        .eq("id", existing.id);

      if (error) {
        toast({
          title: "Error",
          description: "Failed to update material parameters",
          variant: "destructive",
        });
        return;
      }
    } else {
      const { error } = await supabase
        .from("material_process_parameters")
        .insert({
          process_id: processId,
          material_id: materialId,
          ...params,
        });

      if (error) {
        toast({
          title: "Error",
          description: "Failed to save material parameters",
          variant: "destructive",
        });
        return;
      }
    }

    await fetchPricingData();
    toast({
      title: "Success",
      description: "Material parameters saved successfully",
    });
  };

  const resetMaterialProcessParams = async (processId: string, materialId: string) => {
    const existing = getMaterialProcessParam(processId, materialId);
    if (existing) {
      const { error } = await supabase
        .from("material_process_parameters")
        .delete()
        .eq("id", existing.id);

      if (error) {
        toast({
          title: "Error",
          description: "Failed to reset parameters",
          variant: "destructive",
        });
        return;
      }

      await fetchPricingData();
      toast({
        title: "Success",
        description: "Parameters reset to defaults",
      });
    }
  };

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'heat_treatment': return 'Heat Treatment';
      case 'post_process': return 'Post Process';
      case 'surface_treatment': return 'Surface Treatment';
      default: return category;
    }
  };

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = async (event: any) => {
    const { active, over } = event;

    if (active.id !== over.id) {
      const oldIndex = categories.findIndex((cat) => cat.id === active.id);
      const newIndex = categories.findIndex((cat) => cat.id === over.id);

      const reorderedCategories = arrayMove(categories, oldIndex, newIndex);
      
      // Update display_order for all categories
      const updatedCategories = reorderedCategories.map((cat, index) => ({
        ...cat,
        display_order: index + 1
      }));

      setCategories(updatedCategories);

      // Save new order to database
      try {
        for (const cat of updatedCategories) {
          await supabase
            .from('material_categories')
            .update({ display_order: cat.display_order })
            .eq('id', cat.id);
        }

        toast({
          title: 'Success',
          description: 'Category order updated',
        });
      } catch (error: any) {
        toast({
          title: 'Error',
          description: 'Failed to update category order',
          variant: 'destructive',
        });
      }
    }
  };

  if (authLoading || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-2">
              <Settings className="h-6 w-6" />
              <h1 className="text-3xl font-bold">Pricing Settings</h1>
            </div>
            <p className="text-muted-foreground">
              Configure manufacturing processes, material costs, and pricing formulas
            </p>
          </div>

          <Tabs defaultValue="processes" className="space-y-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="processes">Manufacturing Processes</TabsTrigger>
              <TabsTrigger value="materials">Material Costs</TabsTrigger>
              <TabsTrigger value="treatments">Surface Treatments</TabsTrigger>
              <TabsTrigger value="formulas">Pricing Formulas</TabsTrigger>
            </TabsList>

            {/* Manufacturing Processes Tab */}
            <TabsContent value="processes" className="space-y-4">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Manufacturing Processes</CardTitle>
                      <CardDescription>
                        Set hourly rates, setup costs, and complexity multipliers
                      </CardDescription>
                    </div>
                    <Button onClick={addNewProcess} size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Add Process
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <Accordion type="multiple" className="w-full">
                    {processes.map((process) => (
                      <AccordionItem key={process.id} value={process.id}>
                        <AccordionTrigger className="hover:no-underline">
                          <div className="flex items-center gap-3 w-full pr-4">
                            <span className="font-semibold text-lg">{process.name}</span>
                            <Badge variant={process.is_active ? 'default' : 'secondary'}>
                              {process.is_active ? 'Active' : 'Inactive'}
                            </Badge>
                            <span className="ml-auto text-sm text-muted-foreground">
                              ${process.base_rate_per_hour}/hr
                            </span>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent>
                          <div className="space-y-4 pt-4">
                            <div className="flex items-center gap-2">
                              <Input
                                value={process.name}
                                onChange={(e) => updateProcess(process.id, 'name', e.target.value)}
                                className="font-semibold w-64"
                              />
                              <div className="flex items-center gap-2 ml-auto">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => updateProcess(process.id, 'is_active', !process.is_active)}
                                >
                                  {process.is_active ? 'Deactivate' : 'Activate'}
                                </Button>
                                <Button
                                  variant="destructive"
                                  size="sm"
                                  onClick={() => deleteProcess(process.id)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>

                            <div className="grid grid-cols-3 gap-4">
                              <div>
                                <Label htmlFor={`rate-${process.id}`}>Hourly Rate ($)</Label>
                                <Input
                                  id={`rate-${process.id}`}
                                  type="number"
                                  step="0.01"
                                  value={process.base_rate_per_hour}
                                  onChange={(e) =>
                                    updateProcess(process.id, 'base_rate_per_hour', parseFloat(e.target.value))
                                  }
                                />
                                <p className="text-xs text-muted-foreground mt-1">
                                  Cost per hour of machining time
                                </p>
                              </div>
                              <div>
                                <Label htmlFor={`setup-${process.id}`}>Setup Cost ($)</Label>
                                <Input
                                  id={`setup-${process.id}`}
                                  type="number"
                                  step="0.01"
                                  value={process.setup_cost}
                                  onChange={(e) =>
                                    updateProcess(process.id, 'setup_cost', parseFloat(e.target.value))
                                  }
                                />
                                <p className="text-xs text-muted-foreground mt-1">
                                  One-time setup cost per job
                                </p>
                              </div>
                              <div>
                                <Label htmlFor={`complexity-${process.id}`}>Complexity Multiplier</Label>
                                <Input
                                  id={`complexity-${process.id}`}
                                  type="number"
                                  step="0.1"
                                  value={process.complexity_multiplier}
                                  onChange={(e) =>
                                    updateProcess(process.id, 'complexity_multiplier', parseFloat(e.target.value))
                                  }
                                />
                                <p className="text-xs text-muted-foreground mt-1">
                                  Time adjustment factor (1.0 = normal)
                                </p>
                              </div>
                            </div>

                            {/* Machining Parameters Section */}
                            <Separator className="my-4" />
                            <div className="space-y-4">
                              <h4 className="font-semibold text-sm">Machining Parameters</h4>
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <Label htmlFor={`feed-rate-${process.id}`}>Feed Rate (mm/min)</Label>
                                  <Input
                                    id={`feed-rate-${process.id}`}
                                    type="number"
                                    step="10"
                                    value={process.feed_rate_mm_per_min ?? 500}
                                    onChange={(e) =>
                                      updateProcess(process.id, 'feed_rate_mm_per_min', parseFloat(e.target.value))
                                    }
                                  />
                                  <p className="text-xs text-muted-foreground mt-1">
                                    Cutting tool feed rate
                                  </p>
                                </div>
                                <div>
                                  <Label htmlFor={`spindle-speed-${process.id}`}>Spindle Speed (RPM)</Label>
                                  <Input
                                    id={`spindle-speed-${process.id}`}
                                    type="number"
                                    step="100"
                                    value={process.spindle_speed_rpm ?? 3000}
                                    onChange={(e) =>
                                      updateProcess(process.id, 'spindle_speed_rpm', parseFloat(e.target.value))
                                    }
                                  />
                                  <p className="text-xs text-muted-foreground mt-1">
                                    Spindle rotation speed
                                  </p>
                                </div>
                                <div>
                                  <Label htmlFor={`depth-of-cut-${process.id}`}>Depth of Cut (mm)</Label>
                                  <Input
                                    id={`depth-of-cut-${process.id}`}
                                    type="number"
                                    step="0.1"
                                    value={process.depth_of_cut_mm ?? 2.0}
                                    onChange={(e) =>
                                      updateProcess(process.id, 'depth_of_cut_mm', parseFloat(e.target.value))
                                    }
                                  />
                                  <p className="text-xs text-muted-foreground mt-1">
                                    Cut depth per pass
                                  </p>
                                </div>
                                <div>
                                  <Label htmlFor={`tool-change-${process.id}`}>Tool Change Time (min)</Label>
                                  <Input
                                    id={`tool-change-${process.id}`}
                                    type="number"
                                    step="0.5"
                                    value={process.tool_change_time_minutes ?? 2.0}
                                    onChange={(e) =>
                                      updateProcess(process.id, 'tool_change_time_minutes', parseFloat(e.target.value))
                                    }
                                  />
                                  <p className="text-xs text-muted-foreground mt-1">
                                    Time for tool changes
                                  </p>
                                </div>
                                <div className="col-span-2">
                                  <Label htmlFor={`rapid-feed-${process.id}`}>Rapid Feed Rate (mm/min)</Label>
                                  <Input
                                    id={`rapid-feed-${process.id}`}
                                    type="number"
                                    step="100"
                                    value={process.rapid_feed_rate_mm_per_min ?? 5000}
                                    onChange={(e) =>
                                      updateProcess(process.id, 'rapid_feed_rate_mm_per_min', parseFloat(e.target.value))
                                    }
                                  />
                                  <p className="text-xs text-muted-foreground mt-1">
                                    Non-cutting movement speed
                                  </p>
                                </div>
                              </div>
                            </div>

                            {/* Material-Specific Parameters Section */}
                            <Separator className="my-6" />
                            <div className="space-y-4">
                              <h4 className="font-semibold flex items-center gap-2">
                                Material-Specific Parameters
                                <Badge variant="secondary">
                                  {materialProcessParams.filter(p => p.process_id === process.id).length} / {materials.filter(m => m.is_active).length} configured
                                </Badge>
                              </h4>
                              
                              <div className="mb-6">
                                <Label>Select Material to Configure</Label>
                                <Select 
                                  value={selectedMaterials[process.id] || ""} 
                                  onValueChange={(value) => {
                                    setSelectedMaterials({...selectedMaterials, [process.id]: value});
                                    // Initialize local params when material is selected
                                    const key = `${process.id}-${value}`;
                                    if (!localParamsState[key]) {
                                      const existingParams = getMaterialProcessParam(process.id, value);
                                      setLocalParamsState({
                                        ...localParamsState,
                                        [key]: existingParams || {
                                          spindle_speed_rpm: null,
                                          feed_rate_mm_per_min: null,
                                          depth_of_cut_mm: null,
                                          cutting_speed_m_per_min: null,
                                          material_removal_rate_adjustment: 1.0,
                                          tool_wear_multiplier: 1.0,
                                          surface_finish_factor: 1.0,
                                          setup_time_multiplier: 1.0,
                                          cycle_time_multiplier: 1.0,
                                        }
                                      });
                                    }
                                  }}
                                >
                                  <SelectTrigger className="w-full">
                                    <SelectValue placeholder="Choose a material..." />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {materials.filter(m => m.is_active).map(material => {
                                      const hasCustomParams = materialProcessParams.some(
                                        p => p.material_id === material.id && p.process_id === process.id
                                      );
                                      
                                      return (
                                        <SelectItem key={material.id} value={material.id}>
                                          <div className="flex items-center gap-2">
                                            {material.material_name}
                                            {hasCustomParams && <Badge variant="default" className="ml-2 text-xs">Custom</Badge>}
                                          </div>
                                        </SelectItem>
                                      );
                                    })}
                                  </SelectContent>
                                </Select>
                              </div>

                              {selectedMaterials[process.id] && (() => {
                                const selectedMaterial = materials.find(m => m.id === selectedMaterials[process.id]);
                                const key = `${process.id}-${selectedMaterials[process.id]}`;
                                const localParams = localParamsState[key] || {
                                  spindle_speed_rpm: null,
                                  feed_rate_mm_per_min: null,
                                  depth_of_cut_mm: null,
                                  cutting_speed_m_per_min: null,
                                  material_removal_rate_adjustment: 1.0,
                                  tool_wear_multiplier: 1.0,
                                  surface_finish_factor: 1.0,
                                  setup_time_multiplier: 1.0,
                                  cycle_time_multiplier: 1.0,
                                };

                                const updateLocalParam = (field: keyof MaterialProcessParameter, value: number | null) => {
                                  setLocalParamsState({
                                    ...localParamsState,
                                    [key]: { ...localParams, [field]: value }
                                  });
                                };

                                return (
                                  <div className="bg-muted/50 p-4 rounded-lg space-y-4">
                                    <div className="flex items-center justify-between mb-4">
                                      <h5 className="font-medium">Parameters for {selectedMaterial?.material_name}</h5>
                                      {getMaterialProcessParam(process.id, selectedMaterials[process.id]) ? (
                                        <Badge variant="default">Using Custom Parameters</Badge>
                                      ) : (
                                        <Badge variant="secondary">Using Material Defaults</Badge>
                                      )}
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                      <div>
                                        <Label>Spindle Speed (RPM)</Label>
                                        <Input
                                          type="number"
                                          value={localParams.spindle_speed_rpm || ""}
                                          onChange={(e) => updateLocalParam('spindle_speed_rpm', Number(e.target.value) || null)}
                                          placeholder="Enter spindle speed"
                                        />
                                      </div>

                                      <div>
                                        <Label>Feed Rate (mm/min)</Label>
                                        <Input
                                          type="number"
                                          value={localParams.feed_rate_mm_per_min || ""}
                                          onChange={(e) => updateLocalParam('feed_rate_mm_per_min', Number(e.target.value) || null)}
                                          placeholder="Enter feed rate"
                                        />
                                      </div>

                                      <div>
                                        <Label>Depth of Cut (mm)</Label>
                                        <Input
                                          type="number"
                                          step="0.1"
                                          value={localParams.depth_of_cut_mm || ""}
                                          onChange={(e) => updateLocalParam('depth_of_cut_mm', Number(e.target.value) || null)}
                                          placeholder="Enter depth of cut"
                                        />
                                      </div>

                                      <div>
                                        <Label>Cutting Speed (m/min)</Label>
                                        <Input
                                          type="number"
                                          value={localParams.cutting_speed_m_per_min || ""}
                                          onChange={(e) => updateLocalParam('cutting_speed_m_per_min', Number(e.target.value) || null)}
                                          placeholder="Enter cutting speed"
                                        />
                                      </div>

                                      <div>
                                        <Label>MRR Adjustment</Label>
                                        <Input
                                          type="number"
                                          step="0.1"
                                          value={localParams.material_removal_rate_adjustment || 1.0}
                                          onChange={(e) => updateLocalParam('material_removal_rate_adjustment', Number(e.target.value))}
                                        />
                                      </div>

                                      <div>
                                        <Label>Tool Wear Multiplier</Label>
                                        <Input
                                          type="number"
                                          step="0.1"
                                          value={localParams.tool_wear_multiplier || 1.0}
                                          onChange={(e) => updateLocalParam('tool_wear_multiplier', Number(e.target.value))}
                                        />
                                      </div>

                                      <div>
                                        <Label>Surface Finish Factor</Label>
                                        <Input
                                          type="number"
                                          step="0.1"
                                          value={localParams.surface_finish_factor || 1.0}
                                          onChange={(e) => updateLocalParam('surface_finish_factor', Number(e.target.value))}
                                        />
                                      </div>

                                      <div>
                                        <Label>Setup Time Multiplier</Label>
                                        <Input
                                          type="number"
                                          step="0.1"
                                          value={localParams.setup_time_multiplier || 1.0}
                                          onChange={(e) => updateLocalParam('setup_time_multiplier', Number(e.target.value))}
                                        />
                                      </div>

                                      <div>
                                        <Label>Cycle Time Multiplier</Label>
                                        <Input
                                          type="number"
                                          step="0.1"
                                          value={localParams.cycle_time_multiplier || 1.0}
                                          onChange={(e) => updateLocalParam('cycle_time_multiplier', Number(e.target.value))}
                                        />
                                      </div>
                                    </div>

                                    <div className="flex gap-2 pt-4">
                                      <Button 
                                        onClick={() => saveMaterialProcessParams(process.id, selectedMaterials[process.id], localParams)}
                                        className="flex-1"
                                      >
                                        Save Parameters
                                      </Button>
                                      {getMaterialProcessParam(process.id, selectedMaterials[process.id]) && (
                                        <Button 
                                          variant="outline"
                                          onClick={() => resetMaterialProcessParams(process.id, selectedMaterials[process.id])}
                                        >
                                          Reset to Defaults
                                        </Button>
                                      )}
                                    </div>
                                  </div>
                                );
                              })()}
                            </div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>

                  <div className="flex justify-end pt-6">
                    <Button onClick={saveProcesses} disabled={saving}>
                      {saving ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Saving...
                        </>
                      ) : (
                        <>
                          <Save className="h-4 w-4 mr-2" />
                          Save All Processes
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Materials Tab */}
            <TabsContent value="materials" className="space-y-4">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Material Costs</CardTitle>
                      <CardDescription>
                        Set material pricing per volume and surface area
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <Button onClick={addNewCategory} variant="outline" size="sm">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Category
                      </Button>
                      <Button onClick={addNewMaterial} variant="outline" size="sm">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Material
                      </Button>
                      <Button onClick={() => document.getElementById('import-materials-excel')?.click()} size="sm">
                        <Upload className="h-4 w-4 mr-2" />
                        Import from Excel
                      </Button>
                      <input
                        id="import-materials-excel"
                        type="file"
                        accept=".xlsx,.xls"
                        className="hidden"
                        onChange={handleMaterialsExcelImport}
                      />
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <Tabs value={activeTab || categories[0]?.id || 'uncategorized'} onValueChange={setActiveTab} className="w-full">
                    <DndContext
                      sensors={sensors}
                      collisionDetection={closestCenter}
                      onDragEnd={handleDragEnd}
                    >
                      <SortableContext
                        items={categories.map(c => c.id)}
                        strategy={horizontalListSortingStrategy}
                      >
                        <TabsList className="w-full justify-start overflow-x-auto flex-wrap h-auto">
                          {categories.map((category) => {
                            const count = materials.filter(m => m.category_id === category.id).length;
                            return <SortableTabTrigger key={category.id} category={category} count={count} />;
                          })}
                          {materials.filter(m => !m.category_id).length > 0 && (
                            <TabsTrigger value="uncategorized" className="gap-2">
                              Uncategorized
                              <Badge variant="secondary" className="ml-1">
                                {materials.filter(m => !m.category_id).length}
                              </Badge>
                            </TabsTrigger>
                          )}
                        </TabsList>
                      </SortableContext>
                    </DndContext>

                    {categories.map((category) => {
                      const categoryMaterials = materials.filter(m => m.category_id === category.id);

                      return (
                        <TabsContent key={category.id} value={category.id} className="mt-6">
                          {categoryMaterials.length === 0 ? (
                            <div className="text-center py-12 border rounded-lg bg-muted/20">
                              <p className="text-muted-foreground mb-4">
                                No materials in this category yet
                              </p>
                              <Button onClick={addNewMaterial} size="sm">
                                <Plus className="h-4 w-4 mr-2" />
                                Add Material
                              </Button>
                            </div>
                          ) : (
                            <Accordion type="multiple" className="w-full">
                            {categoryMaterials.map((material) => (
                              <AccordionItem key={material.id} value={material.id}>
                                <AccordionTrigger className="hover:no-underline">
                                  <div className="flex items-center gap-3 w-full pr-4">
                                    <span className="font-semibold text-lg">{material.material_name}</span>
                                    <Badge variant={material.is_active ? 'default' : 'secondary'}>
                                      {material.is_active ? 'Active' : 'Inactive'}
                                    </Badge>
                                    <span className="ml-auto text-sm text-muted-foreground">
                                      {material.pricing_method === 'linear_inch' ? 'By Linear Inch' : 
                                       material.pricing_method === 'sheet' ? 'By Sheet' : 'By Weight'}
                                    </span>
                                  </div>
                                </AccordionTrigger>
                                <AccordionContent>
                                  <div className="space-y-4 pt-4">
                                    <div className="flex items-center gap-2">
                                      <Input
                                        value={material.material_name}
                                        onChange={(e) => updateMaterial(material.id, 'material_name', e.target.value)}
                                        className="font-semibold w-64"
                                      />
                                      <div className="flex items-center gap-2 ml-auto">
                                        <Button
                                          variant="outline"
                                          size="sm"
                                          onClick={() => updateMaterial(material.id, 'is_active', !material.is_active)}
                                        >
                                          {material.is_active ? 'Deactivate' : 'Activate'}
                                        </Button>
                                        <Button
                                          variant="destructive"
                                          size="sm"
                                          onClick={() => deleteMaterial(material.id)}
                                        >
                                          <Trash2 className="h-4 w-4" />
                                        </Button>
                                      </div>
                                    </div>

                                    <div className="mb-4">
                                      <Label htmlFor={`category-${material.id}`}>Category</Label>
                                      <Select
                                        value={material.category_id || 'uncategorized'}
                                        onValueChange={(value) => 
                                          updateMaterial(material.id, 'category_id', value === 'uncategorized' ? null : value)
                                        }
                                      >
                                        <SelectTrigger id={`category-${material.id}`}>
                                          <SelectValue placeholder="Select category" />
                                        </SelectTrigger>
                                        <SelectContent className="bg-background z-50">
                                          <SelectItem value="uncategorized">Uncategorized</SelectItem>
                                          {categories.map(cat => (
                                            <SelectItem key={cat.id} value={cat.id}>{cat.name}</SelectItem>
                                          ))}
                                        </SelectContent>
                                      </Select>
                                      <p className="text-xs text-muted-foreground mt-1">
                                        Group materials by type for better organization
                                      </p>
                                    </div>

                                    <div className="mb-4">
                                      <Label htmlFor={`pricing-method-${material.id}`}>Pricing Method</Label>
                                      <Select
                                        value={material.pricing_method || 'weight'}
                                        onValueChange={(value) => updateMaterial(material.id, 'pricing_method', value)}
                                      >
                                        <SelectTrigger id={`pricing-method-${material.id}`}>
                                          <SelectValue placeholder="Select pricing method" />
                                        </SelectTrigger>
                                        <SelectContent className="bg-background z-50">
                                          <SelectItem value="weight">By Weight (Volume)</SelectItem>
                                          <SelectItem value="linear_inch">By Linear Inch</SelectItem>
                                          <SelectItem value="sheet">By Sheet (Nesting)</SelectItem>
                                        </SelectContent>
                                      </Select>
                                      <p className="text-xs text-muted-foreground mt-1">
                                        Choose how to calculate material costs
                                      </p>
                                    </div>

                                    {material.pricing_method === 'weight' ? (
                                      <ResizablePanelGroup direction="horizontal" className="min-h-[120px] border rounded-lg">
                                        <ResizablePanel defaultSize={33} minSize={20}>
                                          <div className="p-4 h-full">
                                            <Label htmlFor={`volume-${material.id}`}>Cost per cm³ ($)</Label>
                                            <Input
                                              id={`volume-${material.id}`}
                                              type="number"
                                              step="0.01"
                                              value={material.cost_per_cubic_cm}
                                              onChange={(e) =>
                                                updateMaterial(material.id, 'cost_per_cubic_cm', parseFloat(e.target.value))
                                              }
                                            />
                                            <p className="text-xs text-muted-foreground mt-1">
                                              Material cost per cubic centimeter
                                            </p>
                                          </div>
                                        </ResizablePanel>
                                        <ResizableHandle withHandle />
                                        <ResizablePanel defaultSize={33} minSize={20}>
                                          <div className="p-4 h-full">
                                            <Label htmlFor={`surface-${material.id}`}>Cost per cm² ($)</Label>
                                            <Input
                                              id={`surface-${material.id}`}
                                              type="number"
                                              step="0.01"
                                              value={material.cost_per_square_cm}
                                              onChange={(e) =>
                                                updateMaterial(material.id, 'cost_per_square_cm', parseFloat(e.target.value))
                                              }
                                            />
                                            <p className="text-xs text-muted-foreground mt-1">
                                              Used for finish/coating costs
                                            </p>
                                          </div>
                                        </ResizablePanel>
                                        <ResizableHandle withHandle />
                                        <ResizablePanel defaultSize={34} minSize={20}>
                                          <div className="p-4 h-full">
                                            <Label htmlFor={`density-${material.id}`}>Density (g/cm³)</Label>
                                            <Input
                                              id={`density-${material.id}`}
                                              type="number"
                                              step="0.01"
                                              value={material.density || ''}
                                              onChange={(e) =>
                                                updateMaterial(material.id, 'density', parseFloat(e.target.value) || null)
                                              }
                                            />
                                            <p className="text-xs text-muted-foreground mt-1">
                                              Material density (optional)
                                            </p>
                                          </div>
                                        </ResizablePanel>
                                      </ResizablePanelGroup>
                                    ) : material.pricing_method === 'linear_inch' ? (
                                      <div className="border rounded-lg p-4 space-y-4">
                                        <div className="space-y-4 mb-4">
                                          <div>
                                            <Label>Price Per Pound ($)</Label>
                                            <Input
                                              type="number"
                                              step="0.01"
                                              value={material.price_per_lb || 1.0}
                                              onChange={(e) => {
                                                const newPrice = parseFloat(e.target.value);
                                                setMaterials(prev =>
                                                  prev.map(m => {
                                                    if (m.id === material.id) {
                                                      const updatedCrossSections = (m.cross_sections || []).map(cs => ({
                                                        ...cs,
                                                        cost_per_inch: cs.weight_per_foot 
                                                          ? (cs.weight_per_foot / 12) * newPrice 
                                                          : cs.cost_per_inch
                                                      }));
                                                      return { ...m, price_per_lb: newPrice, cross_sections: updatedCrossSections };
                                                    }
                                                    return m;
                                                  })
                                                );
                                              }}
                                              className="max-w-xs"
                                            />
                                            <p className="text-xs text-muted-foreground mt-1">
                                              Base price per pound - all cross-section prices will be calculated from this
                                            </p>
                                          </div>
                                        </div>

                                        <div className="flex items-center justify-between mb-2">
                                          <Label>Cross Sections</Label>
                                          <div className="flex gap-2 flex-wrap">
                                            <Button
                                              type="button"
                                              variant="outline"
                                              size="sm"
                                              onClick={() => importEMJData(material.id)}
                                            >
                                              <Download className="h-4 w-4 mr-2" />
                                              Import EMJ Standard Sizes
                                            </Button>
                                            <Button
                                              type="button"
                                              variant="outline"
                                              size="sm"
                                              onClick={() => addCrossSection(material.id)}
                                            >
                                              <Plus className="h-4 w-4 mr-2" />
                                              Add Custom Cross Section
                                            </Button>
                                            <label htmlFor={`upload-excel-${material.id}`}>
                                              <Button
                                                type="button"
                                                size="sm"
                                                variant="outline"
                                                onClick={() => document.getElementById(`upload-excel-${material.id}`)?.click()}
                                              >
                                                <Upload className="h-4 w-4 mr-2" />
                                                Upload Excel File
                                              </Button>
                                            </label>
                                            <input
                                              id={`upload-excel-${material.id}`}
                                              type="file"
                                              accept=".xlsx,.xls"
                                              className="hidden"
                                              onChange={(e) => handleExcelUpload(e, material.id, material)}
                                            />
                                            <label htmlFor={`upload-table-${material.id}`}>
                                              <Button
                                                type="button"
                                                size="sm"
                                                variant="outline"
                                                onClick={() => document.getElementById(`upload-table-${material.id}`)?.click()}
                                              >
                                                <Upload className="h-4 w-4 mr-2" />
                                                Upload Table Image
                                              </Button>
                                            </label>
                                            <input
                                              id={`upload-table-${material.id}`}
                                              type="file"
                                              accept="image/*"
                                              className="hidden"
                                              onChange={(e) => handleTableUpload(e, material.id, material)}
                                            />
                                          </div>
                                        </div>
                                        <p className="text-xs text-muted-foreground mb-3">
                                          Define standard cross-section sizes and their cost per linear inch
                                        </p>
                                        {(material.cross_sections || []).length === 0 ? (
                                          <p className="text-sm text-muted-foreground text-center py-4">
                                            No cross sections defined. Click "Add Cross Section" to create one.
                                          </p>
                                        ) : (
                                          <div className="space-y-4">
                                            {(() => {
                                              const rectangularSections = (material.cross_sections || []).filter(s => !s.shape || s.shape === 'rectangular');
                                              const circularSections = (material.cross_sections || []).filter(s => s.shape === 'circular');
                                              
                                              return (
                                                <div>
                                                  <div className="flex items-center justify-between mb-2">
                                                    <Label className="text-sm font-medium">Select Cross Section to Edit</Label>
                                                    <div className="flex gap-2 text-xs text-muted-foreground">
                                                      {rectangularSections.length > 0 && (
                                                        <span className="flex items-center gap-1">
                                                          <span className="w-3 h-3 bg-blue-500 rounded-sm"></span>
                                                          {rectangularSections.length} Rectangular
                                                        </span>
                                                      )}
                                                      {circularSections.length > 0 && (
                                                        <span className="flex items-center gap-1">
                                                          <span className="w-3 h-3 bg-green-500 rounded-full"></span>
                                                          {circularSections.length} Circular
                                                        </span>
                                                      )}
                                                    </div>
                                                  </div>
                                                  <Select
                                                    value={selectedCrossSections[material.id]?.toString() || '0'}
                                                    onValueChange={(value) => {
                                                      const newIdx = parseInt(value);
                                                      setSelectedCrossSections(prev => ({ ...prev, [material.id]: newIdx }));
                                                    }}
                                                  >
                                                    <SelectTrigger>
                                                      <SelectValue placeholder="Select a cross section" />
                                                    </SelectTrigger>
                                                    <SelectContent className="bg-background z-50">
                                                      {rectangularSections.length > 0 && (
                                                        <>
                                                          <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground bg-muted/50 flex items-center gap-2">
                                                            <span className="w-3 h-3 bg-blue-500 rounded-sm"></span>
                                                            Rectangular (Flat Bar)
                                                          </div>
                                                           {(material.cross_sections || [])
                                                            .map((section, originalIdx) => ({ section, originalIdx }))
                                                            .filter(({ section }) => !section.shape || section.shape === 'rectangular')
                                                            .map(({ section, originalIdx }) => {
                                                            const thickness = section.thickness || 0;
                                                            const width = section.width || 0;
                                                            const costPerInch = section.cost_per_inch || 0;
                                                            const thicknessDisplay = thickness > 0 ? decimalToFraction(thickness) : 'N/A';
                                                            const widthDisplay = width > 0 ? decimalToFraction(width) : 'N/A';
                                                            const costDisplay = costPerInch > 0 ? `$${costPerInch.toFixed(4)}/inch` : 'N/A';
                                                            return (
                                                              <div key={`rect-${originalIdx}`} className="flex items-center group hover:bg-accent">
                                                                <SelectItem value={originalIdx.toString()} className="flex-1 cursor-pointer">
                                                                  {`${thicknessDisplay}" × ${widthDisplay}" - ${costDisplay}`}
                                                                </SelectItem>
                                                                <Button
                                                                  type="button"
                                                                  variant="ghost"
                                                                  size="sm"
                                                                  className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity mr-2"
                                                                  onClick={(e) => {
                                                                    e.preventDefault();
                                                                    e.stopPropagation();
                                                                    removeCrossSection(material.id, originalIdx);
                                                                    setSelectedCrossSections(prev => {
                                                                      const newVal = { ...prev };
                                                                      if (newVal[material.id] >= (material.cross_sections?.length || 1) - 1) {
                                                                        newVal[material.id] = Math.max(0, (material.cross_sections?.length || 1) - 2);
                                                                      }
                                                                      return newVal;
                                                                    });
                                                                  }}
                                                                >
                                                                  <Trash2 className="h-3.5 w-3.5 text-destructive" />
                                                                </Button>
                                                              </div>
                                                            );
                                                          })}
                                                        </>
                                                      )}
                                                      {circularSections.length > 0 && (
                                                        <>
                                                          {rectangularSections.length > 0 && (
                                                            <div className="my-1 border-t"></div>
                                                          )}
                                                          <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground bg-muted/50 flex items-center gap-2">
                                                            <span className="w-3 h-3 bg-green-500 rounded-full"></span>
                                                            Circular (Round Bar)
                                                          </div>
                                                          {(material.cross_sections || [])
                                                            .map((section, originalIdx) => ({ section, originalIdx }))
                                                            .filter(({ section }) => section.shape === 'circular')
                                                            .map(({ section, originalIdx }) => {
                                                            const diameter = section.width || 0;
                                                            const costPerInch = section.cost_per_inch || 0;
                                                            const diameterDisplay = diameter > 0 ? decimalToFraction(diameter) : 'N/A';
                                                            const costDisplay = costPerInch > 0 ? `$${costPerInch.toFixed(4)}/inch` : 'N/A';
                                                            return (
                                                              <div key={`circ-${originalIdx}`} className="flex items-center group hover:bg-accent">
                                                                <SelectItem value={originalIdx.toString()} className="flex-1 cursor-pointer">
                                                                  {`Ø ${diameterDisplay}" - ${costDisplay}`}
                                                                </SelectItem>
                                                                <Button
                                                                  type="button"
                                                                  variant="ghost"
                                                                  size="sm"
                                                                  className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity mr-2"
                                                                  onClick={(e) => {
                                                                    e.preventDefault();
                                                                    e.stopPropagation();
                                                                    removeCrossSection(material.id, originalIdx);
                                                                    setSelectedCrossSections(prev => {
                                                                      const newVal = { ...prev };
                                                                      if (newVal[material.id] >= (material.cross_sections?.length || 1) - 1) {
                                                                        newVal[material.id] = Math.max(0, (material.cross_sections?.length || 1) - 2);
                                                                      }
                                                                      return newVal;
                                                                    });
                                                                  }}
                                                                >
                                                                  <Trash2 className="h-3.5 w-3.5 text-destructive" />
                                                                </Button>
                                                              </div>
                                                            );
                                                          })}
                                                        </>
                                                      )}
                                                    </SelectContent>
                                                  </Select>
                                                </div>
                                              );
                                            })()}

                                            {(() => {
                                              const selectedIdx = selectedCrossSections[material.id] ?? 0;
                                              const section = (material.cross_sections || [])[selectedIdx];
                                              if (!section) return null;

                                              return (
                                                <div className="border rounded-md p-4 space-y-4">
                                                  <div className="flex items-center justify-between">
                                                    <Label className="text-sm font-semibold">
                                                      Cross Section {selectedIdx + 1}
                                                    </Label>
                                                    <Button
                                                      type="button"
                                                      variant="destructive"
                                                      size="sm"
                                                      onClick={() => {
                                                        removeCrossSection(material.id, selectedIdx);
                                                        setSelectedCrossSections(prev => {
                                                          const newVal = { ...prev };
                                                          if (newVal[material.id] >= (material.cross_sections?.length || 1) - 1) {
                                                            newVal[material.id] = Math.max(0, (material.cross_sections?.length || 1) - 2);
                                                          }
                                                          return newVal;
                                                        });
                                                      }}
                                                    >
                                                      <Trash2 className="h-4 w-4 mr-2" />
                                                      Delete
                                                    </Button>
                                                  </div>

                                                  <div className="mb-4 space-y-2">
                                                    <Label className="text-sm font-medium">Shape</Label>
                                                    <select
                                                      value={section.shape || 'rectangular'}
                                                      onChange={(e) => updateCrossSection(material.id, selectedIdx, 'shape', e.target.value as 'rectangular' | 'circular')}
                                                      className="w-full px-3 py-2 border rounded-md bg-background"
                                                    >
                                                      <option value="rectangular">Rectangular (Flat Bar)</option>
                                                      <option value="circular">Circular (Round Bar)</option>
                                                    </select>
                                                  </div>

                                                  <div className="grid grid-cols-3 gap-4">
                                                    {section.shape === 'circular' ? (
                                                      <div className="space-y-2">
                                                        <Label className="text-sm font-medium">Diameter (Ø)</Label>
                                                        <div className="text-lg font-semibold">Ø {section.width > 0 ? decimalToFraction(section.width) : 'N/A'}"</div>
                                                      </div>
                                                    ) : (
                                                      <>
                                                        <div className="space-y-2">
                                                          <Label className="text-sm font-medium">Thickness</Label>
                                                          <div className="text-lg font-semibold">{section.thickness > 0 ? decimalToFraction(section.thickness) : 'N/A'}"</div>
                                                        </div>

                                                        <div className="space-y-2">
                                                          <Label className="text-sm font-medium">Width</Label>
                                                          <div className="text-lg font-semibold">{section.width > 0 ? decimalToFraction(section.width) : 'N/A'}"</div>
                                                        </div>
                                                      </>
                                                    )}

                                                    <div className="space-y-2">
                                                      <Label className="text-sm font-medium">Weight/Foot (lbs)</Label>
                                                      <Input
                                                        type="number"
                                                        step="0.0001"
                                                        value={section.weight_per_foot || ''}
                                                        placeholder="0"
                                                        onChange={(e) => {
                                                          const weight = parseFloat(e.target.value) || 0;
                                                          updateCrossSection(material.id, selectedIdx, 'weight_per_foot', weight);
                                                          const pricePerLb = material.price_per_lb || 1.0;
                                                          updateCrossSection(material.id, selectedIdx, 'cost_per_inch', (weight / 12) * pricePerLb);
                                                        }}
                                                      />
                                                    </div>

                                                    <div className="space-y-2">
                                                      <Label className="text-sm font-medium">Weight/Bar (lbs)</Label>
                                                      <Input
                                                        type="number"
                                                        step="0.0001"
                                                        value={section.weight_per_bar || ''}
                                                        placeholder="0"
                                                        onChange={(e) =>
                                                          updateCrossSection(material.id, selectedIdx, 'weight_per_bar', parseFloat(e.target.value) || 0)
                                                        }
                                                      />
                                                      <p className="text-xs text-muted-foreground">12-ft bar weight</p>
                                                    </div>

                                                    <div className="col-span-2 space-y-2">
                                                      <Label className="text-sm font-medium">Cost per Inch ($)</Label>
                                                      <Input
                                                        type="text"
                                                        value={section.cost_per_inch > 0 ? `$${section.cost_per_inch.toFixed(4)}` : 'N/A'}
                                                        disabled
                                                        className="bg-muted"
                                                        title={section.weight_per_foot > 0 ? `Calculated: (${section.weight_per_foot} lbs/ft ÷ 12) × $${material.price_per_lb || 1.0}/lb` : 'Enter weight per foot to calculate'}
                                                      />
                                                      <p className="text-xs text-muted-foreground">Auto-calculated from weight and price per pound</p>
                                                    </div>
                                                  </div>
                                                </div>
                                              );
                                            })()}
                                          </div>
                                        )}
                                      </div>
                                    ) : material.pricing_method === 'sheet' ? (
                                      <div className="border rounded-lg p-4 space-y-4">
                                        <div>
                                          <Label>Nesting Efficiency (%)</Label>
                                          <p className="text-xs text-muted-foreground mb-2">
                                            Typical: 70-85%. Higher = better material utilization.
                                          </p>
                                          <Input
                                            type="number"
                                            min="0"
                                            max="100"
                                            step="1"
                                            value={((material.default_nesting_efficiency || 0.75) * 100).toFixed(0)}
                                            onChange={(e) =>
                                              updateMaterial(material.id, 'default_nesting_efficiency', parseFloat(e.target.value) / 100)
                                            }
                                          />
                                        </div>

                                        <div className="flex items-center justify-between mb-2">
                                          <Label>Sheet Configurations</Label>
                                          <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={() => addSheetConfiguration(material.id)}
                                          >
                                            <Plus className="h-4 w-4 mr-2" />
                                            Add Sheet Size
                                          </Button>
                                        </div>
                                        <p className="text-xs text-muted-foreground mb-3">
                                          Define available sheet sizes and cost per sheet
                                        </p>
                                        {(material.sheet_configurations || []).length === 0 ? (
                                          <p className="text-sm text-muted-foreground text-center py-4">
                                            No sheet sizes defined. Click "Add Sheet Size" to create one.
                                          </p>
                                        ) : (
                                          <div className="space-y-3">
                                            {(material.sheet_configurations || []).map((sheet, idx) => (
                                              <div key={idx} className="border rounded-md p-4 space-y-3">
                                                <div className="flex items-center justify-between mb-2">
                                                  <Label className="text-sm font-medium">Sheet {idx + 1}</Label>
                                                  <Button
                                                    type="button"
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => removeSheetConfiguration(material.id, idx)}
                                                    className="h-8 w-8 p-0"
                                                  >
                                                    <Trash2 className="h-4 w-4 text-destructive" />
                                                  </Button>
                                                </div>
                                                <div className="grid grid-cols-2 gap-3">
                                                  <div className="space-y-2">
                                                    <Label className="text-xs">Width</Label>
                                                    <Input
                                                      type="number"
                                                      step="0.125"
                                                      value={sheet.width}
                                                      onChange={(e) =>
                                                        updateSheetConfiguration(material.id, idx, 'width', parseFloat(e.target.value))
                                                      }
                                                    />
                                                  </div>
                                                  <div className="space-y-2">
                                                    <Label className="text-xs">Height</Label>
                                                    <Input
                                                      type="number"
                                                      step="0.125"
                                                      value={sheet.height}
                                                      onChange={(e) =>
                                                        updateSheetConfiguration(material.id, idx, 'height', parseFloat(e.target.value))
                                                      }
                                                    />
                                                  </div>
                                                  <div className="space-y-2">
                                                    <Label className="text-xs">Thickness</Label>
                                                    <Input
                                                      type="number"
                                                      step="0.001"
                                                      value={sheet.thickness}
                                                      onChange={(e) =>
                                                        updateSheetConfiguration(material.id, idx, 'thickness', parseFloat(e.target.value))
                                                      }
                                                    />
                                                  </div>
                                                  <div className="space-y-2">
                                                    <Label className="text-xs">Cost/Sheet ($)</Label>
                                                    <Input
                                                      type="number"
                                                      step="0.01"
                                                      value={sheet.cost_per_sheet}
                                                      onChange={(e) =>
                                                        updateSheetConfiguration(material.id, idx, 'cost_per_sheet', parseFloat(e.target.value))
                                                      }
                                                    />
                                                  </div>
                                                  <div className="col-span-2 space-y-2">
                                                    <Label className="text-xs">Unit</Label>
                                                    <Select
                                                      value={sheet.unit}
                                                      onValueChange={(value) =>
                                                        updateSheetConfiguration(material.id, idx, 'unit', value)
                                                      }
                                                    >
                                                      <SelectTrigger>
                                                        <SelectValue />
                                                      </SelectTrigger>
                                                      <SelectContent className="bg-background z-50">
                                                        <SelectItem value="inch">Inches</SelectItem>
                                                        <SelectItem value="cm">Centimeters</SelectItem>
                                                      </SelectContent>
                                                    </Select>
                                                  </div>
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    ) : null}

                                    <div>
                                      <Label>Available Finishes</Label>
                                      <p className="text-sm text-muted-foreground mb-2">
                                        Comma-separated list of finish options
                                      </p>
                                      <Input
                                        value={Array.isArray(material.finish_options) ? material.finish_options.join(', ') : ''}
                                        onChange={(e) =>
                                          updateMaterial(
                                            material.id,
                                            'finish_options',
                                            e.target.value.split(',').map(f => f.trim())
                                          )
                                        }
                                        placeholder="As-machined, Anodized, Powder Coated"
                                      />
                                    </div>
                                  </div>
                                </AccordionContent>
                              </AccordionItem>
                            ))}
                            </Accordion>
                          )}
                        </TabsContent>
                      );
                    })}

                    {/* Uncategorized materials tab */}
                    {materials.filter(m => !m.category_id).length > 0 && (
                      <TabsContent value="uncategorized" className="mt-6">
                        <Accordion type="multiple" className="w-full">
                          {materials.filter(m => !m.category_id).map((material) => (
                            <AccordionItem key={material.id} value={material.id}>
                              <AccordionTrigger className="hover:no-underline">
                                <div className="flex items-center gap-3 w-full pr-4">
                                  <span className="font-semibold text-lg">{material.material_name}</span>
                                  <Badge variant={material.is_active ? 'default' : 'secondary'}>
                                    {material.is_active ? 'Active' : 'Inactive'}
                                  </Badge>
                                  <span className="ml-auto text-sm text-muted-foreground">
                                    {material.pricing_method === 'linear_inch' ? 'By Linear Inch' : 'By Weight'}
                                  </span>
                                </div>
                              </AccordionTrigger>
                              <AccordionContent>
                                {/* Same content as above - keeping it DRY would require extracting to component */}
                                <div className="space-y-4 pt-4">
                                  <div className="flex items-center gap-2">
                                    <Input
                                      value={material.material_name}
                                      onChange={(e) => updateMaterial(material.id, 'material_name', e.target.value)}
                                      className="font-semibold w-64"
                                    />
                                    <div className="flex items-center gap-2 ml-auto">
                                      <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => updateMaterial(material.id, 'is_active', !material.is_active)}
                                      >
                                        {material.is_active ? 'Deactivate' : 'Activate'}
                                      </Button>
                                      <Button
                                        variant="destructive"
                                        size="sm"
                                        onClick={() => deleteMaterial(material.id)}
                                      >
                                        <Trash2 className="h-4 w-4" />
                                      </Button>
                                    </div>
                                  </div>

                                  <div className="mb-4">
                                    <Label htmlFor={`category-${material.id}`}>Category</Label>
                                    <Select
                                      value={material.category_id || 'uncategorized'}
                                      onValueChange={(value) => 
                                        updateMaterial(material.id, 'category_id', value === 'uncategorized' ? null : value)
                                      }
                                    >
                                      <SelectTrigger id={`category-${material.id}`}>
                                        <SelectValue placeholder="Select category" />
                                      </SelectTrigger>
                                      <SelectContent className="bg-background z-50">
                                        <SelectItem value="uncategorized">Uncategorized</SelectItem>
                                        {categories.map(cat => (
                                          <SelectItem key={cat.id} value={cat.id}>{cat.name}</SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                    <p className="text-xs text-muted-foreground mt-1">
                                      Assign this material to a category
                                    </p>
                                  </div>

                                  <div className="text-sm text-muted-foreground">
                                    Configure pricing and other settings after assigning to a category
                                  </div>
                                </div>
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </TabsContent>
                    )}
                  </Tabs>

                  <div className="flex justify-end pt-6">
                    <Button onClick={saveMaterials} disabled={saving}>
                      {saving ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Saving...
                        </>
                      ) : (
                        <>
                          <Save className="h-4 w-4 mr-2" />
                          Save All Materials
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Surface Treatments Tab */}
            <TabsContent value="treatments" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>{editingTreatmentId ? "Edit" : "Add"} Surface Treatment</CardTitle>
                  <CardDescription>
                    Manage heat treatments, post-processes, and surface treatments (priced by surface area)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleTreatmentSubmit} className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="treatment-name">Treatment Name</Label>
                        <Input
                          id="treatment-name"
                          value={treatmentFormData.name}
                          onChange={(e) => setTreatmentFormData({ ...treatmentFormData, name: e.target.value })}
                          placeholder="e.g., Anodizing Type II"
                          required
                        />
                      </div>

                      <div>
                        <Label htmlFor="treatment-category">Category</Label>
                        <Select
                          value={treatmentFormData.category}
                          onValueChange={(value: any) => setTreatmentFormData({ ...treatmentFormData, category: value })}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="heat_treatment">Heat Treatment</SelectItem>
                            <SelectItem value="post_process">Post Process</SelectItem>
                            <SelectItem value="surface_treatment">Surface Treatment</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label htmlFor="treatment-cost">Cost per cm² ($)</Label>
                        <Input
                          id="treatment-cost"
                          type="number"
                          step="0.001"
                          min="0"
                          value={treatmentFormData.cost_per_cm2}
                          onChange={(e) => setTreatmentFormData({ ...treatmentFormData, cost_per_cm2: parseFloat(e.target.value) || 0 })}
                          required
                        />
                      </div>

                      <div className="flex items-center space-x-2">
                        <Switch
                          id="treatment-active"
                          checked={treatmentFormData.is_active}
                          onCheckedChange={(checked) => setTreatmentFormData({ ...treatmentFormData, is_active: checked })}
                        />
                        <Label htmlFor="treatment-active">Active</Label>
                      </div>
                    </div>

                    <div>
                      <Label htmlFor="treatment-description">Description</Label>
                      <Textarea
                        id="treatment-description"
                        value={treatmentFormData.description}
                        onChange={(e) => setTreatmentFormData({ ...treatmentFormData, description: e.target.value })}
                        placeholder="Brief description of the treatment..."
                      />
                    </div>

                    <div className="flex gap-2">
                      <Button type="submit">
                        {editingTreatmentId ? "Update" : "Add"} Treatment
                      </Button>
                      {editingTreatmentId && (
                        <Button type="button" variant="outline" onClick={resetTreatmentForm}>
                          Cancel
                        </Button>
                      )}
                    </div>
                  </form>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Surface Treatments</CardTitle>
                  <CardDescription>
                    All surface treatments grouped by category
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {['heat_treatment', 'post_process', 'surface_treatment'].map((category) => {
                      const categoryTreatments = treatments.filter(t => t.category === category);
                      if (categoryTreatments.length === 0) return null;

                      return (
                        <div key={category}>
                          <h3 className="font-semibold text-lg mb-2">{getCategoryLabel(category)}</h3>
                          <div className="space-y-2">
                            {categoryTreatments.map((treatment) => (
                              <div
                                key={treatment.id}
                                className="flex items-center justify-between p-4 border rounded-lg"
                              >
                                <div className="flex-1">
                                  <div className="flex items-center gap-2">
                                    <h4 className="font-medium">{treatment.name}</h4>
                                    {!treatment.is_active && (
                                      <Badge variant="secondary">Inactive</Badge>
                                    )}
                                  </div>
                                  <p className="text-sm text-muted-foreground">{treatment.description}</p>
                                  <p className="text-sm font-semibold mt-1">
                                    ${treatment.cost_per_cm2.toFixed(3)} per cm²
                                  </p>
                                </div>
                                <div className="flex gap-2">
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => handleEditTreatment(treatment)}
                                  >
                                    <Edit className="h-4 w-4" />
                                  </Button>
                                  <Button
                                    variant="destructive"
                                    size="sm"
                                    onClick={() => handleDeleteTreatment(treatment.id)}
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Formulas Tab */}
            <TabsContent value="formulas" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Pricing Formulas</CardTitle>
                  <CardDescription>
                    View and understand the pricing calculation formulas
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold text-lg mb-2">Unit Price Calculation</h3>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                        Unit Price = Material Cost + Machining Cost + Setup Cost + Finish Cost
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h3 className="font-semibold mb-2">Material Cost</h3>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                        Material Cost = Volume (cm³) × Material Rate ($/cm³)
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h3 className="font-semibold mb-2">Machining Cost</h3>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm space-y-2">
                        <div>Complexity Multiplier = 1 + (Complexity Score - 5) / 10</div>
                        <div>Estimated Hours = (Surface Area / 100) × Complexity Multiplier × Process Multiplier</div>
                        <div>Machining Cost = Estimated Hours × Hourly Rate ($/hr)</div>
                      </div>
                      <p className="text-sm text-muted-foreground mt-2">
                        Complexity Score ranges from 1-10 (5 = average). Higher scores increase time estimates.
                      </p>
                    </div>

                    <Separator />

                    <div>
                      <h3 className="font-semibold mb-2">Setup Cost</h3>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                        Setup Cost per Unit = Process Setup Cost / Quantity
                      </div>
                      <p className="text-sm text-muted-foreground mt-2">
                        Setup costs are amortized across the quantity ordered.
                      </p>
                    </div>

                    <Separator />

                    <div>
                      <h3 className="font-semibold mb-2">Finish Cost</h3>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                        Finish Cost = Surface Area (cm²) × 0.05 (if finish selected)
                      </div>
                      <p className="text-sm text-muted-foreground mt-2">
                        Applies only when a finish other than "As-machined" is selected.
                      </p>
                    </div>

                    <Separator />

                    <div>
                      <h3 className="font-semibold mb-2">Quantity Discounts</h3>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm space-y-1">
                        <div>1,000+ units: 20% discount</div>
                        <div>100-999 units: 15% discount</div>
                        <div>50-99 units: 10% discount</div>
                        <div>10-49 units: 5% discount</div>
                        <div>1-9 units: No discount</div>
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h3 className="font-semibold mb-2">Lead Time Estimation</h3>
                      <div className="bg-muted p-4 rounded-lg font-mono text-sm space-y-1">
                        <div>Total Hours = Estimated Hours × Quantity</div>
                        <div>Lead Time (days) = max(5, ceil(Total Hours / 8) + 2)</div>
                      </div>
                      <p className="text-sm text-muted-foreground mt-2">
                        Minimum 5 business days. Assumes 8 production hours per day plus 2-day buffer.
                      </p>
                    </div>

                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-6">
                      <h4 className="font-semibold text-blue-900 mb-2">Note: Editing Formulas</h4>
                      <p className="text-sm text-blue-800">
                        To modify the pricing formulas (e.g., change the surface area divisor from 100, adjust discount tiers, 
                        or modify the finish cost multiplier), you'll need to edit the edge function at:{' '}
                        <code className="bg-blue-100 px-2 py-1 rounded">
                          supabase/functions/calculate-preliminary-quote/index.ts
                        </code>
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
      <Footer />
    </div>
  );
};

// Sortable Tab Trigger Component
const SortableTabTrigger = ({ category, count }: { category: any; count: number }) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
  } = useSortable({ id: category.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div ref={setNodeRef} style={style} {...attributes} {...listeners} className="cursor-grab active:cursor-grabbing">
      <TabsTrigger value={category.id} className="gap-2">
        {category.name}
        <Badge variant="secondary" className="ml-1">{count}</Badge>
      </TabsTrigger>
    </div>
  );
};

export default PricingSettings;
