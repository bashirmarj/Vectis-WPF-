import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CADViewer } from "./CADViewer";
import FeatureTree from "./FeatureTree";
import { Package, Layers, DollarSign, Loader2, CheckCircle2, AlertCircle, Zap, X } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { cn } from "@/lib/utils";
import { ChevronsUpDown, Check } from "lucide-react";
import * as React from "react";

interface PartDetailCustomerProps {
  file: {
    file: File;
    quantity: number;
    material?: string;
    process?: string;
    meshData?: {
      vertices: number[];
      indices: number[];
      normals: number[];
      vertex_colors?: string[];
      triangle_count: number;
      face_types?: string[];
      feature_edges?: number[][][];
    };
    analysis?: {
      volume_cm3: number;
      surface_area_cm2: number;
      complexity_score: number;
      confidence?: number;
      method?: string;
      recommended_processes?: string[];
      detected_features?: any;
      manufacturing_features?: any;
      feature_tree?: {
        common_dimensions: Array<{
          label: string;
          value: number;
          unit: string;
        }>;
        oriented_sections: Array<{
          orientation: string;
          features: any[];
        }>;
      };
      feature_summary?: any;
      geometric_features?: any;
      recommended_routings?: string[];
      routing_reasoning?: string[];
      machining_summary?: Array<{
        routing: string;
        machining_time_min: number;
        machining_cost: number;
      }>;
      estimated_total_cost_usd?: number;
    };
    quote?: {
      unit_price: number;
      total_price: number;
      breakdown: {
        material_cost: number;
        machining_cost: number;
        setup_cost: number;
        finish_cost: number;
      };
      lead_time_days: number;
    };
    isAnalyzing?: boolean;
  };
  materials: string[];
  onUpdateMaterial: (material: string) => void;
  onAnalyze: () => void;
  onRemove: () => void;
}

export function PartDetailCustomer({
  file,
  materials,
  onUpdateMaterial,
  onAnalyze,
  onRemove,
}: PartDetailCustomerProps) {
  const hasAnalysis = !!file.analysis;
  const hasFeatures =
    hasAnalysis &&
    (file.analysis.manufacturing_features || file.analysis.feature_tree || file.analysis.detected_features);
  const hasQuote = !!file.quote;

  // Debug logging
  React.useEffect(() => {
    console.log("📊 Passing to CADViewer:", {
      hasGeometricFeatures: !!file.analysis?.geometric_features,
      hasFeatureInstances: !!file.analysis?.geometric_features?.feature_instances,
      instanceCount: file.analysis?.geometric_features?.feature_instances?.length || 0,
      structure: file.analysis?.geometric_features,
    });
  }, [file.analysis?.geometric_features]);

  console.log("🔍 Checking for features:", {
    hasAnalysis,
    manufacturing_features: file.analysis?.manufacturing_features,
    feature_summary: file.analysis?.feature_summary,
    hasFeatures,
  });

  // Calculate feature count from either old or new structure
  const featureCount = hasFeatures
    ? file.analysis.feature_tree?.oriented_sections?.reduce((sum, s) => sum + s.features.length, 0) ||
      (file.analysis.feature_summary
        ? (file.analysis.feature_summary.through_holes || 0) +
          (file.analysis.feature_summary.blind_holes || 0) +
          (file.analysis.feature_summary.bores || 0) +
          (file.analysis.feature_summary.bosses || 0)
        : 0)
    : 0;

  return (
    <Card className="w-full">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
              <Package className="h-6 w-6 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base">{file.file.name}</CardTitle>
              <p className="text-sm text-muted-foreground">
                {(file.file.size / 1024 / 1024).toFixed(2)} MB • Qty: {file.quantity}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onRemove}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        {!hasAnalysis ? (
          <div className="space-y-4">
            <MaterialSelector value={file.material} materials={materials} onSelect={onUpdateMaterial} />
            <Button onClick={onAnalyze} disabled={file.isAnalyzing || !file.material} className="w-full">
              {file.isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4" />
                  Analyze CAD File
                </>
              )}
            </Button>
            {!file.material && (
              <p className="text-xs text-muted-foreground text-center">Select a material to enable analysis</p>
            )}
          </div>
        ) : (
          <Tabs defaultValue="model" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="model">
                <Package className="h-4 w-4 mr-2" />
                3D Model & Features
                {featureCount > 0 && (
                  <Badge variant="secondary" className="ml-2">
                    {featureCount}
                  </Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="quote">
                <DollarSign className="h-4 w-4 mr-2" />
                Quote
              </TabsTrigger>
            </TabsList>

            <TabsContent value="model" className="mt-4">
              <div className="space-y-4">
                <MaterialSelector value={file.material} materials={materials} onSelect={onUpdateMaterial} compact />
                
                <div className="h-[600px]">
                  <CADViewer 
                    fileName={file.file.name} 
                    meshData={file.meshData}
                    geometricFeatures={file.analysis?.geometric_features}
                  />
                </div>

                {hasAnalysis && (
                  <>
                    <div className="grid grid-cols-3 gap-3">
                      <StatCard label="Volume" value={`${file.analysis.volume_cm3.toFixed(1)} cm³`} />
                      <StatCard label="Surface Area" value={`${file.analysis.surface_area_cm2.toFixed(1)} cm²`} />
                      <StatCard label="Complexity" value={file.analysis.complexity_score.toString()} />
                    </div>

                    {file.analysis.routing_reasoning && file.analysis.routing_reasoning.length > 0 && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-sm flex items-center gap-2">
                            <Zap className="h-4 w-4 text-purple-600" />
                            Industrial Routing Logic
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ul className="space-y-2">
                            {file.analysis.routing_reasoning.map((reason, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-sm">
                                <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                                <span>{reason}</span>
                              </li>
                            ))}
                          </ul>
                        </CardContent>
                      </Card>
                    )}
                  </>
                )}
              </div>
            </TabsContent>

            <TabsContent value="quote" className="mt-4">
              <div className="space-y-4">
                <MaterialSelector value={file.material} materials={materials} onSelect={onUpdateMaterial} compact />

                {hasQuote ? (
                  <div className="space-y-4">
                    <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-950/20 dark:to-blue-950/20 border-2 border-purple-200 dark:border-purple-800 rounded-lg p-6">
                      <div className="flex items-center justify-between mb-4">
                        <Badge className="bg-purple-600 text-white">
                          <Zap className="h-3 w-3 mr-1" />
                          AI Estimated Price
                        </Badge>
                        {file.analysis?.confidence && (
                          <Badge variant="secondary">
                            {file.analysis.confidence >= 0.85 ? (
                              <>
                                <CheckCircle2 className="h-3 w-3 mr-1" />
                                High Confidence
                              </>
                            ) : (
                              "Estimated"
                            )}
                          </Badge>
                        )}
                      </div>
                      <div className="text-center">
                        <p className="text-sm text-muted-foreground mb-1">Unit Price</p>
                        <p className="text-4xl font-bold text-purple-600 dark:text-purple-400">
                          ${file.quote.unit_price.toFixed(2)}
                        </p>
                        <p className="text-lg text-muted-foreground mt-2">
                          Total: ${file.quote.total_price.toFixed(2)}
                        </p>
                      </div>
                    </div>

                    {file.analysis?.machining_summary && file.analysis.machining_summary.length > 0 && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-sm flex items-center gap-2">
                            <Layers className="h-4 w-4 text-blue-600" />
                            Machining Operations
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {file.analysis.machining_summary.map((op, idx) => (
                              <div key={idx} className="flex justify-between items-center p-3 bg-muted/30 rounded-md">
                                <div>
                                  <div className="font-medium text-sm">{op.routing}</div>
                                  <div className="text-xs text-muted-foreground">
                                    {op.machining_time_min.toFixed(1)} minutes
                                  </div>
                                </div>
                                <div className="text-sm font-semibold">${op.machining_cost.toFixed(2)}</div>
                              </div>
                            ))}
                            {file.analysis.estimated_total_cost_usd && (
                              <div className="pt-2 border-t flex justify-between items-center">
                                <span className="text-sm font-semibold">Total Machining Cost</span>
                                <span className="text-base font-bold text-purple-600">
                                  ${file.analysis.estimated_total_cost_usd.toFixed(2)}
                                </span>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {file.analysis?.recommended_processes && file.analysis.recommended_processes.length > 0 && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-sm">Recommended Manufacturing Processes</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="flex flex-wrap gap-2">
                            {file.analysis.recommended_processes.map((process, idx) => (
                              <Badge key={idx} variant="secondary">
                                {process}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Cost Breakdown</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <CostRow label="Material" value={file.quote.breakdown.material_cost} />
                        <CostRow label="Machining" value={file.quote.breakdown.machining_cost} />
                        <CostRow label="Setup" value={file.quote.breakdown.setup_cost} />
                        <CostRow label="Finish" value={file.quote.breakdown.finish_cost} />
                        <div className="pt-2 border-t">
                          <div className="flex justify-between text-sm font-semibold">
                            <span>Lead Time</span>
                            <span>{file.quote.lead_time_days} days</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
                      <p className="text-xs text-muted-foreground">
                        💡 This is a preliminary estimate. Final pricing will be confirmed after engineering review.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <DollarSign className="h-12 w-12 text-muted-foreground mb-3" />
                    <p className="text-sm text-muted-foreground mb-3">No pricing available yet</p>
                    <Button onClick={onAnalyze} disabled={file.isAnalyzing || !file.material}>
                      {file.isAnalyzing ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Zap className="h-4 w-4" />
                          Calculate Quote
                        </>
                      )}
                    </Button>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card border rounded-lg p-3 text-center">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className="text-sm font-semibold">{value}</p>
    </div>
  );
}

function CostRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">${value.toFixed(2)}</span>
    </div>
  );
}

function MaterialSelector({
  value,
  materials,
  onSelect,
  compact = false,
}: {
  value?: string;
  materials: string[];
  onSelect: (material: string) => void;
  compact?: boolean;
}) {
  const [open, setOpen] = React.useState(false);

  return (
    <div className={compact ? "" : "space-y-2"}>
      {!compact && <label className="text-sm font-medium">Material</label>}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" role="combobox" aria-expanded={open} className="w-full justify-between">
            {value || "Select material..."}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-full p-0">
          <Command>
            <CommandInput placeholder="Search material..." />
            <CommandList>
              <CommandEmpty>No material found.</CommandEmpty>
              <CommandGroup>
                {materials.map((material) => (
                  <CommandItem
                    key={material}
                    value={material}
                    onSelect={() => {
                      onSelect(material);
                      setOpen(false);
                    }}
                  >
                    <Check className={cn("mr-2 h-4 w-4", value === material ? "opacity-100" : "opacity-0")} />
                    {material}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
