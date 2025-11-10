import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { CADViewer } from '@/components/CADViewer';
import FeatureTree from '@/components/FeatureTree';
import { Box, Ruler, Gauge } from 'lucide-react';

interface LineItem {
  id: string;
  file_name: string;
  file_path: string;
  quantity: number;
  unit_price: number | null;
  lead_time_days: number | null;
  notes: string | null;
  estimated_volume_cm3: number | null;
  estimated_surface_area_cm2: number | null;
  estimated_complexity_score: number | null;
  material_cost: number | null;
  machining_cost: number | null;
  setup_cost: number | null;
  finish_cost: number | null;
  preliminary_unit_price: number | null;
  selected_process: string | null;
  material_type: string | null;
  finish_type: string | null;
  mesh_id: string | null;
  machining_operations: any[] | null;
  estimated_machining_cost: number | null;
  recommended_routings: string[] | null;
  routing_reasoning: string[] | null;
}

interface FeatureTree {
  geometric_features?: any;
}

interface PartDetailTabsProps {
  lineItem: LineItem;
  featureTree: FeatureTree | null;
  onUpdateLineItem: (id: string, field: string, value: any) => void;
}

const PartDetailTabs: React.FC<PartDetailTabsProps> = ({
  lineItem,
  featureTree,
  onUpdateLineItem,
}) => {
  const formatNumber = (num: number | null | undefined, decimals: number = 2): string => {
    if (num === null || num === undefined) return 'N/A';
    return num.toFixed(decimals);
  };

  const getComplexityColor = (score: number | null): string => {
    if (!score) return 'text-gray-600';
    if (score <= 3) return 'text-green-600';
    if (score <= 6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getComplexityBadge = (score: number | null): string => {
    if (!score) return 'bg-gray-100 text-gray-800';
    if (score <= 3) return 'bg-green-100 text-green-800';
    if (score <= 6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  // Calculate part dimensions from volume (cubic root approximation)
  const estimatedSize = lineItem.estimated_volume_cm3
    ? Math.cbrt(lineItem.estimated_volume_cm3)
    : null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{lineItem.file_name}</CardTitle>
          <Badge variant="outline">Qty: {lineItem.quantity}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="3d-model" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="3d-model">3D Model</TabsTrigger>
            <TabsTrigger value="features">Features</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="pricing">Pricing</TabsTrigger>
          </TabsList>

          {/* 3D Model Tab */}
          <TabsContent value="3d-model" className="mt-4">
            <div className="h-[600px] border rounded-lg overflow-hidden">
              <CADViewer
                fileName={lineItem.file_name}
                fileUrl={lineItem.file_path}
              />
            </div>
          </TabsContent>

          {/* Features Tab - Geometric Recognition */}
          <TabsContent value="features" className="mt-4">
            {featureTree?.geometric_features ? (
              <FeatureTree
                features={featureTree.geometric_features}
              />
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle>Feature Recognition</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8 text-muted-foreground">
                    <Box className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No feature analysis available for this part.</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="mt-4">
            <div className="space-y-6">
              {/* Geometry Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Box className="w-5 h-5" />
                    Geometry Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Volume</Label>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold">
                          {formatNumber(lineItem.estimated_volume_cm3)}
                        </div>
                        <div className="text-sm text-gray-500">cm³</div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>Surface Area</Label>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold">
                          {formatNumber(lineItem.estimated_surface_area_cm2)}
                        </div>
                        <div className="text-sm text-gray-500">cm²</div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>Complexity Score</Label>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <Badge className={getComplexityBadge(lineItem.estimated_complexity_score)}>
                          {lineItem.estimated_complexity_score || 'N/A'} / 10
                        </Badge>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>Estimated Size</Label>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold">
                          {estimatedSize ? `~${formatNumber(estimatedSize, 1)}` : 'N/A'}
                        </div>
                        <div className="text-sm text-gray-500">cm (approx)</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Material & Process */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Gauge className="w-5 h-5" />
                    Material & Process
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <Label>Material</Label>
                      <div className="p-3 bg-gray-50 rounded-lg mt-2">
                        {lineItem.material_type || 'Not specified'}
                      </div>
                    </div>

                    <div>
                      <Label>Selected Process</Label>
                      <div className="p-3 bg-gray-50 rounded-lg mt-2">
                        {lineItem.selected_process || 'Not specified'}
                      </div>
                    </div>

                    {lineItem.finish_type && (
                      <div>
                        <Label>Surface Finish</Label>
                        <div className="p-3 bg-gray-50 rounded-lg mt-2">
                          {lineItem.finish_type}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Recommended Routings */}
              {lineItem.recommended_routings && lineItem.recommended_routings.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Ruler className="w-5 h-5" />
                      Recommended Manufacturing Routings
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {lineItem.recommended_routings.map((routing, idx) => (
                        <div key={idx} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <div className="font-medium">{routing}</div>
                          {lineItem.routing_reasoning && lineItem.routing_reasoning[idx] && (
                            <div className="text-sm text-gray-600 mt-1">
                              {lineItem.routing_reasoning[idx]}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Machining Operations */}
              {lineItem.machining_operations && lineItem.machining_operations.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Machining Operations Breakdown</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {lineItem.machining_operations.map((op: any, idx: number) => (
                        <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center justify-between">
                            <div className="font-medium">{op.routing}</div>
                            <Badge variant="outline">
                              ${formatNumber(op.machining_cost)}
                            </Badge>
                          </div>
                          <div className="text-sm text-gray-600 mt-1">
                            Time: {formatNumber(op.machining_time_min)} minutes
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Cost Breakdown */}
              <Card>
                <CardHeader>
                  <CardTitle>Cost Breakdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {lineItem.material_cost !== null && (
                      <div className="flex items-center justify-between">
                        <span>Material Cost</span>
                        <span className="font-medium">${formatNumber(lineItem.material_cost)}</span>
                      </div>
                    )}
                    {lineItem.machining_cost !== null && (
                      <div className="flex items-center justify-between">
                        <span>Machining Cost</span>
                        <span className="font-medium">${formatNumber(lineItem.machining_cost)}</span>
                      </div>
                    )}
                    {lineItem.setup_cost !== null && (
                      <div className="flex items-center justify-between">
                        <span>Setup Cost</span>
                        <span className="font-medium">${formatNumber(lineItem.setup_cost)}</span>
                      </div>
                    )}
                    {lineItem.finish_cost !== null && (
                      <div className="flex items-center justify-between">
                        <span>Finish Cost</span>
                        <span className="font-medium">${formatNumber(lineItem.finish_cost)}</span>
                      </div>
                    )}
                    {lineItem.preliminary_unit_price !== null && (
                      <>
                        <div className="border-t pt-3 mt-3"></div>
                        <div className="flex items-center justify-between text-lg">
                          <span className="font-semibold">Preliminary Unit Price</span>
                          <span className="font-bold text-blue-600">
                            ${formatNumber(lineItem.preliminary_unit_price)}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Pricing Tab */}
          <TabsContent value="pricing" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Edit Pricing</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="unit-price">Unit Price ($)</Label>
                  <Input
                    id="unit-price"
                    type="number"
                    step="0.01"
                    value={lineItem.unit_price || ''}
                    onChange={(e) =>
                      onUpdateLineItem(
                        lineItem.id,
                        'unit_price',
                        parseFloat(e.target.value) || null
                      )
                    }
                    placeholder="Enter unit price"
                  />
                  {lineItem.preliminary_unit_price && (
                    <p className="text-sm text-gray-500">
                      Preliminary estimate: ${formatNumber(lineItem.preliminary_unit_price)}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="lead-time">Lead Time (days)</Label>
                  <Input
                    id="lead-time"
                    type="number"
                    value={lineItem.lead_time_days || ''}
                    onChange={(e) =>
                      onUpdateLineItem(
                        lineItem.id,
                        'lead_time_days',
                        parseInt(e.target.value) || null
                      )
                    }
                    placeholder="Enter lead time"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="notes">Notes</Label>
                  <Textarea
                    id="notes"
                    value={lineItem.notes || ''}
                    onChange={(e) =>
                      onUpdateLineItem(lineItem.id, 'notes', e.target.value)
                    }
                    placeholder="Add any notes or special considerations"
                    rows={4}
                  />
                </div>

                {/* Total Price Display */}
                {lineItem.unit_price && (
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-center justify-between text-lg">
                      <span className="font-semibold">Total Price</span>
                      <span className="font-bold text-blue-600">
                        ${formatNumber((lineItem.unit_price || 0) * lineItem.quantity)}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      {lineItem.quantity} units × ${formatNumber(lineItem.unit_price)}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default PartDetailTabs;
