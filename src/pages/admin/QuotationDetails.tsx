import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { Loader2, ArrowLeft, Download, Save, Send, Sparkles } from 'lucide-react';
import { StatusBadge } from '@/components/admin/StatusBadge';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import PartDetailTabs from '@/components/admin/PartDetailTabs';

interface QuotationData {
  id: string;
  email: string;
  customer_name: string;
  customer_company: string | null;
  customer_phone: string;
  shipping_address: string;
  customer_message: string | null;
  status: string;
  quote_number: string;
  submitted_at: string;
  created_at: string;
}

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
  estimated_machine_time_hours: number | null;
  mesh_id: string | null;
  machining_operations: any[] | null;
  estimated_machining_cost: number | null;
  recommended_routings: string[] | null;
  routing_reasoning: string[] | null;
}

interface Quote {
  subtotal: number;
  shipping_cost: number;
  tax_rate: number;
  tax_amount: number;
  total_amount: number;
  estimated_lead_time_days: number | null;
  notes: string | null;
  valid_until: string | null;
}

const QuotationDetails = () => {
  const { id } = useParams();
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [quotation, setQuotation] = useState<QuotationData | null>(null);
  const [lineItems, setLineItems] = useState<LineItem[]>([]);
  const [quote, setQuote] = useState<Quote>({
    subtotal: 0,
    shipping_cost: 0,
    tax_rate: 0,
    tax_amount: 0,
    total_amount: 0,
    estimated_lead_time_days: null,
    notes: null,
    valid_until: null,
  });
  const [featureTrees, setFeatureTrees] = useState<Map<string, any>>(new Map());

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

      // Fetch quotation data
      const { data: quotationData, error: quotationError } = await supabase
        .from('quotation_submissions')
        .select('*')
        .eq('id', id)
        .single();

      if (quotationError || !quotationData) {
        toast({
          title: 'Error',
          description: 'Failed to load quotation details.',
          variant: 'destructive',
        });
        navigate('/admin');
        return;
      }

      setQuotation(quotationData);

      // Fetch line items
      const { data: lineItemsData, error: lineItemsError } = await supabase
        .from('quote_line_items')
        .select('*')
        .eq('quotation_id', id)
        .order('created_at', { ascending: true });

      if (!lineItemsError && lineItemsData) {
        setLineItems(lineItemsData as LineItem[]);

        // Fetch feature trees for each line item from cad_meshes
        const featureTreesMap = new Map();
        for (const item of lineItemsData) {
          let geometric_features = null;
          
          // Fetch from cad_meshes if mesh_id exists
          if (item.mesh_id) {
            const { data: meshData } = await supabase
              .from('cad_meshes')
              .select('geometric_features')
              .eq('id', item.mesh_id)
              .maybeSingle();
            
            if (meshData?.geometric_features) {
              geometric_features = meshData.geometric_features;
            }
          }
          
          // Fallback to old part_features table if needed
          if (!geometric_features) {
            const { data: features } = await supabase
              .from('part_features')
              .select('*')
              .eq('line_item_id', item.id);
            
            if (features && features.length > 0) {
              // Reconstruct old format for backward compatibility
              const manufacturing_features: any = {
                through_holes: [],
                blind_holes: [],
                bores: [],
                bosses: [],
                planar_faces: [],
                fillets: []
              };
              
              let feature_summary: any = null;
              
              features.forEach(feature => {
                if (feature.feature_type === 'summary') {
                  feature_summary = feature.parameters;
                } else if (feature.feature_type === 'through_hole') {
                  manufacturing_features.through_holes.push(feature.parameters);
                } else if (feature.feature_type === 'blind_hole') {
                  manufacturing_features.blind_holes.push(feature.parameters);
                } else if (feature.feature_type === 'bore') {
                  manufacturing_features.bores.push(feature.parameters);
                } else if (feature.feature_type === 'boss') {
                  manufacturing_features.bosses.push(feature.parameters);
                } else if (feature.feature_type === 'planar_face') {
                  manufacturing_features.planar_faces.push(feature.parameters);
                } else if (feature.feature_type === 'fillet') {
                  manufacturing_features.fillets.push(feature.parameters);
                }
              });
            }
          }
          
          if (geometric_features) {
            featureTreesMap.set(item.id, { geometric_features });
          }
        }
        setFeatureTrees(featureTreesMap);
      }

      // Fetch existing quote if available
      const { data: quoteData } = await supabase
        .from('quotes')
        .select('*')
        .eq('quotation_id', id)
        .single();

      if (quoteData) {
        setQuote({
          subtotal: Number(quoteData.subtotal),
          shipping_cost: Number(quoteData.shipping_cost),
          tax_rate: Number(quoteData.tax_rate),
          tax_amount: Number(quoteData.tax_amount),
          total_amount: Number(quoteData.total_amount),
          estimated_lead_time_days: quoteData.estimated_lead_time_days,
          notes: quoteData.notes,
          valid_until: quoteData.valid_until,
        });
      }

      setLoading(false);
    };

    checkAdminAndFetchData();
  }, [user, authLoading, id, navigate, toast]);

  const calculateTotals = (items: LineItem[], shippingCost: number, taxRate: number) => {
    const subtotal = items.reduce((sum, item) => {
      return sum + (Number(item.unit_price || 0) * item.quantity);
    }, 0);

    const taxAmount = (subtotal + shippingCost) * (taxRate / 100);
    const totalAmount = subtotal + shippingCost + taxAmount;

    return { subtotal, taxAmount, totalAmount };
  };

  const updateLineItem = (id: string, field: string, value: any) => {
    const updated = lineItems.map(item => 
      item.id === id ? { ...item, [field]: value } : item
    );
    setLineItems(updated);

    // Recalculate totals
    const { subtotal, taxAmount, totalAmount } = calculateTotals(
      updated,
      quote.shipping_cost,
      quote.tax_rate
    );
    setQuote(prev => ({ ...prev, subtotal, tax_amount: taxAmount, total_amount: totalAmount }));
  };

  const handleShippingChange = (value: string) => {
    const shippingCost = parseFloat(value) || 0;
    const { subtotal, taxAmount, totalAmount } = calculateTotals(lineItems, shippingCost, quote.tax_rate);
    setQuote(prev => ({ ...prev, shipping_cost: shippingCost, subtotal, tax_amount: taxAmount, total_amount: totalAmount }));
  };

  const handleTaxRateChange = (value: string) => {
    const taxRate = parseFloat(value) || 0;
    const { subtotal, taxAmount, totalAmount } = calculateTotals(lineItems, quote.shipping_cost, taxRate);
    setQuote(prev => ({ ...prev, tax_rate: taxRate, subtotal, tax_amount: taxAmount, total_amount: totalAmount }));
  };

  const handleSave = async (sendEmail: boolean = false) => {
    if (!quotation || !user) return;

    setSaving(true);
    try {
      // Update line items
      for (const item of lineItems) {
        const { error } = await supabase
          .from('quote_line_items')
          .update({
            unit_price: item.unit_price,
            lead_time_days: item.lead_time_days,
            notes: item.notes,
          })
          .eq('id', item.id);

        if (error) throw error;
      }

      // Upsert quote
      const { error: quoteError } = await supabase
        .from('quotes')
        .upsert({
          quotation_id: quotation.id,
          quote_number: quotation.quote_number,
          subtotal: quote.subtotal,
          shipping_cost: quote.shipping_cost,
          tax_rate: quote.tax_rate,
          tax_amount: quote.tax_amount,
          total_amount: quote.total_amount,
          estimated_lead_time_days: quote.estimated_lead_time_days,
          notes: quote.notes,
          valid_until: quote.valid_until,
          created_by: user.id,
          sent_at: sendEmail ? new Date().toISOString() : null,
        });

      if (quoteError) throw quoteError;

      // Update quotation status
      const newStatus = sendEmail ? 'quoted' : 'reviewing';
      const { error: statusError } = await supabase
        .from('quotation_submissions')
        .update({
          status: newStatus,
          reviewed_by: user.id,
          reviewed_at: new Date().toISOString(),
        })
        .eq('id', quotation.id);

      if (statusError) throw statusError;

      if (sendEmail) {
        // Call edge function to send quote email
        const { error: emailError } = await supabase.functions.invoke('send-quote-email', {
          body: {
            quotationId: quotation.id,
            customerEmail: quotation.email,
            customerName: quotation.customer_name,
            quoteNumber: quotation.quote_number,
          },
        });

        if (emailError) throw emailError;
      }

      toast({
        title: 'Success',
        description: sendEmail ? 'Quote saved and email sent to customer.' : 'Quote saved as draft.',
      });

      if (sendEmail) {
        navigate('/admin');
      }
    } catch (error: any) {
      console.error('Error saving quote:', error);
      toast({
        title: 'Error',
        description: error.message || 'Failed to save quote.',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const downloadFile = async (filePath: string, fileName: string) => {
    try {
      const { data, error } = await supabase.storage
        .from('part-files')
        .download(filePath);

      if (error) throw error;

      const url = URL.createObjectURL(data);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error: any) {
      toast({
        title: 'Error',
        description: 'Failed to download file.',
        variant: 'destructive',
      });
    }
  };

  if (authLoading || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!quotation) {
    return null;
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      <main className="flex-1 container mx-auto py-12 px-4">
        <Button
          variant="outline"
          onClick={() => navigate('/admin')}
          className="mb-6"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Button>

        <div className="grid gap-6">
          {/* Header */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Quotation {quotation.quote_number}</CardTitle>
                  <CardDescription>
                    Submitted {new Date(quotation.submitted_at).toLocaleString()}
                  </CardDescription>
                </div>
                <StatusBadge status={quotation.status as any} />
              </div>
            </CardHeader>
          </Card>

          {/* Customer Information */}
          <Card>
            <CardHeader>
              <CardTitle>Customer Information</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div>
                <Label>Name</Label>
                <p className="text-sm font-medium">{quotation.customer_name}</p>
              </div>
              {quotation.customer_company && (
                <div>
                  <Label>Company</Label>
                  <p className="text-sm font-medium">{quotation.customer_company}</p>
                </div>
              )}
              <div>
                <Label>Email</Label>
                <p className="text-sm font-medium">{quotation.email}</p>
              </div>
              <div>
                <Label>Phone</Label>
                <p className="text-sm font-medium">{quotation.customer_phone}</p>
              </div>
              <div className="md:col-span-2">
                <Label>Shipping Address</Label>
                <p className="text-sm font-medium whitespace-pre-line">{quotation.shipping_address}</p>
              </div>
              {quotation.customer_message && (
                <div className="md:col-span-2">
                  <Label>Customer Message</Label>
                  <p className="text-sm font-medium whitespace-pre-line">{quotation.customer_message}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Files & Pricing */}
          <Card>
            <CardHeader>
              <CardTitle>Parts & Pricing</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {lineItems.map((item, index) => (
                  <div key={item.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <Badge variant="outline">Part {index + 1}</Badge>
                        <h3 className="font-semibold">{item.file_name}</h3>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => downloadFile(item.file_path, item.file_name)}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download
                      </Button>
                    </div>
                    
                    <PartDetailTabs
                      lineItem={item}
                      onUpdateLineItem={updateLineItem}
                    />
                  </div>
                ))}
              </div>

              {/* Quote Totals */}
              <div className="mt-6 pt-6 border-t space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <Label htmlFor="shipping">Shipping Cost ($)</Label>
                    <Input
                      id="shipping"
                      type="number"
                      step="0.01"
                      value={quote.shipping_cost}
                      onChange={(e) => handleShippingChange(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label htmlFor="tax-rate">Tax Rate (%)</Label>
                    <Input
                      id="tax-rate"
                      type="number"
                      step="0.01"
                      value={quote.tax_rate}
                      onChange={(e) => handleTaxRateChange(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label htmlFor="lead-time">Estimated Lead Time (days)</Label>
                    <Input
                      id="lead-time"
                      type="number"
                      value={quote.estimated_lead_time_days || ''}
                      onChange={(e) => setQuote(prev => ({ ...prev, estimated_lead_time_days: parseInt(e.target.value) || null }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="valid-until">Valid Until</Label>
                    <Input
                      id="valid-until"
                      type="date"
                      value={quote.valid_until || ''}
                      onChange={(e) => setQuote(prev => ({ ...prev, valid_until: e.target.value }))}
                    />
                  </div>
                </div>

                <div>
                  <Label htmlFor="quote-notes">Quote Notes</Label>
                  <Textarea
                    id="quote-notes"
                    value={quote.notes || ''}
                    onChange={(e) => setQuote(prev => ({ ...prev, notes: e.target.value }))}
                    placeholder="Additional terms, conditions, or notes for the customer..."
                    rows={3}
                  />
                </div>

                <div className="bg-muted p-4 rounded-lg space-y-2">
                  <div className="flex justify-between">
                    <span>Subtotal:</span>
                    <span className="font-medium">${quote.subtotal.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Shipping:</span>
                    <span className="font-medium">${quote.shipping_cost.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tax ({quote.tax_rate}%):</span>
                    <span className="font-medium">${quote.tax_amount.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-lg font-bold pt-2 border-t">
                    <span>Total:</span>
                    <span>${quote.total_amount.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="mt-6 flex gap-4">
                <Button
                  variant="outline"
                  onClick={() => handleSave(false)}
                  disabled={saving}
                >
                  {saving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Save className="h-4 w-4 mr-2" />}
                  Save Draft
                </Button>
                <Button
                  onClick={() => handleSave(true)}
                  disabled={saving || quote.total_amount === 0}
                >
                  {saving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Send className="h-4 w-4 mr-2" />}
                  Save & Send Quote Email
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default QuotationDetails;
