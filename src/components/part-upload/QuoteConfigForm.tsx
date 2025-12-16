import { useState } from 'react';
import { Shield, Ruler, Settings, User, Building2, Phone, Mail, MapPin, FileText, Minus, Plus, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { CADViewer } from '@/components/CADViewer';
import { AddressAutocomplete } from './AddressAutocomplete';
import { Loader2 } from 'lucide-react';

const MATERIAL_OPTIONS = [
  "Aluminum (6061)",
  "Stainless Steel (304L / 316L)",
  "Alloy Steel (4140)",
  "Carbon Steel (1018 / 1045)",
  "Tool Steel (A2 / D2 / S7)",
  "Copper Alloy (Brass / Bronze)",
  "Plastic (Delrin / UHMW)",
  "Other",
];

const FINISH_OPTIONS = [
  "As Machined",
  "Anodizing (Type II)",
  "Hard Anodizing (Type III)",
  "Passivation",
  "Polishing",
  "Other",
];

interface QuoteConfigFormProps {
  file: File;
  meshData?: any;
  boundingBox?: { width: number; height: number; depth: number };
  onBack: () => void;
  onSubmitSuccess: () => void;
}

export function QuoteConfigForm({ file, meshData, boundingBox, onBack, onSubmitSuccess }: QuoteConfigFormProps) {
  // Contact info state
  const [name, setName] = useState('');
  const [company, setCompany] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [address, setAddress] = useState('');
  const [projectDescription, setProjectDescription] = useState('');

  // Part details state
  const [partDetailsOpen, setPartDetailsOpen] = useState(false);
  const [partName, setPartName] = useState('');
  const [material, setMaterial] = useState('');
  const [quantity, setQuantity] = useState(1);
  const [finish, setFinish] = useState('As Machined');
  const [heatTreatment, setHeatTreatment] = useState(false);
  const [heatTreatmentDetails, setHeatTreatmentDetails] = useState('');
  const [tolerances, setTolerances] = useState('');

  // Submission state
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate required fields
    if (!name.trim() || !company.trim() || !phone.trim() || !email.trim() || !projectDescription.trim()) {
      toast.error('Please fill in all required fields');
      return;
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      toast.error('Please enter a valid email address');
      return;
    }

    setSubmitting(true);

    try {
      // Build the customer message with all details
      let fullMessage = projectDescription;
      
      if (address) {
        fullMessage += `\nShipping Address: ${address}`;
      }
      
      if (partName || material || quantity > 1 || finish !== 'As Machined' || heatTreatment || tolerances) {
        fullMessage += '\n\n--- Part Details ---';
        if (partName) fullMessage += `\nPart Name: ${partName}`;
        if (material) fullMessage += `\nMaterial: ${material}`;
        fullMessage += `\nQuantity: ${quantity}`;
        fullMessage += `\nFinish: ${finish}`;
        if (heatTreatment) {
          fullMessage += `\nHeat Treatment: Required`;
          if (heatTreatmentDetails.trim()) {
            fullMessage += ` - ${heatTreatmentDetails.trim()}`;
          }
        }
        if (tolerances) fullMessage += `\nThreads/Tolerances: ${tolerances}`;
      }

      // Create IP hash
      const ipHash = btoa(new Date().toISOString()).slice(0, 32);

      // Insert quotation submission
      const { data: quotation, error: quotationError } = await supabase
        .from('quotation_submissions')
        .insert({
          email: email.trim(),
          customer_name: name.trim(),
          customer_company: company.trim(),
          customer_phone: phone.trim(),
          customer_message: fullMessage,
          shipping_address: address.trim() || null,
          ip_hash: ipHash,
        })
        .select()
        .single();

      if (quotationError) throw quotationError;

      // Track uploaded file paths for email attachment
      const uploadedFilePaths: Array<{ name: string; path: string; size: number }> = [];

      // Upload the CAD file
      const filePath = `${quotation.id}/${Date.now()}-${file.name}`;
      
      const { error: uploadError } = await supabase.storage
        .from('cad-files')
        .upload(filePath, file);

      if (!uploadError) {
        uploadedFilePaths.push({
          name: file.name,
          path: filePath,
          size: file.size
        });

        // Create line item
        await supabase
          .from('quote_line_items')
          .insert({
            quotation_id: quotation.id,
            file_name: file.name,
            file_path: filePath,
            quantity: quantity,
            material_type: material || null,
            finish_type: finish,
          });
      }

      // Send email notification
      try {
        const emailPayload = {
          notificationOnly: true,
          name: name.trim(),
          company: company.trim(),
          email: email.trim(),
          phone: phone.trim(),
          address: address.trim(),
          message: fullMessage,
          quoteNumber: quotation.quote_number,
          files: uploadedFilePaths
        };

        await supabase.functions.invoke('send-quotation-request', {
          body: emailPayload
        });
      } catch (emailErr) {
        console.error('Failed to send email notification:', emailErr);
      }

      toast.success('Quote request submitted successfully! We\'ll be in touch within 24 hours.');
      onSubmitSuccess();

    } catch (error) {
      console.error('Submission error:', error);
      toast.error('Failed to submit quote request. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Back button */}
      <button
        onClick={onBack}
        className="text-white/70 hover:text-white text-sm flex items-center gap-2"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to upload
      </button>

      {/* CAD Viewer */}
      <div className="backdrop-blur-sm border border-white/20 rounded-sm overflow-hidden" style={{ backgroundColor: "rgba(60, 60, 60, 0.75)" }}>
        <div className="p-4 border-b border-white/10">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white">{file.name}</h3>
              <p className="text-sm text-gray-400">
                {(file.size / 1024).toFixed(2)} KB
                {boundingBox && ` • ${boundingBox.width.toFixed(1)} × ${boundingBox.height.toFixed(1)} × ${boundingBox.depth.toFixed(1)} mm`}
              </p>
            </div>
          </div>
        </div>
        <div className="h-[400px]">
          {meshData ? (
            <CADViewer
              meshData={meshData}
              fileName={file.name}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-white/50">
              <p>3D preview not available</p>
            </div>
          )}
        </div>
      </div>

      {/* Quote Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Feature Badges */}
        <div className="flex flex-wrap justify-center gap-4 text-sm">
          <div className="flex items-center gap-2 text-white">
            <Shield className="h-4 w-4 text-primary" />
            <span>Confidential by default</span>
          </div>
          <div className="flex items-center gap-2 text-white">
            <Ruler className="h-4 w-4 text-primary" />
            <span>GD&T friendly</span>
          </div>
          <div className="flex items-center gap-2 text-white">
            <Settings className="h-4 w-4 text-primary" />
            <span>HMLV ready</span>
          </div>
        </div>

        {/* Project Details Card */}
        <Card className="bg-black/60 border-white/20 backdrop-blur-sm">
          <CardContent className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-white">Contact Information</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Name */}
              <div className="space-y-1.5">
                <Label htmlFor="name" className="text-sm text-white flex items-center gap-2">
                  <User className="h-4 w-4" />
                  Name *
                </Label>
                <Input
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Jane Smith"
                  className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                  required
                />
              </div>

              {/* Company */}
              <div className="space-y-1.5">
                <Label htmlFor="company" className="text-sm text-white flex items-center gap-2">
                  <Building2 className="h-4 w-4" />
                  Company Name *
                </Label>
                <Input
                  id="company"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  placeholder="Acme Manufacturing Ltd."
                  className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                  required
                />
              </div>

              {/* Phone */}
              <div className="space-y-1.5">
                <Label htmlFor="phone" className="text-sm text-white flex items-center gap-2">
                  <Phone className="h-4 w-4" />
                  Phone Number *
                </Label>
                <Input
                  id="phone"
                  type="tel"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="(555) 123-4567"
                  className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                  required
                />
              </div>

              {/* Email */}
              <div className="space-y-1.5">
                <Label htmlFor="email" className="text-sm text-white flex items-center gap-2">
                  <Mail className="h-4 w-4" />
                  Email *
                </Label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                  required
                />
              </div>

              {/* Address with Autocomplete */}
              <div className="space-y-1.5 md:col-span-2">
                <Label htmlFor="address" className="text-sm text-white flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  Shipping Address
                </Label>
                <AddressAutocomplete
                  value={address}
                  onChange={setAddress}
                  placeholder="Start typing your address..."
                  className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                />
                <p className="text-xs text-white/60">For shipping estimates (optional)</p>
              </div>

              {/* Project Description */}
              <div className="space-y-1.5 md:col-span-2">
                <Label htmlFor="description" className="text-sm text-white flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Project Description *
                </Label>
                <Textarea
                  id="description"
                  value={projectDescription}
                  onChange={(e) => setProjectDescription(e.target.value)}
                  placeholder="Briefly describe the part and requirements (material, qty, finish, critical dims, deadline)."
                  className="bg-white/10 border-white/20 text-white placeholder:text-white/40 min-h-[100px]"
                  required
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Part Details Collapsible */}
        <Collapsible open={partDetailsOpen} onOpenChange={setPartDetailsOpen}>
          <Card className="bg-black/60 border-white/20 backdrop-blur-sm">
            <CollapsibleTrigger asChild>
              <CardContent className="p-4 cursor-pointer hover:bg-white/10 transition-colors">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                      <span className={`transform transition-transform ${partDetailsOpen ? 'rotate-90' : ''}`}>▶</span>
                      Part Details (Optional)
                    </h2>
                    <p className="text-xs text-white/70 mt-1">Click to expand and provide additional details</p>
                  </div>
                </div>
              </CardContent>
            </CollapsibleTrigger>

            <CollapsibleContent>
              <CardContent className="p-6 pt-0 space-y-4 border-t border-white/20">
                {/* Part Name */}
                <div className="space-y-1.5">
                  <Label htmlFor="partName" className="text-sm text-white">Part / Job name</Label>
                  <Input
                    id="partName"
                    value={partName}
                    onChange={(e) => setPartName(e.target.value)}
                    placeholder="e.g., Shaft Holder Rev B"
                    className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                  />
                </div>

                {/* Material & Quantity Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <Label className="text-sm text-white">Material</Label>
                    <Select value={material} onValueChange={setMaterial}>
                      <SelectTrigger className="bg-white/10 border-white/20 text-white">
                        <SelectValue placeholder="Select Material" />
                      </SelectTrigger>
                      <SelectContent>
                        {MATERIAL_OPTIONS.map((option) => (
                          <SelectItem key={option} value={option}>{option}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1.5">
                    <Label className="text-sm text-white">Quantity</Label>
                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        className="h-10 w-10 border-white/20 text-white hover:bg-white/10"
                        onClick={() => setQuantity(Math.max(1, quantity - 1))}
                      >
                        <Minus className="h-4 w-4" />
                      </Button>
                      <Input
                        type="number"
                        min="1"
                        value={quantity}
                        onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                        className="bg-white/10 border-white/20 text-white text-center w-20"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        className="h-10 w-10 border-white/20 text-white hover:bg-white/10"
                        onClick={() => setQuantity(quantity + 1)}
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Finish */}
                <div className="space-y-1.5">
                  <Label className="text-sm text-white">Finish</Label>
                  <Select value={finish} onValueChange={setFinish}>
                    <SelectTrigger className="bg-white/10 border-white/20 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {FINISH_OPTIONS.map((option) => (
                        <SelectItem key={option} value={option}>{option}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Heat Treatment */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="heatTreatment"
                      checked={heatTreatment}
                      onCheckedChange={(checked) => setHeatTreatment(checked as boolean)}
                    />
                    <Label htmlFor="heatTreatment" className="text-sm text-white cursor-pointer">
                      Heat Treatment Required
                    </Label>
                  </div>
                  {heatTreatment && (
                    <Input
                      value={heatTreatmentDetails}
                      onChange={(e) => setHeatTreatmentDetails(e.target.value)}
                      placeholder="Specify heat treatment (e.g., HRC 58-62)"
                      className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                    />
                  )}
                </div>

                {/* Tolerances */}
                <div className="space-y-1.5">
                  <Label className="text-sm text-white">Threads / Tight Tolerances</Label>
                  <Textarea
                    value={tolerances}
                    onChange={(e) => setTolerances(e.target.value)}
                    placeholder="List any threaded holes, tight tolerances, or critical dimensions"
                    className="bg-white/10 border-white/20 text-white placeholder:text-white/40 min-h-[80px]"
                  />
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>

        {/* Submit Button */}
        <Button
          type="submit"
          size="lg"
          disabled={submitting}
          className="w-full bg-primary hover:bg-primary/90 text-white"
        >
          {submitting ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Submitting...
            </>
          ) : (
            'Submit Quote Request'
          )}
        </Button>
      </form>
    </div>
  );
}
