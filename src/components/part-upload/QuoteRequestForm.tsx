import { useState, useRef } from 'react';
import { Shield, Ruler, Settings, Clock, Award, FileText, Upload, Plus, Minus, X, Loader2, User, Building2, Phone, Mail } from 'lucide-react';
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

const ACCEPTED_EXTENSIONS = ['.step', '.stp', '.iges', '.igs', '.dxf', '.dwg', '.pdf', '.png', '.jpg', '.jpeg', '.zip'];

interface UploadedFile {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'done' | 'error';
}

export function QuoteRequestForm() {
  // Contact info state
  const [name, setName] = useState('');
  const [company, setCompany] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [projectDescription, setProjectDescription] = useState('');

  // Part details state
  const [partDetailsOpen, setPartDetailsOpen] = useState(false);
  const [partName, setPartName] = useState('');
  const [material, setMaterial] = useState('');
  const [quantity, setQuantity] = useState(1);
  const [finish, setFinish] = useState('As Machined');
  const [heatTreatment, setHeatTreatment] = useState(false);
  const [tolerances, setTolerances] = useState('');

  // File upload state
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Submission state
  const [submitting, setSubmitting] = useState(false);

  const validateFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      toast.error(`Unsupported file type: ${ext}`);
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast.error(`File too large: ${file.name} (max 10MB)`);
      return false;
    }
    return true;
  };

  const handleFileSelect = (selectedFiles: FileList | null) => {
    if (!selectedFiles) return;
    
    const validFiles: UploadedFile[] = [];
    const currentCount = files.length;
    
    for (let i = 0; i < selectedFiles.length && currentCount + validFiles.length < 10; i++) {
      if (validateFile(selectedFiles[i])) {
        validFiles.push({ file: selectedFiles[i], progress: 0, status: 'pending' });
      }
    }
    
    if (selectedFiles.length > 10 - currentCount) {
      toast.warning('Maximum 10 files allowed');
    }
    
    setFiles(prev => [...prev, ...validFiles]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

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
      
      if (partName || material || quantity > 1 || finish !== 'As Machined' || heatTreatment || tolerances) {
        fullMessage += '\n\n--- Part Details ---';
        if (partName) fullMessage += `\nPart Name: ${partName}`;
        if (material) fullMessage += `\nMaterial: ${material}`;
        fullMessage += `\nQuantity: ${quantity}`;
        fullMessage += `\nFinish: ${finish}`;
        if (heatTreatment) fullMessage += `\nHeat Treatment: Required`;
        if (tolerances) fullMessage += `\nThreads/Tolerances: ${tolerances}`;
      }

      // Create IP hash (simple hash for rate limiting)
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
          ip_hash: ipHash,
        })
        .select()
        .single();

      if (quotationError) throw quotationError;

      // Upload files if any
      if (files.length > 0) {
        for (let i = 0; i < files.length; i++) {
          const fileData = files[i];
          const filePath = `${quotation.id}/${Date.now()}-${fileData.file.name}`;
          
          setFiles(prev => prev.map((f, idx) => 
            idx === i ? { ...f, status: 'uploading', progress: 50 } : f
          ));

          const { error: uploadError } = await supabase.storage
            .from('cad-files')
            .upload(filePath, fileData.file);

          if (uploadError) {
            console.error('File upload error:', uploadError);
            setFiles(prev => prev.map((f, idx) => 
              idx === i ? { ...f, status: 'error' } : f
            ));
            continue;
          }

          // Create line item for each file
          await supabase
            .from('quote_line_items')
            .insert({
              quotation_id: quotation.id,
              file_name: fileData.file.name,
              file_path: filePath,
              quantity: quantity,
              material_type: material || null,
              finish_type: finish,
            });

          setFiles(prev => prev.map((f, idx) => 
            idx === i ? { ...f, status: 'done', progress: 100 } : f
          ));
        }
      }

      toast.success('Quote request submitted successfully! We\'ll be in touch within 24 hours.');
      
      // Reset form
      setName('');
      setCompany('');
      setPhone('');
      setEmail('');
      setProjectDescription('');
      setPartName('');
      setMaterial('');
      setQuantity(1);
      setFinish('As Machined');
      setHeatTreatment(false);
      setTolerances('');
      setFiles([]);
      setPartDetailsOpen(false);

    } catch (error) {
      console.error('Submission error:', error);
      toast.error('Failed to submit quote request. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Request a Quote</h1>
        <p className="text-white/80 mb-4">Tell us about your part. We typically reply within 24 hours.</p>
        
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
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Project Details Card */}
        <Card className="bg-black/60 border-white/20 backdrop-blur-sm">
          <CardContent className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-white">Project Details</h2>
              <span className="text-xs text-white/70">Faster quotes with CAD or dimensions</span>
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
                <p className="text-xs text-white/60">Include extension if applicable</p>
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
                <p className="text-xs text-white/60">If no drawing, a short description is fine.</p>
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
                    <p className="text-xs text-white/70 mt-1">Click to expand and provide additional details if you have them</p>
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
                  <p className="text-xs text-white/60">Enter a descriptive name for your part or job</p>
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
                    <p className="text-xs text-white/60">Select closest match</p>
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
                    <p className="text-xs text-white/60">Total parts needed</p>
                  </div>
                </div>

                {/* Finish & Heat Treatment Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                    <p className="text-xs text-white/60">Leave as 'As Machined' if unsure</p>
                  </div>

                  <div className="space-y-1.5">
                    <Label className="text-sm text-white">Heat Treatment</Label>
                    <div className="flex items-center space-x-2 h-10">
                      <Checkbox
                        id="heatTreatment"
                        checked={heatTreatment}
                        onCheckedChange={(checked) => setHeatTreatment(checked as boolean)}
                        className="border-white/20"
                      />
                      <label htmlFor="heatTreatment" className="text-sm text-white cursor-pointer">
                        Required
                      </label>
                    </div>
                    <p className="text-xs text-white/60">Check if heat treatment is needed</p>
                  </div>
                </div>

                {/* Threads / Tolerances */}
                <div className="space-y-1.5">
                  <Label htmlFor="tolerances" className="text-sm text-white">Threads / Tolerances</Label>
                  <Textarea
                    id="tolerances"
                    value={tolerances}
                    onChange={(e) => setTolerances(e.target.value)}
                    placeholder="e.g., ±0.001&quot;, 1/4-20 UNC, Ø10 H7, true position 0.05"
                    className="bg-white/10 border-white/20 text-white placeholder:text-white/40"
                  />
                  <p className="text-xs text-white/60">Please specify any critical dimensions, threads, or tolerance requirements</p>
                </div>

                {/* File Upload */}
                <div className="space-y-1.5">
                  <Label className="text-sm text-white">Drawings / CAD (optional)</Label>
                  <p className="text-xs text-white/60 mb-2">STEP, IGES, DXF, DWG, PDF, images — up to 10 files / 10MB each</p>
                  
                  <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                      isDragging 
                        ? 'border-primary bg-primary/10' 
                        : 'border-white/30 hover:border-white/50'
                    }`}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="h-8 w-8 mx-auto mb-2 text-white/70" />
                    <p className="text-white">Click to upload files</p>
                    <p className="text-white/60 text-sm">or drag and drop</p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept={ACCEPTED_EXTENSIONS.join(',')}
                      className="hidden"
                      onChange={(e) => handleFileSelect(e.target.files)}
                    />
                  </div>
                  <p className="text-xs text-white/60">ZIP multiple drawings. Filenames help us match parts.</p>

                  {/* Uploaded Files List */}
                  {files.length > 0 && (
                    <div className="space-y-2 mt-3">
                      {files.map((fileData, index) => (
                        <div key={index} className="flex items-center justify-between bg-white/10 rounded p-2">
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <FileText className="h-4 w-4 text-white/70 flex-shrink-0" />
                            <span className="text-sm text-white truncate">{fileData.file.name}</span>
                            {fileData.status === 'uploading' && (
                              <Loader2 className="h-4 w-4 animate-spin text-primary" />
                            )}
                            {fileData.status === 'done' && (
                              <span className="text-xs text-green-500">✓</span>
                            )}
                            {fileData.status === 'error' && (
                              <span className="text-xs text-red-500">Error</span>
                            )}
                          </div>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-white/70 hover:text-white"
                            onClick={() => removeFile(index)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>

        {/* Submit Button */}
        <Button
          type="submit"
          size="lg"
          className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-6"
          disabled={submitting}
        >
          {submitting ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Submitting...
            </>
          ) : (
            'Submit Quote Request'
          )}
        </Button>
      </form>

      {/* Footer Feature Badges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8 pt-8 border-t border-white/20">
        <div className="text-center">
          <Settings className="h-6 w-6 mx-auto mb-2 text-primary" />
          <p className="text-sm font-medium text-white">Tight-tolerance CNC</p>
          <p className="text-xs text-white/60">Precision machining to ±0.0001"</p>
        </div>
        <div className="text-center">
          <Award className="h-6 w-6 mx-auto mb-2 text-primary" />
          <p className="text-sm font-medium text-white">35+ years experience</p>
          <p className="text-xs text-white/60">Decades of precision mfg</p>
        </div>
        <div className="text-center">
          <Clock className="h-6 w-6 mx-auto mb-2 text-primary" />
          <p className="text-sm font-medium text-white">95% on-time delivery</p>
          <p className="text-xs text-white/60">Reliable delivery when you need it</p>
        </div>
        <div className="text-center">
          <Shield className="h-6 w-6 mx-auto mb-2 text-primary" />
          <p className="text-sm font-medium text-white">NDA on request</p>
          <p className="text-xs text-white/60">Confidential by default</p>
        </div>
      </div>
    </div>
  );
}
