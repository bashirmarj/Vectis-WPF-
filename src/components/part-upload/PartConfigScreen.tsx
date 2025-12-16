import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { ChevronLeft, Mail, Phone, Building2, User, Loader2, Minus, Plus } from "lucide-react";
import { CADViewer } from "@/components/CADViewer";
import { AddressAutocomplete } from "@/components/ui/address-autocomplete";

// Material options matching the screenshots
const MATERIALS = [
  { value: "aluminum-6061", label: "Aluminum (6061)" },
  { value: "stainless-steel-304l-316l", label: "Stainless Steel (304L / 316L)" },
  { value: "alloy-steel-4140", label: "Alloy Steel (4140)" },
  { value: "carbon-steel-1018-1045", label: "Carbon Steel (1018 / 1045)" },
  { value: "tool-steel-a2-d2-s7", label: "Tool Steel (A2 / D2 / S7)" },
  { value: "copper-alloy", label: "Copper Alloy (Brass / Bronze)" },
  { value: "plastic-delrin-uhmw", label: "Plastic (Delrin / UHMW)" },
  { value: "other", label: "Other" },
];

// Finish options matching the screenshots
const FINISHES = [
  { value: "as-machined", label: "As Machined" },
  { value: "anodizing-type-ii", label: "Anodizing (Type II)" },
  { value: "hard-anodizing-type-iii", label: "Hard Anodizing (Type III)" },
  { value: "passivation", label: "Passivation" },
  { value: "polishing", label: "Polishing" },
  { value: "other", label: "Other" },
];

interface FileWithData {
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
    face_mapping?: Record<number, { triangle_indices: number[]; triangle_range: [number, number] }>;
    vertex_face_ids?: number[];
    tagged_edges?: Array<{
      feature_id: number;
      start: [number, number, number];
      end: [number, number, number];
      type: "arc" | "circle" | "line";
      iso_type?: string;
      diameter?: number;
      radius?: number;
      length?: number;
    }>;
    edge_classifications?: any[];
    face_classifications?: any[];
    geometric_features?: any;
  };
  analysis?: {
    volume_cm3?: number;
    surface_area_cm2?: number;
    complexity_score?: number;
    confidence?: number;
    method?: string;
    geometric_features?: any;
    recommended_processes?: string[];
    routing_reasoning?: string[];
    machining_summary?: any[];
  };
  quote?: any;
  isAnalyzing?: boolean;
}

interface PartConfigScreenProps {
  files: FileWithData[];
  materials: string[];
  processes: string[];
  onBack: () => void;
  onSubmit: (formData: any) => void;
  onUpdateFile: (index: number, updates: Partial<FileWithData>) => void;
  selectedFileIndex: number;
  onSelectFile: (index: number) => void;
  isSubmitting: boolean;
}

const PartConfigScreen: React.FC<PartConfigScreenProps> = ({
  files,
  materials,
  processes,
  onBack,
  onSubmit,
  onUpdateFile,
  selectedFileIndex,
  onSelectFile,
  isSubmitting,
}) => {
  const [contactInfo, setContactInfo] = useState({
    name: "",
    email: "",
    phone: "",
    company: "",
    address: "",
    projectDescription: "",
  });

  const [partDetails, setPartDetails] = useState({
    partName: "",
    material: "",
    finish: "",
    heatTreatment: false,
    heatTreatmentDetails: "",
    threadsTolerances: "",
  });

  const selectedFile = files[selectedFileIndex];

  const handleSubmit = () => {
    onSubmit({
      files: files,
      contact: contactInfo,
      partDetails: partDetails,
    });
  };

  const handleContactInfoChange = (field: string, value: string) => {
    setContactInfo((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handlePartDetailsChange = (field: string, value: string | boolean) => {
    setPartDetails((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleQuantityChange = (delta: number) => {
    const newQty = Math.max(1, selectedFile.quantity + delta);
    onUpdateFile(selectedFileIndex, { quantity: newQty });
  };

  const isFormValid = () => {
    const contactValid = contactInfo.name && contactInfo.email && contactInfo.phone && contactInfo.company && contactInfo.projectDescription;
    return contactValid;
  };

  return (
    <div className="max-w-5xl mx-auto">
      {/* CAD Viewer Section */}
      <div className="pt-4">
        {/* Compact Header: Back Button + Part Name */}
        <div 
          className="rounded-t-lg py-2 px-3 flex items-center justify-between backdrop-blur-sm border border-white/10 shadow-[0_0_30px_rgba(255,255,255,0.15)]"
          style={{ backgroundColor: 'rgba(60, 60, 60, 0.75)' }}
        >
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onBack}
            className="text-gray-300 hover:text-white hover:bg-white/10 h-7 px-2"
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            Back to Upload
          </Button>
          <span className="text-sm font-medium text-white">
            Part Name: {selectedFile.file.name.replace(/\.[^/.]+$/, "")}
          </span>
        </div>

        {/* CAD Viewer Area */}
        <div className="h-[480px]">
          <div className="h-full rounded-b-lg border border-t-0 border-white/10 bg-muted/20 overflow-hidden">
            {selectedFile.meshData ? (
              <CADViewer
                meshData={selectedFile.meshData}
                fileName={selectedFile.file.name}
                isSidebarCollapsed={false}
                onMeshLoaded={(meshData) => {
                  console.log("✅ Mesh loaded successfully:", {
                    triangles: meshData.triangle_count,
                    hasColors: !!meshData.vertex_colors,
                  });
                }}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-6">
                {selectedFile.isAnalyzing ? (
                  <>
                    <Loader2 className="h-10 w-10 animate-spin text-primary" />
                    <div>
                      <p className="text-sm font-medium text-foreground">Analyzing CAD file...</p>
                      <p className="text-xs text-muted-foreground mt-1">This may take 30-60 seconds</p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-5xl">📦</div>
                    <div>
                      <p className="text-sm font-medium text-foreground">Preview not available</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        You can still submit your quote request
                      </p>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Form Sections Below CAD Viewer */}
      <div className="py-4 space-y-4">
        {/* PROJECT DETAILS Section */}
        <Card>
          <CardContent className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-foreground uppercase tracking-wide">Project Details</h3>
              <span className="text-xs text-muted-foreground">Faster quotes with CAD</span>
            </div>

            {/* Row 1: Name, Company */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="name" className="text-xs flex items-center gap-1">
                  <User className="w-3 h-3" />
                  Name *
                </Label>
                <Input
                  id="name"
                  className="h-9"
                  value={contactInfo.name}
                  onChange={(e) => handleContactInfoChange("name", e.target.value)}
                  placeholder="Jane Smith"
                />
              </div>

              <div className="space-y-1">
                <Label htmlFor="company" className="text-xs flex items-center gap-1">
                  <Building2 className="w-3 h-3" />
                  Company Name *
                </Label>
                <Input
                  id="company"
                  className="h-9"
                  value={contactInfo.company}
                  onChange={(e) => handleContactInfoChange("company", e.target.value)}
                  placeholder="Acme Manufacturing Ltd."
                />
              </div>
            </div>

            {/* Row 2: Phone, Email */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="phone" className="text-xs flex items-center gap-1">
                  <Phone className="w-3 h-3" />
                  Phone Number *
                </Label>
                <Input
                  id="phone"
                  type="tel"
                  className="h-9"
                  value={contactInfo.phone}
                  onChange={(e) => handleContactInfoChange("phone", e.target.value)}
                  placeholder="(555) 123-4567"
                />
              </div>

              <div className="space-y-1">
                <Label htmlFor="email" className="text-xs flex items-center gap-1">
                  <Mail className="w-3 h-3" />
                  Email *
                </Label>
                <Input
                  id="email"
                  type="email"
                  className="h-9"
                  value={contactInfo.email}
                  onChange={(e) => handleContactInfoChange("email", e.target.value)}
                  placeholder="you@company.com"
                />
              </div>
            </div>

            {/* Row 3: Address with Autocomplete */}
            <div className="space-y-1">
              <Label htmlFor="address" className="text-xs">Address</Label>
              <AddressAutocomplete
                value={contactInfo.address}
                onChange={(value) => handleContactInfoChange("address", value)}
                placeholder="Start typing for suggestions..."
              />
            </div>

            {/* Row 4: Project Description */}
            <div className="space-y-1">
              <Label htmlFor="projectDescription" className="text-xs">Project Description *</Label>
              <Textarea
                id="projectDescription"
                value={contactInfo.projectDescription}
                onChange={(e) => handleContactInfoChange("projectDescription", e.target.value)}
                placeholder="Briefly describe the part and requirements (material, qty, finish, critical dims, deadline)."
                rows={3}
                className="resize-none"
              />
              <p className="text-xs text-muted-foreground">If no drawing, a short description is fine.</p>
            </div>
          </CardContent>
        </Card>

        {/* PART DETAILS (Optional) - Always Visible */}
        <Card>
          <CardContent className="p-4 space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-foreground uppercase tracking-wide">Part Details (Optional)</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Provide additional details to help us quote accurately</p>
            </div>

            {/* Part / Job name */}
            <div className="space-y-1">
              <Label htmlFor="partName" className="text-xs">Part / Job name</Label>
              <Input
                id="partName"
                className="h-9"
                value={partDetails.partName}
                onChange={(e) => handlePartDetailsChange("partName", e.target.value)}
                placeholder="e.g., Shaft Holder Rev B"
              />
            </div>

            {/* Material + Quantity */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="material" className="text-xs">Material</Label>
                <Select
                  value={partDetails.material}
                  onValueChange={(value) => handlePartDetailsChange("material", value)}
                >
                  <SelectTrigger className="h-9">
                    <SelectValue placeholder="Select Material" />
                  </SelectTrigger>
                  <SelectContent>
                    {MATERIALS.map((mat) => (
                      <SelectItem key={mat.value} value={mat.value}>{mat.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label className="text-xs">Quantity</Label>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-9 w-9"
                    onClick={() => handleQuantityChange(-1)}
                    disabled={selectedFile.quantity <= 1}
                  >
                    <Minus className="w-4 h-4" />
                  </Button>
                  <Input
                    type="number"
                    min="1"
                    className="h-9 text-center w-20"
                    value={selectedFile.quantity}
                    onChange={(e) => onUpdateFile(selectedFileIndex, { quantity: parseInt(e.target.value) || 1 })}
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-9 w-9"
                    onClick={() => handleQuantityChange(1)}
                  >
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Finish + Heat Treatment */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="finish" className="text-xs">Finish</Label>
                <Select
                  value={partDetails.finish}
                  onValueChange={(value) => handlePartDetailsChange("finish", value)}
                >
                  <SelectTrigger className="h-9">
                    <SelectValue placeholder="Select Finish" />
                  </SelectTrigger>
                  <SelectContent>
                    {FINISHES.map((finish) => (
                      <SelectItem key={finish.value} value={finish.value}>{finish.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label className="text-xs">Heat Treatment</Label>
                <div className="flex items-center space-x-2 h-9">
                  <Checkbox
                    id="heatTreatment"
                    checked={partDetails.heatTreatment}
                    onCheckedChange={(checked) => handlePartDetailsChange("heatTreatment", !!checked)}
                  />
                  <label
                    htmlFor="heatTreatment"
                    className="text-sm text-muted-foreground cursor-pointer"
                  >
                    Required
                  </label>
                </div>
              </div>
            </div>

            {/* Conditional Heat Treatment Details */}
            {partDetails.heatTreatment && (
              <div className="space-y-1">
                <Label htmlFor="heatTreatmentDetails" className="text-xs">Heat Treatment Details</Label>
                <Input
                  id="heatTreatmentDetails"
                  className="h-9"
                  value={partDetails.heatTreatmentDetails}
                  onChange={(e) => handlePartDetailsChange("heatTreatmentDetails", e.target.value)}
                  placeholder="e.g., Hardened to 58-62 HRC, Case Hardened 0.030&quot;"
                />
              </div>
            )}

            {/* Threads / Tolerances */}
            <div className="space-y-1">
              <Label htmlFor="threadsTolerances" className="text-xs">Threads / Tolerances</Label>
              <Textarea
                id="threadsTolerances"
                value={partDetails.threadsTolerances}
                onChange={(e) => handlePartDetailsChange("threadsTolerances", e.target.value)}
                placeholder="e.g., ±0.001&quot;, 1/4-20 UNC, Ø10 H7, true position 0.05"
                rows={2}
                className="resize-none"
              />
            </div>
          </CardContent>
        </Card>

        {/* Submit Button */}
        <div className="flex justify-center pt-2">
          <Button 
            onClick={handleSubmit} 
            disabled={!isFormValid() || isSubmitting} 
            className="w-full md:w-auto min-w-[200px]" 
            size="lg"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Submitting...
              </>
            ) : (
              "Submit Quote Request"
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default PartConfigScreen;
