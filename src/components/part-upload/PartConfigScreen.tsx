import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChevronLeft, Mail, Phone, Building2, MapPin, User, Loader2, FileText } from "lucide-react";
import { CADViewer } from "@/components/CADViewer";

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
    message: "",
  });

  const selectedFile = files[selectedFileIndex];

  const handleSubmit = () => {
    onSubmit({
      files: files,
      contact: contactInfo,
    });
  };

  const handleContactInfoChange = (field: string, value: string) => {
    setContactInfo((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const isFormValid = () => {
    const filesValid = files.every((f) => f.material && f.quantity > 0);
    const contactValid = contactInfo.name && contactInfo.email && contactInfo.phone;
    return filesValid && contactValid;
  };

  return (
    <div className="min-h-screen bg-muted/30 flex flex-col">
      {/* Header with Back Button */}
      <div className="border-b bg-background px-4 py-2 flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ChevronLeft className="w-4 h-4 mr-1" />
          Back to Upload
        </Button>
      </div>

      {/* CAD Viewer Section - Full Width */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Tab-style Header with Part Name */}
        <div className="bg-background border-b">
          <Tabs value={selectedFileIndex.toString()} onValueChange={(v) => onSelectFile(parseInt(v))}>
            <TabsList className="h-10 bg-transparent border-b-0 px-4">
              {files.length === 1 ? (
                <TabsTrigger 
                  value="0" 
                  className="data-[state=active]:bg-background data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-primary rounded-none"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  {selectedFile.file.name}
                </TabsTrigger>
              ) : (
                files.map((file, index) => (
                  <TabsTrigger 
                    key={index} 
                    value={index.toString()}
                    className="data-[state=active]:bg-background data-[state=active]:shadow-none border-b-2 border-transparent data-[state=active]:border-primary rounded-none"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    <span className="max-w-[150px] truncate">{file.file.name}</span>
                  </TabsTrigger>
                ))
              )}
            </TabsList>
          </Tabs>
        </div>

        {/* CAD Viewer Area */}
        <div className="flex-1 p-4 min-h-[350px]">
          <div className="h-full rounded-lg border bg-muted/20 overflow-hidden">
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

      {/* Configuration + Contact Information - Full Width Below CAD Viewer */}
      <div className="border-t bg-background p-4">
        <Card>
          <CardContent className="p-4 space-y-4">
            {/* Configuration Row */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
              <div className="space-y-1">
                <Label htmlFor="material" className="text-xs font-medium">Material *</Label>
                <Select
                  value={selectedFile.material || ""}
                  onValueChange={(value) => onUpdateFile(selectedFileIndex, { material: value })}
                >
                  <SelectTrigger id="material" className="h-9">
                    <SelectValue placeholder="Select material" />
                  </SelectTrigger>
                  <SelectContent>
                    {materials.map((material) => (
                      <SelectItem key={material} value={material}>
                        {material}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label htmlFor="quantity" className="text-xs font-medium">Quantity *</Label>
                <Input
                  id="quantity"
                  type="number"
                  min="1"
                  className="h-9"
                  value={selectedFile.quantity}
                  onChange={(e) => onUpdateFile(selectedFileIndex, { quantity: parseInt(e.target.value) || 1 })}
                />
              </div>

              <div className="space-y-1">
                <Label htmlFor="process" className="text-xs font-medium">Preferred Process</Label>
                <Select
                  value={selectedFile.process || "auto"}
                  onValueChange={(value) =>
                    onUpdateFile(selectedFileIndex, { process: value === "auto" ? undefined : value })
                  }
                >
                  <SelectTrigger id="process" className="h-9">
                    <SelectValue placeholder="Auto-select" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto-select</SelectItem>
                    {processes.map((process) => (
                      <SelectItem key={process} value={process}>
                        {process}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Spacer for alignment */}
              <div className="hidden md:block" />
            </div>

            {/* Divider */}
            <div className="border-t" />

            {/* Contact Information */}
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-foreground">Contact Information</h3>
              
              {/* Row 1: Name, Email, Phone */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="space-y-1">
                  <Label htmlFor="name" className="text-xs flex items-center gap-1">
                    <User className="w-3 h-3" />
                    Full Name *
                  </Label>
                  <Input
                    id="name"
                    className="h-9"
                    value={contactInfo.name}
                    onChange={(e) => handleContactInfoChange("name", e.target.value)}
                    placeholder="John Doe"
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
                    placeholder="john@example.com"
                  />
                </div>

                <div className="space-y-1">
                  <Label htmlFor="phone" className="text-xs flex items-center gap-1">
                    <Phone className="w-3 h-3" />
                    Phone *
                  </Label>
                  <Input
                    id="phone"
                    type="tel"
                    className="h-9"
                    value={contactInfo.phone}
                    onChange={(e) => handleContactInfoChange("phone", e.target.value)}
                    placeholder="+1 (555) 123-4567"
                  />
                </div>
              </div>

              {/* Row 2: Company, Address */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label htmlFor="company" className="text-xs flex items-center gap-1">
                    <Building2 className="w-3 h-3" />
                    Company
                  </Label>
                  <Input
                    id="company"
                    className="h-9"
                    value={contactInfo.company}
                    onChange={(e) => handleContactInfoChange("company", e.target.value)}
                    placeholder="Acme Corp"
                  />
                </div>

                <div className="space-y-1">
                  <Label htmlFor="address" className="text-xs flex items-center gap-1">
                    <MapPin className="w-3 h-3" />
                    Shipping Address
                  </Label>
                  <Input
                    id="address"
                    className="h-9"
                    value={contactInfo.address}
                    onChange={(e) => handleContactInfoChange("address", e.target.value)}
                    placeholder="123 Main St, City, State, ZIP"
                  />
                </div>
              </div>

              {/* Row 3: Notes + Submit */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
                <div className="md:col-span-2 space-y-1">
                  <Label htmlFor="message" className="text-xs">Additional Notes</Label>
                  <Textarea
                    id="message"
                    value={contactInfo.message}
                    onChange={(e) => handleContactInfoChange("message", e.target.value)}
                    placeholder="Any special requirements or notes..."
                    rows={2}
                    className="resize-none"
                  />
                </div>

                <div className="flex justify-end">
                  <Button 
                    onClick={handleSubmit} 
                    disabled={!isFormValid() || isSubmitting} 
                    className="w-full md:w-auto min-w-[180px]" 
                    size="default"
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
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PartConfigScreen;
