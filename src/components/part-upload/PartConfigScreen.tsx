import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ChevronLeft, Mail, Phone, Building2, MapPin, User, Loader2, PanelLeftClose, PanelLeftOpen, FileText } from "lucide-react";
import { CADViewer } from "@/components/CADViewer";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";

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
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
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
    <div className="min-h-screen bg-muted/30">
      <ResizablePanelGroup direction="horizontal" className="h-screen">
        {/* Collapsible Left Sidebar - Compact Configuration */}
        {!isSidebarCollapsed && (
          <>
            <ResizablePanel defaultSize={25} minSize={18} maxSize={35} className="bg-background">
              <div className="h-full overflow-y-auto">
                <div className="p-4 space-y-4">
                  {/* Header with collapse button */}
                  <div className="flex items-center justify-between">
                    <Button variant="ghost" size="sm" onClick={onBack} className="flex-1 justify-start">
                      <ChevronLeft className="w-4 h-4 mr-1" />
                      Back
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setIsSidebarCollapsed(true)}
                      title="Collapse sidebar"
                    >
                      <PanelLeftClose className="h-4 w-4" />
                    </Button>
                  </div>

                  {/* Compact Configuration Section */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-foreground">Configuration</h3>
                    
                    {/* Material Selection */}
                    <div className="space-y-1">
                      <Label htmlFor="material" className="text-xs">Material *</Label>
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

                    {/* Quantity */}
                    <div className="space-y-1">
                      <Label htmlFor="quantity" className="text-xs">Quantity *</Label>
                      <Input
                        id="quantity"
                        type="number"
                        min="1"
                        className="h-9"
                        value={selectedFile.quantity}
                        onChange={(e) => onUpdateFile(selectedFileIndex, { quantity: parseInt(e.target.value) || 1 })}
                      />
                    </div>

                    {/* Process (Optional) */}
                    <div className="space-y-1">
                      <Label htmlFor="process" className="text-xs">Preferred Process</Label>
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
                  </div>

                  {/* Multiple files indicator */}
                  {files.length > 1 && (
                    <div className="pt-2 border-t">
                      <p className="text-xs text-muted-foreground mb-2">
                        {files.length} files uploaded
                      </p>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {files.map((file, index) => (
                          <button
                            key={index}
                            onClick={() => onSelectFile(index)}
                            className={`w-full p-2 text-left rounded text-xs transition-colors flex items-center gap-2 ${
                              index === selectedFileIndex
                                ? "bg-primary/10 text-primary"
                                : "hover:bg-muted"
                            }`}
                          >
                            <FileText className="w-3 h-3 flex-shrink-0" />
                            <span className="truncate">{file.file.name}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Submit Button */}
                  <Button 
                    onClick={handleSubmit} 
                    disabled={!isFormValid() || isSubmitting} 
                    className="w-full" 
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
            </ResizablePanel>

            <ResizableHandle withHandle />
          </>
        )}

        {/* Right Panel - 3D Viewer + Contact Form */}
        <ResizablePanel defaultSize={75} className="bg-background">
          <div className="h-full flex flex-col overflow-y-auto">
            {/* Part Name Header */}
            <div className="border-b px-6 py-3 flex items-center justify-between bg-muted/30">
              {isSidebarCollapsed && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setIsSidebarCollapsed(false)}
                        className="mr-3"
                      >
                        <PanelLeftOpen className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Show Configuration</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
              <h2 className="font-medium text-foreground truncate flex-1">
                {selectedFile.file.name}
              </h2>
              {selectedFile.meshData && (
                <span className="text-xs text-green-600 ml-2">● 3D Ready</span>
              )}
            </div>

            {/* CAD Viewer Area */}
            <div className="flex-1 min-h-[400px] p-4">
              <div className="h-full rounded-lg border bg-muted/20 overflow-hidden">
                {selectedFile.meshData ? (
                  <CADViewer
                    meshData={selectedFile.meshData}
                    fileName={selectedFile.file.name}
                    isSidebarCollapsed={isSidebarCollapsed}
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

            {/* Contact Information - Always Visible Below CAD Viewer */}
            <div className="border-t bg-background p-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Contact Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
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

                  {/* Row 3: Notes */}
                  <div className="space-y-1">
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
                </CardContent>
              </Card>
            </div>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
};

export default PartConfigScreen;
