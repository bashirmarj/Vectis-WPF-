import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, FileText, CheckCircle2, XCircle, Lock, ArrowLeft } from "lucide-react";
import { ValidationReport } from "@/components/validation/ValidationReport";

interface ValidationState {
  stepFile: File | null;
  asJsonFile: File | null;
  supplementaryFiles: File[];
  isValidating: boolean;
  validationReport: any | null;
  error: string | null;
}

const Validation = () => {
  const [state, setState] = useState<ValidationState>({
    stepFile: null,
    asJsonFile: null,
    supplementaryFiles: [],
    isValidating: false,
    validationReport: null,
    error: null,
  });
  
  const { user, loading } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleStepFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (!['step', 'stp', 'iges', 'igs'].includes(ext || '')) {
        toast({
          title: "Invalid file type",
          description: "Please upload a STEP or IGES file",
          variant: "destructive",
        });
        return;
      }
      setState(prev => ({ ...prev, stepFile: file, validationReport: null, error: null }));
    }
  };

  const handleAsJsonFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.json')) {
        toast({
          title: "Invalid file type",
          description: "Please upload a JSON file",
          variant: "destructive",
        });
        return;
      }
      setState(prev => ({ ...prev, asJsonFile: file, validationReport: null, error: null }));
    }
  };

  const handleSupplementaryFilesSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      const validFiles = files.filter(file => {
        const ext = file.name.split('.').pop()?.toLowerCase();
        return ext === 'json' || ext === 'txt';
      });

      if (validFiles.length !== files.length) {
        toast({
          title: "Invalid file types",
          description: "Only JSON and TXT files are accepted for supplementary features",
          variant: "destructive",
        });
      }

      setState(prev => ({ 
        ...prev, 
        supplementaryFiles: [...prev.supplementaryFiles, ...validFiles],
        validationReport: null, 
        error: null 
      }));
    }
  };

  const removeSupplementaryFile = (index: number) => {
    setState(prev => ({
      ...prev,
      supplementaryFiles: prev.supplementaryFiles.filter((_, i) => i !== index)
    }));
  };

  const mergeSupplementaryFeatures = async (
    baseGroundTruth: any,
    supplementaryFiles: File[]
  ): Promise<any> => {
    if (supplementaryFiles.length === 0) {
      return baseGroundTruth;
    }

    const merged = JSON.parse(JSON.stringify(baseGroundTruth)); // Deep clone
    const mergeStats: { [key: string]: number } = {};

    console.log('🔍 MERGE DEBUG: Base ground truth structure:', {
      has_parts: !!merged.parts,
      parts_length: merged.parts?.length,
      has_parts_0: !!merged.parts?.[0],
      has_bodies: !!merged.parts?.[0]?.bodies,
      bodies_length: merged.parts?.[0]?.bodies?.length,
      has_bodies_0: !!merged.parts?.[0]?.bodies?.[0],
      has_features: !!merged.parts?.[0]?.bodies?.[0]?.features,
    });

    // Get the correct nested location for features
    const targetLocation = merged.parts?.[0]?.bodies?.[0]?.features;
    
    if (!targetLocation) {
      console.error('❌ Cannot merge: features location not found in ground truth structure');
      toast({
        title: "Merge error",
        description: "Ground truth JSON doesn't have the expected structure (parts[0].bodies[0].features)",
        variant: "destructive",
      });
      return merged;
    }

    for (const file of supplementaryFiles) {
      try {
        const content = await file.text();
        const data = JSON.parse(content);

        console.log('🔍 MERGE DEBUG: Supplementary file structure:', {
          file_name: file.name,
          data_keys: Object.keys(data),
          has_filletChains: !!data.filletChains,
          filletChains_length: data.filletChains?.length,
        });

        // Auto-detect and merge known feature arrays
        const mergeableArrays = [
          'filletChains',
          'chamferChains',
          'threads',
          'holes',
          'pockets',
          'slots',
          'shoulders',
          'shafts'
        ];

        for (const arrayName of mergeableArrays) {
          if (Array.isArray(data[arrayName]) && data[arrayName].length > 0) {
            if (!targetLocation[arrayName]) {
              targetLocation[arrayName] = [];
            }
            const beforeCount = targetLocation[arrayName].length;
            targetLocation[arrayName].push(...data[arrayName]);
            const afterCount = targetLocation[arrayName].length;
            const addedCount = data[arrayName].length;
            
            mergeStats[arrayName] = (mergeStats[arrayName] || 0) + addedCount;
            
            console.log(`✅ Merged ${addedCount} ${arrayName} from ${file.name}`, {
              before: beforeCount,
              after: afterCount,
              added: addedCount,
              path: 'parts[0].bodies[0].features.' + arrayName,
            });
          }
        }
      } catch (error) {
        console.error(`Failed to merge ${file.name}:`, error);
        toast({
          title: "Merge warning",
          description: `Could not merge ${file.name}: ${error instanceof Error ? error.message : 'Invalid JSON'}`,
          variant: "destructive",
        });
      }
    }

    // Log final merged structure
    console.log('🔍 MERGE DEBUG: Final merged structure:', {
      has_parts: !!merged.parts,
      parts_length: merged.parts?.length,
      filletChains_length: merged.parts?.[0]?.bodies?.[0]?.features?.filletChains?.length,
      chamferChains_length: merged.parts?.[0]?.bodies?.[0]?.features?.chamferChains?.length,
    });

    // Show merge summary
    const mergedFeatures = Object.entries(mergeStats)
      .map(([feature, count]) => `${count} ${feature}`)
      .join(', ');
    
    if (mergedFeatures) {
      toast({
        title: "Supplementary features merged",
        description: `Added: ${mergedFeatures}`,
      });
    }

    return merged;
  };

  const runValidation = async () => {
    if (!state.stepFile || !state.asJsonFile) {
      toast({
        title: "Missing files",
        description: "Please upload both STEP and Analysis Situs JSON files",
        variant: "destructive",
      });
      return;
    }

    setState(prev => ({ ...prev, isValidating: true, error: null }));

    try {
      // Read AS JSON ground truth
      const asJsonText = await state.asJsonFile.text();
      let asGroundTruth = JSON.parse(asJsonText);

      // Merge supplementary feature files if any
      asGroundTruth = await mergeSupplementaryFeatures(asGroundTruth, state.supplementaryFiles);

      // Log what we're sending to the edge function
      console.log('🔍 CLIENT DEBUG: Sending to edge function:', {
        has_as_ground_truth: !!asGroundTruth,
        validation_mode: true,
        filletChains_in_features: asGroundTruth?.parts?.[0]?.bodies?.[0]?.features?.filletChains?.length,
      });

      // Upload STEP file to storage
      const fileExt = state.stepFile.name.split('.').pop()?.toLowerCase() || 'step';
      const fileName = `validation_${Date.now()}.${fileExt}`;
      const filePath = `validation-uploads/${fileName}`;

      const { error: uploadError } = await supabase.storage
        .from('cad-files')
        .upload(filePath, state.stepFile);

      if (uploadError) throw uploadError;

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from('cad-files')
        .getPublicUrl(filePath);

      // Call analyze-cad edge function with validation mode
      const { data, error } = await supabase.functions.invoke('analyze-cad', {
        body: {
          fileName: state.stepFile.name,
          fileUrl: publicUrl,
          as_ground_truth: asGroundTruth,
          validation_mode: true,
        },
      });

      if (error) throw error;

      // Debug logging
      console.log('🔍 Analyze-cad response:', {
        has_validation_report: !!data?.validation_report,
        validation_report_keys: data?.validation_report ? Object.keys(data.validation_report) : null,
        full_data_keys: data ? Object.keys(data) : null
      });

      setState(prev => ({
        ...prev,
        isValidating: false,
        validationReport: data?.validation_report,
      }));

      toast({
        title: "Validation complete",
        description: "Feature recognition validation has finished",
      });

    } catch (error: any) {
      console.error('Validation error:', error);
      setState(prev => ({
        ...prev,
        isValidating: false,
        error: error.message || 'Validation failed',
      }));
      toast({
        title: "Validation failed",
        description: error.message || "An error occurred during validation",
        variant: "destructive",
      });
    }
  };

  const resetValidation = () => {
    setState({
      stepFile: null,
      asJsonFile: null,
      supplementaryFiles: [],
      isValidating: false,
      validationReport: null,
      error: null,
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="container mx-auto px-4 py-24 flex items-center justify-center">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="container mx-auto px-4 py-24">
          <Card className="max-w-md mx-auto">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lock className="h-5 w-5" />
                Authentication Required
              </CardTitle>
              <CardDescription>
                Please sign in to access the validation tool
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={() => navigate('/auth')} className="w-full">
                Sign In
              </Button>
            </CardContent>
          </Card>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container mx-auto px-4 py-24">
        <div className="mb-6">
          <Button
            variant="ghost"
            onClick={() => navigate(-1)}
            className="mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          
          <h1 className="text-4xl font-bold mb-2">Feature Recognition Validation</h1>
          <p className="text-muted-foreground text-lg">
            Compare AAG recognition results against Analysis Situs ground truth
          </p>
        </div>

        {!state.validationReport ? (
          <div className="grid md:grid-cols-2 gap-6 max-w-4xl">
            {/* STEP File Upload */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  STEP File
                </CardTitle>
                <CardDescription>
                  Upload the CAD file to analyze
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                      state.stepFile
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:border-primary/50'
                    }`}
                  >
                    <input
                      type="file"
                      accept=".step,.stp,.iges,.igs"
                      onChange={handleStepFileSelect}
                      className="hidden"
                      id="step-upload"
                    />
                    <label htmlFor="step-upload" className="cursor-pointer">
                      {state.stepFile ? (
                        <div className="flex flex-col items-center gap-2">
                          <CheckCircle2 className="h-12 w-12 text-primary" />
                          <p className="font-medium">{state.stepFile.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {(state.stepFile.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-2">
                          <Upload className="h-12 w-12 text-muted-foreground" />
                          <p className="font-medium">Click to upload STEP file</p>
                          <p className="text-sm text-muted-foreground">
                            .step, .stp, .iges, .igs
                          </p>
                        </div>
                      )}
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Analysis Situs JSON Upload */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Ground Truth JSON
                </CardTitle>
                <CardDescription>
                  Upload Analysis Situs log file
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                      state.asJsonFile
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:border-primary/50'
                    }`}
                  >
                    <input
                      type="file"
                      accept=".json"
                      onChange={handleAsJsonFileSelect}
                      className="hidden"
                      id="json-upload"
                    />
                    <label htmlFor="json-upload" className="cursor-pointer">
                      {state.asJsonFile ? (
                        <div className="flex flex-col items-center gap-2">
                          <CheckCircle2 className="h-12 w-12 text-primary" />
                          <p className="font-medium">{state.asJsonFile.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {(state.asJsonFile.size / 1024).toFixed(2)} KB
                          </p>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-2">
                          <Upload className="h-12 w-12 text-muted-foreground" />
                          <p className="font-medium">Click to upload JSON file</p>
                          <p className="text-sm text-muted-foreground">
                            Analysis Situs format
                          </p>
                        </div>
                      )}
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Supplementary Features Upload (Optional) */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Supplementary Features (Optional)
                </CardTitle>
                <CardDescription>
                  Upload additional feature files (e.g., blends.txt, chamfers.txt) that will be automatically merged with the main ground truth
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                    <input
                      type="file"
                      onChange={handleSupplementaryFilesSelect}
                      className="hidden"
                      id="supplementary-upload"
                      accept=".json,.txt"
                      multiple
                    />
                    <label
                      htmlFor="supplementary-upload"
                      className="cursor-pointer flex flex-col items-center gap-2"
                    >
                      <Upload className="h-8 w-8 text-muted-foreground" />
                      <span className="text-sm font-medium">
                        Click to upload supplementary files
                      </span>
                      <span className="text-xs text-muted-foreground">
                        JSON or TXT files (multiple files supported)
                      </span>
                    </label>
                  </div>

                  {state.supplementaryFiles.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-sm font-medium">
                        {state.supplementaryFiles.length} supplementary file(s) selected:
                      </p>
                      {state.supplementaryFiles.map((file, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-2 bg-muted rounded-md"
                        >
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4 text-primary" />
                            <span className="text-sm">{file.name}</span>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeSupplementaryFile(index)}
                          >
                            <XCircle className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Run Validation Button */}
            <div className="md:col-span-2">
              <Card>
                <CardContent className="pt-6">
                  <Button
                    onClick={runValidation}
                    disabled={!state.stepFile || !state.asJsonFile || state.isValidating}
                    className="w-full"
                    size="lg"
                  >
                    {state.isValidating ? (
                      <>Processing validation...</>
                    ) : (
                      <>Run Validation</>
                    )}
                  </Button>
                  
                  {state.error && (
                    <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-lg flex items-start gap-2">
                      <XCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-destructive">{state.error}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">Validation Results</h2>
              <Button onClick={resetValidation} variant="outline">
                New Validation
              </Button>
            </div>
            
            <ValidationReport report={state.validationReport} />
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
};

export default Validation;
