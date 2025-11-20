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
  isValidating: boolean;
  validationReport: any | null;
  error: string | null;
}

const Validation = () => {
  const [state, setState] = useState<ValidationState>({
    stepFile: null,
    asJsonFile: null,
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
      const asGroundTruth = JSON.parse(asJsonText);

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

      setState(prev => ({
        ...prev,
        isValidating: false,
        validationReport: data.validation_report || data,
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
