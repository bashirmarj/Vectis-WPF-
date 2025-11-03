import { useState } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { Copy, Terminal, AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface StartTrainingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

export function StartTrainingDialog({ open, onOpenChange, onSuccess }: StartTrainingDialogProps) {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [credentials, setCredentials] = useState<any>(null);
  const [trainingCommand, setTrainingCommand] = useState("");
  
  // Training parameters
  const [datasetName, setDatasetName] = useState("MFCAD");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(8);
  const [learningRate, setLearningRate] = useState(0.001);

  const fetchCredentials = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase.functions.invoke('get-training-credentials');
      
      if (error) throw error;
      
      setCredentials(data);
      toast({
        title: "Credentials Retrieved",
        description: "Follow the setup instructions below",
      });
    } catch (error) {
      toast({
        title: "Failed to get credentials",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const generateTrainingCommand = async () => {
    setLoading(true);
    try {
      // Create a training job record
      const { data: jobData, error: jobError } = await supabase
        .from('training_jobs')
        .insert({
          dataset_name: datasetName,
          status: 'pending_local',
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
        })
        .select()
        .single();

      if (jobError) throw jobError;

      const command = `python training/train_local.py --dataset_name "${datasetName}" --batch_size ${batchSize} --epochs ${epochs} --learning_rate ${learningRate} --job_id ${jobData.id}`;
      
      setTrainingCommand(command);
      onSuccess();
      
      toast({
        title: "Training Job Created",
        description: "Copy the command below to start training",
      });
    } catch (error) {
      toast({
        title: "Failed to create job",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied to clipboard",
      description: "Paste this into your terminal",
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Start Local Training</DialogTitle>
          <DialogDescription>
            Configure and start a UV-Net training job on your local RTX 3070
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="setup" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="setup">Setup Credentials</TabsTrigger>
            <TabsTrigger value="configure">Configure Training</TabsTrigger>
          </TabsList>

          <TabsContent value="setup" className="space-y-4">
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                First-time setup only. You'll need to configure your local environment with Supabase credentials.
              </AlertDescription>
            </Alert>

            {!credentials ? (
              <Card>
                <CardHeader>
                  <CardTitle>Step 1: Get Training Credentials</CardTitle>
                  <CardDescription>
                    Retrieve the credentials needed for local training
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button onClick={fetchCredentials} disabled={loading}>
                    {loading ? "Loading..." : "Get Credentials"}
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Step 2: Create .env File</CardTitle>
                    <CardDescription>
                      {credentials.instructions.step1}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-black text-green-400 p-4 rounded-md font-mono text-sm relative">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => copyToClipboard(credentials.instructions.content)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      <pre className="whitespace-pre-wrap">{credentials.instructions.content}</pre>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Step 3: Verify Setup</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-black text-green-400 p-4 rounded-md font-mono text-sm relative">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => copyToClipboard(credentials.instructions.step3)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      <pre>{credentials.instructions.step3}</pre>
                    </div>
                  </CardContent>
                </Card>

                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    {credentials.warning}
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </TabsContent>

          <TabsContent value="configure" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Training Configuration</CardTitle>
                <CardDescription>
                  Set parameters for your training job
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="dataset">Dataset Name</Label>
                    <Input
                      id="dataset"
                      value={datasetName}
                      onChange={(e) => setDatasetName(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="epochs">Epochs</Label>
                    <Input
                      id="epochs"
                      type="number"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value))}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="batchSize">Batch Size (RTX 3070: 8-12)</Label>
                    <Input
                      id="batchSize"
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="learningRate">Learning Rate</Label>
                    <Input
                      id="learningRate"
                      type="number"
                      step="0.0001"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    />
                  </div>
                </div>

                <Button 
                  onClick={generateTrainingCommand} 
                  disabled={loading}
                  className="w-full"
                >
                  {loading ? "Generating..." : "Generate Training Command"}
                </Button>
              </CardContent>
            </Card>

            {trainingCommand && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Terminal className="h-5 w-5" />
                    Run This Command
                  </CardTitle>
                  <CardDescription>
                    Copy and paste this into your terminal (in the project root directory)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="bg-black text-green-400 p-4 rounded-md font-mono text-sm relative">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2"
                      onClick={() => copyToClipboard(trainingCommand)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                    <pre className="whitespace-pre-wrap pr-12">{trainingCommand}</pre>
                  </div>
                  
                  <Alert className="mt-4">
                    <AlertDescription>
                      The training will automatically update the dashboard with real-time progress and results.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
