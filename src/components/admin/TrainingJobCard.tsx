import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Download, Terminal, XCircle } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

interface TrainingJob {
  id: string;
  dataset_name: string;
  status: string;
  created_at: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  best_val_iou?: number;
  best_val_accuracy?: number;
  training_time_seconds?: number;
  model_path?: string;
  error_message?: string;
  logs?: any[];
  train_samples?: number;
  val_samples?: number;
  test_samples?: number;
}

export function TrainingJobCard({ job, onRefresh }: { job: TrainingJob; onRefresh: () => void }) {
  const { toast } = useToast();
  const [showLogs, setShowLogs] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'running':
        return 'bg-blue-500';
      case 'failed':
        return 'bg-red-500';
      case 'cancelled':
        return 'bg-gray-500';
      case 'pending':
      case 'pending_local':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'pending_local':
        return 'Waiting for Local Execution';
      case 'cancelled':
        return 'Cancelled';
      default:
        return status.charAt(0).toUpperCase() + status.slice(1);
    }
  };

  const handleDownloadModel = async () => {
    if (!job.model_path) return;

    try {
      const { data, error } = await supabase.storage
        .from('trained-models')
        .download(job.model_path);

      if (error) throw error;

      const url = window.URL.createObjectURL(data);
      const a = document.createElement('a');
      a.href = url;
      a.download = job.model_path.split('/').pop() || 'model.ckpt';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        title: "Download started",
        description: "Model file is being downloaded",
      });
    } catch (error) {
      toast({
        title: "Download failed",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  const handleCancelTraining = async () => {
    setIsCancelling(true);
    try {
      const { data, error } = await supabase.functions.invoke('cancel-training', {
        body: { job_id: job.id }
      });

      if (error) throw error;

      toast({
        title: "Training cancelled",
        description: data.message || "Training job has been cancelled successfully",
      });
      
      setShowCancelDialog(false);
      onRefresh();
    } catch (error) {
      toast({
        title: "Failed to cancel",
        description: error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsCancelling(false);
    }
  };

  const canCancel = ['pending', 'pending_local', 'running'].includes(job.status);

  const progress = job.logs && job.logs.length > 0
    ? Math.min(100, (job.logs.length / (job.epochs || 100)) * 100)
    : 0;

  return (
    <>
      <Card className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold">{job.dataset_name}</h3>
              <Badge className={getStatusColor(job.status)}>
                {getStatusLabel(job.status)}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Created {formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}
            </p>
          </div>

          <div className="flex gap-2">
            {canCancel && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => setShowCancelDialog(true)}
                className="gap-2"
              >
                <XCircle className="h-4 w-4" />
                Cancel
              </Button>
            )}
            {job.logs && job.logs.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowLogs(true)}
                className="gap-2"
              >
                <Terminal className="h-4 w-4" />
                Logs
              </Button>
            )}
            {job.status === 'completed' && job.model_path && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownloadModel}
                className="gap-2"
              >
                <Download className="h-4 w-4" />
                Download
              </Button>
            )}
          </div>
        </div>

        {/* Training Parameters */}
        <div className="grid grid-cols-3 gap-4 mb-3 text-sm">
          <div>
            <span className="text-muted-foreground">Epochs:</span>{' '}
            <span className="font-medium">{job.epochs}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Batch Size:</span>{' '}
            <span className="font-medium">{job.batch_size}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Learning Rate:</span>{' '}
            <span className="font-medium">{job.learning_rate}</span>
          </div>
        </div>

        {/* Progress Bar for Running Jobs */}
        {job.status === 'running' && (
          <div className="mb-3">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-muted-foreground">Training Progress</span>
              <span className="font-medium">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        )}

        {/* Metrics for Completed Jobs */}
        {job.status === 'completed' && (
          <div className="grid grid-cols-2 gap-4 p-3 bg-muted rounded-lg">
            <div>
              <p className="text-xs text-muted-foreground">Validation IoU</p>
              <p className="text-lg font-bold text-green-600">
                {job.best_val_iou ? (job.best_val_iou * 100).toFixed(2) + '%' : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Validation Accuracy</p>
              <p className="text-lg font-bold text-green-600">
                {job.best_val_accuracy ? (job.best_val_accuracy * 100).toFixed(2) + '%' : 'N/A'}
              </p>
            </div>
            {job.training_time_seconds && (
              <div>
                <p className="text-xs text-muted-foreground">Training Time</p>
                <p className="text-sm font-medium">
                  {Math.round(job.training_time_seconds / 60)} minutes
                </p>
              </div>
            )}
            {job.train_samples && (
              <div>
                <p className="text-xs text-muted-foreground">Samples</p>
                <p className="text-sm font-medium">
                  {job.train_samples} / {job.val_samples} / {job.test_samples}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Error Message for Failed Jobs */}
        {job.status === 'failed' && job.error_message && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-600">{job.error_message}</p>
          </div>
        )}

        {/* Cancelled Message */}
        {job.status === 'cancelled' && (
          <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <p className="text-sm text-gray-600">Training was cancelled</p>
          </div>
        )}
      </Card>

      {/* Logs Dialog */}
      <Dialog open={showLogs} onOpenChange={setShowLogs}>
        <DialogContent className="max-w-3xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle>Training Logs - {job.dataset_name}</DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-[60vh] w-full rounded-md border p-4 bg-black text-green-400 font-mono text-sm">
            {job.logs && job.logs.length > 0 ? (
              <div className="space-y-1">
                {job.logs.map((log, idx) => (
                  <div key={idx}>{JSON.stringify(log)}</div>
                ))}
              </div>
            ) : (
              <div className="text-muted-foreground">No logs available yet</div>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>

      {/* Cancel Confirmation Dialog */}
      <AlertDialog open={showCancelDialog} onOpenChange={setShowCancelDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel Training Job?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to cancel the training job for "{job.dataset_name}"? 
              This action cannot be undone. If the training is currently running, it will be stopped.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isCancelling}>No, keep training</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleCancelTraining}
              disabled={isCancelling}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isCancelling ? "Cancelling..." : "Yes, cancel training"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
