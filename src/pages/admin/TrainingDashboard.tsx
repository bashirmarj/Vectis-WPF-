import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { ArrowLeft, Download, Play, RefreshCw, Terminal } from "lucide-react";
import { StartTrainingDialog } from "@/components/admin/StartTrainingDialog";
import { TrainingJobCard } from "@/components/admin/TrainingJobCard";
import { useTrainingJobs } from "@/hooks/useTrainingJobs";

export default function TrainingDashboard() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [user, setUser] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showStartDialog, setShowStartDialog] = useState(false);
  const { jobs, loading: jobsLoading, refetch } = useTrainingJobs();

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      navigate('/auth');
      return;
    }

    setUser(user);

    const { data: roleData } = await supabase
      .from('user_roles')
      .select('role')
      .eq('user_id', user.id)
      .eq('role', 'admin')
      .single();

    if (!roleData) {
      navigate('/');
      return;
    }

    setIsAdmin(true);
    setLoading(false);
  };

  const handleRefresh = () => {
    refetch();
    toast({
      title: "Refreshed",
      description: "Training jobs updated",
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse">Loading...</div>
      </div>
    );
  }

  if (!isAdmin) {
    return null;
  }

  const pendingJobs = jobs.filter(j => j.status === 'pending' || j.status === 'pending_local');
  const runningJobs = jobs.filter(j => j.status === 'running');
  const completedJobs = jobs.filter(j => j.status === 'completed');
  const failedJobs = jobs.filter(j => j.status === 'failed');

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      
      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              onClick={() => navigate('/admin')}
              className="gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Admin
            </Button>
            <div>
              <h1 className="text-3xl font-bold">ML Training Dashboard</h1>
              <p className="text-muted-foreground">Manage and monitor UV-Net training jobs</p>
            </div>
          </div>
          
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={handleRefresh}
              className="gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
            <Button
              onClick={() => setShowStartDialog(true)}
              className="gap-2"
            >
              <Play className="h-4 w-4" />
              Start Training
            </Button>
          </div>
        </div>

        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Pending</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{pendingJobs.length}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Running</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{runningJobs.length}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Completed</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{completedJobs.length}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Failed</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{failedJobs.length}</div>
            </CardContent>
          </Card>
        </div>

        {/* Training Jobs List */}
        <Card>
          <CardHeader>
            <CardTitle>Training Jobs</CardTitle>
            <CardDescription>
              All training jobs ordered by creation date
            </CardDescription>
          </CardHeader>
          <CardContent>
            {jobsLoading ? (
              <div className="text-center py-8 text-muted-foreground">
                Loading training jobs...
              </div>
            ) : jobs.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No training jobs yet. Click "Start Training" to begin.
              </div>
            ) : (
              <div className="space-y-4">
                {jobs.map((job) => (
                  <TrainingJobCard key={job.id} job={job} onRefresh={refetch} />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      <Footer />

      <StartTrainingDialog
        open={showStartDialog}
        onOpenChange={setShowStartDialog}
        onSuccess={refetch}
      />
    </div>
  );
}
