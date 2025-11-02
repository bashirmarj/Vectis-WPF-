import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { Loader2, Eye, Search, Settings, Upload } from 'lucide-react';
import { StatusBadge } from '@/components/admin/StatusBadge';

interface QuotationSubmission {
  id: string;
  email: string;
  customer_name: string;
  customer_company: string | null;
  status: string;
  quote_number: string;
  submitted_at: string;
  created_at: string;
}

const Admin = () => {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [submissions, setSubmissions] = useState<QuotationSubmission[]>([]);
  const [filteredSubmissions, setFilteredSubmissions] = useState<QuotationSubmission[]>([]);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  useEffect(() => {
    const checkAdminAndFetchData = async () => {
      if (authLoading) return;

      if (!user) {
        navigate('/auth');
        return;
      }

      // Check if user has admin role
      const { data: roles, error: roleError } = await supabase
        .from('user_roles')
        .select('role')
        .eq('user_id', user.id)
        .eq('role', 'admin')
        .single();

      if (roleError || !roles) {
        toast({
          title: 'Access Denied',
          description: 'You do not have permission to access this page.',
          variant: 'destructive',
        });
        navigate('/');
        return;
      }

      setIsAdmin(true);

      // Fetch quotation submissions
      const { data, error } = await supabase
        .from('quotation_submissions')
        .select('*')
        .order('submitted_at', { ascending: false });

      if (error) {
        toast({
          title: 'Error',
          description: 'Failed to load submissions.',
          variant: 'destructive',
        });
      } else {
        setSubmissions(data || []);
        setFilteredSubmissions(data || []);
      }

      setLoading(false);
    };

    checkAdminAndFetchData();
  }, [user, authLoading, navigate, toast]);

  useEffect(() => {
    let filtered = submissions;

    // Filter by status
    if (statusFilter !== 'all') {
      filtered = filtered.filter(s => s.status === statusFilter);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(s =>
        s.customer_name.toLowerCase().includes(query) ||
        s.email.toLowerCase().includes(query) ||
        s.quote_number.toLowerCase().includes(query) ||
        (s.customer_company && s.customer_company.toLowerCase().includes(query))
      );
    }

    setFilteredSubmissions(filtered);
  }, [submissions, searchQuery, statusFilter]);

  if (authLoading || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!isAdmin) {
    return null;
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      <main className="flex-1 container mx-auto py-12 px-4">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Admin Dashboard</CardTitle>
                <CardDescription>View all quotation requests</CardDescription>
              </div>
              <div className="flex gap-2">
                <Button onClick={() => navigate('/admin/dataset-upload')} variant="outline">
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Dataset
                </Button>
                <Button onClick={() => navigate('/admin/pricing-settings')} variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  Pricing Settings
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {/* Admin Actions */}
            <div className="flex gap-2 mb-6">
              <Button
                variant="outline"
                onClick={() => navigate('/admin/pricing-settings')}
              >
                <Settings className="mr-2 h-4 w-4" />
                Pricing Settings
              </Button>
            </div>
            
            {/* Filters */}
            <div className="flex gap-4 mb-6">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by name, email, quote number, or company..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Filter by status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="reviewing">Reviewing</SelectItem>
                  <SelectItem value="quoted">Quoted</SelectItem>
                  <SelectItem value="accepted">Accepted</SelectItem>
                  <SelectItem value="rejected">Rejected</SelectItem>
                  <SelectItem value="expired">Expired</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {submissions.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No submissions yet.</p>
            ) : filteredSubmissions.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No submissions match your filters.</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Quote #</TableHead>
                    <TableHead>Customer</TableHead>
                    <TableHead>Company</TableHead>
                    <TableHead>Email</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Submitted</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredSubmissions.map((submission) => (
                    <TableRow key={submission.id}>
                      <TableCell className="font-mono font-medium">{submission.quote_number}</TableCell>
                      <TableCell>{submission.customer_name}</TableCell>
                      <TableCell>{submission.customer_company || '-'}</TableCell>
                      <TableCell>{submission.email}</TableCell>
                      <TableCell>
                        <StatusBadge status={submission.status as any} />
                      </TableCell>
                      <TableCell>{new Date(submission.submitted_at).toLocaleDateString()}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => navigate(`/admin/quotations/${submission.id}`)}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </main>
      <Footer />
    </div>
  );
};

export default Admin;
