import { Download, FileText, ShoppingCart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ProjectPart } from '@/hooks/useProjectParts';

interface ReportPanelProps {
  parts: ProjectPart[];
  selectedIds: Set<string>;
}

export function ReportPanel({ parts, selectedIds }: ReportPanelProps) {
  const selectedParts = selectedIds.size > 0 
    ? parts.filter((p) => selectedIds.has(p.id)) 
    : parts;

  const totalQuantity = selectedParts.reduce((sum, p) => sum + p.quantity, 0);
  const totalPrice = selectedParts.reduce((sum, p) => sum + (p.subtotal || 0), 0);
  const quotedParts = selectedParts.filter((p) => p.status === 'quoted' || p.status === 'completed');

  const handleDownloadReport = () => {
    // TODO: Implement PDF generation
    console.log('Downloading report for', selectedParts.length, 'parts');
  };

  return (
    <div className="w-72 border-l bg-card p-4 flex flex-col gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <ShoppingCart className="h-4 w-4" />
            Quote Summary
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Parts</span>
            <span className="font-medium">{selectedParts.length}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Total Quantity</span>
            <span className="font-medium">{totalQuantity}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Quoted Parts</span>
            <span className="font-medium">{quotedParts.length}</span>
          </div>
          <Separator />
          <div className="flex justify-between">
            <span className="font-medium">Estimated Total</span>
            <span className="font-bold text-primary">
              ${totalPrice.toFixed(2)}
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Export Options
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button
            onClick={handleDownloadReport}
            className="w-full gap-2"
            disabled={selectedParts.length === 0}
          >
            <Download className="h-4 w-4" />
            Download Report
          </Button>
          <p className="text-xs text-muted-foreground text-center">
            {selectedIds.size > 0
              ? `Export ${selectedIds.size} selected part(s)`
              : `Export all ${parts.length} part(s)`}
          </p>
        </CardContent>
      </Card>

      <div className="mt-auto">
        <Card className="bg-primary/5 border-primary/20">
          <CardContent className="p-4">
            <p className="text-sm text-muted-foreground">
              Need help with your quotation? Contact our engineering team for assistance.
            </p>
            <Button variant="link" className="p-0 h-auto mt-2 text-primary">
              Contact Support →
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
