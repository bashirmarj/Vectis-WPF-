import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { CheckCircle2, XCircle, AlertTriangle, Download } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ValidationReportProps {
  report: any;
}

export const ValidationReport = ({ report }: ValidationReportProps) => {
  if (!report) return null;

  const getCheckIcon = (passed: boolean) => {
    return passed ? (
      <CheckCircle2 className="h-4 w-4 text-green-600" />
    ) : (
      <XCircle className="h-4 w-4 text-destructive" />
    );
  };

  const getPassRateColor = (rate: number) => {
    if (rate >= 90) return "text-green-600";
    if (rate >= 70) return "text-yellow-600";
    return "text-destructive";
  };

  const downloadReport = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `validation-report-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const totalChecks = report.summary?.total_checks || 0;
  const passedChecks = report.summary?.passed || 0;
  const failedChecks = report.summary?.failed || 0;
  const passRate = totalChecks > 0 ? (passedChecks / totalChecks) * 100 : 0;

  // Group checks by category
  const groupedChecks = report.checks ? report.checks.reduce((acc: any, check: any) => {
    const category = check.category || 'Other';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(check);
    return acc;
  }, {}) : {};

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl">Validation Summary</CardTitle>
              <CardDescription>
                Overall validation results against Analysis Situs ground truth
              </CardDescription>
            </div>
            <Button onClick={downloadReport} variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Download JSON
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <p className="text-3xl font-bold">{totalChecks}</p>
              <p className="text-sm text-muted-foreground">Total Checks</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-3xl font-bold text-green-600">{passedChecks}</p>
              <p className="text-sm text-muted-foreground">Passed</p>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-lg">
              <p className="text-3xl font-bold text-destructive">{failedChecks}</p>
              <p className="text-sm text-muted-foreground">Failed</p>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Pass Rate</span>
              <span className={`text-2xl font-bold ${getPassRateColor(passRate)}`}>
                {passRate.toFixed(1)}%
              </span>
            </div>
            <Progress value={passRate} className="h-3" />
          </div>
        </CardContent>
      </Card>

      {/* Category Breakdown */}
      {Object.keys(groupedChecks).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Category Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(groupedChecks).map(([category, categoryChecks]: [string, any]) => {
                const passed = categoryChecks.filter((c: any) => c.passed).length;
                const total = categoryChecks.length;
                const categoryPassRate = total > 0 ? (passed / total) * 100 : 0;
                return (
                  <div key={category} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{category}</span>
                        <Badge variant={categoryPassRate >= 80 ? "default" : "destructive"}>
                          {passed}/{total}
                        </Badge>
                      </div>
                      <span className={`font-bold ${getPassRateColor(categoryPassRate)}`}>
                        {categoryPassRate.toFixed(0)}%
                      </span>
                    </div>
                    <Progress value={categoryPassRate} className="h-2" />
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Detailed Checks */}
      {Object.keys(groupedChecks).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Detailed Validation Checks</CardTitle>
            <CardDescription>
              Expand each section to see individual check results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion type="single" collapsible className="w-full">
              {Object.entries(groupedChecks).map(([category, categoryChecks]: [string, any]) => (
                <AccordionItem key={category} value={category.toLowerCase()}>
                  <AccordionTrigger>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{category} Checks</span>
                      <Badge variant={categoryChecks.every((c: any) => c.passed) ? "default" : "secondary"}>
                        {categoryChecks.filter((c: any) => c.passed).length}/{categoryChecks.length}
                      </Badge>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2">
                      {categoryChecks.map((check: any, idx: number) => (
                        <div key={idx} className="p-3 bg-muted rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              {getCheckIcon(check.passed)}
                              <span className="text-sm font-medium">{check.name}</span>
                            </div>
                            {!check.passed && check.deviation !== undefined && (
                              <Badge variant="destructive">
                                Deviation: {check.deviation}
                              </Badge>
                            )}
                          </div>
                          {(check.expected !== undefined || check.actual !== undefined) && (
                            <div className="flex items-center gap-4 text-xs text-muted-foreground ml-6">
                              {check.expected !== undefined && <span>Expected: {check.expected}</span>}
                              {check.actual !== undefined && <span>Actual: {check.actual}</span>}
                            </div>
                          )}
                          {check.message && (
                            <p className="text-xs text-muted-foreground mt-2 ml-6">{check.message}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>
      )}

      {/* Warnings */}
      {report.warnings && report.warnings.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-600" />
              <CardTitle>Warnings</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {report.warnings.map((warning: any, idx: number) => (
                <div key={idx} className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-yellow-900">{warning.message || warning}</p>
                    {warning.details && (
                      <p className="text-xs text-yellow-700 mt-1">{warning.details}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};