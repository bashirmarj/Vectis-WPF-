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

  const totalChecks = report.total_checks || 0;
  const passedChecks = report.passed_checks || 0;
  const failedChecks = report.failed_checks || 0;
  const passRate = totalChecks > 0 ? (passedChecks / totalChecks) * 100 : 0;

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
      {report.categories && (
        <Card>
          <CardHeader>
            <CardTitle>Category Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(report.categories).map(([category, data]: [string, any]) => {
                const categoryPassRate = data.total > 0 ? (data.passed / data.total) * 100 : 0;
                return (
                  <div key={category} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{category}</span>
                        <Badge variant={categoryPassRate >= 80 ? "default" : "destructive"}>
                          {data.passed}/{data.total}
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
      {report.checks && (
        <Card>
          <CardHeader>
            <CardTitle>Detailed Validation Checks</CardTitle>
            <CardDescription>
              Expand each section to see individual check results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion type="single" collapsible className="w-full">
              {/* Structure Checks */}
              {report.checks.structure && (
                <AccordionItem value="structure">
                  <AccordionTrigger>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Structure Checks</span>
                      <Badge variant={report.checks.structure.every((c: any) => c.passed) ? "default" : "secondary"}>
                        {report.checks.structure.filter((c: any) => c.passed).length}/{report.checks.structure.length}
                      </Badge>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2">
                      {report.checks.structure.map((check: any, idx: number) => (
                        <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                          <div className="flex items-center gap-2">
                            {getCheckIcon(check.passed)}
                            <span className="text-sm">{check.name}</span>
                          </div>
                          {check.message && (
                            <span className="text-xs text-muted-foreground">{check.message}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              )}

              {/* Count Checks */}
              {report.checks.counts && (
                <AccordionItem value="counts">
                  <AccordionTrigger>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Feature Count Checks</span>
                      <Badge variant={report.checks.counts.every((c: any) => c.passed) ? "default" : "secondary"}>
                        {report.checks.counts.filter((c: any) => c.passed).length}/{report.checks.counts.length}
                      </Badge>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2">
                      {report.checks.counts.map((check: any, idx: number) => (
                        <div key={idx} className="p-3 bg-muted rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              {getCheckIcon(check.passed)}
                              <span className="text-sm font-medium">{check.name}</span>
                            </div>
                            {!check.passed && check.deviation !== undefined && (
                              <Badge variant="destructive">
                                {check.deviation > 0 ? '+' : ''}{check.deviation}
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground ml-6">
                            <span>Expected: {check.expected}</span>
                            <span>Actual: {check.actual}</span>
                            {check.deviation !== undefined && (
                              <span className={check.deviation === 0 ? 'text-green-600' : 'text-destructive'}>
                                Deviation: {check.deviation}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              )}

              {/* Parameter Checks */}
              {report.checks.parameters && report.checks.parameters.length > 0 && (
                <AccordionItem value="parameters">
                  <AccordionTrigger>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Parameter Checks</span>
                      <Badge variant={report.checks.parameters.every((c: any) => c.passed) ? "default" : "secondary"}>
                        {report.checks.parameters.filter((c: any) => c.passed).length}/{report.checks.parameters.length}
                      </Badge>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2">
                      {report.checks.parameters.map((check: any, idx: number) => (
                        <div key={idx} className="p-3 bg-muted rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              {getCheckIcon(check.passed)}
                              <span className="text-sm font-medium">{check.name}</span>
                            </div>
                            {check.tolerance && (
                              <Badge variant="outline">±{check.tolerance}%</Badge>
                            )}
                          </div>
                          {check.details && (
                            <p className="text-xs text-muted-foreground ml-6">{check.details}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              )}
            </Accordion>
          </CardContent>
        </Card>
      )}

      {/* Warnings */}
      {report.warnings && report.warnings.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-600" />
              Warnings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {report.warnings.map((warning: string, idx: number) => (
                <div key={idx} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-yellow-800">{warning}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
