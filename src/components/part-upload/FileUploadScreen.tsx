import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, File, X, Loader2, RefreshCw, AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";

interface FileUploadScreenProps {
  files: Array<{ 
    file: File; 
    isAnalyzing?: boolean;
    analysis?: any;
    uploadProgress?: number;
    analysisProgress?: number;
    uploadSpeed?: number;
    analysisStatus?: string;
  }>;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveFile: (index: number) => void;
  onRetryFile?: (index: number) => void;
  onContinue: () => void;
  isAnalyzing: boolean;
}

export const FileUploadScreen = ({
  files,
  onFileSelect,
  onRemoveFile,
  onRetryFile,
  onContinue,
  isAnalyzing
}: FileUploadScreenProps) => {
  const hasFailedFiles = files.some(f => !f.isAnalyzing && !f.analysis);
  
  return (
    <div className="max-w-4xl mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>Upload Your Parts</CardTitle>
          <CardDescription>
            Upload STEP or IGES files to get instant quotes for custom manufacturing
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload Area */}
          <div className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-primary/50 transition-colors">
            <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-lg font-medium mb-2">Drop your CAD files here</p>
            <p className="text-sm text-muted-foreground mb-4">
              or click to browse (STEP, IGES files supported)
            </p>
            <input
              type="file"
              id="cad-files"
              accept=".step,.stp,.iges,.igs"
              multiple
              onChange={onFileSelect}
              className="hidden"
            />
            <Button asChild variant="outline">
              <label htmlFor="cad-files" className="cursor-pointer">
                Select Files
              </label>
            </Button>
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="space-y-3">
              <h3 className="font-semibold">Uploaded Files ({files.length})</h3>
              {files.map((fileItem, index) => {
                const failed = !fileItem.isAnalyzing && !fileItem.analysis;
                const totalProgress = fileItem.uploadProgress !== undefined && fileItem.analysisProgress !== undefined
                  ? (fileItem.uploadProgress * 0.3 + fileItem.analysisProgress * 0.7)
                  : 0;
                
                const formatSpeed = (bytesPerSecond?: number) => {
                  if (!bytesPerSecond) return '';
                  const mbps = bytesPerSecond / (1024 * 1024);
                  return mbps >= 1 ? `${mbps.toFixed(1)} MB/s` : `${(bytesPerSecond / 1024).toFixed(1)} KB/s`;
                };

                return (
                  <div
                    key={index}
                    className={`p-4 rounded-lg space-y-3 ${
                      failed ? 'bg-destructive/10 border-2 border-destructive/30' : 'bg-muted/50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1">
                        {failed ? (
                          <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0" />
                        ) : (
                          <File className="h-5 w-5 text-primary flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">{fileItem.file.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {(fileItem.file.size / 1024).toFixed(2)} KB
                            {fileItem.uploadSpeed && fileItem.isAnalyzing && (
                              <span className="ml-2">• {formatSpeed(fileItem.uploadSpeed)}</span>
                            )}
                            {failed && <span className="text-destructive ml-2">• Analysis failed</span>}
                            {fileItem.analysis && <span className="text-green-600 ml-2">• Complete</span>}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {fileItem.isAnalyzing && (
                          <Loader2 className="h-4 w-4 animate-spin text-primary" />
                        )}
                        {failed && onRetryFile && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onRetryFile(index)}
                            className="gap-2"
                          >
                            <RefreshCw className="h-4 w-4" />
                            Retry
                          </Button>
                        )}
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => onRemoveFile(index)}
                          disabled={fileItem.isAnalyzing}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    {/* Progress bar and status */}
                    {fileItem.isAnalyzing && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">
                            {fileItem.analysisStatus || 'Processing...'}
                          </span>
                          <span className="font-medium text-primary">
                            {Math.round(totalProgress)}%
                          </span>
                        </div>
                        <Progress value={totalProgress} className="h-2" />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Continue Button */}
          {files.length > 0 && (
            <div className="flex justify-end pt-4">
              <Button
                size="lg"
                onClick={onContinue}
                disabled={isAnalyzing || files.some(f => f.isAnalyzing)}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Continue to Configuration'
                )}
              </Button>
            </div>
          )}

          {isAnalyzing && (
            <Alert>
              <AlertDescription>
                Analyzing your CAD files... This may take a moment.
              </AlertDescription>
            </Alert>
          )}

          {hasFailedFiles && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Some files failed to analyze. This may be due to the geometry service starting up. 
                Please use the "Retry" button to try again, or remove the failed files to continue.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
