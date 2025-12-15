import { Button } from "@/components/ui/button";
import { Upload, File, X, Loader2, RefreshCw, AlertCircle } from "lucide-react";
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
      {/* Dark themed card - matches capabilities cards styling */}
      <div className="backdrop-blur-sm border border-white/10 rounded-sm overflow-hidden" style={{ backgroundColor: "rgba(5, 5, 5, 0.4)" }}>
        {/* Header */}
        <div className="p-6 pb-2">
          <h2 className="text-2xl font-bold text-white">Upload Your Parts</h2>
          <p className="text-gray-400 mt-1">
            Upload STEP or IGES files to get instant quotes for custom manufacturing
          </p>
        </div>

        {/* Content */}
        <div className="p-6 pt-4 space-y-6">
          {/* File Upload Area */}
          <div className="border-2 border-dashed border-white/10 rounded-sm p-12 text-center hover:border-primary/50 transition-colors" style={{ backgroundColor: "rgba(5, 5, 5, 0.3)" }}>
            <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
            <p className="text-lg font-medium mb-2 text-white">Drop your CAD files here</p>
            <p className="text-sm text-gray-400 mb-4">
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
            <Button asChild variant="outline" className="border-primary text-primary hover:bg-primary hover:text-white">
              <label htmlFor="cad-files" className="cursor-pointer">
                Select Files
              </label>
            </Button>
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="space-y-3">
              <h3 className="font-semibold text-white">Uploaded Files ({files.length})</h3>
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
                  className={`p-4 rounded-sm space-y-3 border ${
                      failed ? 'border-red-500/30' : 'border-white/10'
                    }`}
                  style={{ backgroundColor: failed ? "rgba(127, 29, 29, 0.2)" : "rgba(5, 5, 5, 0.3)" }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1">
                        {failed ? (
                          <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
                        ) : (
                          <File className="h-5 w-5 text-primary flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate text-white">{fileItem.file.name}</p>
                          <p className="text-sm text-gray-400">
                            {(fileItem.file.size / 1024).toFixed(2)} KB
                            {fileItem.uploadSpeed && fileItem.isAnalyzing && (
                              <span className="ml-2">• {formatSpeed(fileItem.uploadSpeed)}</span>
                            )}
                            {failed && <span className="text-red-400 ml-2">• Preview unavailable</span>}
                            {fileItem.analysis && <span className="text-green-400 ml-2">• Complete</span>}
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
                            className="gap-2 border-white/20 text-white hover:bg-white/10"
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
                          className="text-gray-400 hover:text-white hover:bg-white/10"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    {/* Progress bar and status */}
                    {fileItem.isAnalyzing && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">
                            {fileItem.analysisStatus || 'Processing...'}
                          </span>
                          <span className="font-medium text-primary">
                            {Math.round(totalProgress)}%
                          </span>
                        </div>
                        <Progress value={totalProgress} className="h-2 bg-gray-700" />
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
                className="bg-primary hover:bg-primary/90 text-white"
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
            <div className="border border-white/10 rounded-sm p-4" style={{ backgroundColor: "rgba(5, 5, 5, 0.3)" }}>
              <p className="text-gray-300 text-sm">
                Analyzing your CAD files... This may take a moment.
              </p>
            </div>
          )}

          {hasFailedFiles && (
            <div className="border border-white/10 rounded-sm p-4 flex items-start gap-3" style={{ backgroundColor: "rgba(5, 5, 5, 0.3)" }}>
              <AlertCircle className="h-5 w-5 text-yellow-400 flex-shrink-0 mt-0.5" />
              <p className="text-gray-300 text-sm">
                Failed to load your step file(s). We will still receive your request and provide you with the quotation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
