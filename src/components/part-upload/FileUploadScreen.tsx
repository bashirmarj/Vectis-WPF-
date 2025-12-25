import { useState, useCallback } from "react";
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
  onFileDrop?: (files: FileList) => void;
  onRemoveFile: (index: number) => void;
  onRetryFile?: (index: number) => void;
  onContinue: () => void;
  isAnalyzing: boolean;
  isSaving?: boolean;
  singleFileMode?: boolean;
  continueButtonText?: string;
}

export const FileUploadScreen = ({
  files,
  onFileSelect,
  onFileDrop,
  onRemoveFile,
  onRetryFile,
  onContinue,
  isAnalyzing,
  isSaving = false,
  singleFileMode = false,
  continueButtonText
}: FileUploadScreenProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const hasFailedFiles = files.some(f => !f.isAnalyzing && !f.analysis);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0 && onFileDrop) {
      onFileDrop(droppedFiles);
    }
  }, [onFileDrop]);

  // Hide upload area if in single file mode and already has a file
  const showUploadArea = !singleFileMode || files.length === 0;

  // Determine button text
  const buttonText = continueButtonText || (singleFileMode ? 'Continue to Configure' : 'Save & Continue to Dashboard');
  
  return (
    <div className="max-w-5xl mx-auto">
      {/* Light themed card - matches website beige theme */}
      <div className="backdrop-blur-sm border border-gray-300 rounded-sm overflow-hidden shadow-lg" style={{ backgroundColor: "rgba(245, 245, 242, 0.95)" }}>
        {/* Header */}
        <div className="p-6 pb-2">
          <h2 className="text-2xl font-bold text-gray-900">
            {singleFileMode ? 'Upload Your Part' : 'Upload Your Parts'}
          </h2>
          <p className="text-gray-600 mt-1">
            {singleFileMode 
              ? 'Upload a STEP or IGES file to get an instant preview and quote'
              : 'Upload STEP or IGES files to get instant quotes for custom manufacturing'
            }
          </p>
        </div>

        {/* Content */}
        <div className="p-6 pt-4 space-y-6">
          {/* File Upload Area - with drag-and-drop */}
          {showUploadArea && (
            <div 
              className={`border-2 border-dashed rounded-sm p-12 text-center transition-all duration-200 ${
                isDragging 
                  ? 'border-primary bg-primary/10 shadow-lg' 
                  : 'border-gray-300 hover:border-primary/50'
              }`}
              style={{ backgroundColor: isDragging ? "rgba(255, 255, 255, 0.9)" : "rgba(255, 255, 255, 0.7)" }}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <Upload className={`h-12 w-12 mx-auto mb-4 transition-colors ${isDragging ? 'text-primary' : 'text-gray-500'}`} />
              <p className="text-lg font-medium mb-2 text-gray-900">
                {isDragging ? 'Drop your file here' : `Drop your CAD file${singleFileMode ? '' : 's'} here`}
              </p>
              <p className="text-sm text-gray-600 mb-4">
                or click to browse (STEP, IGES files supported)
              </p>
              <input
                type="file"
                id="cad-files"
                accept=".step,.stp,.iges,.igs"
                multiple={!singleFileMode}
                onChange={onFileSelect}
                className="hidden"
              />
              <Button asChild variant="outline" className="border-primary text-primary hover:bg-primary hover:text-white">
                <label htmlFor="cad-files" className="cursor-pointer">
                  Select File{singleFileMode ? '' : 's'}
                </label>
              </Button>
            </div>
          )}

          {/* File List */}
          {files.length > 0 && (
            <div className="space-y-3">
              <h3 className="font-semibold text-gray-900">
                {singleFileMode ? 'Selected File' : `Uploaded Files (${files.length})`}
              </h3>
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
                      failed ? 'border-red-300 bg-red-50' : 'border-gray-200 bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1">
                        {failed ? (
                          <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                        ) : (
                          <File className="h-5 w-5 text-primary flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate text-gray-900">{fileItem.file.name}</p>
                          <p className="text-sm text-gray-600">
                            {(fileItem.file.size / 1024).toFixed(2)} KB
                            {fileItem.uploadSpeed && fileItem.isAnalyzing && (
                              <span className="ml-2">• {formatSpeed(fileItem.uploadSpeed)}</span>
                            )}
                            {failed && <span className="text-red-500 ml-2">• Preview unavailable</span>}
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
                            className="gap-2 border-gray-300 text-gray-700 hover:bg-gray-100"
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
                          className="text-gray-500 hover:text-gray-900 hover:bg-gray-100"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    {/* Progress bar and status */}
                    {fileItem.isAnalyzing && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-600">
                            {fileItem.analysisStatus || 'Processing...'}
                          </span>
                          <span className="font-medium text-primary">
                            {Math.round(totalProgress)}%
                          </span>
                        </div>
                        <Progress value={totalProgress} className="h-2 bg-gray-200" />
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
                disabled={isAnalyzing || isSaving || files.some(f => f.isAnalyzing)}
                className="bg-primary hover:bg-primary/90 text-white"
              >
                {isSaving ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Saving...
                  </>
                ) : isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  buttonText
                )}
              </Button>
            </div>
          )}

          {isAnalyzing && (
            <div className="border border-gray-200 rounded-sm p-4 bg-white">
              <p className="text-gray-700 text-sm">
                Analyzing your CAD file{singleFileMode ? '' : 's'}... This may take a moment.
              </p>
            </div>
          )}

          {hasFailedFiles && (
            <div className="border border-gray-200 rounded-sm p-4 flex items-start gap-3 bg-amber-50">
              <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <p className="text-gray-700 text-sm">
                Failed to load your step file(s). We will still receive your request and provide you with the quotation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
