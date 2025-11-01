import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, File, X, Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface FileUploadScreenProps {
  files: Array<{ file: File; isAnalyzing?: boolean }>;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveFile: (index: number) => void;
  onContinue: () => void;
  isAnalyzing: boolean;
}

export const FileUploadScreen = ({
  files,
  onFileSelect,
  onRemoveFile,
  onContinue,
  isAnalyzing
}: FileUploadScreenProps) => {
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
              {files.map((fileItem, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-muted/50 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <File className="h-5 w-5 text-primary" />
                    <div>
                      <p className="font-medium">{fileItem.file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(fileItem.file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {fileItem.isAnalyzing && (
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
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
              ))}
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
        </CardContent>
      </Card>
    </div>
  );
};
