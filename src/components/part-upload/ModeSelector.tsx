import { FileText, FolderOpen, Zap, Layout } from "lucide-react";

interface ModeSelectorProps {
  onSelectMode: (mode: "quick" | "project") => void;
}

export const ModeSelector = ({ onSelectMode }: ModeSelectorProps) => {
  return (
    <div className="max-w-5xl mx-auto">
      <div 
        className="backdrop-blur-sm border border-white/20 rounded-sm overflow-hidden shadow-[0_0_30px_rgba(255,255,255,0.15)]" 
        style={{ backgroundColor: "rgba(60, 60, 60, 0.75)" }}
      >
        <div className="p-6 pb-2">
          <h2 className="text-2xl font-bold text-white">How would you like to proceed?</h2>
          <p className="text-gray-400 mt-1">
            Choose the upload option that best fits your needs
          </p>
        </div>
        
        <div className="p-6 pt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Quick Quote Card */}
            <button
              onClick={() => onSelectMode("quick")}
              className="group p-6 rounded-sm border-2 border-white/20 hover:border-primary transition-all duration-300 text-left hover:shadow-[0_0_25px_rgba(255,255,255,0.2)]"
              style={{ backgroundColor: "rgba(50, 50, 50, 0.65)" }}
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-sm bg-primary/20 flex items-center justify-center group-hover:bg-primary/30 transition-colors">
                  <Zap className="w-6 h-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-primary transition-colors">
                    Quick Quote
                  </h3>
                  <ul className="space-y-2 text-sm text-gray-400">
                    <li className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-primary/60" />
                      Upload a single file
                    </li>
                    <li className="flex items-center gap-2">
                      <Layout className="w-4 h-4 text-primary/60" />
                      Instant 3D preview
                    </li>
                    <li className="flex items-center gap-2">
                      <Zap className="w-4 h-4 text-primary/60" />
                      Submit quote request directly
                    </li>
                  </ul>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-white/10">
                <span className="text-xs text-gray-500">
                  Best for: Quick quotes on individual parts
                </span>
              </div>
            </button>

            {/* Project Batch Card */}
            <button
              onClick={() => onSelectMode("project")}
              className="group p-6 rounded-sm border-2 border-white/20 hover:border-primary transition-all duration-300 text-left hover:shadow-[0_0_25px_rgba(255,255,255,0.2)]"
              style={{ backgroundColor: "rgba(50, 50, 50, 0.65)" }}
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-sm bg-primary/20 flex items-center justify-center group-hover:bg-primary/30 transition-colors">
                  <FolderOpen className="w-6 h-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-primary transition-colors">
                    Add to Project
                  </h3>
                  <ul className="space-y-2 text-sm text-gray-400">
                    <li className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-primary/60" />
                      Upload multiple files at once
                    </li>
                    <li className="flex items-center gap-2">
                      <FolderOpen className="w-4 h-4 text-primary/60" />
                      Organize in projects
                    </li>
                    <li className="flex items-center gap-2">
                      <Layout className="w-4 h-4 text-primary/60" />
                      Manage from dashboard
                    </li>
                  </ul>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-white/10">
                <span className="text-xs text-gray-500">
                  Best for: Managing multiple parts and projects
                </span>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
