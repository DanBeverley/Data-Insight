import { useState, useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle2, FileText, X, BarChart3, ArrowRight, AlertTriangle, Download, Globe, FileEdit, Table, Presentation, File, Archive, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ReportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  reportPath: string;
  sessionId: string;
}

type StepStatus = 'pending' | 'active' | 'completed';

interface ProcessStep {
  id: string;
  label: string;
  status: StepStatus;
}

// --- Sub-Components ---

const DataProfileView = ({ data }: { data: any }) => {
  if (!data) return null;
  const qualityIcon = data.completeness >= 95 ? "✓" : data.completeness >= 80 ? "⚠" : "✗";

  return (
    <div className="bg-card/50 border border-border/50 rounded-lg p-4 mb-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-primary" />
        Data Profile
      </h3>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-background/50 p-3 rounded-md border border-border/30">
          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Rows</div>
          <div className="text-2xl font-mono font-bold">{data.rows?.toLocaleString() || 0}</div>
        </div>
        <div className="bg-background/50 p-3 rounded-md border border-border/30">
          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Columns</div>
          <div className="text-2xl font-mono font-bold">{data.columns || 0}</div>
        </div>
        <div className="bg-background/50 p-3 rounded-md border border-border/30">
          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Completeness</div>
          <div className={cn(
            "text-2xl font-mono font-bold flex items-center gap-2",
            data.completeness >= 95 ? "text-green-500" : data.completeness >= 80 ? "text-yellow-500" : "text-red-500"
          )}>
            {data.completeness?.toFixed(1)}% <span className="text-lg">{qualityIcon}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const ArtifactGalleryView = ({ data }: { data: any[] }) => {
  if (!data || data.length === 0) return null;

  return (
    <div className="space-y-6 mb-8">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <FileText className="h-5 w-5 text-primary" />
        Visual Analysis
      </h3>
      <div className="grid gap-6">
        {data.map((artifact, idx) => (
          <div key={idx} className="bg-card/50 border border-border/50 rounded-lg overflow-hidden">
            <div className="p-3 border-b border-border/30 bg-muted/20 font-medium text-sm">
              {artifact.filename}
            </div>
            <div className="p-4 bg-black/20 flex justify-center">
              <img
                src={artifact.url}
                alt={artifact.filename}
                className="max-h-[300px] rounded shadow-lg object-contain"
                onError={(e) => {
                  e.currentTarget.src = "https://placehold.co/600x400/1e293b/475569?text=Image+Not+Found";
                }}
              />
            </div>
            {artifact.explanation && (
              <div className="p-4 bg-muted/10 text-sm text-muted-foreground border-t border-border/30">
                <div className="font-semibold text-xs uppercase tracking-wider mb-1 text-primary/80">AI Insight</div>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{artifact.explanation}</ReactMarkdown>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

const MarkdownSection = ({ title, content, icon: Icon }: { title: string, content: string, icon: any }) => {
  if (!content) return null;
  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
        <Icon className="h-5 w-5 text-primary" />
        {title}
      </h3>
      <div className="bg-card/30 p-4 rounded-lg border border-border/30">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Custom code block styling
            code({ node, inline, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || '')
              return !inline ? (
                <div className="relative my-4 rounded-md bg-black/40 p-4 font-mono text-sm overflow-x-auto border border-white/10">
                  <code className={className} {...props}>
                    {children}
                  </code>
                </div>
              ) : (
                <code className="px-1.5 py-0.5 rounded bg-primary/10 text-primary font-mono text-xs border border-primary/20" {...props}>
                  {children}
                </code>
              )
            },
            // Custom table styling
            table({ children }) {
              return (
                <div className="my-4 w-full overflow-x-auto rounded-lg border border-white/10">
                  <table className="w-full text-sm text-left">
                    {children}
                  </table>
                </div>
              )
            },
            thead({ children }) {
              return <thead className="bg-white/5 text-xs uppercase text-muted-foreground font-semibold">{children}</thead>
            },
            th({ children }) {
              return <th className="px-4 py-3 whitespace-nowrap">{children}</th>
            },
            td({ children }) {
              return <td className="px-4 py-3 border-t border-white/5">{children}</td>
            },
            // Custom list styling
            ul({ children }) {
              return <ul className="list-disc pl-5 space-y-2 my-2 text-muted-foreground text-sm">{children}</ul>
            },
            ol({ children }) {
              return <ol className="list-decimal pl-5 space-y-2 my-2 text-muted-foreground text-sm">{children}</ol>
            },
            // Custom headings
            h1: ({ children }) => <h1 className="text-2xl font-bold mt-6 mb-4 text-foreground">{children}</h1>,
            h2: ({ children }) => <h2 className="text-xl font-semibold mt-5 mb-3 text-foreground">{children}</h2>,
            h3: ({ children }) => <h3 className="text-lg font-medium mt-4 mb-2 text-foreground">{children}</h3>,
            p: ({ children }) => <p className="mb-4 text-muted-foreground text-sm leading-relaxed">{children}</p>,
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  );
};

export function ReportPanel({ isOpen, onClose, reportPath, sessionId }: ReportPanelProps) {
  const [reportHtml, setReportHtml] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const [isDownloading, setIsDownloading] = useState<string | null>(null);
  const downloadMenuRef = useRef<HTMLDivElement>(null);

  const downloadFormats = [
    { id: "html", label: "HTML (Interactive)", icon: Globe, description: "Web viewing" },
    { id: "pdf", label: "PDF Document", icon: FileText, description: "Print & share" },
    { id: "docx", label: "Word Document", icon: FileEdit, description: "Editable" },
    { id: "txt", label: "Plain Text", icon: File, description: "Simple text" },
    { id: "zip", label: "ZIP Bundle", icon: Archive, description: "With artifacts" },
  ];

  const handleDownload = async (format: string) => {
    setIsDownloading(format);
    try {
      const reportId = reportPath.split("/").pop()?.replace(".html", "") || sessionId;
      const response = await fetch(`/api/reports/${sessionId}/export/${format}`);
      if (!response.ok) throw new Error("Download failed");

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report_${sessionId.slice(0, 8)}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download error:", err);
    } finally {
      setIsDownloading(null);
      setShowDownloadMenu(false);
    }
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (downloadMenuRef.current && !downloadMenuRef.current.contains(event.target as Node)) {
        setShowDownloadMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (isOpen) {
      if (!reportPath) {
        setIsLoading(true);
        setReportHtml(""); // Clear previous report if path becomes empty
        setError(null);
        return;
      }

      const loadReport = async () => {
        try {
          setIsLoading(true);
          setError(null);

          const cleanPath = reportPath.startsWith('/') ? reportPath.slice(1) : reportPath;
          const response = await fetch(`/${cleanPath}`);

          if (!response.ok) {
            throw new Error(`Failed to load report: ${response.statusText}`);
          }

          const html = await response.text();
          setReportHtml(html);
        } catch (e: any) {
          console.error('Error loading report:', e);
          setError(e.message || 'Failed to load report');
          setReportHtml(""); // Clear report on error
        } finally {
          setIsLoading(false);
        }
      };

      loadReport();
    } else {
      // When panel closes, reset states
      setReportHtml("");
      setIsLoading(false);
      setError(null);
    }
  }, [isOpen, reportPath]);

  if (!isOpen) return null;

  return (
    <div className="w-[600px] border-l border-white/10 bg-black/40 backdrop-blur-xl h-full flex flex-col animate-in slide-in-from-right duration-500 relative z-20 shadow-2xl">
      <div className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-white/5">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-primary" />
          <span className="font-display font-semibold tracking-wide">Analysis Report</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative" ref={downloadMenuRef}>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowDownloadMenu(!showDownloadMenu)}
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
              disabled={!reportPath || isLoading}
            >
              <Download className="h-4 w-4" />
            </Button>
            {showDownloadMenu && (
              <div className="absolute right-0 top-full mt-1 w-56 bg-card/95 backdrop-blur-xl border border-white/10 rounded-lg shadow-xl z-50 py-1 animate-in fade-in slide-in-from-top-2 duration-200">
                {downloadFormats.map((format) => {
                  const Icon = format.icon;
                  return (
                    <button
                      key={format.id}
                      onClick={() => handleDownload(format.id)}
                      disabled={isDownloading === format.id}
                      className="w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-white/5 transition-colors text-left disabled:opacity-50"
                    >
                      {isDownloading === format.id ? (
                        <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      ) : (
                        <Icon className="h-4 w-4 text-muted-foreground" />
                      )}
                      <div className="flex-1">
                        <div className="text-foreground">{format.label}</div>
                        <div className="text-xs text-muted-foreground">{format.description}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8 text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden">
        {error ? (
          <div className="p-6 h-full flex flex-col items-center justify-center text-center">
            <div className="p-4 rounded-full bg-white/5 mb-4">
              <FileText className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="font-semibold text-lg mb-2">Ready for Analysis</h3>
            <p className="text-muted-foreground max-w-xs mx-auto mb-6">
              Upload a dataset to generate a comprehensive analysis report.
            </p>
            <div className="text-xs text-white/20 font-mono">
              {error.includes("404") ? "Waiting for data..." : error}
            </div>
          </div>
        ) : isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground font-mono">Loading report...</span>
            </div>
          </div>
        ) : reportPath ? (
          <iframe
            src={reportPath.startsWith('/') ? reportPath : `/${reportPath}`}
            className="w-full h-full border-0"
            title="Analysis Report"
          />
        ) : (
          <div className="p-6">
            <div className="text-muted-foreground p-4 border border-border/30 rounded-lg bg-muted/10 flex items-center gap-3">
              <FileText className="h-5 w-5" />
              <div className="text-sm">No report available</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}