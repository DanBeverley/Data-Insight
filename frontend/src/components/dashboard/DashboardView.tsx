import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MoreHorizontal, Download, Share2, Maximize2, FileJson, FileBarChart, FileSpreadsheet, Activity, FileImage, FileText, X } from "lucide-react";
import { ChartBlock } from "../viz/ChartBlock";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { cn } from "@/lib/utils";

interface Artifact {
  id: number | string;
  name: string;
  type: string;
  size: string;
  date: string;
  icon: any;
  color: string;
  chartData?: any[];
  chartTitle?: string;
  chartType?: string;
  previewUrl?: string;
  description?: string;
}

interface DashboardViewProps {
  sessionId: string;
}

export function DashboardView({ sessionId }: DashboardViewProps) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(false);
  const [insights, setInsights] = useState<any[]>([]);
  const [loadingInsights, setLoadingInsights] = useState(true);
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const fetchArtifacts = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/data/${sessionId}/artifacts`);
        const data = await response.json();

        if (data.status === 'success' && data.artifacts) {
          const mappedArtifacts = data.artifacts.map((a: any, index: number) => {
            const ext = a.filename.split('.').pop()?.toLowerCase();
            let icon = FileText;
            let color = "text-gray-400";
            let type = ext || 'file';

            if (['json'].includes(ext)) { icon = FileJson; color = "text-primary"; }
            else if (['csv'].includes(ext)) { icon = FileBarChart; color = "text-purple-400"; }
            else if (['xlsx', 'xls'].includes(ext)) { icon = FileSpreadsheet; color = "text-green-400"; }
            else if (['png', 'jpg', 'jpeg'].includes(ext)) { icon = FileImage; color = "text-pink-400"; }
            else if (['txt', 'log'].includes(ext)) { icon = Activity; color = "text-orange-400"; }

            return {
              id: a.artifact_id || index,
              name: a.filename,
              type: type,
              size: a.size || 'Unknown',
              date: new Date(a.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              icon: icon,
              color: color,
              chartTitle: a.description || a.filename,
              previewUrl: (['png', 'jpg', 'jpeg', 'html'].includes(ext)) ? a.file_path : undefined,
              // Use empty array if no chart data provided
              chartData: a.chartData || [],
              chartType: 'bar'
            };
          });
          setArtifacts(mappedArtifacts);
        }
      } catch (error) {
        console.error("Failed to fetch artifacts:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchArtifacts();
    // Poll for updates
    const interval = setInterval(fetchArtifacts, 5000);
    return () => clearInterval(interval);
  }, [sessionId]);

  // Fetch dynamic insights
  useEffect(() => {
    const fetchInsights = async () => {
      setLoadingInsights(true);
      try {
        const response = await fetch(`/api/data/${sessionId}/insights`);
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'success' && data.insights) {
            setInsights(data.insights);
          }
        }
      } catch (error) {
        console.error("Failed to fetch insights:", error);
      } finally {
        setLoadingInsights(false);
      }
    };

    if (sessionId) {
      fetchInsights();
    }
  }, [sessionId]);

  const handleDownload = (id: string | number) => {
    window.open(`/api/data/${sessionId}/artifacts/download/${id}`, '_blank');
  };

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-display font-bold text-foreground tracking-tight">Mission Control</h2>
          <p className="text-muted-foreground mt-1">Artifacts generated from recent analysis sessions.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="bg-white/5 border-white/10 hover:bg-white/10">
            <Share2 className="mr-2 h-4 w-4" /> Share
          </Button>
          <Button className="bg-primary text-primary-foreground shadow-[0_0_15px_rgba(0,242,234,0.3)] hover:bg-primary/90">
            <Download className="mr-2 h-4 w-4" /> Export Report
          </Button>
        </div>
      </div>

      {/* Artifacts Grid - Compact View with Hover Preview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {artifacts.length === 0 && !loading && (
          <div className="col-span-full text-center text-muted-foreground py-10">
            No artifacts found. Upload data or ask the agent to generate analysis.
          </div>
        )}

        {artifacts.map((artifact) => (
          <HoverCard key={artifact.id} openDelay={100} closeDelay={100}>
            <HoverCardTrigger asChild>
              <div
                className="group relative bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4 cursor-pointer transition-all hover:bg-white/10 hover:border-white/20 hover:-translate-y-1 hover:shadow-lg"
                onClick={() => setSelectedArtifact(artifact)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className={cn("p-2 rounded-lg bg-white/5 ring-1 ring-white/10", artifact.color)}>
                    <artifact.icon className="h-5 w-5" />
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 -mr-2 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => { e.stopPropagation(); handleDownload(artifact.id); }}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-medium truncate pr-2" title={artifact.name}>{artifact.name}</h4>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>{artifact.size}</span>
                    <span>•</span>
                    <span>{artifact.date}</span>
                  </div>
                </div>
              </div>
            </HoverCardTrigger>

            <HoverCardContent side="top" className="w-[400px] p-0 border-white/10 bg-black/90 backdrop-blur-xl shadow-2xl overflow-hidden z-50">
              <div className="p-3 border-b border-white/10 bg-white/5 flex items-center justify-between">
                <span className="text-xs font-mono font-medium text-muted-foreground uppercase tracking-wider">Preview: {artifact.name}</span>
                <Maximize2 className="h-3 w-3 text-muted-foreground" />
              </div>
              <div className="p-4 bg-background/50 flex justify-center items-center min-h-[200px]">
                {artifact.previewUrl ? (
                  artifact.name.endsWith('.html') ? (
                    <iframe
                      src={artifact.previewUrl}
                      className="w-full h-[200px] bg-white rounded border-0"
                      title="Preview"
                      sandbox="allow-scripts allow-same-origin"
                    />
                  ) : (
                    <img src={artifact.previewUrl || ""} alt="Preview" className="max-w-full max-h-[200px] object-contain rounded" />
                  )
                ) : (
                  <ChartBlock
                    title={artifact.chartTitle || artifact.name}
                    type={artifact.chartType as any}
                    data={artifact.chartData || []}
                    color={artifact.color.replace('text-', 'var(--') + ')'}
                  />
                )}
              </div>
            </HoverCardContent>
          </HoverCard>
        ))}
      </div>

      {/* Dynamic Insights from ML Profiling */}
      <div className="bg-gradient-to-r from-primary/10 to-purple-500/10 backdrop-blur-md border border-white/10 rounded-2xl p-6 relative overflow-hidden mt-6">
        <div className="absolute top-0 right-0 p-32 bg-primary/20 blur-[100px] rounded-full pointer-events-none" />

        <h3 className="text-xl font-display font-bold mb-4">Key Insights Summary</h3>
        {loadingInsights ? (
          <div className="flex items-center justify-center py-4 text-muted-foreground">
            <span>Loading analysis...</span>
          </div>
        ) : insights.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {insights.slice(0, 3).map((insight, idx) => (
              <div key={idx} className="bg-black/20 rounded-xl p-4 border border-white/5">
                <p className="text-xs text-muted-foreground uppercase mb-1">{insight.label}</p>
                <p className="text-2xl font-mono text-primary">
                  {insight.value}
                  {insight.change && <span className="text-xs text-green-400 ml-1">↑ {insight.change}</span>}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-muted-foreground py-4">
            No insights available. Run analysis to generate insights.
          </div>
        )}
      </div>

      {selectedArtifact && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-8" onClick={() => setSelectedArtifact(null)}>
          <div className="relative w-full h-full max-w-6xl max-h-[90vh] bg-background/50 border border-white/10 rounded-2xl overflow-hidden shadow-2xl flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-white/10 bg-white/5">
              <div className="flex items-center gap-3">
                <div className={cn("p-2 rounded-lg bg-white/5", selectedArtifact.color)}>
                  <selectedArtifact.icon className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-semibold">{selectedArtifact.name}</h3>
                  <p className="text-xs text-muted-foreground">{selectedArtifact.date} • {selectedArtifact.size}</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="ghost" size="icon" onClick={() => handleDownload(selectedArtifact.id)}>
                  <Download className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" onClick={() => setSelectedArtifact(null)}>
                  <X className="h-5 w-5" />
                </Button>
              </div>
            </div>
            <div className="flex-1 overflow-hidden bg-black/20 p-4 flex items-center justify-center">
              {selectedArtifact.previewUrl ? (
                selectedArtifact.name.endsWith('.html') ? (
                  <iframe
                    src={selectedArtifact.previewUrl}
                    className="w-full h-full bg-white rounded-lg border-0"
                    title="Full View"
                    sandbox="allow-scripts allow-same-origin"
                  />
                ) : (
                  <img src={selectedArtifact.previewUrl} alt="Full View" className="max-w-full max-h-full object-contain rounded-lg shadow-2xl" />
                )
              ) : (
                <div className="text-center">
                  <ChartBlock
                    title={selectedArtifact.chartTitle || ''}
                    type={selectedArtifact.chartType as any}
                    data={selectedArtifact.chartData || []}
                    color={selectedArtifact.color.replace('text-', 'var(--') + ')'}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )
      }
    </div >
  );
}