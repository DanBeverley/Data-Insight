import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { FileText, Upload, Brain, Trash2, Search, Filter, BookOpen, ToggleLeft, ToggleRight, Eye, X } from "lucide-react";
import { Input } from "@/components/ui/input";

interface KnowledgeItem {
  id: string;
  source: string;
  source_name: string;
  added_at: string;
  content_preview: string;
  inject_to_context: boolean;
}

interface DocumentViewerState {
  isOpen: boolean;
  content: string;
  title: string;
  loading: boolean;
}

interface KnowledgeStoreViewProps {
  sessionId: string;
}

export function KnowledgeStoreView({ sessionId }: KnowledgeStoreViewProps) {
  const [items, setItems] = useState<KnowledgeItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [viewer, setViewer] = useState<DocumentViewerState>({ isOpen: false, content: "", title: "", loading: false });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  const fetchItems = async () => {
    try {
      const response = await fetch(`/api/knowledge/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setItems(data.items || []);
      }
    } catch (error) {
      console.error("Failed to fetch knowledge items:", error);
    }
  };

  useEffect(() => {
    if (sessionId) {
      fetchItems();
    }
  }, [sessionId]);

  const handleFileSelect = async (files: File[]) => {
    if (!files.length) return;

    setUploading(true);

    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        await fetch(`/api/knowledge/${sessionId}`, {
          method: 'POST',
          body: formData
        });
      } catch (error) {
        console.error("Upload error:", error);
      }
    }

    setUploading(false);
    fetchItems();
  };

  const handleDelete = async (docId: string) => {
    try {
      await fetch(`/api/knowledge/${sessionId}/${docId}`, {
        method: 'DELETE'
      });
      fetchItems();
    } catch (error) {
      console.error("Delete error:", error);
    }
  };

  const handleToggleInject = async (docId: string, currentState: boolean) => {
    try {
      await fetch(`/api/knowledge/${sessionId}/${docId}?inject_to_context=${!currentState}`, {
        method: 'PATCH'
      });
      setItems(prev => prev.map(item =>
        item.id === docId ? { ...item, inject_to_context: !currentState } : item
      ));
    } catch (error) {
      console.error("Toggle error:", error);
    }
  };

  const handleViewDocument = async (docId: string, title: string) => {
    setViewer({ isOpen: true, content: "", title, loading: true });
    try {
      const response = await fetch(`/api/knowledge/${sessionId}/doc/${docId}`);
      if (response.ok) {
        const data = await response.json();
        setViewer({ isOpen: true, content: data.content, title: data.source_name, loading: false });
      } else {
        setViewer(prev => ({ ...prev, content: "Failed to load document", loading: false }));
      }
    } catch (error) {
      setViewer(prev => ({ ...prev, content: "Error loading document", loading: false }));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    handleFileSelect(files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const getSourceIcon = (source: string) => {
    if (source === "research") return <Brain className="h-4 w-4" />;
    return <FileText className="h-4 w-4" />;
  };

  const getSourceLabel = (source: string) => {
    switch (source) {
      case "research": return "Research";
      case "user_upload": return "Upload";
      case "agent": return "Agent";
      default: return "Other";
    }
  };

  const filteredItems = items.filter(item =>
    item.source_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.content_preview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div>
        <h2 className="text-3xl font-display font-bold text-foreground tracking-tight flex items-center gap-3">
          <BookOpen className="h-8 w-8 text-primary" />
          Knowledge Store
        </h2>
        <p className="text-muted-foreground mt-1">Your research findings and uploaded documents, all in one place.</p>
      </div>

      <input
        type="file"
        multiple
        ref={fileInputRef}
        className="hidden"
        accept=".txt,.md,.pdf,.json,.csv,.tsv,.docx,.pptx,.xlsx,.xls,.html,.htm,.xml,.eml,.odt,.rtf,.rst,.org"
        onChange={(e) => e.target.files && handleFileSelect(Array.from(e.target.files))}
      />

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div
          ref={dropZoneRef}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
          className="md:col-span-3 bg-white/5 backdrop-blur-sm border border-white/10 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center hover:bg-white/10 transition-colors cursor-pointer group"
        >
          <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            <Upload className="h-8 w-8 text-primary" />
          </div>
          <h3 className="text-lg font-medium">{uploading ? 'Uploading...' : 'Drop files to add to knowledge'}</h3>
          <p className="text-sm text-muted-foreground mt-1">Support for TXT, Markdown, PDF, JSON, CSV</p>
        </div>

        <div className="md:col-span-1 space-y-4">
          <div className="bg-primary/10 border border-primary/20 rounded-2xl p-5">
            <p className="text-xs text-primary/80 uppercase font-bold mb-2">Total Items</p>
            <p className="text-3xl font-mono font-bold text-primary">{items.length}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
            <p className="text-xs text-muted-foreground uppercase font-bold mb-2">Research Findings</p>
            <p className="text-3xl font-mono font-bold text-foreground">
              {items.filter(i => i.source === "research").length}
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search knowledge..."
            className="pl-10 bg-white/5 border-white/10 focus-visible:ring-primary/50"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <Button variant="outline" className="bg-white/5 border-white/10">
          <Filter className="mr-2 h-4 w-4" /> Filter
        </Button>
      </div>

      <div className="rounded-2xl border border-white/10 overflow-hidden bg-white/5 backdrop-blur-sm">
        <Table>
          <TableHeader className="bg-white/5">
            <TableRow className="hover:bg-transparent border-white/10">
              <TableHead className="text-muted-foreground">Name</TableHead>
              <TableHead className="text-muted-foreground">Source</TableHead>
              <TableHead className="text-muted-foreground text-center">Memory</TableHead>
              <TableHead className="text-muted-foreground">Preview</TableHead>
              <TableHead className="text-right"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredItems.length === 0 ? (
              <TableRow className="hover:bg-transparent">
                <TableCell colSpan={5} className="text-center text-muted-foreground py-10">
                  No knowledge items yet. Upload files or run research to get started.
                </TableCell>
              </TableRow>
            ) : (
              filteredItems.map((item) => (
                <TableRow key={item.id} className="hover:bg-white/5 border-white/10 group">
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-3">
                      <div className={`h-8 w-8 rounded flex items-center justify-center ${item.source === 'research' ? 'bg-purple-500/20 text-purple-400' : 'bg-primary/10 text-primary'}`}>
                        {getSourceIcon(item.source)}
                      </div>
                      <span className="text-sm text-foreground truncate max-w-[200px]">{item.source_name}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${item.source === 'research' ? 'bg-purple-500/20 text-purple-300' : 'bg-white/5 text-muted-foreground'} border border-white/5`}>
                      {getSourceLabel(item.source)}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <button
                      onClick={() => handleToggleInject(item.id, item.inject_to_context)}
                      className="transition-colors"
                      title={item.inject_to_context ? "In agent memory" : "Not in memory"}
                    >
                      {item.inject_to_context ? (
                        <ToggleRight className="h-6 w-6 text-green-400" />
                      ) : (
                        <ToggleLeft className="h-6 w-6 text-muted-foreground" />
                      )}
                    </button>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground truncate max-w-[300px] cursor-pointer hover:text-foreground" onClick={() => handleViewDocument(item.id, item.source_name)}>
                    {item.content_preview}
                  </TableCell>
                  <TableCell className="text-right space-x-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-primary hover:text-primary hover:bg-primary/10"
                      onClick={() => handleViewDocument(item.id, item.source_name)}
                      title="View document"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-red-400 hover:text-red-300 hover:bg-red-500/10"
                      onClick={() => handleDelete(item.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {viewer.isOpen && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setViewer(prev => ({ ...prev, isOpen: false }))}>
          <div className="bg-background border border-white/10 rounded-2xl max-w-3xl w-full max-h-[80vh] flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-white/10">
              <h3 className="font-semibold text-lg truncate">{viewer.title}</h3>
              <Button variant="ghost" size="icon" onClick={() => setViewer(prev => ({ ...prev, isOpen: false }))}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="p-4 overflow-auto flex-1">
              {viewer.loading ? (
                <div className="text-center text-muted-foreground py-8">Loading...</div>
              ) : (
                <pre className="whitespace-pre-wrap text-sm font-mono text-foreground/90">{viewer.content}</pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}