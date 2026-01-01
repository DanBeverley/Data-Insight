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
import { FileText, Upload, Database, MoreVertical, Search, Filter } from "lucide-react";
import { Input } from "@/components/ui/input";

interface Dataset {
  id: string;
  name: string;
  size: string;
  rows: string;
  type: string;
  date: string;
  status: string;
}

interface DatasetViewProps {
  sessionId: string;
}

export function DatasetView({ sessionId }: DatasetViewProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [storageUsed, setStorageUsed] = useState(0);
  const [storageTotal, setStorageTotal] = useState(100);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Fetch uploaded datasets for this session
    const fetchDatasets = async () => {
      try {
        const response = await fetch(`/api/data/${sessionId}/preview`);
        if (response.ok) {
          const data = await response.json();
          // Backend returns { data: [...rows], shape: [rows, cols], columns: [...] }
          if (data.data && data.data.length > 0) {
            // Create dataset entry from preview data
            const dataset = {
              id: sessionId,
              name: "Uploaded Dataset",
              size: `${(JSON.stringify(data.data).length / 1024).toFixed(1)} KB`,
              rows: data.shape ? data.shape[0].toString() : data.data.length.toString(),
              type: "CSV",
              date: new Date().toLocaleDateString(),
              status: "Active"
            };
            setDatasets([dataset]);
          }
        }
      } catch (error) {
        console.error("Failed to fetch datasets:", error);
      }
    };

    if (sessionId) {
      fetchDatasets();
    }
  }, [sessionId]);

  const handleFileSelect = async (files: File[]) => {
    if (!files.length) return;

    setUploading(true);
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('session_id', sessionId);
    formData.append('enable_profiling', 'true');

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      if (result.status === 'success') {
        // Refresh datasets list
        window.location.reload(); // Simple reload for now
      }
    } catch (error) {
      console.error("Upload error:", error);
    } finally {
      setUploading(false);
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

  const storagePercent = (storageUsed / storageTotal) * 100;

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div>
        <h2 className="text-3xl font-display font-bold text-foreground tracking-tight">Data Nexus</h2>
        <p className="text-muted-foreground mt-1">Manage your raw datasets and connected sources.</p>
      </div>

      {/* Hidden file input */}
      <input
        type="file"
        multiple
        ref={fileInputRef}
        className="hidden"
        accept=".csv,.xlsx,.xls,.json,.parquet"
        onChange={(e) => e.target.files && handleFileSelect(Array.from(e.target.files))}
      />

      {/* Stats / Drag Drop Area */}
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
          <h3 className="text-lg font-medium">{uploading ? 'Uploading...' : 'Drop files to upload'}</h3>
          <p className="text-sm text-muted-foreground mt-1">Support for CSV, JSON, Excel, Parquet (Max 500MB)</p>
        </div>

        <div className="md:col-span-1 space-y-4">
          <div className="bg-primary/10 border border-primary/20 rounded-2xl p-5">
            <p className="text-xs text-primary/80 uppercase font-bold mb-2">Total Storage</p>
            <p className="text-2xl font-mono font-bold text-primary">
              {storageUsed.toFixed(1)} GB
            </p>
            <div className="w-full bg-primary/20 h-1.5 rounded-full mt-3 overflow-hidden">
              <div className="bg-primary h-full transition-all duration-300" style={{ width: `${storagePercent}%` }} />
            </div>
            <p className="text-xs text-primary/60 mt-1">of {storageTotal} GB</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
            <p className="text-xs text-muted-foreground uppercase font-bold mb-2">Uploaded Files</p>
            <p className="text-3xl font-mono font-bold text-foreground">{datasets.length}</p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input placeholder="Search datasets..." className="pl-10 bg-white/5 border-white/10 focus-visible:ring-primary/50" />
        </div>
        <Button variant="outline" className="bg-white/5 border-white/10">
          <Filter className="mr-2 h-4 w-4" /> Filter
        </Button>
      </div>

      {/* Table */}
      <div className="rounded-2xl border border-white/10 overflow-hidden bg-white/5 backdrop-blur-sm">
        <Table>
          <TableHeader className="bg-white/5">
            <TableRow className="hover:bg-transparent border-white/10">
              <TableHead className="text-muted-foreground">Name</TableHead>
              <TableHead className="text-muted-foreground">Type</TableHead>
              <TableHead className="text-muted-foreground">Size</TableHead>
              <TableHead className="text-muted-foreground">Rows</TableHead>
              <TableHead className="text-muted-foreground">Status</TableHead>
              <TableHead className="text-right"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {datasets.length === 0 ? (
              <TableRow className="hover:bg-transparent">
                <TableCell colSpan={6} className="text-center text-muted-foreground py-10">
                  No datasets uploaded yet. Upload files to get started.
                </TableCell>
              </TableRow>
            ) : (
              datasets.map((file) => (
                <TableRow key={file.id} className="hover:bg-white/5 border-white/10 group">
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-3">
                      <div className="h-8 w-8 rounded bg-primary/10 flex items-center justify-center text-primary">
                        <FileText className="h-4 w-4" />
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm text-foreground">{file.name}</span>
                        <span className="text-xs text-muted-foreground">{file.date}</span>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className="inline-flex items-center px-2 py-1 rounded-md bg-white/5 text-xs font-mono text-muted-foreground border border-white/5">
                      {file.type}
                    </span>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground font-mono">{file.size}</TableCell>
                  <TableCell className="text-sm text-muted-foreground font-mono">{file.rows}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className={`h-1.5 w-1.5 rounded-full ${file.status === 'Ready' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' : file.status === 'Processing' ? 'bg-yellow-500 animate-pulse' : 'bg-gray-500'}`} />
                      <span className="text-xs text-muted-foreground">{file.status}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="icon" className="opacity-0 group-hover:opacity-100 transition-opacity">
                      <MoreVertical className="h-4 w-4 text-muted-foreground" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}