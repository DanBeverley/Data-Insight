import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Upload, Loader2, Check, AlertCircle, Table, BarChart3, FileImage } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ImageAnalysisModalProps {
    isOpen: boolean;
    onClose: () => void;
    sessionId: string;
    onDataExtracted?: (datasetPath: string) => void;
}

type AnalysisStep = "upload" | "analyzing" | "preview" | "confirmed" | "error";

interface ExtractedData {
    image_id: string;
    image_type: string;
    title: string | null;
    columns: string[];
    preview_rows: Record<string, any>[];
    row_count: number;
    confidence: number;
    notes: string | null;
}

export function ImageAnalysisModal({ isOpen, onClose, sessionId, onDataExtracted }: ImageAnalysisModalProps) {
    const [step, setStep] = useState<AnalysisStep>("upload");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string>("");
    const [extractedData, setExtractedData] = useState<ExtractedData | null>(null);
    const [error, setError] = useState<string>("");
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!isOpen) {
            setStep("upload");
            setSelectedFile(null);
            setPreviewUrl("");
            setExtractedData(null);
            setError("");
        }
    }, [isOpen]);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        if (!file.type.startsWith("image/")) {
            setError("Please select an image file");
            return;
        }

        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setError("");
    };

    const handleUploadAndAnalyze = async () => {
        if (!selectedFile) return;

        setStep("analyzing");
        setError("");

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            formData.append("session_id", sessionId);
            formData.append("analysis_type", "auto");

            const uploadRes = await fetch("/api/image/upload", {
                method: "POST",
                body: formData,
            });

            if (!uploadRes.ok) throw new Error("Upload failed");

            const uploadData = await uploadRes.json();
            const imageId = uploadData.image_id;

            const analyzeRes = await fetch(`/api/image/${sessionId}/analyze/${imageId}`, {
                method: "POST",
            });

            if (!analyzeRes.ok) throw new Error("Analysis failed");

            const analysisData = await analyzeRes.json();
            setExtractedData(analysisData);
            setStep("preview");
        } catch (err: any) {
            setError(err.message || "Failed to analyze image");
            setStep("error");
        }
    };

    const handleConfirm = async () => {
        if (!extractedData) return;

        try {
            const res = await fetch(`/api/image/${sessionId}/confirm/${extractedData.image_id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({}),
            });

            if (!res.ok) throw new Error("Confirmation failed");

            const data = await res.json();
            setStep("confirmed");

            setTimeout(() => {
                onDataExtracted?.(data.dataset_path);
                onClose();
            }, 1500);
        } catch (err: any) {
            setError(err.message);
            setStep("error");
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="relative w-full max-w-2xl bg-background border border-border rounded-2xl shadow-2xl overflow-hidden"
            >
                <div className="flex items-center justify-between px-6 py-4 border-b border-border">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-emerald-500/10">
                            <FileImage className="w-5 h-5 text-emerald-500" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold">Analyze Image</h2>
                            <p className="text-xs text-muted-foreground">Extract data from charts, tables, or documents</p>
                        </div>
                    </div>
                    <Button variant="ghost" size="icon" onClick={onClose}>
                        <X className="w-5 h-5" />
                    </Button>
                </div>

                <div className="p-6">
                    {step === "upload" && (
                        <div className="space-y-4">
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className={cn(
                                    "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors",
                                    selectedFile ? "border-emerald-500 bg-emerald-500/5" : "border-border hover:border-muted-foreground"
                                )}
                            >
                                {previewUrl ? (
                                    <img src={previewUrl} alt="Preview" className="max-h-48 mx-auto rounded-lg" />
                                ) : (
                                    <>
                                        <Upload className="w-10 h-10 mx-auto mb-3 text-muted-foreground" />
                                        <p className="text-sm font-medium">Click to upload an image</p>
                                        <p className="text-xs text-muted-foreground mt-1">PNG, JPG, WEBP up to 10MB</p>
                                    </>
                                )}
                            </div>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={handleFileSelect}
                                className="hidden"
                            />
                            {error && <p className="text-sm text-destructive">{error}</p>}
                            <Button
                                onClick={handleUploadAndAnalyze}
                                disabled={!selectedFile}
                                className="w-full bg-emerald-600 hover:bg-emerald-700"
                            >
                                Analyze Image
                            </Button>
                        </div>
                    )}

                    {step === "analyzing" && (
                        <div className="py-12 text-center">
                            <Loader2 className="w-12 h-12 mx-auto mb-4 text-emerald-500 animate-spin" />
                            <p className="text-sm font-medium">Analyzing image...</p>
                            <p className="text-xs text-muted-foreground mt-1">Extracting data with vision AI</p>
                        </div>
                    )}

                    {step === "preview" && extractedData && (
                        <div className="space-y-4">
                            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                                {extractedData.image_type === "table" ? (
                                    <Table className="w-5 h-5 text-blue-500" />
                                ) : (
                                    <BarChart3 className="w-5 h-5 text-purple-500" />
                                )}
                                <div>
                                    <p className="text-sm font-medium">
                                        {extractedData.title || `${extractedData.image_type} detected`}
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        {extractedData.row_count} rows × {extractedData.columns.length} columns •
                                        {Math.round(extractedData.confidence * 100)}% confidence
                                    </p>
                                </div>
                            </div>

                            <div className="overflow-x-auto rounded-lg border border-border">
                                <table className="w-full text-sm">
                                    <thead className="bg-muted/50">
                                        <tr>
                                            {extractedData.columns.map((col, i) => (
                                                <th key={i} className="px-3 py-2 text-left font-medium">{col}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {extractedData.preview_rows.slice(0, 5).map((row, i) => (
                                            <tr key={i} className="border-t border-border">
                                                {extractedData.columns.map((col, j) => (
                                                    <td key={j} className="px-3 py-2">{String(row[col] ?? "")}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {extractedData.notes && (
                                <p className="text-xs text-muted-foreground italic">{extractedData.notes}</p>
                            )}

                            <div className="flex gap-3">
                                <Button variant="outline" onClick={() => setStep("upload")} className="flex-1">
                                    Try Another
                                </Button>
                                <Button onClick={handleConfirm} className="flex-1 bg-emerald-600 hover:bg-emerald-700">
                                    Use This Data
                                </Button>
                            </div>
                        </div>
                    )}

                    {step === "confirmed" && (
                        <div className="py-12 text-center">
                            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/10 flex items-center justify-center">
                                <Check className="w-8 h-8 text-emerald-500" />
                            </div>
                            <p className="text-sm font-medium">Data extracted successfully!</p>
                            <p className="text-xs text-muted-foreground mt-1">Ready for analysis</p>
                        </div>
                    )}

                    {step === "error" && (
                        <div className="py-8 text-center">
                            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-destructive" />
                            <p className="text-sm font-medium text-destructive">{error}</p>
                            <Button variant="outline" onClick={() => setStep("upload")} className="mt-4">
                                Try Again
                            </Button>
                        </div>
                    )}
                </div>
            </motion.div>
        </div>
    );
}
