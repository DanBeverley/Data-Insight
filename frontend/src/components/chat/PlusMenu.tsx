
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Paperclip, FileText, Globe, Settings, Microscope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import { TimeSlider } from "./TimeSlider";

interface PlusMenuProps {
    onFileSelect: () => void;
    reportMode: boolean;
    onReportModeChange: (enabled: boolean) => void;
    webSearchMode: boolean;
    onWebSearchModeChange: (enabled: boolean) => void;
    onOpenSearchSettings: () => void;
    researchMode: boolean;
    onResearchModeChange: (enabled: boolean) => void;
    researchTimeBudget: number;
    onResearchTimeBudgetChange: (mins: number) => void;
}

export function PlusMenu({
    onFileSelect,
    reportMode,
    onReportModeChange,
    webSearchMode,
    onWebSearchModeChange,
    onOpenSearchSettings,
    researchMode,
    onResearchModeChange,
    researchTimeBudget,
    onResearchTimeBudgetChange
}: PlusMenuProps) {
    const [isOpen, setIsOpen] = useState(false);
    const menuRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    return (
        <div className="relative" ref={menuRef}>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="absolute bottom-12 left-0 w-64 bg-popover/95 backdrop-blur-lg border border-border rounded-xl shadow-xl overflow-hidden z-50 p-1"
                    >
                        <div className="flex flex-col gap-1">
                            <button
                                onClick={() => {
                                    onFileSelect();
                                    setIsOpen(false);
                                }}
                                className="flex items-center gap-3 px-3 py-2.5 text-sm font-medium text-foreground hover:bg-muted/50 rounded-lg transition-colors w-full text-left bg-transparent border-0"
                            >
                                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary">
                                    <Paperclip className="w-4 h-4" />
                                </div>
                                <div className="flex flex-col">
                                    <span>Add attachments</span>
                                    <span className="text-[10px] text-muted-foreground font-normal">CSV, Excel, JSON, or images</span>
                                </div>
                            </button>

                            <div className="h-px bg-border/50 mx-2 my-1" />

                            <div className="flex items-center justify-between px-3 py-2 text-sm font-medium text-foreground hover:bg-muted/50 rounded-lg transition-colors cursor-pointer"
                                onClick={() => onReportModeChange(!reportMode)}>
                                <div className="flex items-center gap-3">
                                    <div className={cn(
                                        "flex items-center justify-center w-8 h-8 rounded-full transition-colors",
                                        reportMode ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
                                    )}>
                                        <FileText className="w-4 h-4" />
                                    </div>
                                    <div className="flex flex-col">
                                        <span>Report Mode</span>
                                        <span className="text-[10px] text-muted-foreground font-normal">Enables deep thinking</span>
                                    </div>
                                </div>
                                <Switch
                                    checked={reportMode}
                                    onCheckedChange={onReportModeChange}
                                    className="scale-75 data-[state=checked]:bg-primary"
                                />
                            </div>

                            <div className="flex items-center justify-between px-3 py-2 text-sm font-medium text-foreground hover:bg-muted/50 rounded-lg transition-colors cursor-pointer"
                                onClick={() => onWebSearchModeChange(!webSearchMode)}>
                                <div className="flex items-center gap-3">
                                    <div className={cn(
                                        "flex items-center justify-center w-8 h-8 rounded-full transition-colors",
                                        webSearchMode ? "bg-blue-500/20 text-blue-400" : "bg-muted text-muted-foreground"
                                    )}>
                                        <Globe className="w-4 h-4" />
                                    </div>
                                    <div className="flex flex-col">
                                        <span>Web Access</span>
                                        <span className="text-[10px] text-muted-foreground font-normal">Search the internet</span>
                                    </div>
                                </div>
                                <div className="flex items-center gap-1">
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-6 w-6 rounded-md hover:bg-muted"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onOpenSearchSettings();
                                        }}
                                    >
                                        <Settings className="w-3.5 h-3.5 text-muted-foreground" />
                                    </Button>
                                    <Switch
                                        checked={webSearchMode}
                                        onCheckedChange={onWebSearchModeChange}
                                        disabled={researchMode}
                                        className="scale-75 data-[state=checked]:bg-blue-500"
                                    />
                                </div>
                            </div>

                            <div className="h-px bg-border/50 mx-2 my-1" />

                            <div className={cn(
                                "flex flex-col px-3 py-2 text-sm font-medium rounded-lg transition-colors",
                                researchMode ? "bg-purple-500/10" : "hover:bg-muted/50"
                            )}>
                                <div
                                    className="flex items-center justify-between cursor-pointer"
                                    onClick={() => onResearchModeChange(!researchMode)}
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={cn(
                                            "flex items-center justify-center w-8 h-8 rounded-full transition-colors",
                                            researchMode ? "bg-purple-500/20 text-purple-400" : "bg-muted text-muted-foreground"
                                        )}>
                                            <Microscope className="w-4 h-4" />
                                        </div>
                                        <div className="flex flex-col">
                                            <span className={researchMode ? "text-purple-300" : ""}>Deep Research</span>
                                            <span className="text-[10px] text-muted-foreground font-normal">Time-boxed investigation</span>
                                        </div>
                                    </div>
                                    <Switch
                                        checked={researchMode}
                                        onCheckedChange={onResearchModeChange}
                                        className="scale-75 data-[state=checked]:bg-purple-500"
                                    />
                                </div>
                                {researchMode && (
                                    <div className="mt-3 ml-11">
                                        <TimeSlider
                                            value={researchTimeBudget}
                                            onChange={onResearchTimeBudgetChange}
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <Button
                variant="ghost"
                size="icon"
                className={cn(
                    "h-10 w-10 rounded-xl transition-all duration-300",
                    isOpen ? "bg-muted rotate-45" : "hover:bg-muted/50 hover:rotate-90"
                )}
                onClick={() => setIsOpen(!isOpen)}
            >
                <Plus className={cn("h-5 w-5 transition-transform duration-300", isOpen && "text-primary")} />
            </Button>
        </div>
    );
}
