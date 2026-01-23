import { ChevronLeft, ChevronRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface VersionSliderProps {
    messageId: string;
    currentVersion: number;
    totalVersions: number;
    onVersionChange: (version: number) => void;
    direction: 'prev' | 'next' | null;
}

export function VersionSlider({
    messageId,
    currentVersion,
    totalVersions,
    onVersionChange,
    direction
}: VersionSliderProps) {
    if (totalVersions <= 1) return null;

    return (
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <button
                onClick={() => onVersionChange(currentVersion - 1)}
                disabled={currentVersion <= 1}
                className="p-1 hover:bg-accent rounded-md disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                aria-label="Previous version"
            >
                <ChevronLeft className="h-3.5 w-3.5" />
            </button>

            <AnimatePresence mode="wait">
                <motion.span
                    key={currentVersion}
                    initial={{ opacity: 0, y: direction === 'next' ? 5 : -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: direction === 'next' ? -5 : 5 }}
                    transition={{ duration: 0.15 }}
                    className="min-w-[2.5rem] text-center font-medium"
                >
                    {currentVersion}/{totalVersions}
                </motion.span>
            </AnimatePresence>

            <button
                onClick={() => onVersionChange(currentVersion + 1)}
                disabled={currentVersion >= totalVersions}
                className="p-1 hover:bg-accent rounded-md disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                aria-label="Next version"
            >
                <ChevronRight className="h-3.5 w-3.5" />
            </button>
        </div>
    );
}

interface VersionContentProps {
    content: string | React.ReactNode;
    direction: 'prev' | 'next' | null;
    version: number;
}

export function VersionContent({ content, direction, version }: VersionContentProps) {
    return (
        <AnimatePresence mode="wait">
            <motion.div
                key={version}
                initial={{
                    opacity: 0,
                    x: direction === 'next' ? 20 : direction === 'prev' ? -20 : 0
                }}
                animate={{ opacity: 1, x: 0 }}
                exit={{
                    opacity: 0,
                    x: direction === 'next' ? -20 : direction === 'prev' ? 20 : 0
                }}
                transition={{ duration: 0.2, ease: "easeInOut" }}
            >
                {content}
            </motion.div>
        </AnimatePresence>
    );
}
