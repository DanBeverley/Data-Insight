import { cn } from "@/lib/utils";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface ResponseLoadingIndicatorProps {
  modelName?: string;
  status?: string;
  className?: string;
}

function PixelWave({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-end gap-0 h-4", className)}>
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          className="w-[4px] bg-primary rounded-none"
          style={{
            animation: `pixelWave 1s steps(4) infinite`,
            animationDelay: `${i * 0.08}s`,
            height: '30%',
          }}
        />
      ))}
      <style>{`
        @keyframes pixelWave {
          0%, 100% { height: 30%; opacity: 0.5; }
          25% { height: 60%; opacity: 0.7; }
          50% { height: 100%; opacity: 1; }
          75% { height: 60%; opacity: 0.7; }
        }
      `}</style>
    </div>
  );
}

function CollapsibleStatus({ status }: { status: string }) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="relative group max-w-[400px]">
      <div
        className={cn(
          "text-xs text-muted-foreground bg-background/50 rounded-md border border-border/50 transition-all duration-300",
          isExpanded ? "max-h-[200px] overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-primary/20" : "max-h-[32px] px-2 py-1.5 overflow-hidden flex items-center cursor-pointer"
        )}
        onClick={() => !isExpanded && setIsExpanded(true)}
      >
        <pre className={cn("font-mono leading-relaxed", isExpanded ? "whitespace-pre-wrap" : "whitespace-nowrap truncate pr-6")}>
          {status}
        </pre>
      </div>
      <button
        onClick={(e) => { e.stopPropagation(); setIsExpanded(!isExpanded); }}
        className="absolute top-1 right-1 p-1 hover:bg-muted rounded text-muted-foreground transition-colors opacity-60 hover:opacity-100"
      >
        {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>
    </div>
  );
}

export function ResponseLoadingIndicator({
  modelName = "quorvix-1",
  status = "Analyzing sources",
  className,
}: ResponseLoadingIndicatorProps) {
  return (
    <div
      className={cn(
        "inline-flex flex-col gap-1.5 px-4 py-3 rounded-xl",
        "bg-card/90 backdrop-blur-sm border border-border",
        "animate-in fade-in slide-in-from-bottom-2 duration-300",
        "font-mono",
        className
      )}
    >
      <div className="flex items-center gap-2">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
        </span>
        <span className="text-sm font-medium text-foreground">{modelName}</span>
        <PixelWave />
      </div>
      {status && status.length > 30 && (
        <CollapsibleStatus status={status} />
      )}
      {(!status || status.length <= 30) && (
        <div className="text-sm text-muted-foreground">{status || "Processing..."}</div>
      )}
    </div>
  );
}
