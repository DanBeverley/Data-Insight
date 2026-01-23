import { cn } from "@/lib/utils";

export function LoadingGrid() {
    return (
        <div className="grid grid-cols-3 gap-1 p-2">
            {[...Array(9)].map((_, i) => (
                <div
                    key={i}
                    className={cn(
                        "w-1.5 h-1.5 bg-primary rounded-full animate-pulse",
                        i === 4 ? "animate-[pulse_1s_ease-in-out_infinite]" : "", // Center dot
                        i % 2 === 0 ? "animate-[pulse_1s_ease-in-out_0.2s_infinite]" : "animate-[pulse_1s_ease-in-out_0.4s_infinite]" // Outer dots
                    )}
                    style={{
                        animationDelay: `${i * 0.1}s`
                    }}
                />
            ))}
        </div>
    );
}
