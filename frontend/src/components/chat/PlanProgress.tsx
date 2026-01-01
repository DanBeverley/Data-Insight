import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { CheckCircle2, Loader2, Circle, XCircle } from 'lucide-react';

export interface Task {
    id: string;
    description: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    assigned_to?: string;
    result?: string;
    error?: string;
}

interface PlanProgressProps {
    plan: Task[];
    className?: string;
}

export function PlanProgress({ plan, className }: PlanProgressProps) {
    if (!plan || plan.length === 0) return null;

    // Find the current active task or the first pending one
    const activeIndex = plan.findIndex(t => t.status === 'in_progress');
    const displayIndex = activeIndex !== -1 ? activeIndex : plan.findIndex(t => t.status === 'pending');

    // If all completed, show the last one
    const targetIndex = displayIndex !== -1 ? displayIndex : plan.length - 1;
    const activeTask = plan[targetIndex];

    if (!activeTask) return null;

    return (
        <div className={cn("relative w-full max-w-md my-4", className)}>
            <AnimatePresence mode="wait">
                <motion.div
                    key={activeTask.id || targetIndex}
                    initial={{ x: 50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: -50, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    className="relative"
                >
                    {/* Dot Indicator (Outside Border) */}
                    <div className="absolute -left-4 top-1/2 -translate-y-1/2 flex flex-col items-center gap-1">
                        <div className={cn(
                            "w-2 h-2 rounded-full transition-colors duration-300",
                            activeTask.status === 'in_progress' ? "bg-primary animate-pulse" :
                                activeTask.status === 'completed' ? "bg-green-500" :
                                    activeTask.status === 'failed' ? "bg-destructive" : "bg-muted-foreground/30"
                        )} />
                        {/* Connecting Line */}
                        {targetIndex < plan.length - 1 && (
                            <div className="w-0.5 h-full bg-border/50 absolute top-4" />
                        )}
                    </div>

                    {/* Card Content */}
                    <div className={cn(
                        "bg-card/80 backdrop-blur-md border border-border/50 rounded-xl p-4 shadow-sm ml-2",
                        activeTask.status === 'in_progress' && "border-primary/20 shadow-[0_0_15px_rgba(0,0,0,0.05)] dark:shadow-[0_0_15px_rgba(255,255,255,0.02)]"
                    )}>
                        <div className="flex items-start justify-between gap-3">
                            <div className="flex-1">
                                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                                    Step {targetIndex + 1} of {plan.length}
                                </h4>
                                <p className="text-sm font-medium text-foreground leading-snug">
                                    {activeTask.description}
                                </p>
                                {activeTask.status === 'failed' && activeTask.error && (
                                    <p className="text-xs text-destructive mt-2 bg-destructive/10 p-2 rounded">
                                        {activeTask.error}
                                    </p>
                                )}
                            </div>

                            <div className="shrink-0">
                                {activeTask.status === 'in_progress' && (
                                    <Loader2 className="w-5 h-5 text-primary animate-spin" />
                                )}
                                {activeTask.status === 'completed' && (
                                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                                )}
                                {activeTask.status === 'failed' && (
                                    <XCircle className="w-5 h-5 text-destructive" />
                                )}
                                {activeTask.status === 'pending' && (
                                    <Circle className="w-5 h-5 text-muted-foreground/30" />
                                )}
                            </div>
                        </div>
                    </div>
                </motion.div>
            </AnimatePresence>
        </div>
    );
}
