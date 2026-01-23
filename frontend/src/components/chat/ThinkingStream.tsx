import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Check, Circle, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ThinkingSection {
    id: string;
    agent: 'brain' | 'hands' | 'verifier';
    summary: string;
    tokens: string[];
    isComplete: boolean;
}

export interface TaskItem {
    id: string;
    description: string;
    status: 'pending' | 'in_progress' | 'complete';
}

interface ThinkingStreamProps {
    sections: ThinkingSection[];
    tasks: TaskItem[];
    currentAgent?: string;
}

const agentLabels = {
    brain: 'Analyzing',
    hands: 'Executing',
    verifier: 'Validating'
};

export function ThinkingStream({ sections, tasks, currentAgent }: ThinkingStreamProps) {
    const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

    const toggleSection = (id: string) => {
        setExpandedSections(prev => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
    };

    if (sections.length === 0 && tasks.length === 0) {
        return null;
    }

    return (
        <div className="space-y-2 py-3 font-mono text-sm">
            <AnimatePresence mode="popLayout">
                {sections.map((section) => (
                    <ThinkingSectionItem
                        key={section.id}
                        section={section}
                        isExpanded={expandedSections.has(section.id)}
                        onToggle={() => toggleSection(section.id)}
                    />
                ))}
            </AnimatePresence>

            {tasks.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 space-y-1"
                >
                    {tasks.map((task) => (
                        <TaskItemRow key={task.id} task={task} />
                    ))}
                </motion.div>
            )}
        </div>
    );
}

function ThinkingSectionItem({
    section,
    isExpanded,
    onToggle
}: {
    section: ThinkingSection;
    isExpanded: boolean;
    onToggle: () => void;
}) {
    const tokensRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (isExpanded && tokensRef.current) {
            tokensRef.current.scrollTop = tokensRef.current.scrollHeight;
        }
    }, [section.tokens, isExpanded]);

    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.2 }}
            className="relative"
        >
            <div
                className={cn(
                    "flex items-start gap-2 cursor-pointer group",
                    section.isComplete && "opacity-60"
                )}
                onClick={onToggle}
            >
                <motion.span
                    className={cn(
                        "mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0",
                        section.isComplete ? "bg-muted-foreground/50" : "bg-primary"
                    )}
                    animate={section.isComplete ? {} : { scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                />

                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <motion.span
                            className={cn(
                                "text-muted-foreground",
                                section.isComplete && "line-through decoration-muted-foreground/50"
                            )}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            {section.summary || `${agentLabels[section.agent]}...`}
                        </motion.span>

                        <motion.div
                            animate={{ rotate: isExpanded ? 180 : 0 }}
                            transition={{ duration: 0.2 }}
                            className="opacity-50 group-hover:opacity-100 transition-opacity"
                        >
                            <ChevronDown className="h-3 w-3 text-muted-foreground" />
                        </motion.div>
                    </div>

                    <AnimatePresence>
                        {isExpanded && section.tokens.length > 0 && (
                            <motion.div
                                ref={tokensRef}
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="mt-1 pl-2 border-l border-border/50 max-h-32 overflow-y-auto"
                            >
                                <p className="text-xs text-muted-foreground/70 italic leading-relaxed">
                                    {section.tokens.map((token, i) => (
                                        <motion.span
                                            key={i}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ duration: 0.05, delay: i * 0.01 }}
                                        >
                                            {token}
                                        </motion.span>
                                    ))}
                                </p>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>


        </motion.div>
    );
}

function TaskItemRow({ task }: { task: TaskItem }) {
    const statusIcons = {
        pending: <Circle className="h-3 w-3 text-muted-foreground/50" />,
        in_progress: <ArrowRight className="h-3 w-3 text-primary" />,
        complete: <Check className="h-3 w-3 text-green-500" />
    };

    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: -5 }}
            animate={{ opacity: 1, x: 0 }}
            className={cn(
                "flex items-center gap-2 text-xs",
                task.status === 'complete' && "opacity-60"
            )}
        >
            <motion.div
                animate={task.status === 'in_progress' ? { x: [0, 2, 0] } : {}}
                transition={{ repeat: Infinity, duration: 0.8 }}
            >
                {statusIcons[task.status]}
            </motion.div>
            <span className={cn(
                "text-muted-foreground truncate max-w-md",
                task.status === 'complete' && "line-through"
            )} title={task.description}>
                {task.description.split('\n')[0].length > 80
                    ? task.description.split('\n')[0].slice(0, 80) + '...'
                    : task.description.split('\n')[0]}
            </span>
        </motion.div>
    );
}

export function StreamingText({ text, speed = 10 }: { text: string; speed?: number }) {
    const [displayedText, setDisplayedText] = useState('');
    const [currentIndex, setCurrentIndex] = useState(0);

    useEffect(() => {
        if (currentIndex < text.length) {
            const timer = setTimeout(() => {
                setDisplayedText(text.slice(0, currentIndex + 1));
                setCurrentIndex(currentIndex + 1);
            }, speed);
            return () => clearTimeout(timer);
        }
    }, [text, currentIndex, speed]);

    useEffect(() => {
        setCurrentIndex(0);
        setDisplayedText('');
    }, [text]);

    return (
        <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
        >
            {displayedText}
            {currentIndex < text.length && (
                <motion.span
                    className="inline-block w-1 h-4 bg-primary ml-0.5"
                    animate={{ opacity: [1, 0] }}
                    transition={{ repeat: Infinity, duration: 0.5 }}
                />
            )}
        </motion.span>
    );
}
