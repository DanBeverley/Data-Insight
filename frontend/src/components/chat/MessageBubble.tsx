import { cn } from "@/lib/utils";
import { VersionSlider } from "./VersionSlider";
import { Bot, User, Cpu, RefreshCw, Pencil, Check, X, Paperclip, Download, MoreVertical, Image as ImageIcon, FileDown, ExternalLink } from "lucide-react";
import { ReactNode, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { LoadingGrid } from "./LoadingGrid";
import { PlanProgress, Task } from "./PlanProgress";
import { PlotlyIframe } from "@/components/viz/PlotlyIframe";

function ArtifactWithDownload({ children, src, filename, type }: {
  children: ReactNode;
  src: string;
  filename?: string;
  type: 'image' | 'chart';
}) {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const derivedFilename = filename || src.split('/').pop() || 'artifact';

  const handleDownloadFile = async () => {
    try {
      const response = await fetch(src);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = derivedFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
    }
    setShowMenu(false);
  };

  const handleOpenInNewTab = () => {
    window.open(src, '_blank');
    setShowMenu(false);
  };

  return (
    <div className="relative group">
      {children}
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-1.5 bg-background/80 backdrop-blur-sm border border-border rounded-md shadow-sm hover:bg-accent transition-colors"
            title="Download options"
          >
            <Download className="w-4 h-4 text-muted-foreground" />
          </button>
          {showMenu && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowMenu(false)} />
              <div className="absolute right-0 top-full mt-1 z-50 min-w-[160px] bg-popover border border-border rounded-md shadow-lg py-1 animate-in fade-in slide-in-from-top-1 duration-150">
                {type === 'image' && (
                  <button
                    onClick={handleDownloadFile}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent transition-colors text-left"
                  >
                    <ImageIcon className="w-4 h-4" />
                    Download PNG
                  </button>
                )}
                {type === 'chart' && (
                  <button
                    onClick={handleDownloadFile}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent transition-colors text-left"
                  >
                    <FileDown className="w-4 h-4" />
                    Download HTML
                  </button>
                )}
                <button
                  onClick={handleOpenInNewTab}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent transition-colors text-left"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in new tab
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}


function InteractiveChart({ href, title }: { href: string; title: string }) {
  const [isOpen, setIsOpen] = useState(false);
  const [animationKey, setAnimationKey] = useState(0);

  const handleToggle = () => {
    if (!isOpen) {
      setAnimationKey(prev => prev + 1);
    }
    setIsOpen(!isOpen);
  };

  return (
    <div className="my-4 w-full">
      <button
        onClick={handleToggle}
        className="inline-flex items-center gap-2 text-primary hover:underline cursor-pointer select-none text-left"
      >
        <span className={cn("transition-transform duration-300 ease-out", isOpen && "rotate-90")}>▶</span>
        {title}
      </button>
      {isOpen && (
        <ArtifactWithDownload src={href} type="chart">
          <div
            key={animationKey}
            className="mt-3 rounded-xl overflow-hidden border border-border/50 shadow-xl animate-in fade-in slide-in-from-top-2 duration-400"
          >
            <PlotlyIframe
              src={href}
              title={title || 'Interactive Chart'}
              className="bg-transparent"
              height="500px"
            />
          </div>
        </ArtifactWithDownload>
      )}
    </div>
  );
}

interface MessageBubbleProps {
  id: string;
  role: 'user' | 'ai';
  content: string | ReactNode;
  timestamp: string;
  onRegenerate?: () => void;
  onEdit?: (newContent: string) => void;
  onOpenReport?: (path: string) => void;
  isTyping?: boolean;
  isLoading?: boolean;
  plan?: Task[];
  userAvatar?: string;
  messageId?: string;
  currentVersion?: number;
  totalVersions?: number;
  onVersionChange?: (version: number) => void;
}

export function MessageBubble({ id, role, content, timestamp, onRegenerate, onEdit, onOpenReport, isTyping, isLoading, plan, userAvatar, messageId, currentVersion = 1, totalVersions = 1, onVersionChange }: MessageBubbleProps) {
  const isUser = role === 'user';
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(typeof content === 'string' ? content : '');
  const [versionDirection, setVersionDirection] = useState<'prev' | 'next' | null>(null);

  const handleVersionChange = (newVersion: number) => {
    if (!onVersionChange) return;
    setVersionDirection(newVersion > currentVersion ? 'next' : 'prev');
    onVersionChange(newVersion);
  };

  const handleSave = () => {
    if (onEdit && editValue.trim()) {
      onEdit(editValue);
      setIsEditing(false);
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditValue(typeof content === 'string' ? content : '');
  };

  return (
    <div className={cn(
      "group flex w-full gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 relative",
      isUser ? "flex-row-reverse" : "flex-row"
    )}>
      <div className={cn(
        "flex h-10 w-10 shrink-0 items-center justify-center rounded-full border backdrop-blur-md shadow-lg z-10 transition-all duration-300 overflow-hidden",
        isUser
          ? "bg-primary/10 text-primary border-primary/30"
          : isLoading
            ? "bg-transparent border-transparent"
            : "bg-card text-card-foreground border-border"
      )}>
        {isUser ? (
          userAvatar ? (
            <img src={userAvatar} alt="" className="h-full w-full object-cover" />
          ) : (
            <User className="h-5 w-5" />
          )
        ) : isLoading ? null : <Cpu className="h-5 w-5" />}
      </div>

      <div className={cn(
        "relative rounded-2xl px-5 py-4 shadow-sm backdrop-blur-sm border transition-all duration-500 ease-in-out",
        isUser
          ? "max-w-[80%] bg-primary/10 border-primary/20 text-foreground rounded-tr-sm"
          : isLoading
            ? "max-w-[80%] bg-transparent border-transparent shadow-none px-0 py-2"
            : "w-full max-w-[1000px] bg-card border-border text-card-foreground rounded-tl-sm"
      )}>


        {isEditing ? (
          <div className="flex flex-col gap-2 min-w-[300px]">
            <Textarea
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              className="min-h-[100px] bg-muted border-border focus:border-primary/50 text-foreground"
            />
            <div className="flex justify-end gap-2">
              <Button size="sm" variant="ghost" onClick={handleCancel} className="h-7 text-xs hover:bg-muted">
                Cancel
              </Button>
              <Button size="sm" onClick={handleSave} className="h-7 text-xs bg-primary/20 hover:bg-primary/30 text-primary border border-primary/20">
                Save
              </Button>
            </div>
          </div>
        ) : (
          <div className={
            cn(
              "break-words leading-relaxed relative font-['Arimo',sans-serif]",
              !isUser && "prose prose-sm max-w-none prose-slate dark:prose-invert"
            )}>
            {isLoading ? (
              plan && plan.length > 0 ? (
                <PlanProgress plan={plan} />
              ) : (
                <LoadingGrid />
              )
            ) : typeof content === 'string' ? (
              <div className="animate-in fade-in duration-700">
                {plan && plan.length > 0 && (
                  <PlanProgress plan={plan} className="mb-4" />
                )}
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    code({ node, inline, className, children, ...props }: any) {
                      return inline ? (
                        <code className={cn(
                          "px-1.5 py-0.5 rounded font-mono text-sm",
                          isUser ? "bg-primary/20 text-primary-foreground" : "bg-muted text-primary"
                        )} {...props}>
                          {children}
                        </code>
                      ) : (
                        <code className="block bg-muted p-3 rounded-lg overflow-x-auto text-sm font-mono text-foreground" {...props}>
                          {children}
                        </code>
                      );
                    },
                    img({ src, alt, ...props }: any) {
                      const isHtml = src?.endsWith('.html');
                      const normalizedSrc = src?.startsWith('/static/') || src?.startsWith('http')
                        ? src
                        : `/static/plots/${src}`;
                      if (isHtml) {
                        return (
                          <ArtifactWithDownload src={normalizedSrc} type="chart">
                            <div className="my-4 rounded-lg overflow-hidden border border-border">
                              <PlotlyIframe
                                src={normalizedSrc}
                                title={alt || 'Interactive Chart'}
                                className="bg-transparent"
                                height="500px"
                              />
                            </div>
                          </ArtifactWithDownload>
                        );
                      }
                      return (
                        <ArtifactWithDownload src={normalizedSrc} type="image">
                          <img
                            src={normalizedSrc}
                            alt={alt}
                            className="max-w-full h-auto rounded-lg my-4 border border-border shadow-sm"
                            {...props}
                          />
                        </ArtifactWithDownload>
                      );
                    },
                    a({ href, children, ...props }: any) {
                      const isReportLink = href?.startsWith('report:') || href?.startsWith('/reports/');
                      if (isReportLink) {
                        const reportPath = href.startsWith('report:') ? href.replace('report:', '') : href;
                        return (
                          <button
                            onClick={(e) => { e.preventDefault(); e.stopPropagation(); onOpenReport?.(reportPath); }}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary/10 hover:bg-primary/20 text-primary font-medium rounded-md transition-colors"
                          >
                            {children}
                          </button>
                        );
                      }
                      const isHtmlArtifact = href?.endsWith('.html');
                      if (isHtmlArtifact) {
                        const normalizedHref = href?.startsWith('/static/') || href?.startsWith('http')
                          ? href
                          : `/static/plots/${href}`;
                        return <InteractiveChart href={normalizedHref} title={String(children)} />;
                      }
                      return <a href={href} className="text-primary hover:underline" {...props}>{children}</a>;
                    },
                    table({ children }) {
                      return (
                        <div className="overflow-x-auto my-4">
                          <table className="border-collapse border border-border w-full">
                            {children}
                          </table>
                        </div>
                      );
                    },
                    th({ children }) {
                      return <th className="border border-border px-4 py-2 text-left font-bold bg-muted/50">{children}</th>
                    },
                    td({ children }) {
                      return <td className="border border-border px-4 py-2">{children}</td>
                    }
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            ) : (
              content
            )}

            {!isUser && !isLoading && !(typeof content === 'string' && content.includes("System online")) && (
              <div className="flex items-center gap-0.5 mt-3">
                <button
                  onClick={() => {
                    const textContent = typeof content === 'string' ? content : '';
                    navigator.clipboard.writeText(textContent);
                  }}
                  className="p-1.5 text-muted-foreground/50 hover:text-foreground transition-colors"
                  title="Copy to clipboard"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                    <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                  </svg>
                </button>
                {onRegenerate && (
                  <button
                    onClick={onRegenerate}
                    className="p-1.5 text-muted-foreground/50 hover:text-foreground transition-colors"
                    title="Regenerate response"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </button>
                )}
                {totalVersions > 1 && (
                  <VersionSlider
                    messageId={messageId || id}
                    currentVersion={currentVersion}
                    totalVersions={totalVersions}
                    onVersionChange={handleVersionChange}
                    direction={versionDirection}
                  />
                )}
              </div>
            )}
          </div>
        )}

        {!isLoading && (
          <div className={cn(
            "absolute -bottom-7 text-[10px] font-medium text-muted-foreground/50",
            isUser ? "right-0" : "left-0"
          )}>
            {timestamp}
          </div>
        )}
      </div>
    </div >
  );
}