import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Sparkles, Mic, ArrowUp, File as FileIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { MessageBubble } from "./MessageBubble";
import { ChartBlock } from "../viz/ChartBlock";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Task } from "./PlanProgress";
import { useAuth } from "@/contexts/AuthContext";
import { authFetch, getAuthToken } from "@/lib/authFetch";
import { ThinkingStream, ThinkingSection, TaskItem } from "./ThinkingStream";
import { PlusMenu } from "./PlusMenu";

interface Message {
  id: string;
  role: 'user' | 'ai';
  content: string | React.ReactNode;
  timestamp: string;
  plan?: Task[];
  visualizations?: Array<{ data: any; id: string }>;
}

interface ChatAreaProps {
  onTriggerReport?: (reportPath: string) => void;
  sessionId: string;
  onSessionUpdate?: () => void;
}

export function ChatArea({ onTriggerReport, sessionId, onSessionUpdate }: ChatAreaProps) {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [persistedUploads, setPersistedUploads] = useState<{ filename: string; rows: number; columns: number }[]>([]);
  const [reportMode, setReportMode] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [thinkingSections, setThinkingSections] = useState<ThinkingSection[]>([]);
  const [streamTasks, setStreamTasks] = useState<TaskItem[]>([]);
  const [lastReportPath, setLastReportPath] = useState<string>("");
  const [loadingStatus, setLoadingStatus] = useState<string>("Initializing...");
  const [currentModelName, setCurrentModelName] = useState<string>("quorvix-1");

  useEffect(() => {
    if (!user) {
      setMessages([]);
      setPersistedUploads([]);
    }
  }, [user]);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const response = await authFetch(`/api/sessions/${sessionId}/messages`);
        if (response.ok) {
          const data = await response.json();
          const messagesList = Array.isArray(data) ? data : (data.messages || []);

          if (messagesList.length > 0) {
            const loadedMessages = messagesList.map((msg: any, idx: number) => ({
              id: msg.id || `loaded-${idx}`,
              role: (msg.role === 'human' || msg.role === 'user' || msg.type === 'human') ? 'user' : 'ai',
              content: msg.content,
              timestamp: new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              // Note: History loading of complex artifacts like visualizations might need valid JSON parsing from metadata if stored there
              // For now we rely on the live session or re-rendering if stored in content.
              // Ideally, backend should return 'metadata' field with 'plots' or 'visualizations' data.
            }));
            setMessages(loadedMessages);
          } else {
            // No history, do not set default message to show empty state
            setMessages([]);
          }
        }
      } catch (error) {
        console.error("Failed to load message history:", error);
        setMessages([]);
      } finally {
        setLoadingHistory(false);
      }
    };

    setMessages([]);
    setPersistedUploads([]);

    if (sessionId) {
      setLoadingHistory(true);
      loadHistory();
      fetch(`/api/data/${sessionId}/uploads`)
        .then(res => res.json())
        .then(data => {
          if (data.files && data.files.length > 0) {
            setPersistedUploads(data.files);
          }
        })
        .catch(() => { });
    } else {
      setLoadingHistory(false);
    }
  }, [sessionId]);

  // Send heartbeat every 30 seconds while agent is processing (isTyping)
  useEffect(() => {
    if (!isTyping || !sessionId) return;

    const sendHeartbeat = () => {
      fetch(`/api/heartbeat/${sessionId}`, { method: 'POST' })
        .catch(err => console.debug('Heartbeat failed:', err));
    };

    // Send initial heartbeat
    sendHeartbeat();

    // Set interval for subsequent heartbeats
    const interval = setInterval(sendHeartbeat, 30000);

    return () => clearInterval(interval);
  }, [isTyping, sessionId]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      const scrollContainer = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages, isTyping]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setAttachedFiles(prev => [...prev, ...newFiles]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const removeAttachment = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSend = async () => {
    if (!input.trim() && attachedFiles.length === 0) return;

    const userMsgContent = input + (attachedFiles.length > 0 ? `\n\n📎 Attached: ${attachedFiles.map(f => f.name).join(', ')}` : '');
    const userMsgId = Date.now().toString();
    const newMessage: Message = {
      id: userMsgId,
      role: 'user',
      content: userMsgContent,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newMessage]);
    const messagesToSend = userMsgContent;
    const filesToUpload = [...attachedFiles];
    setInput("");
    setAttachedFiles([]);
    setIsTyping(true);
    setLoadingStatus("Analyzing request...");

    // Initial AI message placeholder
    let aiMsgId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: aiMsgId,
      role: 'ai',
      content: "",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);

    try {
      // Upload files first if any
      if (filesToUpload.length > 0) {
        setIsUploading(true);
        // Status message removed per user request
        const formData = new FormData();
        filesToUpload.forEach(file => formData.append('files', file));
        formData.append('session_id', sessionId);
        formData.append('enable_profiling', 'true');

        const token = getAuthToken();
        await fetch('/api/upload', {
          method: 'POST',
          body: formData,
          headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        setIsUploading(false);
      }

      const token = getAuthToken();
      const response = await fetch(`/api/agent/chat-stream?message=${encodeURIComponent(messagesToSend)}&session_id=${sessionId}&token_streaming=true&thinking_mode=${reportMode}`, {
        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'thinking_start') {
                const sectionId = `${data.agent}-${Date.now()}`;
                const statusMap: Record<string, string> = {
                  brain: 'Planning analysis...',
                  hands: 'Executing code...',
                  verifier: 'Verifying results...',
                };
                setLoadingStatus(statusMap[data.agent] || 'Processing...');
                if (data.model_name) {
                  setCurrentModelName(data.model_name);
                }
                setThinkingSections(prev => [...prev, {
                  id: sectionId,
                  agent: data.agent,
                  summary: data.agent === 'brain' ? 'Analyzing...' : 'Executing...',
                  tokens: [],
                  isComplete: false
                }]);
              } else if (data.type === 'thinking_complete') {
                setThinkingSections(prev => prev.map(section =>
                  section.agent === data.agent && !section.isComplete
                    ? { ...section, isComplete: true }
                    : section
                ));
              } else if (data.type === 'task') {
                setLoadingStatus(data.description || 'Working on task...');
                setStreamTasks(prev => [...prev, {
                  id: `task-${Date.now()}`,
                  description: data.description,
                  status: data.status || 'pending'
                }]);
              } else if (data.type === 'task_update') {
                setStreamTasks(prev => prev.map((task, idx) =>
                  idx === prev.length - 1 ? { ...task, status: data.status } : task
                ));
              } else if (data.type === 'plan') {
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? { ...msg, plan: data.plan } : msg
                ));
              } else if (data.type === 'token') {
                setLoadingStatus('Generating response...');
                aiContent += data.content;
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? { ...msg, content: aiContent } : msg
                ));
              } else if (data.type === 'visualization') {
                setLoadingStatus('Creating visualizations...');
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? {
                    ...msg,
                    visualizations: [...(msg.visualizations || []), { data: data.data, id: data.id }]
                  } : msg
                ));
              } else if (data.type === 'report_generation_started') {
                console.log('[SSE] Report generation started');
                if (onTriggerReport) {
                  onTriggerReport("");
                }
              } else if (data.type === 'report_generated') {
                console.log('[SSE] Report generated:', data.report_path);
                setLastReportPath(data.report_path || "");
                if (onTriggerReport) {
                  onTriggerReport(data.report_path || "");
                }
              } else if (data.type === 'final_response') {
                console.log('[SSE] Received final_response:', data);
                aiContent = data.response || aiContent;
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? { ...msg, content: aiContent } : msg
                ));
                setTimeout(() => {
                  setThinkingSections([]);
                  setStreamTasks([]);
                }, 1500);

                if (onSessionUpdate) {
                  setTimeout(() => onSessionUpdate(), 1000);
                }
              } else if (data.type === 'error') {
                aiContent += `\n[Error: ${data.content}]`;
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? { ...msg, content: aiContent } : msg
                ));
              }
            } catch (e) {
              console.error("Error parsing SSE:", e, line);
            }
          }
        }
      }

    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, {
        id: (Date.now() + 2).toString(),
        role: 'ai',
        content: "Error communicating with the agent. Please try again.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally {
      setIsTyping(false);
      setIsUploading(false);
    }
  };

  const handleRegenerate = async (aiMsgId: string) => {
    const msgIndex = messages.findIndex(m => m.id === aiMsgId);
    if (msgIndex <= 0) return;

    let userMsg: Message | null = null;
    for (let i = msgIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        userMsg = messages[i];
        break;
      }
    }

    if (!userMsg || typeof userMsg.content !== 'string') return;

    const userContent = userMsg.content.split('\n\n📎')[0];

    setMessages(prev => prev.map(msg =>
      msg.id === aiMsgId ? { ...msg, content: "", plan: undefined } : msg
    ));
    setIsTyping(true);

    try {
      const token = getAuthToken();
      const response = await fetch(`/api/agent/chat-stream?message=${encodeURIComponent(userContent)}&session_id=${sessionId}&token_streaming=true&regenerate=true&thinking_mode=${reportMode}`, {
        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
      });

      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'plan') {
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? { ...msg, plan: data.plan } : msg
                ));
              } else if (data.type === 'token') {
                aiContent += data.content;
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? { ...msg, content: aiContent } : msg
                ));
              } else if (data.type === 'final_response') {
                aiContent = data.response || aiContent;
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMsgId ? { ...msg, content: aiContent } : msg
                ));
              }
            } catch (e) {
              console.error("Error parsing SSE:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Regenerate error:", error);
      setMessages(prev => prev.map(msg =>
        msg.id === aiMsgId ? { ...msg, content: "Error regenerating response. Please try again." } : msg
      ));
    } finally {
      setIsTyping(false);
    }
  };

  const handleEdit = (id: string, newContent: string) => {
    // Placeholder for edit logic
    console.log("Edit", id, newContent);
  };

  return (
    <div className="flex flex-col h-full relative z-10">
      <input
        type="file"
        multiple
        className="hidden"
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept=".csv,.xlsx,.xls,.json,.parquet,.png,.jpg,.jpeg,.webp,.gif"
      />
      <ScrollArea ref={scrollRef} className="flex-1 p-4 md:p-8">
        <div className="max-w-4xl mx-auto space-y-8 pb-4 h-full">
          {loadingHistory ? (
            <div className="flex items-center justify-center py-10 text-muted-foreground">
              <Sparkles className="h-4 w-4 animate-pulse mr-2" />
              <span>Loading session...</span>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[80vh] text-center space-y-8 animate-in fade-in zoom-in duration-500">
              <div className="relative">
                <div className="absolute -inset-4 bg-primary/20 rounded-full blur-xl animate-pulse"></div>
                <div className="relative bg-card/50 p-6 rounded-3xl border border-border shadow-2xl">
                  <Sparkles className="w-16 h-16 text-primary" />
                </div>
              </div>

              <div className="w-full max-w-2xl relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-primary/30 to-purple-500/30 rounded-2xl opacity-20 group-hover:opacity-50 transition duration-500 blur"></div>
                <div className="relative bg-card/60 backdrop-blur-xl rounded-2xl border border-border p-2 shadow-2xl flex flex-col gap-2">
                  {(persistedUploads.length > 0 || attachedFiles.length > 0) && (
                    <div className="flex flex-wrap gap-2 px-2 pt-2">
                      {persistedUploads.map((upload, idx) => (
                        <div key={`persisted-${idx}`} className="flex items-center gap-2 px-3 py-1.5 bg-primary/10 border border-primary/20 rounded-full text-xs">
                          <FileIcon className="h-3 w-3 text-primary" />
                          <span className="text-primary">{upload.filename}</span>
                          <span className="text-muted-foreground text-[10px]">({upload.rows}×{upload.columns})</span>
                          <button
                            onClick={() => setPersistedUploads(prev => prev.filter((_, i) => i !== idx))}
                            className="text-muted-foreground hover:text-destructive transition-colors"
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      ))}
                      {attachedFiles.map((file, idx) => (
                        <div key={idx} className="flex items-center gap-2 px-3 py-1.5 bg-muted/50 border border-border rounded-full text-xs animate-in fade-in zoom-in duration-200">
                          <FileIcon className="h-3 w-3 text-primary" />
                          <span className="text-foreground/80">{file.name}</span>
                          <button onClick={() => removeAttachment(idx)} className="text-muted-foreground hover:text-destructive transition-colors">
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <PlusMenu
                      onFileSelect={() => fileInputRef.current?.click()}
                      reportMode={reportMode}
                      onReportModeChange={setReportMode}
                    />
                    <input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSend();
                      }}
                      placeholder={reportMode ? "Describe the report you want generated..." : "What do you want to know?"}
                      className={cn(
                        "flex-1 bg-transparent border-none outline-none text-foreground placeholder:text-muted-foreground/50 h-10 px-2 transition-colors duration-300",
                        reportMode && "placeholder:text-primary/50"
                      )}
                      autoFocus
                    />
                    <Button
                      onClick={handleSend}
                      size="icon"
                      disabled={!input.trim() && attachedFiles.length === 0}
                      className={cn(
                        "h-10 w-10 rounded-xl transition-all duration-300 flex-shrink-0",
                        (input.trim() || attachedFiles.length > 0) ? "bg-primary text-primary-foreground shadow-[0_0_15px_rgba(0,242,234,0.4)]" : "bg-muted text-muted-foreground hover:bg-muted/80"
                      )}
                    >
                      <ArrowUp className="h-5 w-5" />
                    </Button>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap justify-center gap-3">
                <Button variant="outline" className="rounded-full border-border bg-card/50 hover:bg-card hover:text-primary transition-all" onClick={() => { setInput("Analyze the uploaded dataset"); }}>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Analyze Data
                </Button>
                <Button variant="outline" className="rounded-full border-border bg-card/50 hover:bg-card hover:text-primary transition-all" onClick={() => { setInput("Generate a comprehensive report"); }}>
                  <FileIcon className="w-4 h-4 mr-2" />
                  Generate Report
                </Button>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) => (
                <div key={msg.id} className="flex flex-col gap-2">
                  {/* ThinkingStream removed - legacy duplicate */}

                  <MessageBubble
                    id={msg.id}
                    role={msg.role}
                    content={msg.content}
                    timestamp={msg.timestamp}
                    onRegenerate={msg.role === 'ai' ? () => handleRegenerate(msg.id) : undefined}
                    onEdit={msg.role === 'user' ? (newContent) => handleEdit(msg.id, newContent) : undefined}
                    onOpenReport={(path) => onTriggerReport?.(path)}
                    isTyping={isTyping && msg.role === 'ai' && idx === messages.length - 1}
                    isLoading={isTyping && msg.role === 'ai' && idx === messages.length - 1 && msg.content === ""}
                    loadingStatus={loadingStatus}
                    modelName={currentModelName}
                    plan={msg.plan}
                    userAvatar={user?.avatar_url}
                  />
                </div>
              ))}
            </>
          )}
        </div>
      </ScrollArea>

      {messages.length > 0 && (
        <div className="p-6">
          <div className="max-w-3xl mx-auto relative group">
            {/* Glowing border effect */}
            <div className="absolute -inset-0.5 bg-gradient-to-r from-primary/30 to-purple-500/30 rounded-2xl opacity-20 group-hover:opacity-50 transition duration-500 blur"></div>

            <div className="relative bg-card/60 backdrop-blur-xl rounded-2xl border border-border p-2">
              {/* Attached files display */}
              {/* Attached & Persisted files display */}
              {/* Attached files display */}
              {attachedFiles.length > 0 && (
                <div className="flex flex-wrap gap-2 p-2 mb-2 border-b border-border">
                  {attachedFiles.map((file, idx) => (
                    <div key={`attached-${idx}`} className="flex items-center gap-2 px-3 py-1.5 bg-muted/50 border border-border rounded-lg text-xs">
                      <FileIcon className="h-3 w-3 text-foreground" />
                      <span className="text-foreground">{file.name}</span>
                      <button onClick={() => removeAttachment(idx)} className="text-muted-foreground hover:text-destructive">
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="flex items-end gap-2">
                <PlusMenu
                  onFileSelect={() => fileInputRef.current?.click()}
                  reportMode={reportMode}
                  onReportModeChange={setReportMode}
                />

                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder={reportMode ? "Describe the report you want generated..." : "Ask anything..."}
                  className={cn(
                    "min-h-[40px] w-full resize-none bg-transparent border-none focus-visible:ring-0 shadow-none py-2 px-4 text-base placeholder:text-muted-foreground/50 text-foreground transition-colors duration-300",
                    reportMode && "placeholder:text-primary/50"
                  )}
                  rows={1}
                />

                <Button
                  onClick={handleSend}
                  size="icon"
                  disabled={!input.trim() && attachedFiles.length === 0}
                  className={cn(
                    "h-10 w-10 rounded-xl transition-all duration-300",
                    (input.trim() || attachedFiles.length > 0) ?
                      (reportMode ? "bg-primary text-primary-foreground shadow-[0_0_15px_rgba(0,242,234,0.4)]" : "bg-primary text-primary-foreground shadow-[0_0_15px_rgba(0,242,234,0.4)]")
                      : "bg-muted text-muted-foreground hover:bg-muted/80"
                  )}
                >
                  <ArrowUp className="h-5 w-5" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}