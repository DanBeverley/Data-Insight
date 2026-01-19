import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { authFetch } from "@/lib/authFetch";
import {
  MessageSquarePlus,
  LayoutDashboard,
  Database,
  Settings,
  Menu,
  X,
  ChevronLeft,
  ChevronRight,
  Brain,
  Pencil,
  Sun,
  Moon,
  Plug,
  LogOut,
  User as UserIcon
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useTheme } from "@/components/theme-provider";
import { SearchSettingsModal, loadSearchSettings, SearchSettings } from "@/components/chat/SearchSettingsModal";

interface Session {
  id: string;
  title: string;
  created_at: string;
}

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  currentView: 'chat' | 'dashboards' | 'datasets';
  onViewChange: (view: 'chat' | 'dashboards' | 'datasets') => void;
  currentSessionId: string;
  onSessionChange: (sessionId: string) => void;
  refreshTrigger?: number;
  onConnectDatabase?: () => void;
}

export function Sidebar({ isOpen, onToggle, currentView, onViewChange, currentSessionId, onSessionChange, refreshTrigger = 0, onConnectDatabase }: SidebarProps) {
  const [, setLocation] = useLocation();
  const { user, isGuest, logout } = useAuth();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [deleteConfirmation, setDeleteConfirmation] = useState<string | null>(null);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const { theme, setTheme } = useTheme();
  const [searchSettingsOpen, setSearchSettingsOpen] = useState(false);
  const [searchSettings, setSearchSettings] = useState<SearchSettings>(loadSearchSettings);

  useEffect(() => {
    fetchSessions();
  }, [currentSessionId, refreshTrigger]);

  const fetchSessions = async () => {
    try {
      const response = await authFetch('/api/sessions');
      if (response.ok) {
        const data = await response.json();
        setSessions(data);
      }
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    }
  };

  const handleNewSession = async () => {
    try {
      const response = await authFetch('/api/sessions/new', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        onSessionChange(data.session_id);
        onViewChange('chat');
        fetchSessions();
      }
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const confirmDeleteSession = async (sessionId: string) => {
    try {
      const response = await authFetch(`/api/sessions/${sessionId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setSessions(prev => {
          const newSessions = prev.filter(s => s.id !== sessionId);
          if (newSessions.length === 0) {
            // Auto-create new session if list is empty
            handleNewSession();
          } else if (currentSessionId === sessionId) {
            // Switch to another session if current one was deleted
            onSessionChange(newSessions[0].id);
          }
          return newSessions;
        });
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    } finally {
      setDeleteConfirmation(null);
    }
  };

  const handleDeleteClick = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    setDeleteConfirmation(sessionId);
  };

  const startEditing = (e: React.MouseEvent, session: Session) => {
    e.stopPropagation();
    setEditingSessionId(session.id);
    setEditTitle(session.title || "New Chat");
  };

  const saveTitle = async (sessionId: string) => {
    if (!editTitle.trim()) return;

    try {
      const response = await authFetch(`/api/sessions/${sessionId}/rename`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: editTitle })
      });

      if (response.ok) {
        setSessions(prev => prev.map(s =>
          s.id === sessionId ? { ...s, title: editTitle } : s
        ));
        setEditingSessionId(null);
      }
    } catch (error) {
      console.error('Failed to rename session:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent, sessionId: string) => {
    if (e.key === 'Enter') {
      saveTitle(sessionId);
    } else if (e.key === 'Escape') {
      setEditingSessionId(null);
    }
  };

  const NavItem = ({ icon: Icon, label, id }: { icon: any, label: string, id: 'chat' | 'dashboards' | 'datasets' }) => (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onViewChange(id)}
          className={cn(
            "h-10 w-10 rounded-xl transition-all duration-300",
            currentView === id
              ? "bg-primary/20 text-primary shadow-[0_0_15px_hsl(var(--primary)/0.3)] scale-105"
              : "text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
          )}
        >
          <Icon className="h-5 w-5" />
          <span className="sr-only">{label}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="right" className="bg-popover border-border text-popover-foreground font-medium">
        {label}
      </TooltipContent>
    </Tooltip>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <div
        className={cn(
          "hidden md:flex relative flex-col h-full bg-sidebar transition-all duration-300 ease-in-out z-50",
          isOpen ? "w-64" : "w-20"
        )}
      >
        {/* Logo Area */}
        <div className="flex items-center justify-center h-16 px-4 border-b border-sidebar-border overflow-hidden">
          <div className={cn("flex items-center gap-3", !isOpen && "justify-center")}>
            <div className="w-10 h-10 flex items-center justify-center bg-primary/10 rounded-xl border border-primary/20">
              <Brain className="w-6 h-6 text-primary" />
            </div>
            {isOpen && (
              <span className="font-bold text-lg tracking-tight text-sidebar-foreground whitespace-nowrap">
                Quorix <span className="text-primary">AI</span>
              </span>
            )}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex-1 py-6 flex flex-col gap-2 overflow-hidden">
          <div className="flex justify-center px-3">
            <Button
              onClick={handleNewSession}
              variant="ghost"
              size="icon"
              className={cn(
                "h-12 rounded-xl transition-all duration-300 bg-primary/10 text-primary hover:bg-primary/20",
                isOpen && "w-full justify-start gap-3 px-3"
              )}
            >
              <MessageSquarePlus className="h-5 w-5 flex-shrink-0" />
              {isOpen && <span className="font-medium">New Chat</span>}
            </Button>
          </div>

          <div className="my-2 px-4">
            <div className="h-px bg-sidebar-border" />
          </div>

          <div className="flex flex-col items-center space-y-1">
            <NavItem icon={MessageSquarePlus} label="Chat" id="chat" />
            <NavItem icon={LayoutDashboard} label="Dashboards" id="dashboards" />
            <NavItem icon={Database} label="Knowledge" id="datasets" />
            {onConnectDatabase && (
              <Tooltip delayDuration={0}>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={onConnectDatabase}
                    className="h-12 w-12 rounded-xl transition-all duration-300 text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
                  >
                    <Plug className="h-5 w-5" />
                    <span className="sr-only">Connect Database</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right" className="bg-popover border-border text-popover-foreground">
                  Connect Database
                </TooltipContent>
              </Tooltip>
            )}
          </div>

          <div className="my-2 px-4">
            <div className="h-px bg-sidebar-border" />
          </div>

          {/* Recent Sessions */}
          <div className="flex-1 min-h-0 overflow-y-auto px-3 space-y-1 custom-scrollbar">
            {isOpen && <div className="px-2 mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">Recent</div>}
            {sessions.map((session) => (
              <HoverCard key={session.id} openDelay={200} closeDelay={100}>
                <HoverCardTrigger asChild>
                  <div
                    className={cn(
                      "group flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-200",
                      currentSessionId === session.id
                        ? "bg-sidebar-accent text-sidebar-foreground"
                        : "text-muted-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent/50",
                      !isOpen && "justify-center px-0"
                    )}
                    onClick={() => onSessionChange(session.id)}
                  >
                    <div className={cn(
                      "relative flex-shrink-0 w-2 h-2 rounded-full transition-all duration-300",
                      currentSessionId === session.id ? "bg-primary scale-125" : "bg-muted-foreground/30 group-hover:bg-primary group-hover:scale-125"
                    )} />

                    {isOpen && (
                      <div className="flex-1 flex items-center justify-between min-w-0">
                        {editingSessionId === session.id ? (
                          <input
                            type="text"
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            onKeyDown={(e) => handleKeyDown(e, session.id)}
                            onClick={(e) => e.stopPropagation()}
                            onBlur={() => setEditingSessionId(null)}
                            autoFocus
                            className="w-full bg-transparent border-b border-primary/50 focus:outline-none text-sm font-medium text-sidebar-foreground px-0 py-0"
                          />
                        ) : (
                          <>
                            <span className="truncate text-sm font-medium">
                              {session.title || "New Session"}
                            </span>
                            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                              <button
                                onClick={(e) => startEditing(e, session)}
                                className="p-1 hover:bg-sidebar-accent rounded text-muted-foreground hover:text-sidebar-foreground transition-colors"
                              >
                                <Pencil className="w-3 h-3" />
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </HoverCardTrigger>
                <HoverCardContent side="right" className="w-64 p-4 bg-popover border-border text-popover-foreground backdrop-blur-xl z-50 shadow-2xl">
                  <div className="space-y-3">
                    <div>
                      {editingSessionId === session.id ? (
                        <input
                          type="text"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onKeyDown={(e) => handleKeyDown(e, session.id)}
                          onClick={(e) => e.stopPropagation()}
                          onBlur={() => setEditingSessionId(null)}
                          autoFocus
                          className="w-full bg-transparent border-b border-primary/50 focus:outline-none text-sm font-semibold text-popover-foreground px-0 py-1"
                        />
                      ) : (
                        <div className="flex items-center justify-between gap-2 group/card-header">
                          <h4 className="text-sm font-semibold text-popover-foreground truncate flex-1">{session.title || "New Session"}</h4>
                          <button
                            onClick={(e) => startEditing(e, session)}
                            className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-popover-foreground transition-colors opacity-0 group-hover/card-header:opacity-100"
                          >
                            <Pencil className="w-3 h-3" />
                          </button>
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground mt-1 flex flex-col gap-0.5">
                        <span>Created: {new Date(session.created_at).toLocaleDateString()}</span>
                        <span>Time: {new Date(session.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>
                    </div>
                    <div className="pt-2 border-t border-border">
                      <Button
                        variant="destructive"
                        size="sm"
                        className="w-full h-8 text-xs bg-destructive/20 text-destructive hover:bg-destructive/30 hover:text-destructive border border-destructive/20"
                        onClick={(e) => handleDeleteClick(e, session.id)}
                      >
                        <X className="w-3 h-3 mr-2" />
                        Delete Session
                      </Button>
                    </div>
                  </div>
                </HoverCardContent>
              </HoverCard>
            ))}
          </div>

          {/* Footer Actions */}
          <div className="mt-auto px-3 py-4 border-t border-sidebar-border flex flex-col items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className={cn(
                "h-10 w-10 rounded-xl text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground",
                isOpen && "w-full justify-start px-3 gap-3"
              )}
            >
              {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              {isOpen && <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSearchSettingsOpen(true)}
              className={cn(
                "h-10 w-10 rounded-xl text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground",
                isOpen && "w-full justify-start px-3 gap-3"
              )}
            >
              <Settings className="h-5 w-5" />
              {isOpen && <span>Settings</span>}
            </Button>

            {isGuest ? (
              <div className={cn("flex gap-2 mt-2", isOpen ? "flex-row" : "flex-col")}>
                <Button
                  variant="outline"
                  size={isOpen ? "default" : "icon"}
                  onClick={() => setLocation('/login')}
                  className={cn(
                    "border-primary/50 text-primary hover:bg-primary/10",
                    isOpen ? "flex-1" : "h-10 w-10"
                  )}
                >
                  {isOpen ? "Sign in" : <UserIcon className="h-5 w-5" />}
                </Button>
                <Button
                  size={isOpen ? "default" : "icon"}
                  onClick={() => setLocation('/signup')}
                  className={cn(
                    "bg-primary hover:bg-primary/90",
                    isOpen ? "flex-1" : "h-10 w-10"
                  )}
                >
                  {isOpen ? "Sign up" : <UserIcon className="h-5 w-5" />}
                </Button>
              </div>
            ) : (
              <div className={cn("flex items-center gap-2 mt-2 p-2 rounded-lg bg-sidebar-accent/50", !isOpen && "flex-col justify-center")}>
                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-medium text-sm">
                  {user?.avatar_url ? (
                    <img src={user.avatar_url} alt="" className="w-full h-full rounded-full object-cover" />
                  ) : (
                    user?.full_name?.charAt(0)?.toUpperCase() || user?.email?.charAt(0)?.toUpperCase() || 'U'
                  )}
                </div>
                {isOpen ? (
                  <>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-sidebar-foreground truncate">
                        {user?.full_name || user?.email?.split('@')[0]}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
                    </div>
                    <Button variant="ghost" size="icon" onClick={logout} className="h-8 w-8 text-muted-foreground hover:text-destructive">
                      <LogOut className="h-4 w-4" />
                    </Button>
                  </>
                ) : (
                  <Button variant="ghost" size="icon" onClick={logout} className="h-8 w-8 text-muted-foreground hover:text-destructive">
                    <LogOut className="h-4 w-4" />
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Sidebar Overlay */}
      <div
        className={cn(
          "fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden",
          isOpen ? "block" : "hidden"
        )}
        onClick={onToggle}
      />

      {/* Mobile Sidebar */}
      <div className={cn(
        "fixed inset-y-0 left-0 z-50 w-72 border-r border-sidebar-border bg-background/95 backdrop-blur-xl transition-transform duration-300 ease-in-out md:hidden",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex h-full flex-col p-4">
          <div className="flex items-center justify-between mb-8">
            <span className="font-display font-bold text-xl tracking-tight text-foreground">Quorix</span>
            <Button variant="ghost" size="icon" onClick={onToggle}>
              <X className="h-5 w-5" />
            </Button>
          </div>

          <Button onClick={handleNewSession} className="w-full mb-4 gap-2">
            <MessageSquarePlus className="h-4 w-4" /> New Chat
          </Button>

          <div className="space-y-1 overflow-y-auto">
            <h3 className="text-sm font-medium text-muted-foreground mb-2 px-2">History</h3>
            {sessions.map(session => (
              <div
                key={session.id}
                onClick={() => { onSessionChange(session.id); onToggle(); }}
                className={cn(
                  "flex items-center justify-between w-full px-3 py-2 rounded-lg text-sm transition-colors group cursor-pointer",
                  currentSessionId === session.id
                    ? "bg-primary/10 text-primary"
                    : "hover:bg-muted text-muted-foreground"
                )}
              >
                <span className="truncate">{session.title || "New Session"}</span>
                <button
                  onClick={(e) => handleDeleteClick(e, session.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-muted rounded transition-all text-muted-foreground hover:text-destructive"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteConfirmation} onOpenChange={(open) => !open && setDeleteConfirmation(null)}>
        <AlertDialogContent className="bg-popover border-border backdrop-blur-xl text-popover-foreground">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete chat?</AlertDialogTitle>
            <AlertDialogDescription className="text-muted-foreground">
              This will permanently delete this session and all its history. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-transparent border-border text-popover-foreground hover:bg-muted">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteConfirmation && confirmDeleteSession(deleteConfirmation)}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground border-none"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <SearchSettingsModal
        open={searchSettingsOpen}
        onOpenChange={setSearchSettingsOpen}
        settings={searchSettings}
        onSave={setSearchSettings}
      />
    </>
  );
}