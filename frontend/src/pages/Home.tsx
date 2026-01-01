import { useState, useEffect, useCallback, useRef } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { DashboardView } from "@/components/dashboard/DashboardView";
import { DatasetView } from "@/components/dashboard/DatasetView";
import { ReportPanel } from "@/components/report/ReportPanel";
import { DatabaseModal } from "@/components/database/DatabaseModal";
import { useNotifications } from "@/hooks/useNotifications";
import { useAuth } from "@/contexts/AuthContext";
import { authFetch } from "@/lib/authFetch";
import { Button } from "@/components/ui/button";
import { Menu, ChevronLeft } from "lucide-react";
import { ParticlesBackground } from "@/components/layout/ParticlesBackground";
import { ScrollArea } from "@/components/ui/scroll-area";

type ViewState = 'chat' | 'dashboards' | 'datasets';

export default function Home() {
  const { user, isLoading: authLoading } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentView, setCurrentView] = useState<ViewState>('chat');
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionRefreshTrigger, setSessionRefreshTrigger] = useState(0);

  const [reportPath, setReportPath] = useState<string>("");
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [isDbModalOpen, setIsDbModalOpen] = useState(false);

  const isLoadingRef = useRef(false);
  useNotifications();

  useEffect(() => {
    if (authLoading) return;

    const initSession = async () => {
      try {
        if (user) {
          const savedSessionId = localStorage.getItem('current_session_id');
          if (savedSessionId && !savedSessionId.startsWith('guest_')) {
            setSessionId(savedSessionId);
            return;
          }

          localStorage.removeItem('current_session_id');

          const response = await authFetch('/api/sessions');
          if (response.ok) {
            const sessions = await response.json();
            if (sessions?.length > 0) {
              setSessionId(sessions[0].id);
              localStorage.setItem('current_session_id', sessions[0].id);
              return;
            }
          }

          const createResponse = await authFetch('/api/sessions/new', { method: 'POST' });
          if (createResponse.ok) {
            const data = await createResponse.json();
            setSessionId(data.session_id);
            localStorage.setItem('current_session_id', data.session_id);
          }
        } else {
          const response = await fetch('/api/sessions/new', { method: 'POST' });
          if (response.ok) {
            const data = await response.json();
            setSessionId(data.session_id);
          }
        }
      } catch (error) {
        console.error('Session initialization error:', error);
        const response = await fetch('/api/sessions/new', { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          setSessionId(data.session_id);
        }
      }
    };

    initSession();
  }, [user, authLoading]);

  // Load report for current session
  useEffect(() => {
    if (!sessionId) return;

    isLoadingRef.current = true;

    // Clear current state first
    setReportPath("");
    setIsReportOpen(false);

    // Load from localStorage for this specific session
    try {
      const saved = localStorage.getItem(`report_${sessionId}`);
      if (saved) {
        const data = JSON.parse(saved);
        if (data.path) {
          setReportPath(data.path);
        }
      }
    } catch {
      localStorage.removeItem(`report_${sessionId}`);
    }

    localStorage.setItem('current_session_id', sessionId);

    // Allow saves after load completes
    setTimeout(() => { isLoadingRef.current = false; }, 100);
  }, [sessionId]);

  // Save report when path changes (but not during load)
  useEffect(() => {
    if (!sessionId || isLoadingRef.current) return;

    if (reportPath) {
      localStorage.setItem(`report_${sessionId}`, JSON.stringify({ path: reportPath }));
    }
  }, [reportPath, sessionId]);

  const handleTriggerReport = useCallback((path: string) => {
    if (path) {
      setReportPath(path);
      setIsReportOpen(true);
    } else if (reportPath) {
      setIsReportOpen(true);
    } else {
      setIsReportOpen(prev => !prev);
    }
  }, [reportPath]);

  const handleSessionUpdate = useCallback(() => {
    setSessionRefreshTrigger(prev => prev + 1);
  }, []);

  const hasReport = reportPath !== "";

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground font-sans selection:bg-primary/30 selection:text-primary">
      <ParticlesBackground />

      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        currentView={currentView}
        onViewChange={setCurrentView}
        currentSessionId={sessionId}
        onSessionChange={setSessionId}
        refreshTrigger={sessionRefreshTrigger}
        onConnectDatabase={() => setIsDbModalOpen(true)}
      />

      <main className="flex-1 flex h-full relative min-w-0 bg-background overflow-hidden">
        <div className="md:hidden absolute top-4 left-4 z-10">
          <Button variant="outline" size="icon" onClick={() => setSidebarOpen(true)} className="bg-background/20 backdrop-blur-md border-white/10">
            <Menu className="h-5 w-5" />
          </Button>
        </div>

        <div className="flex h-full w-full">
          <div className="flex-1 flex flex-col h-full min-w-0 relative transition-all duration-500">
            {currentView === 'chat' && (
              <ChatArea
                onTriggerReport={handleTriggerReport}
                sessionId={sessionId}
                onSessionUpdate={handleSessionUpdate}
              />
            )}

            {currentView !== 'chat' && (
              <ScrollArea className="h-full">
                {currentView === 'dashboards' && <DashboardView sessionId={sessionId} />}
                {currentView === 'datasets' && <DatasetView sessionId={sessionId} />}
              </ScrollArea>
            )}
          </div>

          {hasReport && !isReportOpen && (
            <Button
              variant="outline"
              size="icon"
              onClick={() => setIsReportOpen(true)}
              className="fixed right-0 top-1/2 -translate-y-1/2 z-50 h-12 w-8 rounded-l-lg rounded-r-none bg-primary/90 hover:bg-primary border-0 shadow-lg transition-all duration-300 hover:w-10"
              title="Open Report Panel"
            >
              <ChevronLeft className="h-5 w-5 text-primary-foreground" />
            </Button>
          )}

          <div className={`flex-shrink-0 transition-all duration-300 ease-in-out ${isReportOpen ? 'w-[600px] opacity-100' : 'w-0 opacity-0'}`}>
            {isReportOpen && (
              <ReportPanel
                key={sessionId}
                isOpen={isReportOpen}
                onClose={() => setIsReportOpen(false)}
                reportPath={reportPath}
                sessionId={sessionId}
              />
            )}
          </div>
        </div>
      </main>

      <DatabaseModal
        isOpen={isDbModalOpen}
        onClose={() => setIsDbModalOpen(false)}
        sessionId={sessionId}
        onDataLoaded={(tableName, rowCount) => {
          console.log(`Loaded ${rowCount} rows from ${tableName}`);
          setSessionRefreshTrigger(prev => prev + 1);
        }}
      />
    </div>
  );
}