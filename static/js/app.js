class DataInsightApp {
    constructor() {
        this.currentSessionId = null;
        this.agentSessionId = null;
        this.intelligence = {
            profile: null,
            relationships: null,
            recommendations: null
        };
        this.pipelineMonitor = null;
        this.lastClickTime = null;
        this.activeLines = [];
        this.originalDataShape = null;
        this.fullDataPreview = null;
        this.currentEventSource = null;
        this.loadingMessageElement = null;

        this.init();
    }

    static get STYLES() {
        return {
            RIPPLE_COLOR: 'rgba(200, 200, 200, 0.2)',
            ACCENT_GREEN: 'var(--color-accent-green)',
            GOLD_HIGHLIGHT: 'rgba(255, 215, 0, 0.9)',
            TRANSITION_DURATION: '0.3s ease',
            BORDER_RADIUS: '8px',
            BOX_SHADOW_LIGHT: '0 8px 25px rgba(200, 200, 200, 0.15)',
            BOX_SHADOW_MEDIUM: '0 0 20px rgba(200, 200, 200, 0.25)',
            OPACITY_ANIMATION: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
        };
    }

    async init() {
        this.initializeModules();
        this.setupEventListeners();
        this.initializeUI();
        await this.sessionManager.loadSessions();
    }

    initializeModules() {
        this.animationManager = new AnimationManager();
        this.threeRenderer = new ThreeRenderer();
        this.blackHole = new BlackHole('#blackhole');
        this.apiClient = new ApiClient();
        this.chatInterface = new ChatInterface();
        this.sessionManager = new SessionManager();
        this.themeManager = new ThemeManager();

        this.chatInterface.setApp(this);
        this.sessionManager.setApp(this);
        this.apiClient.setApp(this);
        this.themeManager.setApp(this);
    }

    setupEventListeners() {
        this.setupManualAnalysisButton();
        this.sessionManager.setupEventListeners();
        this.chatInterface.setupHeroChatInterface();
        this.themeManager.setupThemeToggle();
        this.threeRenderer.init();
    }

    setupManualAnalysisButton() {
        const manualAnalyzeBtn = document.getElementById('manualAnalyze');
        if (manualAnalyzeBtn) {
            manualAnalyzeBtn.addEventListener('click', () => {
                window.location.href = '/static/dashboard.html';
            });
        }
    }

    initializeUI() {
        this.updateHeaderVisibility('chat');
        this.updateStatusIndicator('Ready', 'ready');
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    updateStatusIndicator(status, type = 'ready') {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');

        if (indicator && statusText) {
            statusText.textContent = status;
            indicator.className = `status-indicator ${type}`;
            if (type === 'running' || type === 'processing') {
                indicator.style.animation = 'pulseGlow 2s infinite';
            } else {
                indicator.style.animation = 'none';
            }
        }
    }

    updateSidebarStats() {
        const quickStats = document.getElementById('quickStats');
        if (quickStats && this.fullDataPreview) {
            const data = this.fullDataPreview;
            const rows = data.shape?.[0] || 0;
            const cols = data.shape?.[1] || 0;
            const quality = data.validation?.is_valid ? 'Good' : 'Issues';

            document.getElementById('sidebarRows').textContent = rows.toLocaleString();
            document.getElementById('sidebarCols').textContent = cols.toLocaleString();
            document.getElementById('sidebarQuality').textContent = quality;
        }
    }

    updateHeaderVisibility(mode) {
        const header = document.getElementById('mainHeader');
        if (!header) return;

        switch (mode) {
            case 'chat':
                header.style.display = 'block';
                break;
            case 'dashboard':
                header.style.display = 'none';
                break;
            default:
                break;
        }
    }

    restartWorkflow() {
        this.currentSessionId = null;
        this.agentSessionId = null;
        this.intelligence = { profile: null, relationships: null, recommendations: null };
        this.originalDataShape = null;
        this.fullDataPreview = null;

        if (this.pipelineMonitor) {
            clearInterval(this.pipelineMonitor);
            this.pipelineMonitor = null;
        }

        this.clearAllResults();
        this.updateStatusIndicator('Ready', 'ready');
        this.showToast('Workflow restarted', 'success');
    }

    clearAllResults() {
        const chatMessages = document.getElementById('heroChatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '<div class="chat-message bot"><div class="message-content"><p>Welcome to DataInsight AI. Please upload a dataset to begin.</p></div></div>';
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            animation: slideInRight 0.3s ease-out;
        `;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.dataInsightApp = new DataInsightApp();
});