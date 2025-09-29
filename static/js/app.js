class DataInsightApp {
    constructor() {
        this.currentSessionId = null;
        this.agentSessionId = null;
        this.currentStep = 1;
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
        this.apiClient = new ApiClient();
        this.uiManager = new UIManager();
        this.chatInterface = new ChatInterface();
        this.sessionManager = new SessionManager();
        this.themeManager = new ThemeManager();

        this.uiManager.setApp(this);
        this.chatInterface.setApp(this);
        this.sessionManager.setApp(this);
        this.apiClient.setApp(this);
    }

    setupEventListeners() {
        this.setupManualAnalysisButton();
        this.sessionManager.setupEventListeners();
        this.uiManager.setupFileUpload();
        this.uiManager.setupIntelligenceFeatures();
        this.uiManager.setupContinueButton();
        this.uiManager.setupDownloads();
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

    showSection(sectionId) {
        const sections = document.querySelectorAll('.section');
        sections.forEach(section => {
            if (section.id === sectionId) {
                section.classList.remove('hidden');
                section.classList.add('active');
                this.animationManager.addSuccessGlow(section);
            } else {
                section.classList.add('hidden');
                section.classList.remove('active');
            }
        });

        this.currentStep = this.getSectionStep(sectionId);
        this.updateProgressIndicator();
        this.updateSidebarProgress();
    }

    getSectionStep(sectionId) {
        const steps = {
            'dataInputSection': 1,
            'intelligenceSection': 2,
            'taskConfigSection': 3,
            'processingSection': 4,
            'resultsSection': 5
        };
        return steps[sectionId] || 1;
    }

    updateProgressIndicator() {
        const progressSteps = document.querySelectorAll('.progress-step');
        progressSteps.forEach((step, index) => {
            if (index + 1 <= this.currentStep) {
                step.classList.add('active');
                step.classList.add('completed');
            } else {
                step.classList.remove('active');
                step.classList.remove('completed');
            }
        });
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

    updateSidebarProgress() {
        const stepItems = document.querySelectorAll('.step-item');
        stepItems.forEach((item, index) => {
            const stepNumber = parseInt(item.dataset.step);
            if (stepNumber < this.currentStep) {
                item.classList.add('completed');
                item.classList.remove('active');
            } else if (stepNumber === this.currentStep) {
                item.classList.add('active');
                item.classList.remove('completed');
            } else {
                item.classList.remove('active', 'completed');
            }
        });

        if (this.currentStep >= 2 && this.currentSessionId) {
            this.updateSidebarStats();
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
        this.currentStep = 1;
        this.intelligence = { profile: null, relationships: null, recommendations: null };
        this.originalDataShape = null;
        this.fullDataPreview = null;

        if (this.pipelineMonitor) {
            clearInterval(this.pipelineMonitor);
            this.pipelineMonitor = null;
        }

        this.uiManager.clearAllResults();
        this.showSection('dataInputSection');
        this.updateStatusIndicator('Ready', 'ready');
        this.uiManager.showElegantToast('Workflow restarted', 'success');
    }

    animateValueChange(elementId, newValue, className = '') {
        const element = document.getElementById(elementId);
        if (!element) return;

        element.style.opacity = '0.5';
        element.style.transform = 'scale(0.95)';

        setTimeout(() => {
            element.textContent = newValue;
            if (className) {
                element.className = className;
            }
            element.style.opacity = '1';
            element.style.transform = 'scale(1)';
        }, 150);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.dataInsightApp = new DataInsightApp();
});