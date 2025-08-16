/**
 * DataInsight AI - Modern Interactive Frontend
 * Elegant, minimalistic interface with smooth animations and intelligence features
 */

console.log('🚀 DataInsight AI Script Loading...');

class DataInsightApp {
    constructor() {
        console.log('🔧 Initializing DataInsight App...');
        this.currentSessionId = null;
        this.currentStep = 1;
        this.intelligence = {
            profile: null,
            relationships: null,
            recommendations: null
        };
        this.pipelineMonitor = null;
        this.lastClickTime = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeUI();
        this.setupAnimations();
    }

    setupEventListeners() {
        // File upload with elegant interactions
        this.setupFileUpload();
        
        // Intelligence tabs
        this.setupIntelligenceTabs();
        
        // Task configuration
        this.setupTaskConfiguration();
        
        // Processing and monitoring
        this.setupProcessing();
        
        // Downloads
        this.setupDownloads();
        
        // Intelligence features
        this.setupIntelligenceFeatures();
    }

    setupFileUpload() {
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');
        
        if (this.uploadZoneClickHandler) {
            uploadZone.removeEventListener('click', this.uploadZoneClickHandler);
        }
        if (this.fileInputChangeHandler) {
            fileInput.removeEventListener('change', this.fileInputChangeHandler);
        }
        
        this.uploadZoneClickHandler = () => {
            if (this.lastClickTime && Date.now() - this.lastClickTime < 300) {
                return;
            }
            this.lastClickTime = Date.now();
            this.addButtonEffect(uploadZone);
            fileInput.click();
        };
        
        this.fileInputChangeHandler = (e) => {
            this.handleFileSelect(e);
        };
        
        uploadZone.addEventListener('click', this.uploadZoneClickHandler);
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
            this.createRippleEffect(e, uploadZone);
        });
        
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            this.addSuccessGlow(uploadZone);
            this.handleFileDrop(e);
        });
        
        fileInput.addEventListener('change', this.fileInputChangeHandler);

        // URL ingestion with smooth interactions
        const urlSubmit = document.getElementById('urlSubmit');
        const urlInput = document.getElementById('urlInput');
        
        urlSubmit.addEventListener('click', (e) => {
            this.addButtonEffect(urlSubmit);
            this.handleUrlSubmit();
        });
        
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.addInputEffect(urlInput);
                this.handleUrlSubmit();
            }
        });
        
        urlInput.addEventListener('focus', () => this.addInputFocus(urlInput));
        urlInput.addEventListener('blur', () => this.removeInputFocus(urlInput));
    }

    setupIntelligenceTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabPanels = document.querySelectorAll('.tab-panel');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const targetTab = button.dataset.tab;
                this.switchTab(targetTab, button);
            });
        });
    }

    setupTaskConfiguration() {
        const taskSelect = document.getElementById('taskSelect');
        const advancedToggle = document.getElementById('advancedToggle');
        
        taskSelect.addEventListener('change', this.handleTaskChange.bind(this));
        advancedToggle.addEventListener('change', this.toggleAdvancedOptions.bind(this));
        
        // Add smooth interactions to checkboxes
        const checkboxes = document.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.addCheckboxEffect(e.target);
            });
        });
    }

    setupProcessing() {
        const startButton = document.getElementById('startProcessing');
        startButton.addEventListener('click', (e) => {
            this.addButtonEffect(startButton);
            this.startProcessing();
        });
        
        const refreshButton = document.getElementById('refreshStatus');
        const recoveryButton = document.getElementById('triggerRecovery');
        
        if (refreshButton) {
            refreshButton.addEventListener('click', (e) => {
                this.addButtonEffect(refreshButton);
                this.refreshPipelineStatus();
            });
        }
        
        if (recoveryButton) {
            recoveryButton.addEventListener('click', (e) => {
                this.addButtonEffect(recoveryButton);
                this.triggerPipelineRecovery();
            });
        }
    }

    setupDownloads() {
        const downloadButtons = document.querySelectorAll('.download-btn');
        downloadButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                this.addDownloadEffect(button);
                this.handleDownload(button.id);
            });
        });
        
        const restartButton = document.getElementById('processNewData');
        if (restartButton) {
            restartButton.addEventListener('click', (e) => {
                this.addButtonEffect(restartButton);
                this.restartWorkflow();
            });
        }
    }

    setupIntelligenceFeatures() {
        // Deep profiling
        const deepProfileButton = document.getElementById('deepProfile');
        if (deepProfileButton) {
            deepProfileButton.addEventListener('click', (e) => {
                this.addButtonEffect(deepProfileButton);
                this.performDeepProfiling();
            });
        }
        
        // Feature recommendations
        const getRecommendations = document.getElementById('getRecommendations');
        if (getRecommendations) {
            getRecommendations.addEventListener('click', (e) => {
                this.addButtonEffect(getRecommendations);
                this.getFeatureRecommendations();
            });
        }
        
        // Relationship graph
        const generateGraph = document.getElementById('generateGraph');
        if (generateGraph) {
            generateGraph.addEventListener('click', (e) => {
                this.addButtonEffect(generateGraph);
                this.generateRelationshipGraph();
            });
        }
        
        // EDA generation with smooth effects
        const generateEDA = document.getElementById('generateEDA');
        if (generateEDA) {
            generateEDA.addEventListener('click', (e) => {
                this.addButtonEffect(generateEDA);
                this.generateEDA();
            });
        }
    }

    initializeUI() {
        this.updateStatusIndicator('Ready', 'ready');
        this.showSection('dataInputSection');
    }

    setupAnimations() {
        // Add smooth transitions to all elements
        this.addGlobalTransitions();
        
        // Setup intersection observer for scroll animations
        this.setupScrollAnimations();
        
        // Add elegant loading animations
        this.setupLoadingAnimations();
    }

    // Elegant Animation Effects
    addButtonEffect(button) {
        button.style.transform = 'scale(0.95)';
        button.style.transition = 'transform 0.1s ease';
        
        setTimeout(() => {
            button.style.transform = 'scale(1)';
            button.style.transition = 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)';
        }, 100);
        
        this.createRippleEffect(null, button);
    }

    addInputEffect(input) {
        input.style.transform = 'translateY(-2px)';
        input.style.boxShadow = '0 8px 25px rgba(0, 255, 136, 0.15)';
        input.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        
        setTimeout(() => {
            input.style.transform = 'translateY(0)';
            input.style.boxShadow = 'none';
        }, 200);
    }

    addInputFocus(input) {
        input.style.borderColor = 'var(--color-accent-green)';
        input.style.boxShadow = '0 0 0 2px rgba(0, 255, 136, 0.2)';
        input.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
    }

    removeInputFocus(input) {
        input.style.borderColor = 'var(--color-light-grey)';
        input.style.boxShadow = 'none';
    }

    addCheckboxEffect(checkbox) {
        const checkmark = checkbox.nextElementSibling;
        if (checkmark) {
            checkmark.style.transform = 'scale(1.1)';
            checkmark.style.transition = 'transform 0.2s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
            
            setTimeout(() => {
                checkmark.style.transform = 'scale(1)';
            }, 200);
        }
    }

    addSuccessGlow(element) {
        element.style.boxShadow = '0 0 20px rgba(0, 255, 136, 0.4)';
        element.style.borderColor = 'var(--color-accent-green)';
        element.style.transition = 'all 0.3s ease';
        
        setTimeout(() => {
            element.style.boxShadow = 'none';
            element.style.borderColor = 'var(--color-light-grey)';
        }, 1000);
    }

    addDownloadEffect(button) {
        this.addButtonEffect(button);
        
        // Add download progress effect
        const progressBar = document.createElement('div');
        progressBar.className = 'download-progress';
        progressBar.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            height: 2px;
            background: var(--color-accent-green);
            transition: width 1s ease;
            width: 0%;
        `;
        
        button.style.position = 'relative';
        button.appendChild(progressBar);
        
        setTimeout(() => progressBar.style.width = '100%', 50);
        setTimeout(() => progressBar.remove(), 1500);
    }

    createRippleEffect(event, element) {
        const ripple = document.createElement('div');
        ripple.className = 'ripple-effect';
        
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event ? event.clientX - rect.left - size / 2 : rect.width / 2 - size / 2;
        const y = event ? event.clientY - rect.top - size / 2 : rect.height / 2 - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: rgba(0, 255, 136, 0.3);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
            z-index: 10;
        `;
        
        element.style.position = element.style.position || 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    }

    addGlobalTransitions() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(30px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            @keyframes pulseGlow {
                0%, 100% {
                    box-shadow: 0 0 5px rgba(0, 255, 136, 0.3);
                }
                50% {
                    box-shadow: 0 0 20px rgba(0, 255, 136, 0.6);
                }
            }
            
            .drag-over {
                border-color: var(--color-accent-green) !important;
                background: rgba(0, 255, 136, 0.05) !important;
                animation: pulseGlow 1s infinite;
            }
            
            .fade-in-up {
                animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .slide-in-right {
                animation: slideInRight 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            }
        `;
        document.head.appendChild(style);
    }

    setupScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in-up');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        document.querySelectorAll('.section').forEach(section => {
            observer.observe(section);
        });
    }

    setupLoadingAnimations() {
        const processingCircle = document.querySelector('.processing-circle');
        if (processingCircle) {
            processingCircle.addEventListener('animationend', () => {
                processingCircle.style.animation = 'none';
                setTimeout(() => {
                    processingCircle.style.animation = '';
                }, 100);
            });
        }
    }

    // Tab Management with Smooth Transitions
    switchTab(targetTab, button) {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabPanels = document.querySelectorAll('.tab-panel');
        
        // Remove active states with fade effect
        tabButtons.forEach(btn => {
            btn.classList.remove('active');
            btn.style.transition = 'all 0.3s ease';
        });
        
        tabPanels.forEach(panel => {
            if (panel.classList.contains('active')) {
                panel.style.opacity = '0';
                panel.style.transform = 'translateY(10px)';
                setTimeout(() => {
                    panel.classList.remove('active');
                }, 150);
            }
        });
        
        // Add active state to clicked button
        button.classList.add('active');
        this.addButtonEffect(button);
        
        // Show target panel with slide effect
        setTimeout(() => {
            const targetPanel = document.getElementById(targetTab + 'Tab');
            if (targetPanel) {
                targetPanel.classList.add('active');
                targetPanel.style.opacity = '0';
                targetPanel.style.transform = 'translateY(10px)';
                targetPanel.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
                
                setTimeout(() => {
                    targetPanel.style.opacity = '1';
                    targetPanel.style.transform = 'translateY(0)';
                }, 50);
            }
        }, 150);
    }

    // Intelligence Features with Elegant Loading
    async performDeepProfiling() {
        if (!this.currentSessionId) return;
        
        this.showElegantLoading('Performing deep intelligence analysis...');
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/profile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    deep_analysis: true,
                    include_relationships: true,
                    include_domain_detection: true
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.intelligence.profile = data.intelligence_profile;
                this.displayIntelligenceProfile(data.intelligence_profile);
                this.showElegantSuccess('Deep profiling completed');
            } else {
                this.showElegantError('Profiling failed: ' + data.detail);
            }
        } catch (error) {
            this.showElegantError('Network error during profiling');
        } finally {
            this.hideElegantLoading();
        }
    }

    async getFeatureRecommendations() {
        if (!this.currentSessionId) return;
        
        const priorityFilter = document.getElementById('priorityFilter').value;
        const targetColumn = document.getElementById('targetSelect').value;
        
        this.showElegantLoading('Generating AI feature recommendations...');
        
        try {
            const params = new URLSearchParams({
                max_recommendations: '10'
            });
            
            if (priorityFilter) params.append('priority_filter', priorityFilter);
            if (targetColumn) params.append('target_column', targetColumn);
            
            const response = await fetch(`/api/data/${this.currentSessionId}/feature-recommendations?${params}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.intelligence.recommendations = data;
                this.displayFeatureRecommendations(data);
                this.showElegantSuccess(`Generated ${data.recommendations.length} recommendations`);
            } else {
                this.showElegantError('Failed to generate recommendations');
            }
        } catch (error) {
            this.showElegantError('Network error during recommendation generation');
        } finally {
            this.hideElegantLoading();
        }
    }

    async generateRelationshipGraph() {
        if (!this.currentSessionId) return;
        
        this.showElegantLoading('Building relationship graph...');
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/relationship-graph`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.intelligence.relationships = data.graph_data;
                this.renderRelationshipGraph(data.graph_data);
                this.displayRelationshipDetails(data.graph_data);
                this.showElegantSuccess('Relationship graph generated');
            } else {
                this.showElegantError('Failed to generate relationship graph');
            }
        } catch (error) {
            this.showElegantError('Network error during graph generation');
        } finally {
            this.hideElegantLoading();
        }
    }

    // Elegant Loading States
    showElegantLoading(message) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        
        loadingText.textContent = message;
        loadingOverlay.classList.remove('hidden');
        loadingOverlay.style.opacity = '0';
        loadingOverlay.style.transition = 'opacity 0.3s ease';
        
        setTimeout(() => {
            loadingOverlay.style.opacity = '1';
        }, 50);
    }

    hideElegantLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        
        loadingOverlay.style.opacity = '0';
        setTimeout(() => {
            loadingOverlay.classList.add('hidden');
        }, 300);
    }

    showElegantSuccess(message) {
        this.showElegantToast(message, 'success');
    }

    showElegantError(message) {
        this.showElegantToast(message, 'error');
    }

    showElegantToast(message, type) {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-icon">
                    <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
                </div>
                <div class="toast-message">
                    <div class="toast-text">${message}</div>
                </div>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('show');
        }, 50);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // EDA Generation with elegant loading
    async generateEDA() {
        if (!this.currentSessionId) {
            this.showElegantError('No data session found');
            return;
        }
        
        this.showElegantLoading('Generating comprehensive EDA report...');
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/eda`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    include_plots: true,
                    include_statistics: true,
                    plot_style: 'seaborn'
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayEDAResults(data);
                this.showSection('edaSection');
                this.showElegantSuccess('EDA report generated successfully');
            } else {
                this.showElegantError('EDA generation failed: ' + data.detail);
            }
        } catch (error) {
            this.showElegantError('Network error during EDA generation');
        } finally {
            this.hideElegantLoading();
        }
    }
    
    // Processing Pipeline with real-time monitoring
    async startProcessing() {
        if (!this.currentSessionId) {
            this.showElegantError('No data session found');
            return;
        }
        
        const taskType = document.getElementById('taskSelect').value;
        const targetColumn = document.getElementById('targetSelect').value;
        
        if (!taskType) {
            this.showElegantError('Please select a task type');
            return;
        }
        
        this.showElegantLoading(`Starting ${taskType} pipeline...`);
        
        try {
            // Get processing configuration
            const config = this.getProcessingConfig();
            
            const response = await fetch(`/api/data/${this.currentSessionId}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    task_type: taskType,
                    target_column: targetColumn,
                    use_robust_pipeline: true,
                    ...config
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayProcessingResults(data);
                this.startPipelineMonitoring(data.execution_id);
                this.showSection('resultsSection');
                this.showElegantSuccess('Processing pipeline started');
            } else {
                this.showElegantError('Processing failed: ' + data.detail);
            }
        } catch (error) {
            this.showElegantError('Network error during processing');
        } finally {
            this.hideElegantLoading();
        }
    }
    
    // Real-time pipeline monitoring
    async refreshPipelineStatus() {
        if (!this.currentSessionId) return;
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/pipeline-status`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updatePipelineMonitor(data.pipeline_status);
                
                // Continue monitoring if still running
                if (data.pipeline_status.status === 'running') {
                    setTimeout(() => this.refreshPipelineStatus(), 2000);
                }
            }
        } catch (error) {
            console.error('Pipeline monitoring error:', error);
        }
    }
    
    // Pipeline recovery
    async triggerPipelineRecovery() {
        if (!this.currentSessionId) return;
        
        this.showElegantLoading('Attempting pipeline recovery...');
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/pipeline-recovery`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    recovery_strategy: 'auto',
                    use_checkpoints: true
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updatePipelineMonitor(data.recovery_status);
                this.showElegantSuccess('Pipeline recovery initiated');
            } else {
                this.showElegantError('Recovery failed: ' + data.detail);
            }
        } catch (error) {
            this.showElegantError('Network error during recovery');
        } finally {
            this.hideElegantLoading();
        }
    }
    
    // Download handling with progress animation
    async handleDownload(downloadType) {
        if (!this.currentSessionId) {
            this.showElegantError('No data session found');
            return;
        }
        
        const endpoints = {
            'downloadCleaned': '/api/data/export/cleaned',
            'downloadFeatures': '/api/data/export/features',
            'downloadModel': '/api/data/export/model',
            'downloadResults': '/api/data/export/results'
        };
        
        const endpoint = endpoints[downloadType];
        if (!endpoint) return;
        
        try {
            const response = await fetch(`${endpoint}/${this.currentSessionId}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = this.getDownloadFilename(downloadType);
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showElegantSuccess('Download completed successfully');
            } else {
                this.showElegantError('Download failed');
            }
        } catch (error) {
            this.showElegantError('Network error during download');
        }
    }
    
    // File Upload Handlers
    handleFileDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        console.log('📁 Processing file:', file.name);
        this.showElegantLoading('Uploading and analyzing your data...');
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('enable_profiling', 'true');
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.session_id) {
                this.currentSessionId = data.session_id;
                this.displayDataPreview(data);
                this.displayIntelligenceSummary(data.intelligence_summary);
                this.populateTargetSelect(data.columns);
                this.showSection('dataPreviewSection');
                this.showElegantSuccess(`File uploaded successfully: ${file.name}`);
                
                // Auto-advance to configuration after 2 seconds
                setTimeout(() => {
                    console.log('Auto-advancing to task configuration...');
                    this.showSection('taskConfigSection');
                    this.showElegantSuccess('Ready for configuration! Select your task type below.');
                }, 2000);
            } else {
                this.showElegantError('Upload failed: ' + data.detail);
            }
        } catch (error) {
            this.showElegantError('Network error during upload');
        } finally {
            this.hideElegantLoading();
        }
    }

    async handleUrlSubmit() {
        const urlInput = document.getElementById('urlInput');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.showElegantError('Please enter a valid URL');
            return;
        }
        
        this.showElegantLoading('Fetching data from URL...');
        
        try {
            const response = await fetch('/api/ingest-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: url,
                    data_type: 'csv',
                    enable_profiling: true
                })
            });
            
            const data = await response.json();
            
            if (data.session_id) {
                this.currentSessionId = data.session_id;
                this.displayDataPreview(data);
                this.displayIntelligenceSummary(data.intelligence_summary);
                this.populateTargetSelect(data.columns);
                this.showSection('dataPreviewSection');
                this.showElegantSuccess('Data fetched successfully from URL');
                
                // Auto-advance to configuration after 2 seconds
                setTimeout(() => {
                    console.log('Auto-advancing to task configuration...');
                    this.showSection('taskConfigSection');
                    this.showElegantSuccess('Ready for configuration! Select your task type below.');
                }, 2000);
            } else {
                this.showElegantError('URL ingestion failed: ' + data.detail);
            }
        } catch (error) {
            this.showElegantError('Network error during URL fetch');
        } finally {
            this.hideElegantLoading();
        }
    }

    // Display Methods with Smooth Animations
    displayDataPreview(data) {
        // Update data info cards with smooth animations
        this.animateValueChange('dataShape', `${data.shape[0]} × ${data.shape[1]}`);
        this.animateValueChange('columnCount', data.shape[1]);
        
        const qualityStatus = data.validation.is_valid ? 'Good' : 'Issues Detected';
        const qualityClass = data.validation.is_valid ? 'text-success' : 'text-warning';
        this.animateValueChange('qualityStatus', qualityStatus, qualityClass);
        
        // Display data preview table with fade-in effect
        this.displayDataTable(data);
        
        // Display validation issues if any
        this.displayValidationIssues(data.validation.issues);
    }

    displayIntelligenceSummary(intelligenceSummary) {
        if (!intelligenceSummary || !intelligenceSummary.profiling_completed) return;
        
        // Update intelligence indicators with animations
        this.animateValueChange('domainDetected', intelligenceSummary.primary_domain || 'Unknown');
        this.animateValueChange('relationshipCount', intelligenceSummary.relationships_found || 0);
        
        // Populate semantic types if available
        if (intelligenceSummary.semantic_types) {
            this.displaySemanticTypes(intelligenceSummary.semantic_types);
        }
    }

    animateValueChange(elementId, newValue, className = '') {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.style.transform = 'scale(0.8)';
        element.style.opacity = '0.5';
        element.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        
        setTimeout(() => {
            element.textContent = newValue;
            if (className) {
                element.className = `info-value ${className}`;
            }
            element.style.transform = 'scale(1)';
            element.style.opacity = '1';
        }, 150);
    }

    // Data Display Methods
    displayDataTable(data) {
        const table = document.getElementById('dataPreviewTable');
        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');
        
        // Clear existing content
        thead.innerHTML = '';
        tbody.innerHTML = '';
        
        if (!data.columns || data.columns.length === 0) return;
        
        // Create header with smooth animation
        const headerRow = document.createElement('tr');
        data.columns.forEach((column, index) => {
            const th = document.createElement('th');
            th.textContent = column;
            th.style.opacity = '0';
            th.style.transform = 'translateY(-10px)';
            th.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            headerRow.appendChild(th);
            
            setTimeout(() => {
                th.style.opacity = '1';
                th.style.transform = 'translateY(0)';
            }, index * 50);
        });
        thead.appendChild(headerRow);
        
        // Fetch and display preview data
        this.fetchDataPreview();
    }

    async fetchDataPreview() {
        if (!this.currentSessionId) return;
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/preview`);
            const data = await response.json();
            
            const tbody = document.getElementById('dataPreviewTable').querySelector('tbody');
            
            data.data.forEach((row, rowIndex) => {
                const tr = document.createElement('tr');
                tr.style.opacity = '0';
                tr.style.transform = 'translateX(-20px)';
                tr.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
                
                data.columns.forEach(column => {
                    const td = document.createElement('td');
                    td.textContent = row[column] || '';
                    tr.appendChild(td);
                });
                
                tbody.appendChild(tr);
                
                setTimeout(() => {
                    tr.style.opacity = '1';
                    tr.style.transform = 'translateX(0)';
                }, rowIndex * 100);
            });
        } catch (error) {
            console.error('Error fetching data preview:', error);
        }
    }

    displaySemanticTypes(semanticTypes) {
        const tbody = document.querySelector('#semanticTypesTable tbody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        Object.entries(semanticTypes).forEach(([column, type], index) => {
            const row = document.createElement('tr');
            row.style.opacity = '0';
            row.style.transform = 'translateY(20px)';
            row.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
            
            row.innerHTML = `
                <td>${column}</td>
                <td>
                    <span class="type-badge">
                        <span class="type-color" style="background: ${this.getSemanticTypeColor(type)}"></span>
                        ${type.replace(/_/g, ' ')}
                    </span>
                </td>
                <td>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: 85%"></div>
                    </div>
                </td>
                <td><small>Auto-detected</small></td>
            `;
            
            tbody.appendChild(row);
            
            setTimeout(() => {
                row.style.opacity = '1';
                row.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    getSemanticTypeColor(type) {
        const colors = {
            'primary_key': '#ff6b6b',
            'foreign_key': '#4ecdc4',
            'email': '#45b7d1',
            'currency': '#ffa726',
            'datetime': '#ab47bc',
            'text': '#66bb6a',
            'categorical': '#26a69a',
            'default': '#78909c'
        };
        
        for (const [key, color] of Object.entries(colors)) {
            if (type.includes(key)) return color;
        }
        return colors.default;
    }

    displayIntelligenceProfile(profile) {
        if (profile.column_profiles) {
            this.displaySemanticTypes(
                Object.fromEntries(
                    Object.entries(profile.column_profiles).map(([col, prof]) => [col, prof.semantic_type])
                )
            );
        }
        
        if (profile.overall_recommendations) {
            this.displayKeyInsights(profile.overall_recommendations);
        }
    }

    displayKeyInsights(insights) {
        const container = document.getElementById('keyInsights');
        if (!container) return;
        
        container.innerHTML = '';
        
        insights.slice(0, 5).forEach((insight, index) => {
            const item = document.createElement('div');
            item.className = 'insight-item';
            item.textContent = insight;
            item.style.opacity = '0';
            item.style.transform = 'translateX(-20px)';
            item.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
            
            container.appendChild(item);
            
            setTimeout(() => {
                item.style.opacity = '1';
                item.style.transform = 'translateX(0)';
            }, index * 150);
        });
    }

    displayFeatureRecommendations(data) {
        const container = document.getElementById('featureRecommendations');
        if (!container) return;
        
        container.innerHTML = '';
        
        data.recommendations.forEach((rec, index) => {
            const item = document.createElement('div');
            item.className = 'recommendation-item';
            item.style.opacity = '0';
            item.style.transform = 'translateY(30px)';
            item.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
            
            item.innerHTML = `
                <div class="recommendation-header">
                    <div>
                        <div class="recommendation-title">${rec.feature_type.replace(/_/g, ' ').toUpperCase()}</div>
                        <div class="recommendation-meta">
                            <span class="priority-badge ${rec.priority}">${rec.priority}</span>
                            <span class="cost-badge">${rec.computational_cost} cost</span>
                        </div>
                    </div>
                </div>
                <div class="recommendation-description">${rec.description}</div>
                <div class="recommendation-implementation">${rec.implementation}</div>
                <div class="recommendation-benefit">${rec.expected_benefit}</div>
            `;
            
            container.appendChild(item);
            
            setTimeout(() => {
                item.style.opacity = '1';
                item.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }

    renderRelationshipGraph(graphData) {
        const container = document.getElementById('relationshipGraph');
        if (!container || !graphData.nodes || graphData.nodes.length === 0) return;
        
        // Clear existing content
        container.innerHTML = '';
        
        const width = container.clientWidth;
        const height = 400;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('background', 'var(--color-medium-grey)')
            .style('border-radius', 'var(--border-radius-md)');
        
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(graphData.edges).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));
        
        // Add links
        const link = svg.append('g')
            .selectAll('line')
            .data(graphData.edges)
            .join('line')
            .attr('stroke', 'var(--color-accent-green)')
            .attr('stroke-width', d => Math.sqrt(d.strength * 5) || 2)
            .attr('stroke-opacity', 0.6)
            .style('opacity', 0)
            .transition()
            .duration(1000)
            .style('opacity', 1);
        
        // Add nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(graphData.nodes)
            .join('circle')
            .attr('r', 8)
            .attr('fill', 'var(--color-accent-green)')
            .attr('stroke', 'var(--color-white)')
            .attr('stroke-width', 2)
            .style('opacity', 0)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended))
            .transition()
            .duration(1000)
            .delay((d, i) => i * 100)
            .style('opacity', 1);
        
        // Add labels
        const label = svg.append('g')
            .selectAll('text')
            .data(graphData.nodes)
            .join('text')
            .text(d => d.id)
            .attr('font-size', '12px')
            .attr('fill', 'var(--color-text-grey)')
            .attr('text-anchor', 'middle')
            .attr('dy', -15)
            .style('opacity', 0)
            .transition()
            .duration(1000)
            .delay((d, i) => i * 100)
            .style('opacity', 1);
        
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }

    displayRelationshipDetails(graphData) {
        const container = document.getElementById('relationshipsList');
        if (!container || !graphData.edges) return;
        
        container.innerHTML = '';
        
        graphData.edges.forEach((edge, index) => {
            const item = document.createElement('div');
            item.className = 'relationship-item';
            item.style.opacity = '0';
            item.style.transform = 'translateX(20px)';
            item.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
            
            item.innerHTML = `
                <div class="relationship-header">
                    <span class="relationship-type">${edge.type.replace(/_/g, ' ')}</span>
                    <span class="relationship-strength">${(edge.strength * 100).toFixed(1)}%</span>
                </div>
                <div class="relationship-description">
                    <strong>${edge.source}</strong> → <strong>${edge.target}</strong><br>
                    ${edge.description || 'Statistical relationship detected'}
                </div>
            `;
            
            container.appendChild(item);
            
            setTimeout(() => {
                item.style.opacity = '1';
                item.style.transform = 'translateX(0)';
            }, index * 150);
        });
    }

    // Task Configuration
    handleTaskChange() {
        const taskSelect = document.getElementById('taskSelect');
        const targetGroup = document.getElementById('targetColumnGroup');
        
        if (['classification', 'regression', 'timeseries'].includes(taskSelect.value)) {
            targetGroup.style.display = 'block';
            targetGroup.style.opacity = '0';
            targetGroup.style.transform = 'translateY(-10px)';
            targetGroup.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            
            setTimeout(() => {
                targetGroup.style.opacity = '1';
                targetGroup.style.transform = 'translateY(0)';
            }, 50);
        } else {
            targetGroup.style.opacity = '0';
            setTimeout(() => {
                targetGroup.style.display = 'none';
            }, 300);
        }
    }

    toggleAdvancedOptions() {
        const advancedOptions = document.getElementById('advancedOptions');
        const isVisible = advancedOptions.style.display !== 'none';
        
        if (isVisible) {
            advancedOptions.style.opacity = '0';
            advancedOptions.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                advancedOptions.style.display = 'none';
            }, 300);
        } else {
            advancedOptions.style.display = 'block';
            advancedOptions.style.opacity = '0';
            advancedOptions.style.transform = 'translateY(-10px)';
            advancedOptions.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            
            setTimeout(() => {
                advancedOptions.style.opacity = '1';
                advancedOptions.style.transform = 'translateY(0)';
            }, 50);
        }
    }

    // Utility methods for processing and monitoring
    getProcessingConfig() {
        const config = {
            data_cleaning: {
                handle_missing: document.getElementById('handleMissing')?.checked || true,
                remove_outliers: document.getElementById('removeOutliers')?.checked || false,
                normalize_text: document.getElementById('normalizeText')?.checked || true
            },
            feature_engineering: {
                auto_feature_generation: document.getElementById('autoFeatures')?.checked || true,
                polynomial_features: document.getElementById('polyFeatures')?.checked || false,
                interaction_features: document.getElementById('interactionFeatures')?.checked || false
            },
            validation: {
                cross_validation: document.getElementById('crossValidation')?.checked || true,
                test_size: parseFloat(document.getElementById('testSize')?.value) || 0.2
            }
        };
        
        return config;
    }
    
    startPipelineMonitoring(executionId) {
        this.pipelineMonitor = setInterval(() => {
            this.refreshPipelineStatus();
        }, 2000);
        
        setTimeout(() => {
            if (this.pipelineMonitor) {
                clearInterval(this.pipelineMonitor);
            }
        }, 300000); // Stop monitoring after 5 minutes
    }
    
    updatePipelineMonitor(status) {
        this.updateStatusIndicator(status.current_stage || 'Processing', status.status || 'running');
        
        const progressBar = document.getElementById('pipelineProgress');
        if (progressBar && status.progress) {
            progressBar.style.width = `${status.progress}%`;
            progressBar.style.transition = 'width 0.5s ease';
        }
        
        const logContainer = document.getElementById('pipelineLog');
        if (logContainer && status.recent_logs) {
            status.recent_logs.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${log.level}`;
                logEntry.textContent = `[${log.timestamp}] ${log.message}`;
                logContainer.appendChild(logEntry);
                logContainer.scrollTop = logContainer.scrollHeight;
            });
        }
    }
    
    displayEDAResults(data) {
        const edaContainer = document.getElementById('edaResults');
        if (!edaContainer) return;
        
        edaContainer.innerHTML = '';
        
        // Display plots if available
        if (data.plots) {
            Object.entries(data.plots).forEach(([plotType, plotData]) => {
                const plotContainer = document.createElement('div');
                plotContainer.className = 'eda-plot-container';
                plotContainer.innerHTML = `
                    <h4>${plotType.replace(/_/g, ' ').toUpperCase()}</h4>
                    <div class="plot-content">${plotData}</div>
                `;
                edaContainer.appendChild(plotContainer);
            });
        }
        
        // Display statistics
        if (data.statistics) {
            const statsContainer = document.createElement('div');
            statsContainer.className = 'eda-statistics';
            statsContainer.innerHTML = `
                <h4>STATISTICAL SUMMARY</h4>
                <pre>${JSON.stringify(data.statistics, null, 2)}</pre>
            `;
            edaContainer.appendChild(statsContainer);
        }
    }
    
    displayProcessingResults(data) {
        // Update result metrics with animations
        if (data.metrics) {
            Object.entries(data.metrics).forEach(([metric, value]) => {
                this.animateValueChange(metric, value);
            });
        }
        
        // Display model performance if available
        if (data.model_performance) {
            this.displayModelPerformance(data.model_performance);
        }
        
        // Display feature importance
        if (data.feature_importance) {
            this.displayFeatureImportance(data.feature_importance);
        }
    }
    
    displayModelPerformance(performance) {
        const container = document.getElementById('modelPerformance');
        if (!container) return;
        
        container.innerHTML = `
            <div class="performance-metrics">
                ${Object.entries(performance).map(([metric, value]) => `
                    <div class="metric-card">
                        <div class="metric-label">${metric.replace(/_/g, ' ')}</div>
                        <div class="metric-value">${typeof value === 'number' ? value.toFixed(4) : value}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    displayFeatureImportance(importance) {
        const container = document.getElementById('featureImportance');
        if (!container) return;
        
        const sortedFeatures = Object.entries(importance)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10);
        
        container.innerHTML = `
            <div class="importance-chart">
                ${sortedFeatures.map(([feature, score]) => `
                    <div class="importance-bar">
                        <div class="feature-name">${feature}</div>
                        <div class="importance-value">
                            <div class="bar-fill" style="width: ${score * 100}%"></div>
                            <span class="score">${score.toFixed(3)}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    getDownloadFilename(downloadType) {
        const timestamp = new Date().toISOString().slice(0, 10);
        const filenames = {
            'downloadCleaned': `cleaned_data_${timestamp}.csv`,
            'downloadFeatures': `engineered_features_${timestamp}.csv`,
            'downloadModel': `trained_model_${timestamp}.pkl`,
            'downloadResults': `results_${timestamp}.json`
        };
        return filenames[downloadType] || 'download.csv';
    }

    // UI Management methods
    showSection(sectionId) {
        console.log(`Showing section: ${sectionId}`);
        const sections = document.querySelectorAll('.section');
        console.log(`Found ${sections.length} sections`);
        
        sections.forEach(section => {
            if (section.id === sectionId) {
                console.log(`Making section ${sectionId} visible`);
                section.classList.remove('hidden');
                // Override CSS positioning
                section.style.position = 'static';
                section.style.left = 'auto';
                section.style.opacity = '0';
                section.style.transform = 'translateY(20px)';
                section.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
                
                setTimeout(() => {
                    section.style.opacity = '1';
                    section.style.transform = 'translateY(0)';
                }, 50);
            } else {
                section.classList.add('hidden');
            }
        });
        
        this.currentStep = this.getSectionStep(sectionId);
        this.updateProgressIndicator();
    }

    getSectionStep(sectionId) {
        const stepMap = {
            'dataInputSection': 1,
            'dataPreviewSection': 2,
            'intelligenceSection': 3,
            'taskConfigSection': 4,
            'resultsSection': 5
        };
        return stepMap[sectionId] || 1;
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
            
            // Add pulse animation for processing
            if (type === 'running' || type === 'processing') {
                indicator.style.animation = 'pulseGlow 2s infinite';
            } else {
                indicator.style.animation = 'none';
            }
        }
    }

    populateTargetSelect(columns) {
        const targetSelect = document.getElementById('targetSelect');
        if (!targetSelect || !columns) return;
        
        targetSelect.innerHTML = '<option value="">Select target column...</option>';
        
        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            targetSelect.appendChild(option);
        });
    }

    displayValidationIssues(issues) {
        const container = document.getElementById('validationIssues');
        if (!container || !issues || issues.length === 0) {
            if (container) container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        container.innerHTML = `
            <h4>Data Quality Issues</h4>
            <ul class="issue-list">
                ${issues.map(issue => `
                    <li class="issue-item">
                        <span class="issue-type">${issue.type}</span>
                        <span class="issue-description">${issue.description}</span>
                    </li>
                `).join('')}
            </ul>
        `;
    }

    restartWorkflow() {
        this.currentSessionId = null;
        this.intelligence = {
            profile: null,
            relationships: null,
            recommendations: null
        };
        
        if (this.pipelineMonitor) {
            clearInterval(this.pipelineMonitor);
            this.pipelineMonitor = null;
        }
        
        // Clear all data displays
        document.getElementById('dataPreviewTable')?.querySelector('tbody')?.replaceChildren();
        document.getElementById('semanticTypesTable')?.querySelector('tbody')?.replaceChildren();
        document.getElementById('relationshipGraph').innerHTML = '';
        document.getElementById('featureRecommendations').innerHTML = '';
        
        // Reset form values
        document.getElementById('fileInput').value = '';
        document.getElementById('urlInput').value = '';
        document.getElementById('taskSelect').value = '';
        document.getElementById('targetSelect').innerHTML = '<option value="">Select target column...</option>';
        
        // Return to first section
        this.showSection('dataInputSection');
        this.updateStatusIndicator('Ready', 'ready');
        this.showElegantSuccess('Workflow restarted');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 DOM Content Loaded - Starting App...');
    window.dataInsightApp = new DataInsightApp();
    console.log('✅ App initialized successfully');
});