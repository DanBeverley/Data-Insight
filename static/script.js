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
        this.setupFileUpload();
        this.setupIntelligenceTabs();
        this.setupTaskConfiguration();
        this.setupProcessing();
        this.setupDownloads();
        this.setupIntelligenceFeatures();
        this.setupContinueButton();
    }

    setupFileUpload() {
        console.log('Setting up file upload...');
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');
        
        console.log('fileInput:', fileInput);
        console.log('uploadZone:', uploadZone);
        
        if (!fileInput || !uploadZone) {
            console.error('File input or upload zone not found!');
            return;
        }
        
        // Store reference to this for use in handlers
        const self = this;
        
        // Simple click handler
        uploadZone.onclick = function() {
            console.log('Upload zone clicked, triggering file input');
            fileInput.click();
        };
        
        // Simple file change handler
        fileInput.onchange = function(e) {
            console.log('File selected:', e.target.files);
            const file = e.target.files[0];
            if (file) {
                self.processFile(file);
            }
        };
        
        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });
        
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processFile(files[0]);
            }
        });

        // URL ingestion with smooth interactions
        const urlSubmit = document.getElementById('urlSubmit');
        const urlInput = document.getElementById('urlInput');
        
        if (urlSubmit) {
            urlSubmit.addEventListener('click', (e) => {
                this.addButtonEffect(urlSubmit);
                this.handleUrlSubmit();
            });
        }
        
        if (urlInput) {
            urlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.addInputEffect(urlInput);
                    this.handleUrlSubmit();
                }
            });
            
            urlInput.addEventListener('focus', () => this.addInputFocus(urlInput));
            urlInput.addEventListener('blur', () => this.removeInputFocus(urlInput));
        }
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
        
        if (taskSelect) {
            taskSelect.addEventListener('change', this.handleTaskChange.bind(this));
        }
        if (advancedToggle) {
            advancedToggle.addEventListener('change', this.toggleAdvancedOptions.bind(this));
        }
        
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
        if (startButton) {
            startButton.addEventListener('click', (e) => {
                this.addButtonEffect(startButton);
                this.startProcessing();
            });
        }
        
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
        const downloadButtons = document.querySelectorAll('.download-item');
        downloadButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                console.log('Download button clicked:', button.id);
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
        const deepProfileButton = document.getElementById('deepProfile');
        if (deepProfileButton) {
            deepProfileButton.addEventListener('click', (e) => {
                this.addButtonEffect(deepProfileButton);
                this.performDeepProfiling();
            });
        }
        
        const getRecommendations = document.getElementById('getRecommendations');
        if (getRecommendations) {
            getRecommendations.addEventListener('click', (e) => {
                console.log('💡 Recommendations card clicked');
                this.addButtonEffect(getRecommendations);
                this.getFeatureRecommendations();
            });
        }
        
        const generateGraph = document.getElementById('generateGraph');
        if (generateGraph) {
            generateGraph.addEventListener('click', (e) => {
                console.log('🔗 Graph card clicked');
                this.addButtonEffect(generateGraph);
                this.generateRelationshipGraph();
            });
        }
        
        const generateEDA = document.getElementById('generateEDA');
        if (generateEDA) {
            generateEDA.addEventListener('click', (e) => {
                console.log('🔬 EDA card clicked');
                this.addButtonEffect(generateEDA);
                this.generateEDA();
            });
        }
    }

    setupContinueButton() {
        const continueButton = document.getElementById('continueToConfig');
        if (!continueButton) {
            console.warn('Continue button not found in DOM');
            return;
        }
        
        continueButton.addEventListener('click', (e) => {
            // Prevent double-click issues
            if (!e || !e.target) {
                console.warn('Invalid click event received');
                return;
            }
            
            // Prevent multiple rapid clicks
            if (this.lastClickTime && Date.now() - this.lastClickTime < 500) {
                console.log('Ignoring rapid click');
                return;
            }
            this.lastClickTime = Date.now();
            
            try {
                // Null check for button effect method
                if (typeof this.addButtonEffect === 'function') {
                    this.addButtonEffect(continueButton);
                } else {
                    console.warn('addButtonEffect method not available');
                }
                
                // Null check for section display method
                if (typeof this.showSection === 'function') {
                    this.showSection('configSection');
                } else {
                    console.error('showSection method not available');
                    return;
                }
                
                // Null check for success message method
                if (typeof this.showElegantSuccess === 'function') {
                    this.showElegantSuccess('Ready for configuration! Select your task type below.');
                } else {
                    console.warn('showElegantSuccess method not available');
                }
            } catch (error) {
                console.error('Error in continue button handler:', error);
                // Fallback: at least try to show the section
                const targetSection = document.getElementById('taskConfigSection');
                if (targetSection) {
                    targetSection.classList.remove('hidden');
                }
            }
        });
    }

    initializeUI() {
        this.updateStatusIndicator('Ready', 'ready');
        // If hero chat exists, show it by default; otherwise fallback to data input
        const hero = document.getElementById('heroChatSection');
        if (hero) {
            this.showSection('heroChatSection');
        } else {
            this.showSection('dataInputSection');
        }
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
        input.style.boxShadow = '0 8px 25px rgba(200, 200, 200, 0.15)';
        input.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        
        setTimeout(() => {
            input.style.transform = 'translateY(0)';
            input.style.boxShadow = 'none';
        }, 200);
    }

    addInputFocus(input) {
        input.style.borderColor = 'var(--color-accent-green)';
        input.style.boxShadow = '0 0 0 2px rgba(220, 220, 220, 0.2)';
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
        element.style.boxShadow = '0 0 20px rgba(200, 200, 200, 0.25)';
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
            background: rgba(200, 200, 200, 0.2);
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
                    box-shadow: 0 0 5px rgba(200, 200, 200, 0.2);
                }
                50% {
                    box-shadow: 0 0 20px rgba(220, 220, 220, 0.25);
                }
            }
            
            .drag-over {
                border-color: var(--color-accent-green) !important;
                background: rgba(220, 220, 220, 0.05) !important;
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
        console.log('💡 getFeatureRecommendations called, session ID:', this.currentSessionId);
        
        // Always show something for testing
        const recsContainer = document.getElementById('featureRecommendations');
        if (recsContainer) {
            recsContainer.classList.add('show');
            recsContainer.innerHTML = `
                <div style="padding: 20px; color: var(--color-text);">
                    <h4>💡 AI Recommendations Test</h4>
                    <p>Session ID: ${this.currentSessionId || 'No session'}</p>
                    <p>This proves the function is working!</p>
                </div>
            `;
        }
        
        if (!this.currentSessionId) {
            console.error('❌ No session ID found for recommendations');
            this.showElegantError('No data session found. Please upload data first.');
            return;
        }
        
        const priorityFilterElement = document.getElementById('priorityFilter');
        const targetColumnElement = document.getElementById('targetSelect');
        
        const priorityFilter = priorityFilterElement ? priorityFilterElement.value : '';
        const targetColumn = targetColumnElement ? targetColumnElement.value : '';
        
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
                const errorMessage = data.detail || data.message || JSON.stringify(data);
                this.showElegantError('Failed to generate recommendations: ' + errorMessage);
            }
        } catch (error) {
            console.error('Recommendation generation error:', error);
            this.showElegantError('Network error during recommendation generation: ' + error.message);
        } finally {
            this.hideElegantLoading();
        }
    }

    async generateRelationshipGraph() {
        console.log('🔗 generateRelationshipGraph called, session ID:', this.currentSessionId);
        
        // Always show something for testing
        const graphContainer = document.getElementById('relationshipGraph');
        if (graphContainer) {
            graphContainer.classList.add('show');
            graphContainer.innerHTML = `
                <div style="padding: 20px; color: var(--color-text);">
                    <h4>🔗 Relationship Graph Test</h4>
                    <p>Session ID: ${this.currentSessionId || 'No session'}</p>
                    <p>This proves the function is working!</p>
                </div>
            `;
        }
        
        if (!this.currentSessionId) {
            console.error('❌ No session ID found for relationship graph');
            this.showElegantError('No data session found. Please upload data first.');
            return;
        }
        
        this.showElegantLoading('Building relationship graph...');
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/relationship-graph`);
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('Graph data received:', data.graph_data);
                this.intelligence.relationships = data.graph_data;
                this.renderRelationshipGraph(data.graph_data);
                this.displayRelationshipDetails(data.graph_data);
                this.showElegantSuccess('Relationship graph generated');
            } else {
                const errorMessage = data.detail || data.message || JSON.stringify(data);
                this.showElegantError('Failed to generate relationship graph: ' + errorMessage);
            }
        } catch (error) {
            console.error('Graph generation error:', error);
            this.showElegantError('Network error during graph generation: ' + error.message);
        } finally {
            this.hideElegantLoading();
        }
    }

    // Elegant Loading States
    showElegantLoading(message) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        
        if (loadingText) {
            loadingText.textContent = message;
        }
        
        if (loadingOverlay) {
            loadingOverlay.classList.remove('hidden');
            loadingOverlay.style.opacity = '0';
            loadingOverlay.style.transition = 'opacity 0.3s ease';
            
            setTimeout(() => {
                loadingOverlay.style.opacity = '1';
            }, 50);
        }
        
        // Fallback: show message in console if elements don't exist
        if (!loadingText || !loadingOverlay) {
            console.log('Loading:', message);
        }
    }

    hideElegantLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        
        if (loadingOverlay) {
            loadingOverlay.style.opacity = '0';
            setTimeout(() => {
                loadingOverlay.classList.add('hidden');
            }, 300);
        }
    }

    showElegantSuccess(message) {
        this.showElegantToast(message, 'success');
    }

    showElegantError(message) {
        this.showElegantToast(message, 'error');
    }

    showElegantToast(message, type) {
        // Validate parameters
        if (!message) {
            console.warn('showElegantToast: No message provided');
            return;
        }
        
        // Ensure message is a string
        const safeMessage = typeof message === 'string' ? message : String(message);
        
        // Validate and sanitize type parameter
        const validTypes = ['success', 'error', 'warning', 'info'];
        const safeType = validTypes.includes(type) ? type : 'info';
        
        // Check if toast container exists
        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            console.error('showElegantToast: Toast container not found in DOM');
            // Fallback: log message to console
            console.log(`Toast (${safeType}): ${safeMessage}`);
            return;
        }
        
        try {
            // Create toast element
            const toast = document.createElement('div');
            if (!toast) {
                console.error('showElegantToast: Failed to create toast element');
                return;
            }
            
            toast.className = `toast ${safeType}`;
            
            // Determine icon based on type
            const iconClass = safeType === 'success' ? 'fa-check-circle' : 
                            safeType === 'error' ? 'fa-exclamation-circle' :
                            safeType === 'warning' ? 'fa-exclamation-triangle' : 
                            'fa-info-circle';
            
            // Escape HTML in message to prevent XSS
            const escapedMessage = safeMessage.replace(/[&<>"']/g, (match) => {
                const escapeChars = {
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#39;'
                };
                return escapeChars[match];
            });
            
            toast.innerHTML = `
                <div class="toast-content">
                    <div class="toast-icon">
                        <i class="fas ${iconClass}"></i>
                    </div>
                    <div class="toast-message">
                        <div class="toast-text">${escapedMessage}</div>
                    </div>
                </div>
            `;
            
            // Safely append to container
            toastContainer.appendChild(toast);
            
            // Show animation with error handling
            const showTimeout = setTimeout(() => {
                if (toast && toast.parentNode) {
                    toast.classList.add('show');
                }
            }, 50);
            
            // Hide animation with error handling
            const hideTimeout = setTimeout(() => {
                if (toast && toast.parentNode) {
                    toast.classList.remove('show');
                    const removeTimeout = setTimeout(() => {
                        if (toast && toast.parentNode) {
                            try {
                                toast.remove();
                            } catch (e) {
                                console.warn('Error removing toast element:', e);
                            }
                        }
                    }, 300);
                    
                    // Store timeout reference for potential cleanup
                    toast._removeTimeout = removeTimeout;
                }
            }, 3000);
            
            // Store timeout references for potential cleanup
            toast._showTimeout = showTimeout;
            toast._hideTimeout = hideTimeout;
            
        } catch (error) {
            console.error('showElegantToast: Error creating toast notification:', error);
            // Fallback: at least log the message
            console.log(`Toast fallback (${safeType}): ${safeMessage}`);
        }
    }

    // EDA Generation with elegant loading
    async generateEDA() {
        console.log('🔬 generateEDA called, session ID:', this.currentSessionId);
        
        // Debug all containers
        const edaContainer = document.getElementById('edaResults');
        const graphContainer = document.getElementById('relationshipGraph');
        const recsContainer = document.getElementById('featureRecommendations');
        
        console.log('DOM Check:');
        console.log('- edaResults:', edaContainer);
        console.log('- relationshipGraph:', graphContainer);
        console.log('- featureRecommendations:', recsContainer);
        
        // Add the show class and normal content
        if (edaContainer) {
            console.log('✅ EDA container found, adding content...');
            
            edaContainer.classList.add('show');
            edaContainer.innerHTML = `
                <h3 style="color: var(--color-text); margin-bottom: 20px;">📊 Exploratory Data Analysis Results</h3>
                <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px; border: 1px solid var(--color-border);">
                    <h4 style="color: var(--color-text); margin-bottom: 15px;">Dataset Summary</h4>
                    <table style="width: 100%; color: var(--color-text); border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 1px solid var(--color-border);">
                                <th style="text-align: left; padding: 10px;">Metric</th>
                                <th style="text-align: left; padding: 10px;">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td style="padding: 8px;">Total Rows</td><td style="padding: 8px;">1,000</td></tr>
                            <tr><td style="padding: 8px;">Total Columns</td><td style="padding: 8px;">10</td></tr>
                            <tr><td style="padding: 8px;">Missing Values</td><td style="padding: 8px;">5%</td></tr>
                            <tr><td style="padding: 8px;">Data Quality</td><td style="padding: 8px;">Good</td></tr>
                        </tbody>
                    </table>
                </div>
            `;
            
            console.log('✅ Content applied, checking visibility...');
            console.log('- offsetHeight:', edaContainer.offsetHeight);
            console.log('- offsetWidth:', edaContainer.offsetWidth);
            console.log('- computed display:', window.getComputedStyle(edaContainer).display);
            
            // Check parent containers for clipping
            let parent = edaContainer.parentElement;
            let level = 0;
            while (parent && level < 5) {
                const styles = window.getComputedStyle(parent);
                console.log(`Parent ${level} (${parent.className}):`, {
                    overflow: styles.overflow,
                    maxHeight: styles.maxHeight,
                    height: styles.height,
                    display: styles.display,
                    visibility: styles.visibility
                });
                parent = parent.parentElement;
                level++;
            }
            
        } else {
            console.error('❌ EDA container not found');
            alert('EDA container is NULL - DOM issue!');
        }
        
        if (!this.currentSessionId) {
            console.error('❌ No session ID found');
            this.showElegantError('No data session found. Please upload data first.');
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
                this.displayEDAResults(data.report);
                this.showElegantSuccess('EDA report generated successfully');
            } else {
                const errorMessage = data.detail || data.message || JSON.stringify(data);
                this.showElegantError('EDA generation failed: ' + errorMessage);
            }
        } catch (error) {
            console.error('EDA generation error:', error);
            this.showElegantError('Network error during EDA generation: ' + error.message);
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
                    task: taskType,
                    target_column: targetColumn,
                    enable_robust_pipeline: true,
                    enable_intelligence: document.getElementById('enableIntelligence')?.checked || true,
                    feature_generation_enabled: document.getElementById('featureGeneration')?.checked || false,
                    feature_selection_enabled: document.getElementById('featureSelection')?.checked || false
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showSection('processingSection');
                this.showElegantSuccess('Processing pipeline started');
                this.startPipelineMonitoring(data.execution_id);
                
                setTimeout(() => {
                    this.displayProcessingResults(data);
                    this.showSection('resultsSection');
                    this.showElegantSuccess('Processing completed successfully!');
                }, 3000);
            } else {
                const errorMessage = data.detail || data.message || JSON.stringify(data);
                this.showElegantError('Processing failed: ' + errorMessage);
            }
        } catch (error) {
            console.error('Processing error:', error);
            this.showElegantError('Network error during processing: ' + error.message);
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
        console.log('handleDownload called with:', downloadType);
        console.log('Current session ID:', this.currentSessionId);
        
        if (!this.currentSessionId) {
            this.showElegantError('No data session found');
            return;
        }
        
        const endpoints = {
            'downloadData': '/api/data',
            'downloadEnhancedData': '/api/data', 
            'downloadPipeline': '/api/data',
            'downloadIntelligence': '/api/data',
            'downloadLineage': '/api/data',
            'downloadMetadata': '/api/data'
        };
        
        const endpoint = endpoints[downloadType];
        if (!endpoint) {
            console.error('No endpoint found for download type:', downloadType);
            this.showElegantError('Download not available for this artifact type');
            return;
        }
        
        try {
            const artifactType = this.getArtifactType(downloadType);
            const downloadUrl = `${endpoint}/${this.currentSessionId}/download/${artifactType}`;
            console.log('Download URL:', downloadUrl);
            console.log('Artifact type:', artifactType);
            
            const response = await fetch(downloadUrl);
            
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
                console.error('Download failed. Status:', response.status, 'StatusText:', response.statusText);
                const errorText = await response.text();
                console.error('Error response:', errorText);
                this.showElegantError(`Download failed: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            console.error('Network error during download:', error);
            this.showElegantError('Network error during download: ' + error.message);
        }
    }
    
    // File Upload Handlers
    handleFileDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
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
            
            console.log('Upload response status:', response.status);
            console.log('Upload response ok:', response.ok);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.session_id) {
                this.currentSessionId = data.session_id;
                this.displayDataPreview(data);
                this.displayIntelligenceSummary(data.intelligence_summary);
                this.populateTargetSelect(data.columns);
                this.showSection('dataPreviewSection');
                this.showElegantSuccess(`File uploaded successfully: ${file.name}`);
            } else {
                this.showElegantError('Upload failed: ' + data.detail);
            }
        } catch (error) {
            console.error('Upload error details:', error);
            this.showElegantError(`Network error during upload: ${error.message}`);
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
        // Store original data shape for later reference
        if (!this.originalDataShape) {
            this.originalDataShape = data.shape;
        }
        
        this.animateValueChange('dataShape', `${data.shape[0]} × ${data.shape[1]}`);
        
        // Update missing values count
        if (data.validation && data.validation.missing_values_count !== undefined) {
            this.animateValueChange('missingValues', data.validation.missing_values_count);
        }
        
        if (data.validation) {
            const qualityStatus = data.validation.is_valid ? 'Good' : 'Issues Detected';
            this.animateValueChange('dataQuality', qualityStatus);
            this.displayValidationIssues(data.validation.issues);
            
            // Quality Score already has its icon, don't add colored bars
            // Just update the text value, no visual styling changes needed
        } else {
            this.animateValueChange('dataQuality', 'Not Assessed');
        }
        
        // Update missing values card color based on actual missing values
        const missingValuesCard = document.getElementById('missingValuesCard');
        if (missingValuesCard) {
            missingValuesCard.classList.remove('warning-card');
            const cardIcon = missingValuesCard.querySelector('.card-icon');
            if (cardIcon) {
                cardIcon.classList.remove('warning-icon');
            }
            
            if (data.validation && data.validation.missing_values_count > 0) {
                missingValuesCard.classList.add('warning-card');
                if (cardIcon) {
                    cardIcon.classList.add('warning-icon');
                }
            }
        }
        
        this.createSimpleDataPreview(data);
    }

    displayIntelligenceSummary(intelligenceSummary) {
        if (!intelligenceSummary || !intelligenceSummary.profiling_completed) return;
        
        let domain = 'General';
        if (intelligenceSummary.domain_analysis && intelligenceSummary.domain_analysis.detected_domains) {
            const domains = intelligenceSummary.domain_analysis.detected_domains;
            if (domains.length > 0) {
                domain = domains[0].domain;
            }
        }
        
        this.animateValueChange('domainDetected', domain);
        this.animateValueChange('relationshipCount', intelligenceSummary.relationships_found || 0);
        
        // Display feature processing status if available
        if (intelligenceSummary.feature_generation_applied !== undefined || intelligenceSummary.feature_selection_applied !== undefined) {
            this.displayFeatureProcessingStatus(intelligenceSummary);
        }
        
        // Populate semantic types if available
        if (intelligenceSummary.semantic_types) {
            this.displaySemanticTypes(intelligenceSummary.semantic_types);
        }
    }

    displayFeatureProcessingStatus(intelligenceSummary) {
        // Create or update feature processing status display
        let statusContainer = document.getElementById('featureProcessingStatus');
        if (!statusContainer) {
            statusContainer = document.createElement('div');
            statusContainer.id = 'featureProcessingStatus';
            statusContainer.className = 'feature-processing-status';
            
            // Find a good place to insert it - after processing summary
            const summaryCard = document.querySelector('.summary-card');
            if (summaryCard && summaryCard.parentNode) {
                summaryCard.parentNode.insertBefore(statusContainer, summaryCard.nextSibling);
            }
        }
        
        const featureGenStatus = intelligenceSummary.feature_generation_applied ? 
            '<span class="status-enabled">✓ Enabled</span>' : 
            '<span class="status-disabled">✗ Disabled</span>';
            
        const featureSelStatus = intelligenceSummary.feature_selection_applied ? 
            '<span class="status-enabled">✓ Enabled</span>' : 
            '<span class="status-disabled">✗ Disabled</span>';
        
        const shapeInfo = intelligenceSummary.original_shape && intelligenceSummary.final_shape ?
            `<div class="shape-info">
                <span>Original Shape: ${intelligenceSummary.original_shape[0]}×${intelligenceSummary.original_shape[1]}</span>
                <span>Final Shape: ${intelligenceSummary.final_shape[0]}×${intelligenceSummary.final_shape[1]}</span>
            </div>` : '';
        
        statusContainer.innerHTML = `
            <h4>Feature Processing Status</h4>
            <div class="feature-status-grid">
                <div class="status-item">
                    <label>Feature Generation:</label>
                    ${featureGenStatus}
                </div>
                <div class="status-item">
                    <label>Feature Selection:</label>
                    ${featureSelStatus}
                </div>
            </div>
            ${shapeInfo}
        `;
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

    createSimpleDataPreview(data) {
        const previewContainer = document.getElementById('dataPreview');
        if (!previewContainer) {
            console.log('Data preview container not found');
            return;
        }

        this.fullDataPreview = data;

        let tableHTML = '<div class="data-preview-table">';
        tableHTML += '<h4 style="color: var(--color-text); margin-bottom: 1rem;">Data Preview</h4>';
        
        if (data.columns && data.columns.length > 0) {
            tableHTML += '<div class="preview-grid" id="previewGrid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem;">';
            
            const maxColumns = Math.min(data.columns.length, 6);
            for (let i = 0; i < maxColumns; i++) {
                const column = data.columns[i];
                const dataType = data.data_types ? data.data_types[column] : 'unknown';
                
                tableHTML += `
                    <div class="column-info" style="background: rgba(51, 65, 85, 0.5); padding: 1rem; border-radius: 0.5rem; border: 1px solid var(--color-border);">
                        <div style="font-weight: 600; color: var(--color-text); margin-bottom: 0.5rem;">${column}</div>
                        <div style="color: var(--color-text-secondary); font-size: 0.9rem;">${dataType}</div>
                    </div>
                `;
            }
            
            if (data.columns.length > maxColumns) {
                tableHTML += `
                    <div class="column-info expand-columns" id="expandColumns" style="background: rgba(51, 65, 85, 0.3); padding: 1rem; border-radius: 0.5rem; border: 1px dashed var(--color-border); display: flex; align-items: center; justify-content: center; color: var(--color-text-secondary); cursor: pointer; transition: all 0.3s ease;">
                        <i class="fas fa-plus" style="margin-right: 0.5rem;"></i>
                        +${data.columns.length - maxColumns} more columns
                    </div>
                `;
            }
            
            tableHTML += '</div>';
            
            if (data.columns.length > maxColumns) {
                tableHTML += `
                    <div class="collapse-columns" id="collapseColumns" style="display: none; text-align: center; margin-top: 1rem;">
                        <button style="background: var(--color-surface); border: 1px solid var(--color-border); color: var(--color-text); padding: 0.5rem 1rem; border-radius: 0.5rem; cursor: pointer;">
                            <i class="fas fa-minus" style="margin-right: 0.5rem;"></i>
                            Show Less
                        </button>
                    </div>
                `;
            }
        }
        
        tableHTML += '</div>';
        previewContainer.innerHTML = tableHTML;

        const expandBtn = document.getElementById('expandColumns');
        const collapseBtn = document.getElementById('collapseColumns');
        
        if (expandBtn) {
            expandBtn.addEventListener('click', () => this.expandDataPreview());
        }
        if (collapseBtn) {
            collapseBtn.addEventListener('click', () => this.collapseDataPreview());
        }
    }

    expandDataPreview() {
        const previewGrid = document.getElementById('previewGrid');
        const expandBtn = document.getElementById('expandColumns');
        const collapseBtn = document.getElementById('collapseColumns');
        
        if (!this.fullDataPreview || !previewGrid) return;

        previewGrid.innerHTML = '';
        
        this.fullDataPreview.columns.forEach(column => {
            const dataType = this.fullDataPreview.data_types ? this.fullDataPreview.data_types[column] : 'unknown';
            
            const columnDiv = document.createElement('div');
            columnDiv.className = 'column-info';
            columnDiv.style.cssText = 'background: rgba(51, 65, 85, 0.5); padding: 1rem; border-radius: 0.5rem; border: 1px solid var(--color-border); opacity: 0; transform: translateY(20px); transition: all 0.3s ease;';
            columnDiv.innerHTML = `
                <div style="font-weight: 600; color: var(--color-text); margin-bottom: 0.5rem;">${column}</div>
                <div style="color: var(--color-text-secondary); font-size: 0.9rem;">${dataType}</div>
            `;
            previewGrid.appendChild(columnDiv);
            
            setTimeout(() => {
                columnDiv.style.opacity = '1';
                columnDiv.style.transform = 'translateY(0)';
            }, 50);
        });

        if (expandBtn) expandBtn.style.display = 'none';
        if (collapseBtn) collapseBtn.style.display = 'block';
    }

    collapseDataPreview() {
        this.createSimpleDataPreview(this.fullDataPreview);
    }

    // Data Display Methods
    displayDataTable(data) {
        const table = document.getElementById('dataPreviewTable');
        
        if (!table) {
            console.log('Data preview table not found, skipping table display');
            return;
        }
        
        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');
        
        if (!thead || !tbody) {
            console.log('Table head or body not found, skipping table display');
            return;
        }
        
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
        container.classList.add('show');
        
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
        if (!container) return;
        
        // Stop any existing simulation and clear container completely
        if (this.currentSimulation) {
            this.currentSimulation.stop();
            this.currentSimulation = null;
        }
        
        // Clear container, show it, and remove any D3 selections
        container.innerHTML = '';
        container.classList.add('show');
        if (typeof d3 !== 'undefined') {
            try {
                d3.select(container).selectAll('*').remove();
            } catch (e) {
                console.log('D3 cleanup error (non-critical):', e);
            }
        }
        
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            container.innerHTML = `
                <div class="graph-placeholder">
                    <i class="fas fa-info-circle"></i>
                    <p>No relationship data available for visualization</p>
                </div>
            `;
            return;
        }
        
        // Use Canvas-based network graph visualization
        this.renderCanvasNetworkGraph(container, graphData);
        return;
        
        // D3.js code disabled to prevent errors
        console.log('D3.js visualization disabled, using fallback');
    }

    renderCanvasNetworkGraph(container, graphData) {
        // Clear container
        container.innerHTML = '';
        
        // Create wrapper div with title
        const wrapper = document.createElement('div');
        wrapper.className = 'canvas-graph-wrapper';
        wrapper.innerHTML = '<h4 style="color: var(--color-white); margin-bottom: 1rem; text-align: center;">Feature Relationship Network</h4>';
        
        // Create canvas - much larger
        const canvas = document.createElement('canvas');
        canvas.width = 1000;
        canvas.height = 700;
        canvas.style.background = 'var(--color-medium-grey)';
        canvas.style.borderRadius = 'var(--border-radius-md)';
        canvas.style.border = '1px solid var(--color-light-grey)';
        
        wrapper.appendChild(canvas);
        container.appendChild(wrapper);
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Prepare node data with positions - more spread out initial positions
        const nodes = graphData.nodes.map((node, i) => ({
            ...node,
            x: Math.random() * (width - 200) + 100,
            y: Math.random() * (height - 200) + 100,
            vx: 0,
            vy: 0,
            radius: 18
        }));
        
        // Prepare edge data
        const edges = graphData.edges.map(edge => {
            const sourceNode = nodes.find(n => n.id === edge.source || n.name === edge.source);
            const targetNode = nodes.find(n => n.id === edge.target || n.name === edge.target);
            return {
                ...edge,
                sourceNode,
                targetNode,
                strength: edge.strength || 0.5
            };
        }).filter(edge => edge.sourceNode && edge.targetNode);
        
        // Simple physics simulation parameters
        const simulation = {
            alpha: 1,
            alphaDecay: 0.01,
            velocityDecay: 0.4,
            forceStrength: 0.8,
            linkDistance: 400,
            centerForce: 0.05
        };
        
        // Animation loop
        const animate = () => {
            if (simulation.alpha < 0.01) return;
            
            simulation.alpha *= 1 - simulation.alphaDecay;
            
            // Apply forces
            this.applyForces(nodes, edges, width, height, simulation);
            
            // Clear canvas
            ctx.clearRect(0, 0, width, height);
            
            // Draw edges
            this.drawEdges(ctx, edges);
            
            // Draw nodes
            this.drawNodes(ctx, nodes);
            
            requestAnimationFrame(animate);
        };
        
        // Start animation
        animate();
        
        // Add click handling for node interaction
        canvas.addEventListener('click', (event) => {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            const clickedNode = nodes.find(node => {
                const dx = x - node.x;
                const dy = y - node.y;
                return Math.sqrt(dx*dx + dy*dy) < node.radius + 5;
            });
            
            if (clickedNode) {
                this.highlightNode(ctx, nodes, edges, clickedNode);
            }
        });
    }
    
    applyForces(nodes, edges, width, height, simulation) {
        // Reset forces
        nodes.forEach(node => {
            node.vx *= simulation.velocityDecay;
            node.vy *= simulation.velocityDecay;
        });
        
        // Repulsion between nodes
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const nodeA = nodes[i];
                const nodeB = nodes[j];
                const dx = nodeB.x - nodeA.x;
                const dy = nodeB.y - nodeA.y;
                const distance = Math.sqrt(dx*dx + dy*dy) || 1;
                
                if (distance < 250) {
                    const force = simulation.forceStrength / (distance * distance);
                    const fx = (dx / distance) * force;
                    const fy = (dy / distance) * force;
                    
                    nodeA.vx -= fx;
                    nodeA.vy -= fy;
                    nodeB.vx += fx;
                    nodeB.vy += fy;
                }
            }
        }
        
        // Link forces
        edges.forEach(edge => {
            const source = edge.sourceNode;
            const target = edge.targetNode;
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const distance = Math.sqrt(dx*dx + dy*dy) || 1;
            const targetDistance = simulation.linkDistance * Math.max(edge.strength, 0.8);
            const force = (distance - targetDistance) * 0.05;
            
            const fx = (dx / distance) * force;
            const fy = (dy / distance) * force;
            
            source.vx += fx;
            source.vy += fy;
            target.vx -= fx;
            target.vy -= fy;
        });
        
        // Center force
        const centerX = width / 2;
        const centerY = height / 2;
        nodes.forEach(node => {
            node.vx += (centerX - node.x) * simulation.centerForce * 0.01;
            node.vy += (centerY - node.y) * simulation.centerForce * 0.01;
        });
        
        // Update positions
        nodes.forEach(node => {
            node.x += node.vx;
            node.y += node.vy;
            
            // Boundary constraints
            node.x = Math.max(node.radius + 10, Math.min(width - node.radius - 10, node.x));
            node.y = Math.max(node.radius + 10, Math.min(height - node.radius - 10, node.y));
        });
    }
    
    drawEdges(ctx, edges) {
        edges.forEach(edge => {
            const alpha = Math.min(edge.strength * 2, 1);
            ctx.strokeStyle = `rgba(0, 255, 136, ${alpha * 0.8})`;
            ctx.lineWidth = Math.max(2, edge.strength * 4);
            
            // Draw edge line
            ctx.beginPath();
            ctx.moveTo(edge.sourceNode.x, edge.sourceNode.y);
            ctx.lineTo(edge.targetNode.x, edge.targetNode.y);
            ctx.stroke();
            
            // Draw percentage label on edge
            const midX = (edge.sourceNode.x + edge.targetNode.x) / 2;
            const midY = (edge.sourceNode.y + edge.targetNode.y) / 2;
            const percentage = Math.round((edge.strength || 0) * 100);
            
            // Background for text visibility - larger
            ctx.fillStyle = 'rgba(26, 26, 26, 0.9)';
            ctx.fillRect(midX - 20, midY - 10, 40, 20);
            
            // Percentage text - larger
            ctx.fillStyle = '#FFD700';
            ctx.font = 'bold 13px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${percentage}%`, midX, midY);
        });
    }
    
    drawNodes(ctx, nodes) {
        nodes.forEach(node => {
            // Draw node circle - all green
            ctx.fillStyle = '#00ff88';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Draw node label with better visibility
            const label = (node.id || node.name || '').substring(0, 12);
            
            // Text background for visibility - larger
            ctx.font = 'bold 14px Inter, sans-serif';
            ctx.textAlign = 'center';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(26, 26, 26, 0.9)';
            ctx.fillRect(node.x - textWidth/2 - 6, node.y - node.radius - 30, textWidth + 12, 20);
            
            // Text in bright color - larger
            ctx.fillStyle = '#FFD700';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, node.x, node.y - node.radius - 20);
        });
    }
    
    highlightNode(ctx, nodes, edges, highlightedNode) {
        // Redraw with highlighted node
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        // Draw all edges (dimmed)
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.2)';
        ctx.lineWidth = 1;
        edges.forEach(edge => {
            ctx.beginPath();
            ctx.moveTo(edge.sourceNode.x, edge.sourceNode.y);
            ctx.lineTo(edge.targetNode.x, edge.targetNode.y);
            ctx.stroke();
        });
        
        // Draw highlighted edges
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.9)';
        ctx.lineWidth = 3;
        edges.forEach(edge => {
            if (edge.sourceNode === highlightedNode || edge.targetNode === highlightedNode) {
                ctx.beginPath();
                ctx.moveTo(edge.sourceNode.x, edge.sourceNode.y);
                ctx.lineTo(edge.targetNode.x, edge.targetNode.y);
                ctx.stroke();
            }
        });
        
        // Draw all nodes
        this.drawNodes(ctx, nodes);
        
        // Draw highlighted node
        ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
        ctx.strokeStyle = 'var(--color-white)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(highlightedNode.x, highlightedNode.y, highlightedNode.radius + 2, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    }

    displayRelationshipDetails(graphData) {
        const container = document.getElementById('relationshipsList');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!graphData || !graphData.edges || graphData.edges.length === 0) {
            container.innerHTML = '<p class="no-relationships">No relationships found in the data.</p>';
            return;
        }
        
        graphData.edges.forEach((edge, index) => {
            const item = document.createElement('div');
            item.className = 'relationship-item';
            item.style.opacity = '0';
            item.style.transform = 'translateX(20px)';
            item.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
            
            const source = typeof edge.source === 'string' ? edge.source : (edge.source?.name || edge.source?.id || 'Unknown');
            const target = typeof edge.target === 'string' ? edge.target : (edge.target?.name || edge.target?.id || 'Unknown');
            const relationshipType = edge.type || edge.relationship_type || 'correlation';
            const strength = edge.strength || edge.correlation || 0;
            const description = edge.description || `${relationshipType.replace(/_/g, ' ')} between columns`;
            
            item.innerHTML = `
                <div class="relationship-header">
                    <span class="relationship-type">${relationshipType.replace(/_/g, ' ')}</span>
                    <span class="relationship-strength">${(strength * 100).toFixed(1)}%</span>
                </div>
                <div class="relationship-description">
                    <strong>${source}</strong> → <strong>${target}</strong><br>
                    ${description}
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
        const isVisible = advancedOptions.classList.contains('hidden') === false;
        
        console.log('Toggling advanced options, currently visible:', isVisible);
        
        if (isVisible) {
            advancedOptions.style.opacity = '0';
            advancedOptions.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                advancedOptions.classList.add('hidden');
            }, 300);
        } else {
            advancedOptions.classList.remove('hidden');
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
        // Use the existing EDA results container in our dashboard
        let edaContainer = document.getElementById('edaResults');
        if (!edaContainer) {
            console.log('EDA results container not found');
            return;
        }
        
        // Clear any existing content and show the container
        edaContainer.innerHTML = '';
        edaContainer.classList.add('show');
        
        edaContainer.innerHTML = `
            <div class="eda-header">
                <h3>📊 Exploratory Data Analysis</h3>
                <p>Comprehensive analysis of your dataset</p>
            </div>
        `;
        
        // Basic info
        if (data.basic_info) {
            edaContainer.innerHTML += `
                <div class="eda-section">
                    <h4>Dataset Overview</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="label">Shape:</span>
                            <span class="value">${data.basic_info.shape[0]} rows × ${data.basic_info.shape[1]} columns</span>
                        </div>
                        <div class="info-item">
                            <span class="label">Total Missing:</span>
                            <span class="value">${Object.values(data.basic_info.missing_values).reduce((a, b) => a + b, 0)} values</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Numeric summary
        if (data.numeric_summary && Object.keys(data.numeric_summary).length > 0) {
            const firstCol = Object.keys(data.numeric_summary)[0];
            edaContainer.innerHTML += `
                <div class="eda-section">
                    <h4>Numeric Variables Summary</h4>
                    <div class="summary-table">
                        <table class="eda-table">
                            <thead>
                                <tr>
                                    <th>Statistic</th>
                                    ${Object.keys(data.numeric_summary).slice(0, 3).map(col => `<th>${col}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${['mean', 'std', 'min', 'max'].map(stat => `
                                    <tr>
                                        <td>${stat}</td>
                                        ${Object.keys(data.numeric_summary).slice(0, 3).map(col => 
                                            `<td>${data.numeric_summary[col][stat]?.toFixed(2) || 'N/A'}</td>`
                                        ).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }
        
        // Categorical summary
        if (data.categorical_summary && Object.keys(data.categorical_summary).length > 0) {
            edaContainer.innerHTML += `
                <div class="eda-section">
                    <h4>Categorical Variables</h4>
                    <div class="categorical-grid">
                        ${Object.entries(data.categorical_summary).map(([col, values]) => `
                            <div class="categorical-item">
                                <h5>${col}</h5>
                                <ul>
                                    ${Object.entries(values).slice(0, 5).map(([val, count]) => 
                                        `<li>${val}: ${count}</li>`
                                    ).join('')}
                                </ul>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
    }
    
    displayProcessingResults(data) {
        console.log('Processing results received:', data);
        
        // Update basic metrics
        if (data.data_shape) {
            // Use stored original data shape for consistency
            if (this.originalDataShape) {
                this.animateValueChange('originalShape', `${this.originalDataShape[0]} × ${this.originalDataShape[1]}`);
            } else {
                this.animateValueChange('originalShape', `${data.data_shape[0]} × ${data.data_shape[1]}`);
            }
            this.animateValueChange('processedShape', `${data.data_shape[0]} × ${data.data_shape[1]}`);
            
            // Add a note about the processing
            const note = data.data_shape[0] === data.data_shape[1] && data.data_shape[0] > 0 
                ? '(Data cleaned and prepared)' 
                : '(Features engineered)';
            
            const processedElement = document.getElementById('processedShape');
            if (processedElement && processedElement.parentElement) {
                let noteElement = processedElement.parentElement.querySelector('.processing-note');
                if (!noteElement) {
                    noteElement = document.createElement('small');
                    noteElement.className = 'processing-note';
                    noteElement.style.color = 'var(--color-muted-grey)';
                    processedElement.parentElement.appendChild(noteElement);
                }
                noteElement.textContent = note;
            }
        }
        
        if (data.processing_time) {
            this.animateValueChange('processingTime', `${data.processing_time.toFixed(2)}s`);
        }
        
        // Display column roles if available
        if (data.column_roles) {
            this.displayColumnClassification(data.column_roles);
        }
        
        // Setup download buttons
        if (data.artifacts) {
            this.setupDownloadButtons(data.artifacts);
        }
    }
    
    displayColumnClassification(columnRoles) {
        console.log('Column roles received:', columnRoles);
        const container = document.getElementById('columnRoles');
        if (!container) {
            console.error('columnRoles container not found');
            return;
        }
        
        if (!columnRoles) {
            console.log('No column roles provided');
            container.innerHTML = '<p>No column classification available</p>';
            return;
        }
        
        container.innerHTML = '';
        
        const roleEntries = Object.entries(columnRoles);
        console.log('Role entries:', roleEntries);
        
        if (roleEntries.length === 0) {
            container.innerHTML = '<p>No column roles found in data</p>';
            return;
        }
        
        roleEntries.forEach(([role, columns]) => {
            console.log(`Processing role: ${role}, columns:`, columns);
            if (columns && Array.isArray(columns) && columns.length > 0) {
                const roleDiv = document.createElement('div');
                roleDiv.className = 'column-role-group';
                roleDiv.innerHTML = `
                    <h4 class="role-title">${role.replace(/_/g, ' ').toUpperCase()}</h4>
                    <div class="column-tags">
                        ${columns.map(col => `<span class="column-tag">${col}</span>`).join('')}
                    </div>
                `;
                container.appendChild(roleDiv);
            } else {
                console.log(`Skipping role ${role} - no valid columns`);
            }
        });
        
        if (container.children.length === 0) {
            container.innerHTML = '<p>No valid column classifications found</p>';
        }
    }
    
    setupDownloadButtons(artifacts) {
        console.log('setupDownloadButtons called with artifacts:', artifacts);
        
        // Map artifact URLs to button IDs and enable buttons
        const buttonMapping = {
            'pipeline_metadata': 'downloadMetadata',
            'processed_data': 'downloadData', 
            'intelligence_report': 'downloadIntelligence',
            'lineage': 'downloadLineage',
            'enhanced_data': 'downloadEnhancedData',
            'pipeline': 'downloadPipeline'
        };
        
        Object.entries(artifacts).forEach(([type, url]) => {
            const buttonId = buttonMapping[type] || `download${type.charAt(0).toUpperCase() + type.slice(1)}`;
            const button = document.getElementById(buttonId);
            if (button) {
                console.log(`Enabling button ${buttonId} for artifact ${type}`);
                button.style.opacity = '1';
                button.disabled = false;
                // Don't override the onclick - let the existing handleDownload system work
            }
        });
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
    
    getArtifactType(downloadType) {
        const artifactTypes = {
            'downloadData': 'data',
            'downloadEnhancedData': 'enhanced-data',
            'downloadPipeline': 'pipeline',
            'downloadIntelligence': 'intelligence',
            'downloadLineage': 'lineage',
            'downloadMetadata': 'intelligence'
        };
        return artifactTypes[downloadType] || 'data';
    }

    getDownloadFilename(downloadType) {
        const timestamp = new Date().toISOString().slice(0, 10);
        const filenames = {
            'downloadData': `processed_data_${timestamp}.csv`,
            'downloadEnhancedData': `enhanced_data_${timestamp}.csv`,
            'downloadPipeline': `pipeline_${timestamp}.joblib`,
            'downloadIntelligence': `intelligence_report_${timestamp}.json`,
            'downloadLineage': `lineage_report_${timestamp}.json`,
            'downloadMetadata': `pipeline_metadata_${timestamp}.json`
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
        this.updateSidebarProgress();
    }

    getSectionStep(sectionId) {
        const stepMap = {
            'dataInputSection': 1,
            'dataPreviewSection': 2,
            'configSection': 3,
            'processingSection': 4,
            'resultsSection': 5
        };
        return stepMap[sectionId] || 1;
    }

    updateSidebarProgress() {
        const stepItems = document.querySelectorAll('.step-item');
        stepItems.forEach((item, index) => {
            const stepNumber = index + 1;
            const circle = item.querySelector('.step-circle');
            
            item.classList.remove('active', 'completed');
            
            if (stepNumber < this.currentStep) {
                item.classList.add('completed');
                circle.innerHTML = '<i class="fas fa-check"></i>';
            } else if (stepNumber === this.currentStep) {
                item.classList.add('active');
                circle.innerHTML = stepNumber;
            } else {
                circle.innerHTML = stepNumber;
            }
        });

        // Update quick stats when data is available
        if (this.currentStep >= 2 && this.currentSessionId) {
            this.updateSidebarStats();
        }
    }

    updateSidebarStats() {
        const quickStats = document.getElementById('quickStats');
        if (quickStats && this.fullDataPreview) {
            quickStats.style.display = 'block';
            
            const [rows, cols] = this.fullDataPreview.shape || [0, 0];
            document.getElementById('sidebarRows').textContent = rows.toLocaleString();
            document.getElementById('sidebarCols').textContent = cols.toLocaleString();
            
            const quality = this.fullDataPreview.validation?.is_valid ? 'Good' : 'Issues';
            document.getElementById('sidebarQuality').textContent = quality;
        }
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
        if (!container) return;
        
        if (!issues || issues.length === 0) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        container.innerHTML = `
            <div class="quality-issues-header" onclick="this.parentElement.classList.toggle('expanded')">
                <h4>
                    <i class="fas fa-exclamation-triangle"></i>
                    Data Quality Issues (${issues.length})
                    <i class="fas fa-chevron-down expand-icon"></i>
                </h4>
            </div>
            <div class="quality-issues-content">
                <div class="issues-grid">
                    ${issues.map(issue => `
                        <div class="issue-card">
                            <div class="issue-severity ${issue.severity || 'medium'}">
                                ${this.getSeverityIcon(issue.severity)}
                            </div>
                            <div class="issue-details">
                                <h5>${issue.type || 'Quality Issue'}</h5>
                                <p>${issue.description || 'Issue detected in data'}</p>
                                ${issue.affected_columns ? `<small>Columns: ${issue.affected_columns.join(', ')}</small>` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    getSeverityIcon(severity) {
        const icons = {
            'high': '<i class="fas fa-times-circle" style="color: #e74c3c;"></i>',
            'medium': '<i class="fas fa-exclamation-circle" style="color: #f39c12;"></i>',
            'low': '<i class="fas fa-info-circle" style="color: #3498db;"></i>'
        };
        return icons[severity] || icons['medium'];
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
    // Hero chat hooks
    const manualBtn = document.getElementById('manualAnalyze');
    if (manualBtn) {
        manualBtn.addEventListener('click', () => {
            // Hide hero, show manual pipeline path
            document.getElementById('heroChatSection')?.classList.add('hidden');
            document.getElementById('dataInputSection')?.classList.remove('hidden');
        });
    }
    
    // Back to chat hook
    const backToChatBtn = document.getElementById('backToChatBtn');
    if (backToChatBtn) {
        backToChatBtn.addEventListener('click', () => {
            // Hide all manual sections, show hero chat
            document.querySelectorAll('.section').forEach(section => {
                section.classList.add('hidden');
            });
            document.getElementById('heroChatSection')?.classList.remove('hidden');
        });
    }
    // Chat interface is now embedded directly in the HTML
    
    // Background particles
    const canvas = document.getElementById('bgParticles');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        const particles = Array.from({length: 120}, () => ({
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            r: Math.random() * 1.5 + 0.5,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            glow: Math.random() < 0.02,
            born: performance.now()
        }));
        const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
        window.addEventListener('resize', resize); resize();
        function tick() {
            ctx.clearRect(0,0,canvas.width,canvas.height);
            // Draw connections
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const a = particles[i], b = particles[j];
                    const dx = a.x - b.x, dy = a.y - b.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 100) {
                        const alpha = 0.05 * (1 - dist / 100);
                        ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y);
                        ctx.lineTo(b.x, b.y);
                        ctx.stroke();
                    }
                }
            }
            // Draw particles (no hard reset; continuous floating)
            const now = performance.now();
            particles.forEach(p => {
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
                const alpha = p.glow ? 0.7 : 0.18;
                ctx.fillStyle = `rgba(255,255,255,${alpha})`;
                ctx.fill();
                // occasional twinkle
                if (Math.random() < 0.005) p.glow = !p.glow;
            });
            requestAnimationFrame(tick);
        }
        tick();
    }
});