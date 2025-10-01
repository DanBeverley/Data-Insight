class UIManager {
    constructor() {
        this.app = null;
        this.loadingMessageElement = null;
    }

    setApp(app) {
        this.app = app;
    }

    // File Upload Setup
    setupFileUpload() {
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');

        if (!fileInput || !uploadZone) {
            return;
        }

        uploadZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.processFile(file);
            }
        });

        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        uploadZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processFile(files[0]);
            }
        });

        // URL input functionality
        const urlSubmit = document.getElementById('urlSubmit');
        if (urlSubmit) {
            urlSubmit.addEventListener('click', () => {
                this.app.animationManager.animateButton(urlSubmit);
                this.handleUrlSubmit();
            });
        }

        const urlInput = document.getElementById('urlInput');
        if (urlInput) {
            urlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.app.animationManager.animateInput(urlInput);
                    this.handleUrlSubmit();
                }
            });

            urlInput.addEventListener('focus', () => this.app.animationManager.animateInput(urlInput, 'focus'));
            urlInput.addEventListener('blur', () => this.app.animationManager.animateInput(urlInput, 'blur'));
        }
    }

    async processFile(file) {
        this.showElegantLoading('Uploading and analyzing your data...');

        try {
            const data = await this.app.apiClient.uploadFile(file, this.app.currentSessionId || this.app.agentSessionId);

            if (!data.session_id) {
                this.app.currentSessionId = data.session_id;
                this.app.agentSessionId = data.session_id;
            }

            if (data.status === 'success') {
                this.displayDataPreview(data);
                this.displayIntelligenceSummary(data.intelligence_summary);
                this.showElegantToast(`File uploaded successfully: ${file.name}`, 'success');
            } else {
                this.showElegantToast('Upload failed: ' + data.detail, 'error');
            }
        } catch (error) {
            console.error('Upload error details:', error);
            this.showElegantToast(`Network error during upload: ${error.message}`, 'error');
        } finally {
            this.hideElegantLoading();
        }
    }

    async handleUrlSubmit() {
        const urlInput = document.getElementById('urlInput');
        const url = urlInput?.value.trim();

        if (!url) {
            this.showElegantToast('Please enter a valid URL', 'error');
            return;
        }

        this.showElegantLoading('Fetching data from URL...');

        try {
            const data = await this.app.apiClient.ingestFromUrl(url);

            if (data.session_id) {
                this.app.currentSessionId = data.session_id;
                this.app.agentSessionId = data.session_id;
                this.displayDataPreview(data);
                this.displayIntelligenceSummary(data.intelligence_summary);
                this.showElegantToast('Data fetched successfully from URL', 'success');
            } else {
                this.showElegantToast('URL ingestion failed: ' + data.detail, 'error');
            }
        } catch (error) {
            this.showElegantToast('Network error during URL fetch', 'error');
        } finally {
            this.hideElegantLoading();
        }
    }

    // Intelligence Features Setup
    setupIntelligenceFeatures() {
        const deepProfileButton = document.getElementById('deepProfile');
        if (deepProfileButton) {
            deepProfileButton.addEventListener('click', () => {
                this.app.animationManager.animateButton(deepProfileButton);
                this.performDeepProfiling();
            });
        }

        const getRecommendations = document.getElementById('getRecommendations');
        if (getRecommendations) {
            getRecommendations.addEventListener('click', () => {
                this.app.animationManager.animateButton(getRecommendations);
                this.getFeatureRecommendations();
            });
        }

        const generateGraph = document.getElementById('generateGraph');
        if (generateGraph) {
            generateGraph.addEventListener('click', () => {
                this.app.animationManager.animateButton(generateGraph);
                this.generateRelationshipGraph();
            });
        }

        const generateEDA = document.getElementById('generateEDA');
        if (generateEDA) {
            generateEDA.addEventListener('click', () => {
                this.app.animationManager.animateButton(generateEDA);
                this.generateEDA();
            });
        }
    }

    async performDeepProfiling() {
        this.showElegantLoading('Performing deep intelligence analysis...');

        try {
            const data = await this.app.apiClient.performDeepProfiling(this.app.currentSessionId);
            if (data.status === 'success') {
                this.displayIntelligenceProfile(data.intelligence_profile);
                this.showElegantToast('Deep profiling completed', 'success');
            } else {
                this.showElegantToast('Profiling failed: ' + data.detail, 'error');
            }
        } catch (error) {
            this.showElegantToast('Network error during profiling', 'error');
        } finally {
            this.hideElegantLoading();
        }
    }

    async getFeatureRecommendations() {
        const recsContainer = document.getElementById('featureRecommendations');
        if (recsContainer) {
            recsContainer.innerHTML = `
                <div class="loading-message">
                    <div class="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <p>This proves the function is working!</p>
                </div>
            `;
        }

        if (!this.app.currentSessionId) {
            this.showElegantToast('No data session found. Please upload data first.', 'error');
            return;
        }

        const priorityFilterElement = document.getElementById('priorityFilter');
        const targetColumnElement = document.getElementById('targetSelect');

        const priorityFilter = priorityFilterElement?.value;
        const targetColumn = targetColumnElement?.value;

        this.showElegantLoading('Generating AI feature recommendations...');

        try {
            const data = await this.app.apiClient.getFeatureRecommendations(this.app.currentSessionId, {
                priorityFilter,
                targetColumn
            });

            if (data.status === 'success') {
                this.displayFeatureRecommendations(data);
                this.showElegantToast(`Generated ${data.recommendations.length} recommendations`, 'success');
            } else {
                const errorMessage = data.detail || data.message || JSON.stringify(data);
                this.showElegantToast('Failed to generate recommendations: ' + errorMessage, 'error');
            }
        } catch (error) {
            console.error('Recommendation generation error:', error);
            this.showElegantToast('Network error during recommendation generation: ' + error.message, 'error');
        } finally {
            this.hideElegantLoading();
        }
    }

    async generateRelationshipGraph() {
        const graphContainer = document.getElementById('relationshipGraph');
        if (graphContainer) {
            graphContainer.innerHTML = `
                <div class="loading-message">
                    <div class="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <p>This proves the function is working!</p>
                </div>
            `;
        }

        if (!this.app.currentSessionId) {
            this.showElegantToast('No data session found. Please upload data first.', 'error');
            return;
        }

        this.showElegantLoading('Building relationship graph...');

        try {
            const data = await this.app.apiClient.generateRelationshipGraph(this.app.currentSessionId);

            if (data.status === 'success') {
                this.renderRelationshipGraph(data.graph_data);
                this.displayRelationshipDetails(data.graph_data);
                this.showElegantToast('Relationship graph generated', 'success');
            } else {
                const errorMessage = data.detail || data.message || JSON.stringify(data);
                this.showElegantToast('Failed to generate relationship graph: ' + errorMessage, 'error');
            }
        } catch (error) {
            console.error('Graph generation error:', error);
            this.showElegantToast('Network error during graph generation: ' + error.message, 'error');
        } finally {
            this.hideElegantLoading();
        }
    }

    async generateEDA() {
        let edaContainer = document.getElementById('edaResults');
        if (!edaContainer) {
            console.log('EDA results container not found');
            return;
        }

        edaContainer.innerHTML = `
            <div class="loading-message">
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
                <p>Generating comprehensive EDA report...</p>
            </div>
        `;

        if (!this.app.currentSessionId) {
            this.showElegantToast('No data session found. Please upload data first.', 'error');
            return;
        }

        this.showElegantLoading('Generating comprehensive EDA report...');

        try {
            const data = await this.app.apiClient.generateEDA(this.app.currentSessionId);

            if (data.status === 'success') {
                this.displayEDAResults(data.report);
                this.showElegantToast('EDA report generated successfully', 'success');
            } else {
                const errorMessage = data.detail || data.message || JSON.stringify(data);
                this.showElegantToast('EDA generation failed: ' + errorMessage, 'error');
            }
        } catch (error) {
            console.error('EDA generation error:', error);
            this.showElegantToast('Network error during EDA generation: ' + error.message, 'error');
        } finally {
            this.hideElegantLoading();
        }
    }

    // Continue Button Setup
    setupContinueButton() {
        const continueButton = document.getElementById('continueToConfig');
        if (!continueButton) {
            return;
        }

        continueButton.addEventListener('click', (e) => {
            if (!e || !e.target) {
                return;
            }

            if (this.app.lastClickTime && Date.now() - this.app.lastClickTime < 500) {
                return;
            }
            this.app.lastClickTime = Date.now();

            try {
                if (typeof this.app.animationManager.animateButton === 'function') {
                    this.app.animationManager.animateButton(continueButton);
                } else {
                    console.warn('animateButton method not available');
                }

                if (typeof this.app.showSection === 'function') {
                    this.app.showSection('taskConfigSection');
                } else {
                    console.warn('showSection method not available');
                }

                this.showElegantToast('Ready for configuration! Select your task type below.', 'success');
            } catch (error) {
                console.error('Error in continue button handler:', error);
                const targetSection = document.getElementById('taskConfigSection');
                if (targetSection) {
                    targetSection.classList.remove('hidden');
                }
                this.showElegantToast('Configuration section opened', 'info');
            }
        });
    }

    // Downloads Setup
    setupDownloads() {
        ['downloadData', 'downloadModel', 'downloadReport', 'downloadCode', 'downloadInsights'].forEach(id => {
            const button = document.getElementById(id);
            if (button) {
                button.addEventListener('click', () => {
                    this.app.animationManager.animateButton(button, { showProgress: true });
                    this.handleDownload(button.id);
                });
            }
        });

        const restartButton = document.getElementById('processNewData');
        if (restartButton) {
            restartButton.addEventListener('click', () => {
                this.app.animationManager.animateButton(restartButton);
                this.app.restartWorkflow();
            });
        }
    }

    async handleDownload(downloadType) {
        if (!this.app.currentSessionId) {
            this.showElegantToast('No data session found', 'error');
            return;
        }

        const typeMap = {
            'downloadData': 'data',
            'downloadModel': 'model',
            'downloadReport': 'report',
            'downloadCode': 'code',
            'downloadInsights': 'insights'
        };

        const artifactType = typeMap[downloadType];
        if (!artifactType) {
            this.showElegantToast('Download not available for this artifact type', 'error');
            return;
        }

        try {
            await this.app.apiClient.downloadArtifact(this.app.currentSessionId, artifactType);
            this.showElegantToast('Download completed successfully', 'success');
        } catch (error) {
            this.showElegantToast('Download failed: ' + error.message, 'error');
        }
    }

    // Loading and Toast Methods
    showElegantLoading(message) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');

        if (loadingText) {
            loadingText.textContent = message;
        }

        if (loadingOverlay) {
            loadingOverlay.classList.remove('hidden');
            loadingOverlay.style.opacity = '1';
        }

        if (!loadingText || !loadingOverlay) {
            console.log(`Loading: ${message}`);
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

    showElegantToast(message, type) {
        if (!message) {
            console.warn('showElegantToast: No message provided');
            return;
        }

        const safeMessage = typeof message === 'string' ? message : String(message);
        const validTypes = ['success', 'error', 'warning', 'info'];
        const safeType = validTypes.includes(type) ? type : 'info';

        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            console.error('showElegantToast: Toast container not found in DOM');
            console.log(`Toast (${safeType}): ${safeMessage}`);
            return;
        }

        try {
            const toast = document.createElement('div');
            if (!toast) {
                console.error('showElegantToast: Failed to create toast element');
                return;
            }

            toast.className = `toast ${safeType}`;
            const iconClass = safeType === 'success' ? 'fa-check-circle' :
                             safeType === 'error' ? 'fa-exclamation-circle' :
                             safeType === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';

            toast.innerHTML = `
                <i class="fas ${iconClass}"></i>
                <span class="toast-message">${safeMessage}</span>
                <button class="toast-close" onclick="this.parentElement.remove()">×</button>
            `;

            toastContainer.appendChild(toast);

            setTimeout(() => {
                if (toast && toast.parentNode) {
                    toast.style.opacity = '0';
                    setTimeout(() => {
                        if (toast && toast.parentNode) {
                            try {
                                toast.remove();
                            } catch (removeError) {
                                if (toast && toast.parentNode) {
                                    toast.parentNode.removeChild(toast);
                                }
                            }
                        }
                    }, 300);
                }
            }, 5000);

        } catch (error) {
            console.error('showElegantToast: Error creating toast notification:', error);
            console.log(`Toast fallback (${safeType}): ${safeMessage}`);
        }
    }

    // Display Methods
    displayDataPreview(data) {
        if (!this.app.originalDataShape) {
            this.app.originalDataShape = data.shape;
        }

        this.app.animateValueChange('dataShape', `${data.shape[0]} × ${data.shape[1]}`);

        if (data.validation && data.validation.missing_values_count !== undefined) {
            this.app.animateValueChange('missingValues', data.validation.missing_values_count);
        }

        if (data.validation) {
            const qualityStatus = data.validation.is_valid ? 'Good' : 'Issues Detected';
            this.app.animateValueChange('dataQuality', qualityStatus);
            if (data.validation.issues) {
                this.displayValidationIssues(data.validation.issues);
            }
        } else {
            this.app.animateValueChange('dataQuality', 'Not Assessed');
        }

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

        this.app.animateValueChange('domainDetected', domain);
        this.app.animateValueChange('relationshipCount', intelligenceSummary.relationships_found || 0);

        if (intelligenceSummary.feature_generation_applied !== undefined || intelligenceSummary.feature_selection_applied !== undefined) {
            this.displayFeatureProcessingStatus(intelligenceSummary);
        }

        // Populate semantic types if available
        if (intelligenceSummary.semantic_types) {
            this.displaySemanticTypes(intelligenceSummary.semantic_types);
        }
    }

    displayFeatureProcessingStatus(intelligenceSummary) {
        let statusContainer = document.getElementById('featureProcessingStatus');
        if (!statusContainer) {
            statusContainer = document.createElement('div');
            statusContainer.id = 'featureProcessingStatus';
            statusContainer.className = 'feature-processing-status';

            const summaryCard = document.getElementById('intelligenceSummary');
            if (summaryCard && summaryCard.parentNode) {
                summaryCard.parentNode.insertBefore(statusContainer, summaryCard.nextSibling);
            }
        }

        const featureGenStatus = intelligenceSummary.feature_generation_applied ? 'Applied' : 'Not Applied';
        const featureSelStatus = intelligenceSummary.feature_selection_applied ? 'Applied' : 'Not Applied';

        statusContainer.innerHTML = `
            <div class="status-header">
                <h4>🧠 Feature Intelligence</h4>
            </div>
            <div class="status-grid">
                <div class="status-item">
                    <span class="status-label">Feature Generation:</span>
                    <span class="status-value ${intelligenceSummary.feature_generation_applied ? 'success' : 'pending'}">${featureGenStatus}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Feature Selection:</span>
                    <span class="status-value ${intelligenceSummary.feature_selection_applied ? 'success' : 'pending'}">${featureSelStatus}</span>
                </div>
            </div>
        `;
    }

    createSimpleDataPreview(data) {
        const previewContainer = document.getElementById('dataPreview');
        if (!previewContainer) {
            return;
        }

        previewContainer.innerHTML = `
            <div class="preview-header">
                <h4>📊 Data Preview</h4>
                <div class="preview-controls">
                    <button id="expandColumns" class="btn-expand">Expand</button>
                    <button id="collapseColumns" class="btn-collapse" style="display: none;">Collapse</button>
                </div>
            </div>
        `;

        if (data.columns && data.columns.length > 0) {
            const maxColumns = 4;
            const previewGrid = document.createElement('div');
            previewGrid.className = 'preview-grid';
            previewGrid.id = 'previewGrid';

            for (let i = 0; i < maxColumns; i++) {
                if (i < data.columns.length) {
                    const colDiv = document.createElement('div');
                    colDiv.className = 'preview-column';
                    colDiv.innerHTML = `
                        <div class="column-header">${data.columns[i]}</div>
                        <div class="column-sample">${data.sample_data && data.sample_data[0] ? data.sample_data[0][i] || 'N/A' : 'N/A'}</div>
                    `;
                    previewGrid.appendChild(colDiv);
                }
            }

            if (data.columns.length > maxColumns) {
                const moreDiv = document.createElement('div');
                moreDiv.className = 'preview-column more-columns';
                moreDiv.innerHTML = `
                    <div class="column-header">+${data.columns.length - maxColumns} more</div>
                    <div class="column-sample">...</div>
                `;
                previewGrid.appendChild(moreDiv);
            }

            previewContainer.appendChild(previewGrid);

            if (data.columns.length > maxColumns) {
                this.setupPreviewExpansion(data);
            }
        }

        this.app.fullDataPreview = data;

        const expandBtn = document.getElementById('expandColumns');
        if (expandBtn) {
            expandBtn.addEventListener('click', () => this.expandDataPreview());
        }

        const collapseBtn = document.getElementById('collapseColumns');
        if (collapseBtn) {
            collapseBtn.addEventListener('click', () => this.collapseDataPreview());
        }
    }

    setupPreviewExpansion(data) {
        // Implementation for preview expansion functionality
    }

    expandDataPreview() {
        const previewGrid = document.getElementById('previewGrid');
        const expandBtn = document.getElementById('expandColumns');
        const collapseBtn = document.getElementById('collapseColumns');

        if (previewGrid && this.app.fullDataPreview) {
            previewGrid.innerHTML = '';
            this.app.fullDataPreview.columns.forEach((col, index) => {
                const colDiv = document.createElement('div');
                colDiv.className = 'preview-column';
                colDiv.innerHTML = `
                    <div class="column-header">${col}</div>
                    <div class="column-sample">${this.app.fullDataPreview.sample_data && this.app.fullDataPreview.sample_data[0] ? this.app.fullDataPreview.sample_data[0][index] || 'N/A' : 'N/A'}</div>
                `;
                previewGrid.appendChild(colDiv);
            });

            if (expandBtn) expandBtn.style.display = 'none';
            if (collapseBtn) collapseBtn.style.display = 'inline-block';
        }
    }

    collapseDataPreview() {
        if (this.app.fullDataPreview) {
            this.createSimpleDataPreview(this.app.fullDataPreview);
        }
    }

    displayValidationIssues(issues) {
        const container = document.getElementById('validationIssues');
        if (!container) {
            return;
        }

        if (!issues || !Array.isArray(issues) || issues.length === 0) {
            container.innerHTML = '<div class="no-issues">✅ No validation issues found</div>';
            return;
        }

        container.innerHTML = `
            <div class="issues-header">
                <h4>⚠️ Data Quality Issues (${issues.length})</h4>
            </div>
            <div class="issues-list">
                ${issues.map(issue => `
                    <div class="issue-item ${issue.severity}">
                        <div class="issue-icon">${this.getSeverityIcon(issue.severity)}</div>
                        <div class="issue-content">
                            <div class="issue-title">${issue.column || 'General'}</div>
                            <div class="issue-description">${issue.message}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    getSeverityIcon(severity) {
        switch (severity) {
            case 'high': return '🔴';
            case 'medium': return '🟡';
            case 'low': return '🟢';
            default: return 'ℹ️';
        }
    }

    // Additional display methods would be implemented here
    displaySemanticTypes(semanticTypes) {
        // Implementation for semantic types display
        const container = document.getElementById('semanticTypesTable');
        if (!container) return;

        const tbody = container.querySelector('tbody');
        if (!tbody) return;

        tbody.innerHTML = '';
        Object.entries(semanticTypes).forEach(([column, type]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${column}</td>
                <td><span class="semantic-type-badge ${type.toLowerCase()}">${type}</span></td>
            `;
            tbody.appendChild(row);
        });
    }

    displayIntelligenceProfile(profile) {
        // Implementation for intelligence profile display
    }

    displayFeatureRecommendations(data) {
        // Implementation for feature recommendations display
    }

    displayEDAResults(data) {
        // Implementation for EDA results display
    }

    renderRelationshipGraph(graphData) {
        // Implementation for relationship graph rendering
    }

    displayRelationshipDetails(graphData) {
        // Implementation for relationship details display
    }

    // Utility method to clear all results
    clearAllResults() {
        const elementsToReset = [
            'dataPreviewTable', 'semanticTypesTable', 'relationshipGraph',
            'featureRecommendations', 'fileInput', 'urlInput', 'taskSelect', 'targetSelect'
        ];

        elementsToReset.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                if (element.tagName === 'INPUT' || element.tagName === 'SELECT') {
                    element.value = '';
                } else {
                    element.innerHTML = '';
                }
            }
        });

        // Reset target select
        const targetSelect = document.getElementById('targetSelect');
        if (targetSelect) {
            targetSelect.innerHTML = '<option value="">Select target column...</option>';
        }
    }
}