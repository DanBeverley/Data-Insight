/**
 * DataInsight AI - Frontend JavaScript
 * Modern interactive frontend for the DataInsight AI platform
 */

class DataInsightApp {
    constructor() {
        this.currentSessionId = null;
        this.currentStep = 1;
        this.processingSteps = [
            'Profiling data structure...',
            'Classifying columns...',
            'Building preprocessing pipeline...',
            'Applying transformations...',
            'Generating features...',
            'Selecting optimal features...',
            'Finalizing results...'
        ];
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeUI();
    }

    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');
        
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadZone.addEventListener('drop', this.handleFileDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // URL ingestion
        document.getElementById('urlSubmit').addEventListener('click', this.handleUrlSubmit.bind(this));
        document.getElementById('urlInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleUrlSubmit();
        });

        // EDA generation
        document.getElementById('generateEDA').addEventListener('click', this.generateEDA.bind(this));

        // Task configuration
        document.getElementById('taskSelect').addEventListener('change', this.handleTaskChange.bind(this));
        document.getElementById('advancedToggle').addEventListener('change', this.toggleAdvancedOptions.bind(this));

        // Processing
        document.getElementById('startProcessing').addEventListener('click', this.startProcessing.bind(this));

        // Downloads
        document.getElementById('downloadData').addEventListener('click', () => this.downloadArtifact('data'));
        document.getElementById('downloadPipeline').addEventListener('click', () => this.downloadArtifact('pipeline'));
        document.getElementById('downloadLineage').addEventListener('click', () => this.downloadArtifact('lineage'));

        // Restart
        document.getElementById('processNewData').addEventListener('click', this.restart.bind(this));
    }

    initializeUI() {
        this.showSection('dataInputSection');
        this.updateStatus('Ready');
    }

    // File handling methods
    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadZone').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadZone').classList.remove('dragover');
    }

    handleFileDrop(e) {
        e.preventDefault();
        document.getElementById('uploadZone').classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        this.showLoading('Uploading and analyzing file...');
        
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.currentSessionId = result.session_id;
                this.currentSessionData = result; // Store for task configuration
                this.displayDataOverview(result);
                this.showSection('dataPreviewSection');
                this.showToast('File uploaded successfully!', 'success');
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async handleUrlSubmit() {
        const url = document.getElementById('urlInput').value.trim();
        if (!url) {
            this.showToast('Please enter a valid URL', 'warning');
            return;
        }

        this.showLoading('Fetching data from URL...');

        try {
            const response = await fetch('/api/ingest-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: url,
                    data_type: 'csv'
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.currentSessionId = result.session_id;
                this.displayDataOverview(result);
                this.showSection('dataPreviewSection');
                this.showToast('Data fetched successfully!', 'success');
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.showToast(`URL ingestion failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayDataOverview(data) {
        // Update data info cards
        document.getElementById('dataShape').textContent = `${data.shape[0]} rows × ${data.shape[1]} columns`;
        document.getElementById('columnCount').textContent = data.shape[1];
        
        // Update quality status
        const qualityIcon = document.querySelector('.quality-icon i');
        const qualityStatus = document.getElementById('qualityStatus');
        
        if (data.validation.is_valid) {
            qualityStatus.textContent = 'Excellent';
            qualityStatus.className = 'info-value text-success';
            qualityIcon.className = 'fas fa-shield-alt text-success';
        } else {
            qualityStatus.textContent = `${data.validation.issues.length} Issues`;
            qualityStatus.className = 'info-value text-warning';
            qualityIcon.className = 'fas fa-exclamation-triangle text-warning';
            this.displayValidationIssues(data.validation.issues);
        }

        // Load and display data preview
        this.loadDataPreview();
        
        // Populate target column options
        this.populateTargetColumns(data.columns);
    }

    async loadDataPreview() {
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/preview?rows=10`);
            const result = await response.json();

            if (response.ok) {
                this.displayDataTable(result.data, result.columns);
            }
        } catch (error) {
            console.error('Failed to load data preview:', error);
        }
    }

    displayDataTable(data, columns) {
        const table = document.getElementById('dataPreviewTable');
        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');

        // Clear existing content
        thead.innerHTML = '';
        tbody.innerHTML = '';

        // Create header
        const headerRow = document.createElement('tr');
        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Create body rows
        data.forEach(row => {
            const tr = document.createElement('tr');
            columns.forEach(col => {
                const td = document.createElement('td');
                const value = row[col];
                td.textContent = value !== null && value !== undefined ? value : 'null';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    displayValidationIssues(issues) {
        const container = document.getElementById('validationIssues');
        container.innerHTML = '<h3>Data Quality Issues</h3>';
        
        issues.forEach(issue => {
            const issueDiv = document.createElement('div');
            issueDiv.className = 'validation-issue';
            issueDiv.innerHTML = `
                <div class="issue-header">
                    <i class="fas fa-exclamation-triangle text-warning"></i>
                    <strong>${issue.name}</strong>
                </div>
                <p>${issue.message}</p>
            `;
            container.appendChild(issueDiv);
        });
    }

    populateTargetColumns(columns) {
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = '<option value="">Select target column...</option>';
        
        columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            targetSelect.appendChild(option);
        });
    }

    handleTaskChange() {
        const task = document.getElementById('taskSelect').value;
        const targetGroup = document.getElementById('targetColumnGroup');
        
        if (task === 'clustering') {
            targetGroup.style.display = 'none';
        } else {
            targetGroup.style.display = 'block';
            
            // Update target column options based on task
            if (task === 'timeseries' || task === 'regression') {
                this.populateNumericTargets();
            } else {
                this.populateAllTargets();
            }
        }
        
        this.showSection('taskConfigSection');
    }
    
    populateNumericTargets() {
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = '<option value="">Select numeric target column...</option>';
        
        // Assuming we have stored column info from data upload
        if (this.currentSessionData && this.currentSessionData.data_types) {
            Object.entries(this.currentSessionData.data_types).forEach(([col, dtype]) => {
                if (dtype.includes('int') || dtype.includes('float')) {
                    const option = document.createElement('option');
                    option.value = col;
                    option.textContent = col;
                    targetSelect.appendChild(option);
                }
            });
        }
    }
    
    populateAllTargets() {
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = '<option value="">Select target column...</option>';
        
        if (this.currentSessionData && this.currentSessionData.columns) {
            this.currentSessionData.columns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                targetSelect.appendChild(option);
            });
        }
    }

    toggleAdvancedOptions() {
        const toggle = document.getElementById('advancedToggle');
        const options = document.getElementById('advancedOptions');
        
        if (toggle.checked) {
            options.classList.remove('hidden');
        } else {
            options.classList.add('hidden');
        }
    }

    async generateEDA() {
        this.showLoading('Generating comprehensive EDA report...');
        
        try {
            const response = await fetch(`/api/data/${this.currentSessionId}/eda`, {
                method: 'POST'
            });

            const result = await response.json();

            if (response.ok) {
                this.showToast('EDA report generated successfully!', 'success');
                // Here you could display the EDA report in a modal or new tab
                console.log('EDA Report:', result.report);
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.showToast(`EDA generation failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async startProcessing() {
        const task = document.getElementById('taskSelect').value;
        const targetColumn = document.getElementById('targetSelect').value;
        const featureGeneration = document.getElementById('featureGeneration').checked;
        const featureSelection = document.getElementById('featureSelection').checked;

        if (!task) {
            this.showToast('Please select a task', 'warning');
            return;
        }

        if (task !== 'clustering' && !targetColumn) {
            this.showToast('Please select a target column', 'warning');
            return;
        }

        this.showSection('processingSection');
        this.animateProcessing();

        try {
            const config = {
                task: task,
                target_column: targetColumn || null,
                feature_generation_enabled: featureGeneration,
                feature_selection_enabled: featureSelection
            };

            const response = await fetch(`/api/data/${this.currentSessionId}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result);
                this.showSection('resultsSection');
                this.showToast('Processing completed successfully!', 'success');
            } else {
                throw new Error(result.detail);
            }
        } catch (error) {
            this.showToast(`Processing failed: ${error.message}`, 'error');
            this.showSection('taskConfigSection');
        }
    }

    animateProcessing() {
        let currentStep = 0;
        const totalSteps = this.processingSteps.length;
        
        const updateProgress = () => {
            if (currentStep < totalSteps) {
                const progress = ((currentStep + 1) / totalSteps) * 100;
                document.getElementById('progressFill').style.width = `${progress}%`;
                document.getElementById('processingStepText').textContent = this.processingSteps[currentStep];
                currentStep++;
                setTimeout(updateProgress, 1500);
            }
        };
        
        updateProgress();
    }

    displayResults(result) {
        // Update summary stats
        document.getElementById('processedShape').textContent = `${result.data_shape[0]} × ${result.data_shape[1]}`;
        document.getElementById('processingTime').textContent = `${result.processing_time.toFixed(2)}s`;

        // Display column roles
        const rolesContainer = document.getElementById('columnRoles');
        rolesContainer.innerHTML = '';
        
        Object.entries(result.column_roles).forEach(([role, columns]) => {
            if (columns.length > 0) {
                const roleDiv = document.createElement('div');
                roleDiv.className = 'role-item';
                roleDiv.innerHTML = `
                    <span class="role-name">${role.replace('_', ' ').toUpperCase()}</span>
                    <span class="role-count">${columns.length} columns</span>
                `;
                rolesContainer.appendChild(roleDiv);
            }
        });

        // Store artifacts for download
        this.artifacts = result.artifacts;
    }

    async downloadArtifact(type) {
        if (!this.artifacts || !this.artifacts[type]) {
            this.showToast('Artifact not available', 'error');
            return;
        }

        try {
            const response = await fetch(this.artifacts[type]);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = this.getDownloadFilename(type);
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showToast(`${type} downloaded successfully!`, 'success');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            this.showToast(`Download failed: ${error.message}`, 'error');
        }
    }

    getDownloadFilename(type) {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const extensions = {
            'data': 'csv',
            'pipeline': 'joblib',
            'lineage': 'json'
        };
        
        return `datainsight_${type}_${timestamp}.${extensions[type]}`;
    }

    restart() {
        this.currentSessionId = null;
        this.artifacts = null;
        
        // Reset form inputs
        document.getElementById('fileInput').value = '';
        document.getElementById('urlInput').value = '';
        document.getElementById('taskSelect').value = 'classification';
        document.getElementById('targetSelect').innerHTML = '<option value="">Select target column...</option>';
        document.getElementById('advancedToggle').checked = false;
        document.getElementById('featureGeneration').checked = false;
        document.getElementById('featureSelection').checked = false;
        document.getElementById('advancedOptions').classList.add('hidden');
        
        this.showSection('dataInputSection');
        this.updateStatus('Ready');
    }

    // Utility methods
    showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.add('hidden');
        });
        
        // Show target section
        document.getElementById(sectionId).classList.remove('hidden');
    }

    showLoading(message = 'Loading...') {
        document.getElementById('loadingText').textContent = message;
        document.getElementById('loadingOverlay').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }

    updateStatus(status, type = 'ready') {
        const statusText = document.querySelector('.status-text');
        const statusDot = document.querySelector('.status-dot');
        
        statusText.textContent = status;
        
        // Update status dot color based on type
        statusDot.className = 'status-dot';
        if (type === 'processing') {
            statusDot.style.background = '#ffa726';
        } else if (type === 'error') {
            statusDot.style.background = '#ff4757';
        } else if (type === 'success') {
            statusDot.style.background = '#00ff88';
        } else {
            statusDot.style.background = '#00ff88';
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        document.getElementById('toastContainer').appendChild(toast);
        
        // Auto-remove toast after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }

    getToastIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-circle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DataInsightApp();
});