class ApiClient {
    constructor() {
        this.app = null;
        this.currentEventSource = null;
    }

    setApp(app) {
        this.app = app;
    }

    // Session Management APIs
    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            if (!response.ok) throw new Error('Failed to fetch sessions');
            return await response.json();
        } catch (error) {
            console.error('Error loading sessions:', error);
            throw error;
        }
    }

    async createNewSession() {
        try {
            const response = await fetch('/api/sessions/new', { method: 'POST' });
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error creating new session:', error);
            throw error;
        }
    }

    async renameSession(sessionId, newTitle) {
        try {
            await fetch(`/api/sessions/${sessionId}/rename`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle })
            });
        } catch (error) {
            console.error('Error renaming session:', error);
            throw error;
        }
    }

    async deleteSession(sessionId) {
        try {
            await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        } catch (error) {
            console.error('Error deleting session:', error);
            throw error;
        }
    }

    async loadSessionMessages(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}/messages`);
            return await response.json();
        } catch (error) {
            console.error('Error loading session messages:', error);
            throw error;
        }
    }

    // Data Processing APIs
    async uploadFile(file, sessionId) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('enable_profiling', 'true');
            formData.append('session_id', sessionId || '');

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            return await response.json();
        } catch (error) {
            console.error('Error uploading file:', error);
            throw error;
        }
    }

    async ingestFromUrl(url) {
        try {
            const response = await fetch('/api/ingest-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });

            return await response.json();
        } catch (error) {
            console.error('Error ingesting from URL:', error);
            throw error;
        }
    }

    async fetchDataPreview(sessionId) {
        try {
            const response = await fetch(`/api/data/${sessionId}/preview`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching data preview:', error);
            throw error;
        }
    }

    // Intelligence & Analysis APIs
    async performDeepProfiling(sessionId) {
        try {
            const response = await fetch(`/api/data/${sessionId}/profile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            });

            return await response.json();
        } catch (error) {
            console.error('Error performing deep profiling:', error);
            throw error;
        }
    }

    async getFeatureRecommendations(sessionId, options = {}) {
        try {
            const params = new URLSearchParams({
                max_recommendations: '10'
            });

            if (options.priorityFilter) {
                params.append('priority_filter', options.priorityFilter);
            }
            if (options.targetColumn) {
                params.append('target_column', options.targetColumn);
            }

            const response = await fetch(`/api/data/${sessionId}/feature-recommendations?${params}`);
            return await response.json();
        } catch (error) {
            console.error('Error getting feature recommendations:', error);
            throw error;
        }
    }

    async generateRelationshipGraph(sessionId) {
        try {
            const response = await fetch(`/api/data/${sessionId}/relationship-graph`);
            return await response.json();
        } catch (error) {
            console.error('Error generating relationship graph:', error);
            throw error;
        }
    }

    async generateEDA(sessionId) {
        try {
            const response = await fetch(`/api/data/${sessionId}/eda`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            });

            return await response.json();
        } catch (error) {
            console.error('Error generating EDA:', error);
            throw error;
        }
    }

    // Download APIs
    async downloadArtifact(sessionId, artifactType) {
        try {
            const endpoints = {
                'downloadData': '/api/data',
                'downloadModel': '/api/model',
                'downloadReport': '/api/report',
                'downloadCode': '/api/code',
                'downloadInsights': '/api/insights'
            };

            const endpoint = endpoints[`download${artifactType.charAt(0).toUpperCase() + artifactType.slice(1)}`];
            if (!endpoint) {
                throw new Error('Download not available for this artifact type');
            }

            const downloadUrl = `${endpoint}/${sessionId}/download/${artifactType}`;
            const response = await fetch(downloadUrl);

            if (!response.ok) {
                throw new Error(`Download failed: ${response.status} ${response.statusText}`);
            }

            const blob = await response.blob();
            const filename = this.getDownloadFilename(artifactType);

            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            return { success: true };
        } catch (error) {
            console.error('Error downloading artifact:', error);
            throw error;
        }
    }

    // Chat & Streaming APIs
    streamAgentResponse(message, sessionId, onStatus, onFinalResponse, onError) {
        // Close any existing connection to prevent overlaps
        if (this.currentEventSource) {
            this.currentEventSource.close();
        }

        this.currentEventSource = new EventSource(
            `/api/agent/chat-stream?message=${encodeURIComponent(message)}&session_id=${sessionId}`
        );

        this.currentEventSource.onopen = () => {
            console.log("SSE Connection opened");
            onStatus?.('Thinking...', 'processing');
        };

        this.currentEventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case 'status':
                    onStatus?.(data.message, 'active');
                    break;
                case 'final_response':
                    this.currentEventSource.close();
                    this.currentEventSource = null;
                    onFinalResponse?.(data);
                    break;
                case 'error':
                    this.currentEventSource.close();
                    this.currentEventSource = null;
                    onError?.(data.message);
                    break;
            }
        };

        this.currentEventSource.onerror = (err) => {
            console.error("EventSource failed:", err);
            this.currentEventSource.close();
            this.currentEventSource = null;
            onError?.('Connection error occurred');
        };
    }

    // Task Polling APIs
    async pollTaskStatus(taskId) {
        try {
            const response = await fetch(`/api/agent/task-status/${taskId}`);
            return await response.json();
        } catch (error) {
            console.error('Error polling task status:', error);
            throw error;
        }
    }

    // Privacy & PII APIs
    async handlePIIConsent(sessionId, applyProtection) {
        try {
            const response = await fetch('/api/privacy/consent', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    apply_protection: applyProtection
                })
            });

            return await response.json();
        } catch (error) {
            console.error('Error handling PII consent:', error);
            throw error;
        }
    }

    // Helper methods
    getDownloadFilename(downloadType) {
        const timestamp = new Date().toISOString().slice(0, 10);
        const filenames = {
            'data': `processed_data_${timestamp}.csv`,
            'model': `trained_model_${timestamp}.pkl`,
            'report': `analysis_report_${timestamp}.pdf`,
            'code': `generated_code_${timestamp}.py`,
            'insights': `insights_${timestamp}.json`
        };
        return filenames[downloadType] || `download_${timestamp}.txt`;
    }

    // Utility method to close streaming connection
    closeStreamingConnection() {
        if (this.currentEventSource) {
            this.currentEventSource.close();
            this.currentEventSource = null;
        }
    }

    // Generic API request helper
    async makeRequest(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${url}`, error);
            throw error;
        }
    }
}