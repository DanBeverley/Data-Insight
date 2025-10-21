class SessionManager {
    constructor() {
        this.app = null;
    }

    setApp(app) {
        this.app = app;
    }

    setupEventListeners() {
        const newChatBtn = document.getElementById('newChatBtn');
        const sidebarToggle = document.getElementById('sidebarToggle');

        if (newChatBtn) {
            newChatBtn.addEventListener('click', () => this.createNewSession());
        }

        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        }
    }

    async loadSessions() {
        try {
            const sessions = await this.app.apiClient.loadSessions();
            console.log('[SessionManager] Loaded sessions from database:', sessions.length);

            if (sessions.length === 0) {
                console.log('[SessionManager] No sessions found, creating new session');
                await this.createNewSession();
                return;
            }

            this.displaySessions(sessions);

            const lastActiveSessionId = localStorage.getItem('lastActiveSessionId');
            const sessionExists = sessions.some(s => s.id === lastActiveSessionId);

            if (lastActiveSessionId && sessionExists) {
                console.log('[SessionManager] Restoring last active session:', lastActiveSessionId);
                this.app.currentSessionId = lastActiveSessionId;
                this.app.agentSessionId = lastActiveSessionId;
                await this.loadSessionMessages(lastActiveSessionId);
            } else if (!this.app.currentSessionId) {
                const mostRecentSession = sessions[0];
                console.log('[SessionManager] No active session, using most recent:', mostRecentSession.id);
                this.app.currentSessionId = mostRecentSession.id;
                this.app.agentSessionId = mostRecentSession.id;
                localStorage.setItem('lastActiveSessionId', mostRecentSession.id);
                await this.loadSessionMessages(mostRecentSession.id);
            } else {
                console.log('[SessionManager] Using existing currentSessionId:', this.app.currentSessionId);
                if (this.app.currentSessionId && !this.app.agentSessionId) {
                    this.app.agentSessionId = this.app.currentSessionId;
                }
            }

            if (window.artifactStorage && this.app.currentSessionId) {
                window.artifactStorage.setSessionId(this.app.currentSessionId);
            }
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }

    displaySessions(sessions) {
        const sessionList = document.getElementById('sessionList');
        if (!sessionList) return;

        sessionList.innerHTML = '';

        sessions.forEach(session => {
            const sessionItem = document.createElement('div');
            sessionItem.className = `session-item ${session.id === this.app.currentSessionId ? 'active' : ''}`;
            sessionItem.dataset.sessionId = session.id;
            sessionItem.innerHTML = `
                <span class="session-title">${session.title || 'New Chat'}</span>
                <div class="session-actions">
                    <button class="btn-rename" title="Rename">
                        <i class="fa-solid fa-pencil"></i>
                    </button>
                    <button class="btn-delete" title="Delete">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
            `;
            sessionList.appendChild(sessionItem);
        });

        // Set up event listeners for each session item
        sessionList.querySelectorAll('.session-item').forEach(item => {
            const sessionId = item.dataset.sessionId;

            item.addEventListener('click', (e) => {
                if (!e.target.closest('.session-actions')) {
                    this.switchToSession(sessionId);
                }
            });

            item.querySelector('.btn-rename').addEventListener('click', (e) => {
                e.stopPropagation();
                this.renameSession(item);
            });

            item.querySelector('.btn-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteSession(sessionId);
            });
        });
    }

    renameSession(sessionItem) {
        const sessionId = sessionItem.dataset.sessionId;
        const titleSpan = sessionItem.querySelector('.session-title');
        const currentTitle = titleSpan.textContent;

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'session-title-input';
        input.value = currentTitle;

        titleSpan.replaceWith(input);
        input.focus();
        input.select();

        const saveRename = async () => {
            const newTitle = input.value.trim();
            if (newTitle && newTitle !== currentTitle) {
                try {
                    await this.app.apiClient.renameSession(sessionId, newTitle);
                } catch (error) {
                    console.error('Failed to rename session:', error);
                }
            }
            this.loadSessions();
        };

        input.addEventListener('blur', saveRename);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') input.blur();
            if (e.key === 'Escape') this.loadSessions();
        });
    }

    async createNewSession() {
        try {
            const data = await this.app.apiClient.createNewSession();
            this.app.currentSessionId = data.session_id;
            this.app.agentSessionId = data.session_id;
            localStorage.setItem('lastActiveSessionId', data.session_id);
            this.clearChatMessages();

            localStorage.removeItem('hasFirstMessage');
            if (this.app.blackHole) {
                this.app.blackHole.resetBlackHole();
            }

            if (window.artifactStorage) {
                window.artifactStorage.setSessionId(data.session_id);
            }

            this.loadSessions();
        } catch (error) {
            console.error('Failed to create new session:', error);
        }
    }

    async deleteSession(sessionId) {
        try {
            await this.app.apiClient.deleteSession(sessionId);

            if (sessionId === this.app.currentSessionId) {
                await this.createNewSession();
            } else {
                this.loadSessions();
            }
        } catch (error) {
            console.error('Failed to delete session:', error);
        }
    }

    async switchToSession(sessionId) {
        this.app.currentSessionId = sessionId;
        this.app.agentSessionId = sessionId;
        localStorage.setItem('lastActiveSessionId', sessionId);

        if (window.artifactStorage) {
            window.artifactStorage.setSessionId(sessionId);
        }

        await this.loadSessionMessages(sessionId);
        this.loadSessions();
    }

    async loadSessionMessages(sessionId) {
        try {
            console.log('[SessionManager] Loading messages for session:', sessionId);
            const messages = await this.app.apiClient.loadSessionMessages(sessionId);
            console.log('[SessionManager] Loaded messages:', messages.length);
            this.displaySessionMessages(messages);
        } catch (error) {
            console.error('Failed to load session messages:', error);
        }
    }

    displaySessionMessages(messages) {
        const chatMessages = document.getElementById('heroChatMessages');
        if (!chatMessages) return;

        if (!messages || messages.length === 0) {
            this.clearChatMessages();
            return;
        }

        // Clear existing messages
        chatMessages.innerHTML = '';

        messages.forEach(message => {
            const messageEl = document.createElement('div');
            messageEl.className = `chat-message ${message.type === 'human' ? 'user' : 'bot'}`;
            messageEl.innerHTML = `
                <div class="message-content">
                    <p>${message.content}</p>
                </div>
            `;
            chatMessages.appendChild(messageEl);

            // Apply fade-in animation if available
            if (this.app.animationManager) {
                this.app.animationManager.animateMessageFadeIn(messageEl);
            }
        });

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    clearChatMessages() {
        const chatMessages = document.getElementById('heroChatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = `
                <div class="chat-message bot">
                    <div class="message-content">
                        <p>Welcome to DataInsight AI. Please upload a dataset to begin.</p>
                    </div>
                </div>
            `;
        }
    }

    toggleSidebar() {
        document.body.classList.toggle('sidebar-open');
    }

    // Utility methods for session state management
    getCurrentSessionId() {
        return this.app.currentSessionId;
    }

    getAgentSessionId() {
        return this.app.agentSessionId;
    }

    setCurrentSessionId(sessionId) {
        this.app.currentSessionId = sessionId;
        this.app.agentSessionId = sessionId; // Keep in sync
    }

    // Session validation and recovery
    async validateCurrentSession() {
        if (!this.app.currentSessionId) {
            await this.createNewSession();
            return false;
        }

        try {
            const sessions = await this.app.apiClient.loadSessions();
            const currentSessionExists = sessions.some(session => session.id === this.app.currentSessionId);

            if (!currentSessionExists) {
                console.warn('Current session no longer exists, creating new session');
                await this.createNewSession();
                return false;
            }

            return true;
        } catch (error) {
            console.error('Failed to validate session:', error);
            await this.createNewSession();
            return false;
        }
    }

    // Auto-save session title based on conversation
    async autoSaveSessionTitle(messageContent) {
        if (!this.app.currentSessionId || !messageContent) return;

        // Extract meaningful title from first user message
        const words = messageContent.trim().split(' ');
        if (words.length >= 3) {
            const potentialTitle = words.slice(0, 5).join(' ');
            if (potentialTitle.length > 10 && potentialTitle.length < 50) {
                try {
                    await this.app.apiClient.renameSession(this.app.currentSessionId, potentialTitle);
                    this.loadSessions(); // Refresh to show new title
                } catch (error) {
                    console.error('Failed to auto-save session title:', error);
                }
            }
        }
    }

    // Session cleanup for memory management
    cleanup() {
        this.app = null;
    }
}