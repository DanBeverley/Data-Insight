/**
 * DataInsight AI - Embedded Chat Interface
 * Integrates with the hero landing page chat container
 */

class ChatInterface {
    constructor() {
        this.currentSessionId = null;
        this.chatHistory = [];
        this.isProcessing = false;
        this.messagesContainer = null;
        this.inputField = null;
        this.sendButton = null;
        this.init();
    }
    
    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupEmbeddedChat());
        } else {
            this.setupEmbeddedChat();
        }
    }
    
    setupEmbeddedChat() {
        this.messagesContainer = document.getElementById('heroChatMessages');
        this.inputField = document.getElementById('heroChatInput');
        this.sendButton = document.getElementById('heroSendMessage');
        this.uploadBtn = document.getElementById('heroUploadBtn');
        this.fileInput = document.getElementById('heroFileInput');
        
        if (!this.messagesContainer || !this.inputField || !this.sendButton) {
            console.warn('Chat elements not found, waiting...');
            setTimeout(() => this.setupEmbeddedChat(), 500);
            return;
        }
        
        this.setupEventListeners();
        this.initializeChat();
    }
    
    setupEventListeners() {
        this.sendButton.addEventListener('click', () => {
            this.sendMessage();
        });
        
        if (this.uploadBtn && this.fileInput) {
            this.uploadBtn.addEventListener('click', () => this.fileInput.click());
            this.fileInput.addEventListener('change', (e) => {
                const file = e.target.files && e.target.files[0];
                if (file) this.uploadDataset(file);
            });
        }
        
        
        this.inputField.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        
        this.inputField.addEventListener('input', () => {
            this.autoResizeTextarea();
        });
        
        const backToChatBtn = document.getElementById('backToChatBtn');
        if (backToChatBtn) {
            backToChatBtn.addEventListener('click', () => {
                this.showChatLanding();
            });
        }
    }

    async uploadDataset(file) {
        try {
            this.addMessage('assistant', `Uploading dataset: ${file.name} ...`);
            const form = new FormData();
            form.append('file', file);
            const resp = await fetch('/api/upload', { method: 'POST', body: form });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (data && data.session_id) {
                this.currentSessionId = data.session_id;
            }
            this.addMessage('assistant', 'Dataset uploaded successfully. You can now ask questions or start processing.');
        } catch (err) {
            console.error('Upload failed:', err);
            this.addMessage('assistant', 'Upload failed. Please try again with a CSV/XLSX file.');
        }
    }
    
    autoResizeTextarea() {
        this.inputField.style.height = 'auto';
        this.inputField.style.height = Math.min(this.inputField.scrollHeight, 120) + 'px';
    }
    
    initializeChat() {
        this.addMessage('assistant', `Hello! I'm your AI assistant. I can help you analyze your data through natural conversation.

Try saying something like:
• "I want to predict customer churn"
• "Help me analyze sales data"  
• "Create a classification model"

Or upload your data first and I'll guide you through the analysis.`);
    }
    
    async sendMessage() {
        const message = this.inputField.value.trim();
        if (!message || this.isProcessing) return;
        
        this.addMessage('user', message);
        this.inputField.value = '';
        this.autoResizeTextarea();
        
        this.isProcessing = true;
        this.showTypingIndicator();
        
        try {
            const response = await this.callChatAPI(message);
            this.hideTypingIndicator();
            
            // Handle response - debug and fix
            console.log('Full API response:', response);
            
            if (response && response.status === 'success') {
                const assistantText = response.response?.assistant_text;
                const action = response.response?.action;
                const suggestions = response.response?.suggestions;
                
                // Always show the assistant text
                this.addMessage('assistant', assistantText || 'I processed your request.');
                
                // Handle actions
                if (action === 'execute_pipeline') {
                    console.log('Auto-executing pipeline...');
                    setTimeout(() => {
                        this.executeStrategy();
                    }, 1000);
                } else if (action === 'ready_to_execute') {
                    this.addExecuteButton();
                } else if (action === 'upload_prompt') {
                    console.log('Prompting for upload...');
                }
                
                // Handle suggestions
                if (suggestions && suggestions.length > 0) {
                    this.addSuggestions(suggestions);
                }
            } else if (response && response.status === 'error') {
                this.addMessage('assistant', `Error: ${response.error || 'Unknown error occurred'}`);
            } else {
                console.error('Unexpected response format:', response);
                this.addMessage('assistant', 'I received an unexpected response format. Please check the console for details.');
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again or check if the server is running.');
        }
        
        this.isProcessing = false;
    }
    
    async callChatAPI(message) {
        const endpoint = this.currentSessionId 
            ? `/api/chat/${this.currentSessionId}/continue`
            : '/api/chat/start_project';
        
        const requestBody = {
            message: message
        };
        
        // Add session_id for start_project endpoint
        if (!this.currentSessionId) {
            requestBody.session_id = null; // Let backend auto-select or create
        }
            
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!this.currentSessionId && data.session_id) {
            this.currentSessionId = data.session_id;
        }
        
        return data;
    }
    
    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'user' 
            ? '<i class="fas fa-user"></i>'
            : '<i class="fas fa-brain"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(bubble);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        if (role === 'assistant') {
            this.typeMessage(bubble, content);
        } else {
            bubble.innerHTML = content.replace(/\n/g, '<br>');
        }
    }
    
    typeMessage(element, text) {
        let i = 0;
        element.innerHTML = '';
        
        const typeInterval = setInterval(() => {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                this.scrollToBottom();
                i++;
            } else {
                clearInterval(typeInterval);
                element.innerHTML = text.replace(/\n/g, '<br>');
            }
        }, 20);
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-brain"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble typing';
        bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
        
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(bubble);
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showChatLanding() {
        document.querySelectorAll('.section').forEach(section => {
            section.classList.add('hidden');
        });
        
        const heroSection = document.getElementById('heroChatSection');
        if (heroSection) {
            heroSection.classList.remove('hidden');
        }
    }
    
    handleStrategyExecution(strategy) {
        const executeBtn = document.createElement('button');
        executeBtn.className = 'btn btn-primary strategy-execute-btn';
        executeBtn.innerHTML = '<i class="fas fa-play"></i> Execute Strategy';
        executeBtn.onclick = () => this.executeStrategy(strategy);
        
        const lastMessage = this.messagesContainer.lastElementChild;
        if (lastMessage) {
            lastMessage.querySelector('.message-bubble').appendChild(executeBtn);
        }
    }
    
    addExecuteButton() {
        const executeBtn = document.createElement('button');
        executeBtn.className = 'btn btn-primary execute-strategy-btn';
        executeBtn.innerHTML = '<i class="fas fa-play"></i> Run Analysis';
        executeBtn.onclick = () => this.executeStrategy();
        
        const lastMessage = this.messagesContainer.lastElementChild;
        if (lastMessage && lastMessage.classList.contains('assistant')) {
            lastMessage.querySelector('.message-bubble').appendChild(executeBtn);
        }
    }
    
    addSuggestions(suggestions) {
        const suggestionsDiv = document.createElement('div');
        suggestionsDiv.className = 'chat-suggestions';
        
        suggestions.forEach(suggestion => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-outline suggestion-btn';
            btn.textContent = suggestion;
            btn.onclick = () => {
                this.inputField.value = suggestion;
                this.sendMessage();
            };
            suggestionsDiv.appendChild(btn);
        });
        
        const lastMessage = this.messagesContainer.lastElementChild;
        if (lastMessage && lastMessage.classList.contains('assistant')) {
            lastMessage.querySelector('.message-bubble').appendChild(suggestionsDiv);
        }
    }

    async executeStrategy() {
        this.addMessage('assistant', 'Executing your analysis... This may take a few moments.');
        
        try {
            const response = await fetch('/api/execute-strategy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSessionId
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.addMessage('assistant', result.message || 'Analysis completed successfully!');
                if (result.redirect_to_results) {
                    setTimeout(() => {
                        window.dataInsightApp?.showSection('resultsSection');
                    }, 2000);
                }
            } else {
                this.addMessage('assistant', `Execution failed: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Strategy execution error:', error);
            this.addMessage('assistant', 'Failed to execute analysis. Please try again or use manual analysis.');
        }
    }
}

// Initialize chat interface when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});