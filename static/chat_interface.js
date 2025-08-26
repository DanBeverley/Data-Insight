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
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupEmbeddedChat());
        } else {
            this.setupEmbeddedChat();
        }
    }
    
    setupEmbeddedChat() {
        // Find the embedded chat elements
        this.messagesContainer = document.getElementById('heroChatMessages');
        this.inputField = document.getElementById('heroChatInput');
        this.sendButton = document.getElementById('heroSendMessage');
        
        if (!this.messagesContainer || !this.inputField || !this.sendButton) {
            console.warn('Chat elements not found, waiting...');
            setTimeout(() => this.setupEmbeddedChat(), 500);
            return;
        }
        
        this.setupEventListeners();
        this.initializeChat();
    }
    
    setupEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Send message on Enter (but allow Shift+Enter for new lines)
        this.inputField.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.inputField.addEventListener('input', () => {
            this.autoResizeTextarea();
        });
        
        // Back to chat button (when in manual analysis mode)
        const backToChatBtn = document.getElementById('backToChatBtn');
        if (backToChatBtn) {
            backToChatBtn.addEventListener('click', () => {
                this.showChatLanding();
            });
        }
    }
    
    autoResizeTextarea() {
        this.inputField.style.height = 'auto';
        this.inputField.style.height = Math.min(this.inputField.scrollHeight, 120) + 'px';
    }
    
    initializeChat() {
        // Add welcome message
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
        
        // Add user message
        this.addMessage('user', message);
        this.inputField.value = '';
        this.autoResizeTextarea();
        
        // Show typing indicator
        this.isProcessing = true;
        this.showTypingIndicator();
        
        try {
            // Send to backend
            const response = await this.callChatAPI(message);
            this.hideTypingIndicator();
            
            // Add assistant response with streaming effect
            if (response.response) {
                this.addMessage('assistant', response.response);
                
                // Handle any special actions
                if (response.action === 'execute_strategy' && response.strategy) {
                    this.handleStrategyExecution(response.strategy);
                }
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
            : '/api/chat/start';
            
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                context_type: 'conversational'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Update session ID if this was the first message
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
        
        // Simulate typing effect for assistant messages
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
                // Ensure proper line breaks
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
        // Hide all manual sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.add('hidden');
        });
        
        // Show hero chat section
        const heroSection = document.getElementById('heroChatSection');
        if (heroSection) {
            heroSection.classList.remove('hidden');
        }
    }
    
    handleStrategyExecution(strategy) {
        // Add strategy execution button
        const executeBtn = document.createElement('button');
        executeBtn.className = 'btn btn-primary strategy-execute-btn';
        executeBtn.innerHTML = '<i class="fas fa-play"></i> Execute Strategy';
        executeBtn.onclick = () => this.executeStrategy(strategy);
        
        const lastMessage = this.messagesContainer.lastElementChild;
        if (lastMessage) {
            lastMessage.querySelector('.message-bubble').appendChild(executeBtn);
        }
    }
    
    async executeStrategy(strategy) {
        this.addMessage('assistant', 'Executing your strategy... This may take a few moments.');
        
        try {
            const response = await fetch('/api/execute-strategy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSessionId,
                    strategy: strategy
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addMessage('assistant', `Strategy executed successfully! ${result.message || ''}`);
                
                // Optionally switch to results view
                if (result.redirect_to_results) {
                    setTimeout(() => {
                        window.dataInsightApp?.showSection('resultsSection');
                    }, 2000);
                }
            } else {
                this.addMessage('assistant', `Strategy execution failed: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Strategy execution error:', error);
            this.addMessage('assistant', 'Failed to execute strategy. Please try again or use manual analysis.');
        }
    }
}

// Initialize chat interface when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});