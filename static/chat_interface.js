/**
 * DataInsight AI - Conversational Chat Interface
 * Integrates natural language interaction with the existing frontend
 */

class ChatInterface {
    constructor() {
        this.currentSessionId = null;
        this.chatHistory = [];
        this.isProcessing = false;
        this.init();
    }
    
    init() {
        this.createChatUI();
        this.setupEventListeners();
        this.initializeChat();
    }
    
    createChatUI() {
        // Create chat toggle button
        const chatToggle = document.createElement('div');
        chatToggle.id = 'chatToggle';
        chatToggle.className = 'chat-toggle';
        chatToggle.innerHTML = `
            <i class="fas fa-comments"></i>
            <span class="chat-badge" id="chatBadge" style="display: none;">!</span>
        `;
        
        // Create chat panel
        const chatPanel = document.createElement('div');
        chatPanel.id = 'chatPanel';
        chatPanel.className = 'chat-panel';
        chatPanel.innerHTML = `
            <div class="chat-header">
                <h3><i class="fas fa-brain"></i> AI Assistant</h3>
                <button class="chat-close" id="chatClose">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <div class="message-content">
                        <p>Hello! I'm your AI assistant. I can help you analyze your data through natural conversation.</p>
                        <p>Try saying something like:</p>
                        <ul>
                            <li>"I want to predict customer churn"</li>
                            <li>"Help me analyze sales data"</li>
                            <li>"Create a classification model"</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <div class="quick-actions" id="quickActions">
                    <button class="quick-action" data-action="analyze">
                        <i class="fas fa-chart-bar"></i>
                        Analyze Data
                    </button>
                    <button class="quick-action" data-action="predict">
                        <i class="fas fa-crystal-ball"></i>
                        Predict
                    </button>
                    <button class="quick-action" data-action="explain">
                        <i class="fas fa-question-circle"></i>
                        Explain Results
                    </button>
                </div>
                
                <div class="chat-input-wrapper">
                    <textarea id="chatInput" placeholder="Ask me about your data analysis needs..." rows="1"></textarea>
                    <button id="sendMessage" class="send-btn">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        `;
        
        // Add styles
        const styles = document.createElement('style');
        styles.textContent = `
            .chat-toggle {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                background: var(--gradient-primary);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: var(--shadow-xl);
                color: white;
                font-size: 1.5rem;
                z-index: 1000;
                transition: var(--transition);
                position: relative;
            }
            
            .chat-toggle:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            }
            
            .chat-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                background: #ff4757;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.8rem;
                font-weight: bold;
                animation: pulse 2s infinite;
            }
            
            .chat-panel {
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 400px;
                height: 600px;
                background: var(--gradient-surface);
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-2xl);
                box-shadow: var(--shadow-xl);
                z-index: 999;
                display: none;
                flex-direction: column;
                backdrop-filter: blur(20px);
                overflow: hidden;
            }
            
            .chat-panel.open {
                display: flex;
                animation: slideUp 0.3s ease;
            }
            
            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .chat-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: var(--spacing-lg);
                border-bottom: 1px solid var(--color-border);
                background: rgba(99, 102, 241, 0.1);
            }
            
            .chat-header h3 {
                margin: 0;
                color: var(--color-text);
                font-size: 1.1rem;
                display: flex;
                align-items: center;
                gap: var(--spacing-sm);
            }
            
            .chat-close {
                background: none;
                border: none;
                color: var(--color-text-secondary);
                cursor: pointer;
                padding: var(--spacing-sm);
                border-radius: var(--border-radius);
                transition: var(--transition);
            }
            
            .chat-close:hover {
                background: var(--color-surface-light);
                color: var(--color-text);
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: var(--spacing-lg);
                display: flex;
                flex-direction: column;
                gap: var(--spacing-md);
            }
            
            .welcome-message {
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.2);
                border-radius: var(--border-radius-lg);
                padding: var(--spacing-lg);
            }
            
            .welcome-message .message-content {
                color: var(--color-text);
            }
            
            .welcome-message ul {
                margin: var(--spacing-sm) 0 0 var(--spacing-lg);
                color: var(--color-text-secondary);
            }
            
            .message {
                display: flex;
                gap: var(--spacing-md);
                margin-bottom: var(--spacing-md);
            }
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message.assistant {
                justify-content: flex-start;
            }
            
            .message-bubble {
                max-width: 80%;
                padding: var(--spacing-md) var(--spacing-lg);
                border-radius: var(--border-radius-lg);
                position: relative;
            }
            
            .message.user .message-bubble {
                background: var(--gradient-primary);
                color: white;
                border-bottom-right-radius: var(--spacing-sm);
            }
            
            .message.assistant .message-bubble {
                background: var(--color-surface-elevated);
                color: var(--color-text);
                border-bottom-left-radius: var(--spacing-sm);
                border: 1px solid var(--color-border);
            }
            
            .message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.9rem;
                flex-shrink: 0;
            }
            
            .message.user .message-avatar {
                background: var(--color-surface-light);
                color: var(--color-text);
            }
            
            .message.assistant .message-avatar {
                background: var(--gradient-primary);
                color: white;
            }
            
            .quick-actions {
                display: flex;
                gap: var(--spacing-sm);
                padding: var(--spacing-md) var(--spacing-lg);
                border-bottom: 1px solid var(--color-border);
                flex-wrap: wrap;
            }
            
            .quick-action {
                background: var(--color-surface-elevated);
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-lg);
                padding: var(--spacing-sm) var(--spacing-md);
                color: var(--color-text);
                cursor: pointer;
                transition: var(--transition);
                display: flex;
                align-items: center;
                gap: var(--spacing-xs);
                font-size: 0.9rem;
            }
            
            .quick-action:hover {
                background: var(--color-primary);
                color: white;
                transform: translateY(-1px);
            }
            
            .chat-input-container {
                border-top: 1px solid var(--color-border);
            }
            
            .chat-input-wrapper {
                display: flex;
                padding: var(--spacing-lg);
                gap: var(--spacing-md);
                align-items: flex-end;
            }
            
            #chatInput {
                flex: 1;
                background: var(--color-surface);
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-lg);
                padding: var(--spacing-md);
                color: var(--color-text);
                resize: none;
                font-family: inherit;
                max-height: 120px;
                min-height: 40px;
            }
            
            #chatInput:focus {
                outline: none;
                border-color: var(--color-primary);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            }
            
            .send-btn {
                background: var(--gradient-primary);
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: var(--transition);
            }
            
            .send-btn:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }
            
            .send-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading-indicator {
                display: flex;
                align-items: center;
                gap: var(--spacing-sm);
                color: var(--color-text-secondary);
                font-style: italic;
            }
            
            .loading-dots {
                display: flex;
                gap: 2px;
            }
            
            .loading-dots span {
                width: 4px;
                height: 4px;
                border-radius: 50%;
                background: var(--color-primary);
                animation: loadingDots 1.4s ease-in-out infinite;
            }
            
            .loading-dots span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .loading-dots span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes loadingDots {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            .follow-up-questions {
                margin-top: var(--spacing-md);
                display: flex;
                flex-direction: column;
                gap: var(--spacing-sm);
            }
            
            .follow-up-question {
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.2);
                border-radius: var(--border-radius);
                padding: var(--spacing-sm) var(--spacing-md);
                color: var(--color-primary);
                cursor: pointer;
                transition: var(--transition);
                font-size: 0.9rem;
            }
            
            .follow-up-question:hover {
                background: rgba(99, 102, 241, 0.15);
                border-color: rgba(99, 102, 241, 0.3);
            }
            
            .strategy-preview {
                background: var(--color-surface-elevated);
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-lg);
                padding: var(--spacing-md);
                margin-top: var(--spacing-md);
            }
            
            .strategy-preview h4 {
                margin: 0 0 var(--spacing-sm) 0;
                color: var(--color-text);
                font-size: 0.95rem;
            }
            
            .strategy-preview .strategy-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: var(--spacing-xs) 0;
                font-size: 0.85rem;
                color: var(--color-text-secondary);
            }
            
            .execute-strategy-btn {
                background: var(--gradient-primary);
                border: none;
                border-radius: var(--border-radius);
                padding: var(--spacing-sm) var(--spacing-md);
                color: white;
                cursor: pointer;
                margin-top: var(--spacing-md);
                width: 100%;
                font-weight: 600;
                transition: var(--transition);
            }
            
            .execute-strategy-btn:hover {
                transform: translateY(-1px);
                box-shadow: var(--shadow-lg);
            }
            
            @media (max-width: 768px) {
                .chat-panel {
                    width: 100vw;
                    height: 100vh;
                    bottom: 0;
                    right: 0;
                    border-radius: 0;
                }
                
                .chat-toggle {
                    bottom: 80px;
                }
            }
        `;
        
        document.head.appendChild(styles);
        document.body.appendChild(chatToggle);
        document.body.appendChild(chatPanel);
    }
    
    setupEventListeners() {
        // Chat toggle
        document.getElementById('chatToggle').addEventListener('click', () => {
            this.toggleChat();
        });
        
        // Chat close
        document.getElementById('chatClose').addEventListener('click', () => {
            this.toggleChat();
        });
        
        // Send message
        document.getElementById('sendMessage').addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Enter key in chat input
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        document.getElementById('chatInput').addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
        });
        
        // Quick actions
        document.getElementById('quickActions').addEventListener('click', (e) => {
            const action = e.target.closest('.quick-action');
            if (action) {
                this.handleQuickAction(action.dataset.action);
            }
        });
    }
    
    initializeChat() {
        // Check if there's an active session
        if (window.currentSessionId) {
            this.currentSessionId = window.currentSessionId;
            this.loadChatHistory();
        }
    }
    
    toggleChat() {
        const chatPanel = document.getElementById('chatPanel');
        const chatBadge = document.getElementById('chatBadge');
        
        chatPanel.classList.toggle('open');
        
        if (chatPanel.classList.contains('open')) {
            chatBadge.style.display = 'none';
            document.getElementById('chatInput').focus();
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message || this.isProcessing) return;
        
        this.isProcessing = true;
        input.value = '';
        input.style.height = 'auto';
        
        // Add user message to UI
        this.addMessage('user', message);
        
        // Show loading indicator
        const loadingId = this.addLoadingMessage();
        
        try {
            let response;
            
            if (!this.currentSessionId || this.chatHistory.length === 0) {
                // Start new conversation
                response = await this.startProjectConversation(message);
                if (response.status === 'success') {
                    this.currentSessionId = response.session_id;
                    window.currentSessionId = this.currentSessionId;
                }
            } else {
                // Continue existing conversation
                response = await this.continueConversation(message);
            }
            
            // Remove loading indicator
            this.removeLoadingMessage(loadingId);
            
            if (response.status === 'success') {
                this.handleSuccessResponse(response);
            } else {
                this.addMessage('assistant', `I encountered an error: ${response.error}. Please try again.`);
            }
            
        } catch (error) {
            this.removeLoadingMessage(loadingId);
            this.addMessage('assistant', 'I\'m having trouble processing your request. Please check your connection and try again.');
            console.error('Chat error:', error);
        }
        
        this.isProcessing = false;
    }
    
    async startProjectConversation(message) {
        const response = await fetch('/api/chat/start_project', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: this.currentSessionId
            })
        });
        
        return await response.json();
    }
    
    async continueConversation(message) {
        const response = await fetch(`/api/chat/${this.currentSessionId}/continue`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        return await response.json();
    }
    
    handleSuccessResponse(response) {
        const chatResponse = response.response;
        
        // Create response message
        let responseText = this.generateResponseText(chatResponse);
        
        this.addMessage('assistant', responseText);
        
        // Add follow-up questions if available
        if (chatResponse.follow_up_questions && chatResponse.follow_up_questions.length > 0) {
            this.addFollowUpQuestions(chatResponse.follow_up_questions);
        }
        
        // Add strategy preview if strategy is complete
        if (chatResponse.strategy && chatResponse.strategy.target_column) {
            this.addStrategyPreview(chatResponse.strategy);
        }
        
        // Show notification badge if chat is closed
        if (!document.getElementById('chatPanel').classList.contains('open')) {
            document.getElementById('chatBadge').style.display = 'flex';
        }
    }
    
    generateResponseText(chatResponse) {
        const intent = chatResponse.intent;
        const strategy = chatResponse.strategy;
        
        let responseText = '';
        
        if (intent.confidence && intent.confidence > 0.7) {
            responseText += `I understand you want to ${intent.task_type === 'classification' ? 'classify' : intent.task_type === 'regression' ? 'predict' : 'analyze'} your data. `;
        } else {
            responseText += "I'm working to understand your analysis needs. ";
        }
        
        if (strategy.target_column) {
            responseText += `I've identified "${strategy.target_column}" as your target variable for ${strategy.task_type}. `;
        } else if (intent.clarification_needed) {
            responseText += "I need a bit more information to set up your analysis properly. ";
        }
        
        if (chatResponse.suggested_actions && chatResponse.suggested_actions.length > 0) {
            responseText += "\n\nNext steps:\n";
            chatResponse.suggested_actions.forEach((action, i) => {
                responseText += `${i + 1}. ${action}\n`;
            });
        }
        
        return responseText;
    }
    
    addMessage(role, content) {
        const messagesContainer = document.getElementById('chatMessages');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-brain"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = content.replace(/\n/g, '<br>');
        
        if (role === 'user') {
            messageDiv.appendChild(bubble);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(bubble);
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Remove welcome message if this is the first real message
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage && messagesContainer.children.length > 1) {
            welcomeMessage.remove();
        }
    }
    
    addLoadingMessage() {
        const messagesContainer = document.getElementById('chatMessages');
        
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant loading-message';
        loadingDiv.id = `loading-${Date.now()}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-brain"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = `
            <div class="loading-indicator">
                Thinking
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        loadingDiv.appendChild(avatar);
        loadingDiv.appendChild(bubble);
        messagesContainer.appendChild(loadingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return loadingDiv.id;
    }
    
    removeLoadingMessage(loadingId) {
        const loadingMessage = document.getElementById(loadingId);
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    addFollowUpQuestions(questions) {
        const messagesContainer = document.getElementById('chatMessages');
        
        const questionsDiv = document.createElement('div');
        questionsDiv.className = 'follow-up-questions';
        
        questions.forEach(question => {
            const questionBtn = document.createElement('div');
            questionBtn.className = 'follow-up-question';
            questionBtn.textContent = question;
            questionBtn.addEventListener('click', () => {
                document.getElementById('chatInput').value = question;
                this.sendMessage();
            });
            questionsDiv.appendChild(questionBtn);
        });
        
        // Add to last assistant message
        const lastMessage = messagesContainer.querySelector('.message.assistant:last-of-type .message-bubble');
        if (lastMessage) {
            lastMessage.appendChild(questionsDiv);
        }
    }
    
    addStrategyPreview(strategy) {
        const messagesContainer = document.getElementById('chatMessages');
        
        const previewDiv = document.createElement('div');
        previewDiv.className = 'strategy-preview';
        previewDiv.innerHTML = `
            <h4><i class="fas fa-cogs"></i> Strategy Preview</h4>
            <div class="strategy-item">
                <span>Task Type:</span>
                <span>${strategy.task_type}</span>
            </div>
            <div class="strategy-item">
                <span>Target Column:</span>
                <span>${strategy.target_column || 'Not specified'}</span>
            </div>
            <div class="strategy-item">
                <span>Feature Generation:</span>
                <span>${strategy.configuration?.enable_feature_generation ? 'Enabled' : 'Disabled'}</span>
            </div>
            <button class="execute-strategy-btn" onclick="chatInterface.executeStrategy()">
                <i class="fas fa-play"></i> Execute Strategy
            </button>
        `;
        
        // Add to last assistant message
        const lastMessage = messagesContainer.querySelector('.message.assistant:last-of-type .message-bubble');
        if (lastMessage) {
            lastMessage.appendChild(previewDiv);
        }
    }
    
    async executeStrategy() {
        if (!this.currentSessionId) {
            this.addMessage('assistant', 'No active session found. Please start a new conversation.');
            return;
        }
        
        this.isProcessing = true;
        const loadingId = this.addLoadingMessage();
        
        try {
            const response = await fetch(`/api/chat/${this.currentSessionId}/execute`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            this.removeLoadingMessage(loadingId);
            
            if (result.status === 'success') {
                this.addMessage('assistant', `Great! I've successfully executed your analysis strategy. 
                
The pipeline completed with status: ${result.execution_result.status}

You can now view your results in the main interface or ask me to explain specific aspects of the analysis.`);
                
                // Switch to results view in main interface
                this.switchToResultsView();
                
            } else {
                this.addMessage('assistant', `I encountered an issue executing your strategy: ${result.error}`);
            }
            
        } catch (error) {
            this.removeLoadingMessage(loadingId);
            this.addMessage('assistant', 'I had trouble executing the strategy. Please try again.');
            console.error('Strategy execution error:', error);
        }
        
        this.isProcessing = false;
    }
    
    handleQuickAction(action) {
        const quickMessages = {
            'analyze': 'I want to analyze my data and understand the patterns',
            'predict': 'Help me create a predictive model',
            'explain': 'Can you explain the results from my analysis?'
        };
        
        const message = quickMessages[action];
        if (message) {
            document.getElementById('chatInput').value = message;
            this.sendMessage();
        }
    }
    
    switchToResultsView() {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.add('hidden');
        });
        
        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.classList.remove('hidden');
            
            // Update step navigation
            document.querySelectorAll('.step-item').forEach((item, index) => {
                if (index < 4) {
                    item.classList.add('completed');
                    item.classList.remove('active');
                }
                if (index === 4) {
                    item.classList.add('active');
                    item.classList.remove('completed');
                }
            });
        }
    }
    
    async loadChatHistory() {
        if (!this.currentSessionId) return;
        
        try {
            const response = await fetch(`/api/chat/${this.currentSessionId}/history`);
            const result = await response.json();
            
            if (result.status === 'success' && result.conversation_history) {
                // Load conversation history into UI
                const history = result.conversation_history.history || [];
                history.forEach(message => {
                    if (message.role === 'user' || message.role === 'assistant') {
                        this.addMessage(message.role, message.content);
                    }
                });
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
        }
    }
}

// Initialize chat interface when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
    
    // Integrate with existing DataInsight app if available
    if (window.dataInsightApp) {
        // Listen for data upload events
        document.addEventListener('dataUploaded', (event) => {
            if (event.detail && event.detail.sessionId) {
                window.chatInterface.currentSessionId = event.detail.sessionId;
            }
        });
    }
});