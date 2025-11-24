marked.setOptions({
    gfm: true,
    breaks: true,
    headerIds: false,
    tables: true,
    langPrefix: 'language-'
});

class ChatInterface {
    constructor() {
        this.app = null;
        this.loadingMessageElement = null;
        this.selectedFile = null;
        this.attachmentBadge = null;
        this.statusMessageElement = null;
        this.messageVersions = new Map();
        this.emojiConverter = new EmojiConvertor();
        this.emojiConverter.replace_mode = 'unified';
    }

    saveVersionHistory(sessionId) {
        if (!sessionId) return;
        const versionsObj = {};
        this.messageVersions.forEach((versions, messageId) => {
            versionsObj[messageId] = versions;
        });
        try {
            localStorage.setItem(`message_versions_${sessionId}`, JSON.stringify(versionsObj));
        } catch (e) {
            console.warn('Failed to save version history:', e);
        }
    }

    loadVersionHistory(sessionId) {
        if (!sessionId) return;
        try {
            const saved = localStorage.getItem(`message_versions_${sessionId}`);
            if (saved) {
                const versionsObj = JSON.parse(saved);
                this.messageVersions = new Map(Object.entries(versionsObj));
            } else {
                this.messageVersions = new Map();
            }
        } catch (e) {
            console.warn('Failed to load version history:', e);
            this.messageVersions = new Map();
        }
    }

    clearVersionHistory() {
        this.messageVersions = new Map();
    }

    setApp(app) {
        this.app = app;
    }

    closeSettingsPanel() {
        const settingsPanel = document.getElementById('settingsPanel');
        const settingsToggleBtn = document.getElementById('settingsToggleBtn');
        if (settingsPanel && settingsToggleBtn && settingsPanel.classList.contains('open')) {
            settingsPanel.classList.remove('open');
            const arrow = settingsToggleBtn.querySelector('.settings-arrow');
            if (arrow) {
                arrow.classList.remove('rotated');
            }
        }
    }

    setupHeroChatInterface() {
        const heroSendBtn = document.getElementById('heroSendMessage');
        const heroChatInput = document.getElementById('heroChatInput');
        const heroUploadBtn = document.getElementById('heroUploadBtn');
        const heroFileInput = document.getElementById('heroFileInput');
        const heroChatMessages = document.getElementById('heroChatMessages');
        const settingsToggleBtn = document.getElementById('settingsToggleBtn');
        const settingsPanel = document.getElementById('settingsPanel');
        const webSearchToggle = document.getElementById('webSearchToggle');

        if (!heroSendBtn || !heroChatInput || !heroChatMessages) {
            console.warn('Hero chat elements not found');
            return;
        }

        if (settingsToggleBtn && settingsPanel && webSearchToggle) {
            const webSearchEnabled = localStorage.getItem('webSearchEnabled') === 'true';
            webSearchToggle.checked = webSearchEnabled;

            const tokenStreamingToggle = document.getElementById('tokenStreamingToggle');
            const tokenStreamingEnabled = localStorage.getItem('tokenStreamingEnabled') !== 'false';
            if (tokenStreamingToggle) {
                tokenStreamingToggle.checked = tokenStreamingEnabled;
            }

            settingsToggleBtn.addEventListener('click', () => {
                settingsPanel.classList.toggle('open');
                const arrow = settingsToggleBtn.querySelector('.settings-arrow');
                if (arrow) {
                    arrow.classList.toggle('rotated');
                }
            });

            webSearchToggle.addEventListener('change', (e) => {
                localStorage.setItem('webSearchEnabled', e.target.checked);
                console.log('Web search', e.target.checked ? 'enabled' : 'disabled');
            });

            if (tokenStreamingToggle) {
                tokenStreamingToggle.addEventListener('change', (e) => {
                    localStorage.setItem('tokenStreamingEnabled', e.target.checked);
                    console.log('Token streaming', e.target.checked ? 'enabled' : 'disabled');
                });
            }

            const exportConversationBtn = document.getElementById('exportConversationBtn');
            const exportFormatMenu = document.getElementById('exportFormatMenu');

            if (exportConversationBtn && exportFormatMenu) {
                exportConversationBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    exportFormatMenu.classList.toggle('show');
                });

                const exportFormatBtns = exportFormatMenu.querySelectorAll('.export-format-btn');
                exportFormatBtns.forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        const format = btn.dataset.format;
                        await this.exportConversation(format);
                        exportFormatMenu.classList.remove('show');
                        settingsPanel.classList.remove('open');
                    });
                });
            }
        }

        // Ensure we have a session ID for agent communication
        if (!this.app.agentSessionId) {
            this.app.agentSessionId = this.app.currentSessionId || '';
        }

        const handleButtonClick = () => {
            if (heroSendBtn.classList.contains('stop-mode')) {
                this.stopGeneration();
            } else {
                sendMessage();
            }
        };

        const sendMessage = () => {
            const message = heroChatInput.value.trim();
            if (!message && !this.selectedFile) return;

            const isFirstMessage = localStorage.getItem('hasFirstMessage') !== 'true';
            if (isFirstMessage && this.app.blackHole) {
                this.app.blackHole.triggerExpansion();
                localStorage.setItem('hasFirstMessage', 'true');
            }

            if (this.selectedFile) {
                this.sendMessageWithAttachment(message);
            } else {
                const userMessageId = this.addChatMessage('user', message);
                heroChatInput.value = '';
                this.streamAgentResponse(message, userMessageId);
            }
        };

        heroSendBtn.addEventListener('click', handleButtonClick);
        heroChatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        if (heroUploadBtn && heroFileInput) {
            heroUploadBtn.addEventListener('click', () => {
                heroFileInput.click();
            });

            heroFileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (!file) return;

                this.selectedFile = file;
                this.showAttachmentBadge(file);
                e.target.value = '';
            });
        }
        document.addEventListener('click', (e) => {
            if (!settingsPanel || !settingsPanel.classList.contains('open')) { return; }
            const isClickInside = settingsPanel.contains(e.target) || settingsToggleBtn.contains(e.target);
            if (!isClickInside) { this.closeSettingsPanel(); }
        });
    }

    streamAgentResponse(message, userMessageId = null) {
        this.statusMessageElement = null;
        const webSearchEnabled = localStorage.getItem('webSearchEnabled') === 'true';

        this.transformButtonToStop();

        let streamingMessageElement = null;
        let streamingContent = '';

        this.app.apiClient.streamAgentResponse(
            message,
            this.app.agentSessionId,
            webSearchEnabled,
            (statusText, statusType) => {
                if (!this.statusMessageElement) {
                    this.statusMessageElement = this.showStreamingStatusPlaceholder();
                }
                this.updateStreamingStatusWithType(this.statusMessageElement, statusText, statusType, 'Processing');
            },
            (tokenData) => {
                if (tokenData.type === 'start') {
                    streamingContent = '';
                    streamingMessageElement = this.addChatMessage('bot', '');

                    if (userMessageId && streamingMessageElement) {
                        streamingMessageElement.dataset.linkedUserMessageId = userMessageId;
                    }
                } else if (tokenData.content) {
                    streamingContent += tokenData.content;
                    if (streamingMessageElement) {
                        const contentDiv = streamingMessageElement.querySelector('.message-content');
                        if (contentDiv) {
                            contentDiv.textContent = streamingContent;
                        }
                    }
                } else if (tokenData.type === 'end') {
                    streamingMessageElement = null;
                    streamingContent = '';
                }
            },
            (data) => {
                setTimeout(() => {
                    if (this.statusMessageElement) {
                        const remainingLines = this.statusMessageElement.querySelectorAll('.status-line.active');
                        remainingLines.forEach(line => {
                            line.classList.remove('active');
                            line.classList.add('completed');
                            const spinner = line.querySelector('.status-spinner');
                            const statusText = line.querySelector('.status-text');
                            if (spinner) {
                                spinner.style.opacity = '0';
                            }
                            if (statusText) {
                                statusText.style.animation = 'none';
                            }
                        });
                    }

                    let botMessageId;
                    if (streamingMessageElement) {
                        const contentDiv = streamingMessageElement.querySelector('.message-content');
                        if (contentDiv) {
                            contentDiv.innerHTML = marked.parse(data.response || '');
                        }

                        if (data.plots && data.plots.length > 0) {
                            const plotsContainer = streamingMessageElement.querySelector('.message-plots');
                            if (plotsContainer) {
                                plotsContainer.innerHTML = '';
                                data.plots.forEach(plotPath => {
                                    const img = document.createElement('img');
                                    img.src = plotPath;
                                    img.alt = 'Generated Plot';
                                    img.className = 'plot-image';
                                    plotsContainer.appendChild(img);
                                });
                            }
                        }

                        botMessageId = streamingMessageElement.dataset.messageId;
                        streamingMessageElement = null;
                    } else {
                        botMessageId = this.addChatMessage('bot', data.response, data.plots);
                    }

                    const statusContainerId = this.statusMessageElement ? this.statusMessageElement.dataset.statusId : null;

                    if (botMessageId && statusContainerId) {
                        const botMsg = document.querySelector(`[data-message-id="${botMessageId}"]`);
                        if (botMsg) {
                            botMsg.dataset.linkedStatusId = statusContainerId;
                        }
                    }

                    if (userMessageId && botMessageId) {
                        const userMsg = document.querySelector(`[data-message-id="${userMessageId}"]`);
                        const botMsg = document.querySelector(`[data-message-id="${botMessageId}"]`);
                        if (userMsg) {
                            userMsg.dataset.linkedBotMessageId = botMessageId;
                        }
                        if (botMsg) {
                            botMsg.dataset.linkedUserMessageId = userMessageId;
                        }
                    }

                    if (this._waitingForVersionResponse && this._waitingForVersionResponse.tempUserMessageId === userMessageId) {
                        const versionData = this._waitingForVersionResponse;
                        const versions = this.messageVersions.get(versionData.messageId);

                        versions.push({
                            userText: versionData.newText,
                            botMessageId: botMessageId,
                            attachment: versionData.newAttachment,
                            statusContainerId: statusContainerId
                        });

                        this.messageVersions.set(versionData.messageId, versions);
                        versionData.messageDiv.dataset.currentVersion = String(versions.length - 1);
                        this.showVersion(versionData.messageDiv, versions.length - 1);

                        const tempUserDiv = document.querySelector(`[data-message-id="${versionData.tempUserMessageId}"]`);
                        if (tempUserDiv) tempUserDiv.remove();

                        this._waitingForVersionResponse = null;

                        if (this.app.currentSessionId) {
                            this.saveVersionHistory(this.app.currentSessionId);
                        }
                    }

                    this.statusMessageElement = null;
                    this.transformButtonToSend();

                    if (this.app.sessionManager) {
                        let agentResponseText = '';
                        if (typeof data.response === 'string') {
                            agentResponseText = data.response;
                        } else if (data.response && data.response.content) {
                            agentResponseText = data.response.content
                                .filter(item => item.type === 'text')
                                .map(item => item.text)
                                .join(' ');
                        }
                        this.app.sessionManager.autoSaveSessionTitle(message, agentResponseText);
                    }

                    if (this.app.currentSessionId) {
                        this.saveVersionHistory(this.app.currentSessionId);
                    }
                }, 150);
            },
            (errorMessage) => {
                setTimeout(() => {
                    this.updateStreamingStatusWithType(this.statusMessageElement, `Error: ${errorMessage}`, 'error', 'Error');
                    if (this.statusMessageElement) {
                        this.statusMessageElement.style.display = 'none';
                    }
                    this.addChatMessage('bot', `Error: ${errorMessage}`);
                    this.statusMessageElement = null;
                    this.transformButtonToSend();
                }, 150);
            }
        );
    }

    async pollProfilingStatus(sessionId) {
        const maxAttempts = 60;
        let attempts = 0;

        const poll = async () => {
            if (attempts >= maxAttempts) {
                console.log('Profiling status polling timeout');
                return;
            }

            attempts++;

            try {
                const response = await fetch(`/api/profiling-status/${sessionId}`);
                const data = await response.json();

                if (!this.statusMessageElement) {
                    this.statusMessageElement = this.showStreamingStatusPlaceholder();
                }

                if (this.statusMessageElement) {
                    const statusText = data.message || 'Processing...';
                    this.updateStreamingStatusWithType(this.statusMessageElement, statusText, 'profiling', 'Profiling');
                }

                if (data.status === 'completed' || data.status === 'failed') {
                    console.log('Profiling completed:', data.status);
                    return;
                }

                setTimeout(poll, 500);
            } catch (error) {
                console.error('Error polling profiling status:', error);
            }
        };

        await poll();
    }

    transformButtonToStop() {
        const sendBtn = document.getElementById('heroSendMessage');
        if (!sendBtn) return;

        sendBtn.classList.add('stop-mode');
        sendBtn.innerHTML = '<i class="fa-solid fa-square"></i>';
        sendBtn.title = 'Stop Generation';
    }

    transformButtonToSend() {
        const sendBtn = document.getElementById('heroSendMessage');
        if (!sendBtn) return;

        sendBtn.classList.remove('stop-mode');
        sendBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i>';
        sendBtn.title = 'Send Message';
    }

    async stopGeneration() {
        await this.app.apiClient.cancelTask(this.app.agentSessionId || this.app.currentSessionId);

        const stopped = this.app.apiClient.stopStreaming();
        if (stopped) {
            if (this.statusMessageElement) {
                const wrapper = this.statusMessageElement.querySelector('.status-lines-wrapper');
                if (wrapper) {
                    wrapper.innerHTML = '';
                }
                this.updateStreamingStatusWithType(this.statusMessageElement, 'Generation stopped by user', 'stopped', 'Stopped');
                setTimeout(() => {
                    if (this.statusMessageElement) {
                        this.statusMessageElement.style.display = 'none';
                        this.statusMessageElement = null;
                    }
                }, 2500);
            }
            this.transformButtonToSend();
        }
    }


    showStreamingStatusPlaceholder() {
        const heroChatMessages = document.getElementById('heroChatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot streaming-status-container';
        const statusId = `status-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        messageDiv.dataset.statusId = statusId;
        messageDiv.innerHTML = `
            <div class="status-lines-wrapper">
                <!-- Status lines will be added dynamically -->
            </div>
        `;

        heroChatMessages.appendChild(messageDiv);

        heroChatMessages.scrollTop = heroChatMessages.scrollHeight;
        return messageDiv;
    }

    setupStatusToggle(containerElement) {
        const toggleBtn = containerElement.querySelector('.status-toggle-btn');
        const wrapper = containerElement.querySelector('.status-lines-wrapper');

        if (!toggleBtn || !wrapper) return;

        toggleBtn.addEventListener('click', () => {
            const isCollapsed = wrapper.classList.contains('collapsed');

            if (isCollapsed) {
                wrapper.classList.remove('collapsed');
                toggleBtn.innerHTML = '<i class="fa-solid fa-chevron-down"></i>';
            } else {
                wrapper.classList.add('collapsed');
                toggleBtn.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
            }
        });

        // Auto-collapse when there are many status lines
        const observer = new MutationObserver(() => {
            if (wrapper.querySelectorAll('.status-line').length > 0 && isCollapsed) {
                wrapper.classList.add('collapsed');
                toggleBtn.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
            }
        });

        observer.observe(wrapper, { childList: true, subtree: true });
    }

    updateStreamingStatus(containerElement, newStatusText, statusType = 'active') {
        const wrapper = containerElement.querySelector('.status-lines-wrapper');
        if (!wrapper) return;

        const existingLines = wrapper.querySelectorAll('.status-line.active');
        existingLines.forEach(line => {
            line.classList.remove('active');
            line.classList.add('completed');
        });

        const cleanedText = newStatusText.replace(/^Processing:\s*/i, '').trim();

        const statusLine = document.createElement('div');
        statusLine.className = `status-line ${statusType === 'active' ? 'active' : statusType}`;
        statusLine.innerHTML = `
            <div class="status-dot"></div>
            <div class="status-text">${cleanedText}</div>
        `;

        wrapper.appendChild(statusLine);

        if (statusType === 'completed') {
            statusLine.classList.remove('active');
            statusLine.classList.add('completed');
        } else if (statusType === 'active') {
            statusLine.classList.add('active');
        }

        this.updateTaskProgress(wrapper, statusType);

        const allLines = wrapper.querySelectorAll('.status-line');
        if (allLines.length > 2) {
            allLines.forEach((line, index) => {
                if (index < allLines.length - 2) {
                    line.classList.add('elevated');
                } else {
                    line.classList.remove('elevated');
                }
            });
        }
    }

    updateTaskProgress(wrapper, statusType) {
        const allLines = wrapper.querySelectorAll('.status-line');
        const completedLines = wrapper.querySelectorAll('.status-line.completed');
        const activeLines = allLines.length - completedLines.length;

        const title = wrapper.closest('.streaming-status-container').querySelector('.status-title span');
        if (activeLines + completedLines.length > 1 && title) {
            let progressText = title.nextElementSibling;
            if (!progressText) {
                progressText = document.createElement('span');
                progressText.className = 'progress-text';
                title.parentNode.appendChild(progressText);
            }
            progressText.textContent = ` (${completedLines.length}/${allLines.length})`;
        }
    }

    updateStreamingStatusWithType(containerElement, newStatusText, statusType = 'active', taskName = '') {
        let prefix = '';
        let typeClass = statusType;

        switch (statusType) {
            case 'data_processing':
                prefix = '';
                typeClass = 'active';
                break;
            case 'analysis':
                prefix = '';
                typeClass = 'active';
                break;
            case 'model_training':
                prefix = '';
                typeClass = 'active';
                break;
            case 'processing':
                prefix = '';
                typeClass = 'active';
                break;
            case 'completed':
                prefix = '';
                typeClass = 'completed';
                break;
            case 'error':
                prefix = 'ERROR: ';
                typeClass = 'error';
                break;
        }

        const fullStatusText = taskName ? `${taskName}: ${newStatusText}` : newStatusText;
        this.updateStreamingStatus(containerElement, `${prefix}${fullStatusText}`, typeClass);
    }

    addChatMessage(sender, responseData, plots = null, attachmentFileName = null, messageId = null) {
        const heroChatMessages = document.getElementById('heroChatMessages');
        if (!heroChatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;

        if (!messageId) {
            messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        }
        messageDiv.dataset.messageId = messageId;

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content-wrapper';

        if (attachmentFileName && sender === 'user') {
            const attachmentDiv = document.createElement('div');
            attachmentDiv.className = 'message-attachment';
            attachmentDiv.innerHTML = `
                <i class="fa-solid fa-file"></i>
                <span>${attachmentFileName}</span>`;
            contentWrapper.appendChild(attachmentDiv);
        }
        const content = document.createElement('div');
        content.className = 'message-content';

        let messageText = '';
        if (typeof responseData === 'string') {
            messageText = responseData;
        } else if (responseData && typeof responseData === 'object' && responseData.content) {
            // Handle structured response data by joining text parts
            messageText = responseData.content
                .filter(item => item.type === 'text')
                .map(item => item.text)
                .join('\n\n');
        }

        if (sender === 'bot') {
            content.innerHTML = this.formatBotResponse(messageText);
        } else {
            content.innerHTML = this.convertMarkdownToHtml(messageText);
        }
        contentWrapper.appendChild(content);

        if (plots && Array.isArray(plots) && plots.length > 0) {
            plots.forEach(plotUrl => {
                const plotDiv = document.createElement('div');
                plotDiv.className = 'plot-container';
                if (plotUrl.endsWith('.html')) {
                    plotDiv.innerHTML = `
                        <iframe src="${plotUrl}" width="100%" height="400" frameborder="0">
                            Your browser does not support iframes.
                        </iframe>`;
                } else {
                    plotDiv.innerHTML = `
                        <img src="${plotUrl}" alt="Generated Plot" style="max-width: 100%; height: auto;" />`;
                }
                contentWrapper.appendChild(plotDiv);
            });
        }

        messageDiv.appendChild(contentWrapper);
        // Find all elements that should become collapsible headers (e.g., h4 tags)
        const headers = messageDiv.querySelectorAll('h4');
        headers.forEach(header => {
            // Check if it's already a header, if so, skip
            if (header.closest('.collapsible-header')) return;

            const section = document.createElement('div');
            section.className = 'collapsible-section open'; // Start open by default

            // Wrap the header
            const newHeader = document.createElement('div');
            newHeader.className = 'collapsible-header';
            newHeader.innerHTML = `<h4>${header.innerHTML}</h4><i class="fa-solid fa-chevron-down collapsible-toggle"></i>`;

            const content = document.createElement('div');
            content.className = 'collapsible-content';

            // Move all sibling elements until the next h4 into the content div
            let currentElement = header.nextElementSibling;
            while (currentElement && currentElement.tagName !== 'H4') {
                content.appendChild(currentElement);
                currentElement = header.nextElementSibling; // Re-evaluate next sibling
            }

            // Replace the original header with the new collapsible section
            header.parentNode.replaceChild(section, header);
            section.appendChild(newHeader);
            section.appendChild(content);

            // Add the click listener to the new header
            newHeader.addEventListener('click', () => {
                section.classList.toggle('open');
            });
        });

        if (sender === 'user') {
            const editControls = document.createElement('div');
            editControls.className = 'message-edit-controls';
            editControls.innerHTML = `
                <button class="edit-btn" title="Edit message">
                    <i class="fa-solid fa-pen"></i>
                </button>
            `;
            messageDiv.appendChild(editControls);

            const editBtn = editControls.querySelector('.edit-btn');
            editBtn.addEventListener('click', () => this.enterEditMode(messageDiv, attachmentFileName));
        }

        if (sender === 'bot') {
            const regenerateControls = document.createElement('div');
            regenerateControls.className = 'message-regenerate-controls';
            regenerateControls.innerHTML = `
                <button class="regenerate-btn" title="Regenerate response">
                    <i class="fa-solid fa-arrows-rotate"></i>
                </button>
            `;
            contentWrapper.appendChild(regenerateControls);

            const regenerateBtn = regenerateControls.querySelector('.regenerate-btn');
            if (regenerateBtn) {
                regenerateBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Regenerate button clicked for message:', messageDiv.dataset.messageId);
                    this.regenerateResponse(messageDiv);
                });
            } else {
                console.error('Failed to create regenerate button');
            }
        }

        heroChatMessages.appendChild(messageDiv);

        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([messageDiv]).catch(err => console.warn('MathJax error:', err));
        }
        if (sender === "bot") {
            const codeBlocks = messageDiv.querySelectorAll("pre code");
            codeBlocks.forEach((block) => {
                if (window.Prism) {
                    Prism.highlightElement(block);
                }
            });
        }
        heroChatMessages.scrollTop = heroChatMessages.scrollHeight;
        if (sender === "bot") {
            const codeBlocks = messageDiv.querySelectorAll("pre"); // Find <pre> elements
            codeBlocks.forEach((preElement) => {
                // Create the button
                const copyBtn = document.createElement("button");
                copyBtn.className = "copy-code-btn";
                copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i> Copy';
                preElement.appendChild(copyBtn);

                // Add click event listener
                copyBtn.addEventListener("click", () => {
                    const codeToCopy = preElement.querySelector("code").innerText;
                    navigator.clipboard.writeText(codeToCopy).then(() => {
                        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
                        setTimeout(() => {
                            copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i> Copy';
                        }, 2000);
                    }).catch(err => {
                        copyBtn.textContent = 'Failed to copy';
                        console.error('Failed to copy text: ', err);
                    });
                });
            });

            const codeHighlightElements = messageDiv.querySelectorAll("pre code");
            codeHighlightElements.forEach((block) => {
                if (window.Prism) {
                    Prism.highlightElement(block);
                }
            });
        }

        heroChatMessages.scrollTop = heroChatMessages.scrollHeight;
        return messageId;
    }

    enterEditMode(messageDiv, currentAttachment) {
        const contentWrapper = messageDiv.querySelector('.message-content-wrapper');
        const messageContent = messageDiv.querySelector('.message-content');
        const editControls = messageDiv.querySelector('.message-edit-controls');

        if (!contentWrapper || !messageContent) return;

        const currentText = messageContent.textContent.trim();
        messageDiv.dataset.originalText = currentText;

        messageDiv.classList.add('editing');
        if (editControls) {
            editControls.style.display = 'none';
        }

        const textarea = document.createElement('textarea');
        textarea.className = 'edit-textarea';
        textarea.value = currentText;
        textarea.rows = Math.min(Math.max(currentText.split('\n').length, 3), 10);

        messageContent.replaceWith(textarea);
        textarea.focus();

        const editActions = document.createElement('div');
        editActions.className = 'edit-actions';

        let editAttachmentName = currentAttachment;
        const attachmentControls = document.createElement('div');
        attachmentControls.className = 'edit-attachment-controls';

        if (currentAttachment) {
            attachmentControls.innerHTML = `
                <div class="current-attachment">
                    <i class="fa-solid fa-file"></i>
                    <span>${currentAttachment}</span>
                    <button class="remove-attachment-btn" title="Remove attachment">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
            `;
            const removeBtn = attachmentControls.querySelector('.remove-attachment-btn');
            removeBtn.addEventListener('click', () => {
                editAttachmentName = null;
                attachmentControls.innerHTML = '';
                const addAttachmentBtn = document.createElement('button');
                addAttachmentBtn.className = 'add-attachment-btn';
                addAttachmentBtn.innerHTML = '<i class="fa-solid fa-paperclip"></i> Add attachment';
                addAttachmentBtn.addEventListener('click', () => this.addAttachmentInEdit(attachmentControls, (fileName) => {
                    editAttachmentName = fileName;
                }));
                attachmentControls.appendChild(addAttachmentBtn);
            });
        } else {
            const addAttachmentBtn = document.createElement('button');
            addAttachmentBtn.className = 'add-attachment-btn';
            addAttachmentBtn.innerHTML = '<i class="fa-solid fa-paperclip"></i> Add attachment';
            addAttachmentBtn.addEventListener('click', () => this.addAttachmentInEdit(attachmentControls, (fileName) => {
                editAttachmentName = fileName;
            }));
            attachmentControls.appendChild(addAttachmentBtn);
        }

        const actionButtons = document.createElement('div');
        actionButtons.className = 'edit-action-buttons';
        actionButtons.innerHTML = `
            <button class="cancel-edit-btn">Cancel</button>
            <button class="save-edit-btn">Save & Resend</button>
        `;

        editActions.appendChild(attachmentControls);
        editActions.appendChild(actionButtons);
        contentWrapper.appendChild(editActions);

        const cancelBtn = actionButtons.querySelector('.cancel-edit-btn');
        const saveBtn = actionButtons.querySelector('.save-edit-btn');

        cancelBtn.addEventListener('click', () => this.cancelEdit(messageDiv, currentText, currentAttachment));
        saveBtn.addEventListener('click', () => this.saveEditedMessage(messageDiv, textarea.value, editAttachmentName));
    }

    addAttachmentInEdit(attachmentControls, callback) {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.csv,.xlsx,.xls,.json,.tsv,.parquet,.zip,.txt';
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.selectedFile = file;
                attachmentControls.innerHTML = `
                    <div class="current-attachment">
                        <i class="fa-solid fa-file"></i>
                        <span>${file.name}</span>
                        <button class="remove-attachment-btn" title="Remove attachment">
                            <i class="fa-solid fa-times"></i>
                        </button>
                    </div>
                `;
                const removeBtn = attachmentControls.querySelector('.remove-attachment-btn');
                removeBtn.addEventListener('click', () => {
                    this.selectedFile = null;
                    callback(null);
                    attachmentControls.innerHTML = '';
                    const addAttachmentBtn = document.createElement('button');
                    addAttachmentBtn.className = 'add-attachment-btn';
                    addAttachmentBtn.innerHTML = '<i class="fa-solid fa-paperclip"></i> Add attachment';
                    addAttachmentBtn.addEventListener('click', () => this.addAttachmentInEdit(attachmentControls, callback));
                    attachmentControls.appendChild(addAttachmentBtn);
                });
                callback(file.name);
            }
        });
        fileInput.click();
    }

    cancelEdit(messageDiv, originalText, originalAttachment) {
        const contentWrapper = messageDiv.querySelector('.message-content-wrapper');
        const textarea = messageDiv.querySelector('.edit-textarea');
        const editActions = messageDiv.querySelector('.edit-actions');
        const editControls = messageDiv.querySelector('.message-edit-controls');

        if (!textarea || !contentWrapper) return;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.convertMarkdownToHtml(originalText);

        textarea.replaceWith(messageContent);
        editActions.remove();

        messageDiv.classList.remove('editing');
        editControls.style.display = '';
    }

    async saveEditedMessage(messageDiv, newText, newAttachment) {
        if (!newText.trim() && !this.selectedFile) return;

        const heroChatMessages = document.getElementById('heroChatMessages');
        const heroChatInput = document.getElementById('heroChatInput');
        if (!heroChatMessages) return;

        if (heroChatInput) {
            heroChatInput.value = '';
        }

        const messageId = messageDiv.dataset.messageId;
        const linkedBotMessageId = messageDiv.dataset.linkedBotMessageId;

        const allMessages = Array.from(heroChatMessages.querySelectorAll('.chat-message:not(.streaming-status-container)'));
        const messageIndex = allMessages.indexOf(messageDiv);

        if (messageIndex === -1) {
            console.error('Could not find message index');
            return;
        }

        try {
            const sessionId = this.app.currentSessionId || this.app.agentSessionId;
            const response = await fetch(`/api/sessions/${sessionId}/revert/${messageIndex}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error('Failed to revert conversation state');
            }

            const result = await response.json();
            console.log('Reverted to message', messageIndex, result);

            for (let i = allMessages.length - 1; i > messageIndex; i--) {
                allMessages[i].remove();
            }

            const contentWrapper = messageDiv.querySelector('.message-content-wrapper');
            const textarea = messageDiv.querySelector('.edit-textarea');
            const editControls = messageDiv.querySelector('.message-edit-controls');
            const editActions = messageDiv.querySelector('.edit-actions');

            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = this.convertMarkdownToHtml(newText);

            if (textarea && contentWrapper) {
                textarea.replaceWith(messageContent);
            }

            if (editActions) {
                editActions.remove();
            }

            const existingAttachment = contentWrapper.querySelector('.message-attachment');
            if (existingAttachment) {
                existingAttachment.remove();
            }

            if (newAttachment || this.selectedFile) {
                const attachmentFileName = this.selectedFile ? this.selectedFile.name : newAttachment;
                const attachmentDiv = document.createElement('div');
                attachmentDiv.className = 'message-attachment';
                attachmentDiv.innerHTML = `
                    <i class="fa-solid fa-file"></i>
                    <span>${attachmentFileName}</span>`;
                contentWrapper.insertBefore(attachmentDiv, messageContent);
            }

            if (editControls) {
                editControls.style.display = '';
            }
            messageDiv.classList.remove('editing');

            this.streamAgentResponse(newText, messageId);

        } catch (error) {
            console.error('Error reverting conversation:', error);
            const errorMsg = error.message || 'Unknown error';
            alert(`Failed to revert: ${errorMsg}\n\nIf this is an old session, try refreshing the page.`);
        }
    }

    async regenerateResponse(botMessageDiv) {
        const linkedUserMessageId = botMessageDiv.dataset.linkedUserMessageId;

        if (!linkedUserMessageId) {
            console.error('Regenerate failed: No linked user message ID on bot message', botMessageDiv);
            alert('Cannot regenerate: Message link not found. This may be an older message.');
            return;
        }

        const userMessageDiv = document.querySelector(`[data-message-id="${linkedUserMessageId}"]`);
        if (!userMessageDiv) {
            console.error('Regenerate failed: User message not found for ID:', linkedUserMessageId);
            alert('Cannot regenerate: Original message not found.');
            return;
        }

        const messageContent = userMessageDiv.querySelector('.message-content');
        if (!messageContent) {
            console.error('Regenerate failed: Message content not found in:', userMessageDiv);
            alert('Cannot regenerate: Message content is missing.');
            return;
        }

        const userText = messageContent.textContent.trim();

        if (!userText) {
            console.error('Regenerate failed: User text is empty');
            alert('Cannot regenerate: Message text is empty.');
            return;
        }

        botMessageDiv.style.opacity = '0.5';

        const regenerateBtn = botMessageDiv.querySelector('.regenerate-btn');
        if (regenerateBtn) {
            regenerateBtn.disabled = true;
            regenerateBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
        }

        console.log('Regenerating response for message:', userText);

        this.streamAgentResponse(userText, linkedUserMessageId);

        setTimeout(() => {
            botMessageDiv.remove();
        }, 500);
    }

    showVersion(messageDiv, versionIndex) {
        const messageId = messageDiv.dataset.messageId;
        const versions = this.messageVersions.get(messageId);

        if (!versions || versionIndex < 0 || versionIndex >= versions.length) return;

        const version = versions[versionIndex];
        messageDiv.dataset.currentVersion = String(versionIndex);

        const messageContent = messageDiv.querySelector('.message-content');
        if (messageContent) {
            messageContent.innerHTML = this.convertMarkdownToHtml(version.userText);
        }

        versions.forEach((v, idx) => {
            if (v.botMessageId) {
                const botMsg = document.querySelector(`[data-message-id="${v.botMessageId}"]`);
                if (botMsg) {
                    botMsg.style.display = idx === versionIndex ? '' : 'none';
                }
            }
            if (v.statusContainerId) {
                const statusContainer = document.querySelector(`[data-status-id="${v.statusContainerId}"]`);
                if (statusContainer) {
                    statusContainer.style.display = idx === versionIndex ? '' : 'none';
                }
            }
        });

        if (version.botMessageId) {
            messageDiv.dataset.linkedBotMessageId = version.botMessageId;
        }

        messageDiv.style.opacity = '1';

        this.updateVersionNavigation(messageDiv);
    }

    updateVersionNavigation(messageDiv) {
        const messageId = messageDiv.dataset.messageId;
        const versions = this.messageVersions.get(messageId);

        if (!versions || versions.length <= 1) {
            const existingNav = messageDiv.querySelector('.version-navigation');
            if (existingNav) existingNav.remove();
            return;
        }

        const currentVersion = parseInt(messageDiv.dataset.currentVersion || '0');
        let versionNav = messageDiv.querySelector('.version-navigation');

        if (!versionNav) {
            versionNav = document.createElement('div');
            versionNav.className = 'version-navigation';
            const editControls = messageDiv.querySelector('.message-edit-controls');
            if (editControls) {
                editControls.before(versionNav);
            } else {
                messageDiv.appendChild(versionNav);
            }
        }

        versionNav.innerHTML = `
            <button class="version-btn version-prev" ${currentVersion === 0 ? 'disabled' : ''}>
                <i class="fa-solid fa-chevron-left"></i>
            </button>
            <span class="version-counter">${currentVersion + 1}/${versions.length}</span>
            <button class="version-btn version-next" ${currentVersion === versions.length - 1 ? 'disabled' : ''}>
                <i class="fa-solid fa-chevron-right"></i>
            </button>
        `;

        const prevBtn = versionNav.querySelector('.version-prev');
        const nextBtn = versionNav.querySelector('.version-next');

        prevBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (currentVersion > 0) {
                this.showVersion(messageDiv, currentVersion - 1);
            }
        });

        nextBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (currentVersion < versions.length - 1) {
                this.showVersion(messageDiv, currentVersion + 1);
            }
        });
    }

    showLoadingMessage() {
        const messagesContainer = document.getElementById('heroChatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot loading-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        this.loadingMessageElement = messageDiv;

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideLoadingMessage() {
        if (this.loadingMessageElement) {
            this.loadingMessageElement.remove();
            this.loadingMessageElement = null;
        }
    }

    formatMessage(message) {
        const parts = message.split('\n\n');

        if (parts.length > 1) {
            let formattedParts = [];
            for (let i = 0; i < parts.length; i++) {
                const part = parts[i];

                if (this.isDataFrameOutput(part)) {
                    formattedParts.push(this.convertDataFrameToTable(part));
                } else if (this.isFirstRowsResponse(part)) {
                    formattedParts.push(this.convertFirstRowsToTable(part));
                } else {
                    formattedParts.push(this.convertMarkdownToHtml(part));
                }
            }
            return formattedParts.join('<br><br>');
        }

        if (this.isDataFrameOutput(message)) {
            return this.convertDataFrameToTable(message);
        }

        if (this.isFirstRowsResponse(message)) {
            return this.convertFirstRowsToTable(message);
        }

        if (message.includes('|') && message.includes('-')) {
            return this.convertMarkdownTableToHtml(message);
        }

        return this.convertMarkdownToHtml(message);
    }

    isDataFrameOutput(message) {
        return message.includes('pandas.DataFrame') ||
            (message.includes('Index:') && message.includes('Columns:')) ||
            (message.includes('rows') && message.includes('columns')) ||
            (message.includes('Name:') && message.includes('dtype:')) ||
            message.includes('pandas.Series');
    }

    isFirstRowsResponse(message) {
        return message.includes('.head()') ||
            (message.includes('first') && message.includes('rows')) ||
            (message.includes('showing') && message.includes('rows'));
    }

    formatBotResponse(message) {
        const analysisPattern = /ANALYSIS_RESULTS:\{.*?\}\n/gs;
        const plotPattern = /📊 Generated (\d+) visualization\(s\): \[(.*?)\]/;
        const imagePattern = /!\[.*?\]\((.*?)\)/g;

        let cleanMessage = message;
        let analysisData = null;
        let plotUrls = [];

        const analysisMatch = message.match(analysisPattern);
        if (analysisMatch) {
            cleanMessage = cleanMessage.replace(analysisPattern, '');
            try {
                const jsonStr = analysisMatch[0].replace('ANALYSIS_RESULTS:', '').trim();
                analysisData = JSON.parse(jsonStr);
            } catch (e) {
                console.warn('Failed to parse analysis data');
            }
        }

        const plotMatch = message.match(plotPattern);
        if (plotMatch) {
            cleanMessage = cleanMessage.replace(plotPattern, '');
        }

        cleanMessage = cleanMessage.replace(/```[\s\S]*?```/g, '').trim();

        let formatted = '';

        const commentaryMatch = cleanMessage.match(/\*\*Commentary\*\*([\s\S]*?)$/);
        let commentary = '';
        if (commentaryMatch) {
            commentary = commentaryMatch[1].trim();
            cleanMessage = cleanMessage.replace(/\*\*Commentary\*\*[\s\S]*$/, '').trim();
        }

        if (cleanMessage && cleanMessage.length > 10) {
            formatted += `<div>${this.convertMarkdownToHtml(cleanMessage)}</div>`;
        }

        if (commentary) {
            formatted += `<div>${this.convertMarkdownToHtml(commentary)}</div>`;
        }

        return formatted || this.convertMarkdownToHtml(message);
    }

    convertMarkdownToHtml(markdown) {
        if (!markdown) return '';

        marked.use({
            gfm: true,
            breaks: true,
            headerIds: false,
            mangle: false,
            pedantic: false
        });

        const sanitizedMarkdown = markdown
            .replace(/[\u200B-\u200D\uFEFF]/g, '')
            .replace(/^[\s\n]*Below[\s\u200B-\u200D\uFEFF]*\*\*[\s\u200B-\u200D\uFEFF]*\*\*[\s\n]*/i, '')
            .trim();

        const dirty = marked.parse(sanitizedMarkdown);

        const clean = DOMPurify.sanitize(dirty, {
            ALLOWED_TAGS: ['p', 'br', 'strong', 'b', 'em', 'i', 'u', 'code', 'pre', 'a', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'hr', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'div', 'span', 'iframe', 'img'],
            ALLOWED_ATTR: ['href', 'target', 'rel', 'src', 'alt', 'allowfullscreen', 'class', 'align', 'style', 'width', 'height'],
            KEEP_CONTENT: true
        });
        const withEmojis = this.emojiConverter.replace_colons(clean);
        return withEmojis.replace(/<table>/g, '<div class="table-wrapper"><table>')
            .replace(/<\/table>/g, '</table></div>');
    }

    convertInlineMarkdown(text) {
        return text
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>');
    }

    convertMarkdownTableToHtml(message) {
        const lines = message.split('\n');
        let result = '';
        let i = 0;

        while (i < lines.length) {
            const line = lines[i].trim();

            if (line.includes('|') && line.split('|').length > 2) {
                const headers = line.split('|').map(h => h.trim()).filter(h => h);

                if (i + 1 < lines.length && lines[i + 1].includes('|') && lines[i + 1].includes('-')) {
                    let tableHTML = '<div class="table-responsive-container"><div class="table-scroll-wrapper"><table class="data-table">';
                    tableHTML += '<thead><tr>';
                    headers.forEach(header => {
                        tableHTML += `<th>${this.convertInlineMarkdown(header)}</th>`;
                    });
                    tableHTML += '</tr></thead><tbody>';

                    i += 2;

                    while (i < lines.length && lines[i].includes('|')) {
                        const cells = lines[i].split('|').map(c => c.trim()).filter(c => c);
                        if (cells.length > 0) {
                            tableHTML += '<tr>';
                            cells.forEach(cell => {
                                tableHTML += `<td>${this.convertInlineMarkdown(cell)}</td>`;
                            });
                            tableHTML += '</tr>';
                        }
                        i++;
                    }

                    tableHTML += '</tbody></table></div></div>';
                    result += tableHTML;
                    continue;
                }
            }

            result += this.convertMarkdownToHtml(line) + '<br>';
            i++;
        }

        return result;
    }

    convertDataFrameToTable(message) {
        const lines = message.split('\n');
        let headers = [];
        let dataRows = [];
        let maxColumns = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            if (/^\s*\d+\s+/.test(line)) {
                const cells = line.split(/\s+/);
                if (cells.length > 0) {
                    dataRows.push({
                        index: cells[0],
                        data: cells.slice(1)
                    });
                    maxColumns = Math.max(maxColumns, cells.length - 1);
                }
            }
        }

        if (headers.length !== maxColumns) {
            headers = [];
            if (dataRows.length > 0 && dataRows[0].data.length > headers.length) {
                for (let i = headers.length; i < dataRows[0].data.length; i++) {
                    headers.push(`Col_${i}`);
                }
            }
            while (headers.length < maxColumns) {
                headers.push(`Col_${headers.length}`);
            }
        }

        let tableHTML = '<div class="table-responsive-container"><div class="table-scroll-wrapper">';
        tableHTML += '<table class="dataframe-table"><thead><tr><th>Index</th>';

        headers.forEach(header => {
            tableHTML += `<th>${header}</th>`;
        });

        tableHTML += '</tr></thead><tbody>';

        dataRows.forEach(row => {
            tableHTML += `<tr><td>${row.index}</td>`;
            for (let j = 0; j < headers.length; j++) {
                const cellValue = row.data[j] || '';
                tableHTML += `<td>${cellValue}</td>`;
            }
            tableHTML += '</tr>';
        });

        tableHTML += '</tbody></table></div></div>';
        return tableHTML;
    }

    convertFirstRowsToTable(message) {
        return this.convertDataFrameToTable(message);
    }

    formatTableContent(tableData) {
        if (!tableData || !tableData.headers || !tableData.rows) {
            return '<p>Invalid table data</p>';
        }

        let tableHTML = '<div class="table-responsive-container"><div class="table-scroll-wrapper">';
        tableHTML += '<table class="data-table"><thead><tr>';

        tableData.headers.forEach(header => {
            tableHTML += `<th>${header}</th>`;
        });

        tableHTML += '</tr></thead><tbody>';

        tableData.rows.forEach(row => {
            tableHTML += '<tr>';
            row.forEach(cell => {
                tableHTML += `<td>${cell}</td>`;
            });
            tableHTML += '</tr>';
        });

        tableHTML += '</tbody></table></div></div>';
        return tableHTML;
    }

    showPIIConsentDialog(piiData) {
        const heroChatMessages = document.getElementById('heroChatMessages');
        if (!heroChatMessages) return;

        const noticeDiv = document.createElement('div');
        noticeDiv.className = 'pii-notice';

        const detectedColumns = piiData.detected_columns || {};
        const columnCount = Object.keys(detectedColumns).length;

        const sortedColumns = Object.entries(detectedColumns)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5);

        let columnsHtml = '';
        if (sortedColumns.length > 0) {
            columnsHtml = '<div class="pii-columns-list">';
            sortedColumns.forEach(([colName, sensitivity]) => {
                const percentage = Math.round(sensitivity * 100);
                const barColor = sensitivity > 0.8 ? '#ff6b6b' : sensitivity > 0.6 ? '#ffa500' : '#4caf50';
                columnsHtml += `
                    <div class="pii-column-item">
                        <div class="pii-column-header">
                            <span class="pii-column-name">${colName}</span>
                            <span class="pii-sensitivity-value">${percentage}%</span>
                        </div>
                        <div class="pii-sensitivity-bar">
                            <div class="pii-sensitivity-fill" style="width: ${percentage}%; background-color: ${barColor};"></div>
                        </div>
                    </div>
                `;
            });
            if (columnCount > 5) {
                columnsHtml += `<p class="pii-more-columns">... and ${columnCount - 5} more column(s)</p>`;
            }
            columnsHtml += '</div>';
        }

        noticeDiv.innerHTML = `
            <h4><i class="fa-solid fa-shield-halved"></i> Privacy Notice</h4>
            <p>Detected ${columnCount} column(s) with sensitive data. Review sensitivity scores below:</p>
            ${columnsHtml}
            <div class="pii-notice-buttons">
                <button id="applyProtection">Apply Protection</button>
                <button id="continueWithoutProtection">Continue Without</button>
            </div>
        `;

        heroChatMessages.appendChild(noticeDiv);
        heroChatMessages.scrollTop = heroChatMessages.scrollHeight;

        const applyBtn = noticeDiv.querySelector('#applyProtection');
        const continueBtn = noticeDiv.querySelector('#continueWithoutProtection');

        applyBtn.addEventListener('click', () => {
            this.handlePIIConsent(true);
            noticeDiv.innerHTML = `<p>Privacy protection will be applied.</p>`;
        }, { once: true });

        continueBtn.addEventListener('click', () => {
            this.handlePIIConsent(false);
            noticeDiv.innerHTML = `<p>INFO: Continuing analysis without privacy protection.</p>`;
        }, { once: true });
    }

    async handlePIIConsent(applyProtection) {
        try {
            const data = await this.app.apiClient.handlePIIConsent(this.app.agentSessionId, applyProtection);

            if (data.status === 'success') {
                const message = applyProtection
                    ? 'Privacy protection applied. Your data has been anonymized for analysis.'
                    : 'Continuing without privacy protection. Your original data will be used.';
                this.addChatMessage('bot', message);
            }
        } catch (error) {
            console.error('Privacy consent error:', error);
            this.addChatMessage('bot', 'Error processing privacy preference.');
        }
    }

    showAttachmentBadge(file) {
        const chatInputArea = document.querySelector('.chat-input-area');
        if (!chatInputArea) return;

        this.removeAttachmentBadge();

        const badge = document.createElement('div');
        badge.className = 'attachment-badge';
        badge.innerHTML = `
            <i class="fa-solid fa-file"></i>
            <span class="attachment-name">${file.name}</span>
            <button class="attachment-remove" title="Remove attachment">
                <i class="fa-solid fa-times"></i>
            </button>
        `;

        const textarea = chatInputArea.querySelector('textarea');
        chatInputArea.insertBefore(badge, textarea);

        this.attachmentBadge = badge;

        badge.querySelector('.attachment-remove').addEventListener('click', () => {
            this.removeAttachment();
        });
    }

    removeAttachment() {
        this.selectedFile = null;
        this.removeAttachmentBadge();
    }

    removeAttachmentBadge() {
        if (this.attachmentBadge) {
            this.attachmentBadge.remove();
            this.attachmentBadge = null;
        }
    }

    async sendMessageWithAttachment(message) {
        const file = this.selectedFile;
        const heroChatInput = document.getElementById('heroChatInput');

        const displayMessage = message
            ? `${message}`
            : `Uploading ${file.name}...`;

        this.addChatMessage('user', displayMessage, null, file.name);

        if (heroChatInput) {
            heroChatInput.value = '';
        }
        this.removeAttachment();

        const sessionId = this.app.agentSessionId || this.app.currentSessionId;
        if (!sessionId) {
            console.error('No session ID available for upload');
            this.addChatMessage('bot', 'Error: Please refresh the page and try again.');
            return;
        }

        try {
            this.showLoadingMessage();
            const data = await this.app.apiClient.uploadFile(file, sessionId);

            if (data.status === 'success') {
                this.hideLoadingMessage();

                this.app.currentSessionId = data.session_id;
                this.app.agentSessionId = data.agent_session_id || data.session_id;

                if (window.artifactStorage) {
                    window.artifactStorage.setSessionId(this.app.agentSessionId);
                }

                if (data.pii_detection && data.pii_detection.requires_consent) {
                    this.showPIIConsentDialog(data.pii_detection);
                }

                if (data.report_created && data.report_id && window.createReport) {
                    console.log('Opening report panel for upload:', data.report_id);
                    window.createReport({
                        id: data.report_id,
                        dataset_name: file.name,
                        session_id: this.app.agentSessionId
                    });
                }

                if (message) {
                    this.streamAgentResponse(message);
                }
            } else {
                this.hideLoadingMessage();
                this.addChatMessage('bot', `Upload error: ${data.detail || 'Unknown error'}`);
            }
        } catch (error) {
            this.hideLoadingMessage();
            this.addChatMessage('bot', `Upload error: ${error.message}`);
        }
    }

    async exportConversation(format) {
        try {
            const sessionId = this.app.currentSessionId || this.app.agentSessionId;
            if (!sessionId) {
                console.warn('No active session to export');
                return;
            }

            const response = await fetch(`/api/sessions/${sessionId}/export?format=${format}`);
            if (!response.ok) {
                throw new Error(`Export failed: ${response.status}`);
            }

            const blob = await response.blob();
            const timestamp = new Date().toISOString().slice(0, 10);
            const extensions = {
                'markdown': 'md',
                'json': 'json',
                'html': 'html',
                'pdf': 'pdf'
            };
            const filename = `conversation_${timestamp}.${extensions[format] || 'txt'}`;

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            console.log(`Conversation exported as ${format}`);
        } catch (error) {
            console.error('Error exporting conversation:', error);
        }
    }
}
