class ChatInterface {
    constructor() {
        this.app = null;
        this.loadingMessageElement = null;
    }

    setApp(app) {
        this.app = app;
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
        }

        // Ensure we have a session ID for agent communication
        if (!this.app.agentSessionId) {
            this.app.agentSessionId = this.app.currentSessionId || '';
        }

        const sendMessage = () => {
            const message = heroChatInput.value.trim();
            if (!message) return;

            const isFirstMessage = localStorage.getItem('hasFirstMessage') !== 'true';
            if (isFirstMessage && this.app.blackHole) {
                this.app.blackHole.triggerExpansion();
                localStorage.setItem('hasFirstMessage', 'true');
            }

            this.addChatMessage('user', message);
            heroChatInput.value = '';
            this.streamAgentResponse(message);
        };

        heroSendBtn.addEventListener('click', sendMessage);
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

            heroFileInput.addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                const sessionId = this.app.agentSessionId || this.app.currentSessionId;
                if (!sessionId) {
                    console.error('No session ID available for upload');
                    this.addChatMessage('bot', 'Error: Please refresh the page and try again.');
                    return;
                }

                console.log(`Uploading ${file.name} to session ${sessionId}`);
                this.addChatMessage('user', `Uploading ${file.name}...`);

                try {
                    this.showLoadingMessage();
                    const data = await this.app.apiClient.uploadFile(file, sessionId);

                    if (data.status === 'success') {
                        this.app.currentSessionId = data.session_id;
                        this.app.agentSessionId = data.agent_session_id || data.session_id;

                        if (data.agent_analysis) {
                            this.addChatMessage('bot', data.agent_analysis);
                        } else {
                            const profileSummary = data.profiling_summary || {};
                            let message = data.shape && data.shape.length >= 2
                                ? `Dataset uploaded successfully! Shape: ${data.shape[0]} rows x ${data.shape[1]} columns.`
                                : `Dataset uploaded successfully!`;

                            if (profileSummary.quality_score) {
                                message += ` Quality Score: ${profileSummary.quality_score}/100.`;
                            }
                            if (profileSummary.anomalies_detected) {
                                message += ` Detected ${profileSummary.anomalies_detected} anomalies.`;
                            }
                            if (profileSummary.profiling_time) {
                                message += ` Analysis completed in ${profileSummary.profiling_time}s.`;
                            }
                            if (data.intelligence_summary && data.intelligence_summary.profiling_completed) {
                                message += ` Domain: ${data.intelligence_summary.primary_domain}.`;
                            }
                            message += ` You can now ask me questions about your data.`;
                            this.addChatMessage('bot', message);
                        }

                        if (data.pii_detection && data.pii_detection.requires_consent) {
                            this.showPIIConsentDialog(data.pii_detection);
                        }
                    } else {
                        this.addChatMessage('bot', `Upload error: ${data.detail || 'Unknown error'}`);
                    }
                } catch (error) {
                    this.hideLoadingMessage();
                    this.addChatMessage('bot', `Upload error: ${error.message}`);
                }
            });
        }
    }

    streamAgentResponse(message) {
        const statusMessageElement = this.showStreamingStatusPlaceholder();
        const webSearchEnabled = localStorage.getItem('webSearchEnabled') === 'true';

        this.app.apiClient.streamAgentResponse(
            message,
            this.app.agentSessionId,
            webSearchEnabled,
            (statusText, statusType) => {
                this.updateStreamingStatusWithType(statusMessageElement, statusText, statusType, 'Processing');
            },
            (data) => {
                setTimeout(() => {
                    const remainingLines = statusMessageElement.querySelectorAll('.status-line:not(.completed)');
                    remainingLines.forEach(line => {
                        line.classList.add('completed');
                        const statusText = line.querySelector('.status-text');
                        if (statusText) {
                            setTimeout(() => {
                                this.createScratchLines(statusText);
                            }, 100);
                        }
                    });
                    this.addChatMessage('bot', data.response, data.plots);
                }, 150);
            },
            (errorMessage) => {
                setTimeout(() => {
                    this.updateStreamingStatusWithType(statusMessageElement, `Error: ${errorMessage}`, 'error', 'Error');
                    statusMessageElement.style.display = 'none';
                    this.addChatMessage('bot', `Error: ${errorMessage}`);
                }, 150);
            }
        );
    }

    showStreamingStatusPlaceholder() {
        const heroChatMessages = document.getElementById('heroChatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot streaming-status-container';
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

        const existingLines = wrapper.querySelectorAll('.status-line');
        existingLines.forEach(line => {
            line.classList.remove('active');
            line.classList.add('completed');
            const sandclock = line.querySelector('.sandclock-icon');
            if (sandclock) {
                sandclock.style.opacity = '0';
            }
        });

        const cleanedText = newStatusText.replace(/^Processing:\s*/i, '').trim();

        const statusLine = document.createElement('div');
        statusLine.className = `status-line ${statusType === 'active' ? 'active' : statusType}`;
        statusLine.innerHTML = `
            <i class="sandclock-icon fa-solid fa-hourglass-half"></i>
            <div class="status-text">${cleanedText}</div>
        `;

        wrapper.appendChild(statusLine);

        if (statusType === 'completed') {
            statusLine.classList.remove('active');
            statusLine.classList.add('completed');
            const sandclock = statusLine.querySelector('.sandclock-icon');
            if (sandclock) {
                sandclock.style.opacity = '0';
            }
        } else if (statusType === 'active') {
            statusLine.classList.add('active');
        }

        this.updateTaskProgress(wrapper, statusType);

        const allLines = wrapper.querySelectorAll('.status-line');
        if (allLines.length > 5) {
            const pastLines = wrapper.querySelectorAll('.status-line.completed');
            if (pastLines.length > 8) {
                for (let i = 0; i < pastLines.length - 6; i++) {
                    pastLines[i].remove();
                }
            }
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

    createScratchLines(statusText) {
        const text = statusText.textContent.trim();
        if (!text) return;

        const words = text.split(' ');
        const lines = [];
        let currentLine = '';

        // Create a temporary element to measure text width
        const tempDiv = document.createElement('div');
        tempDiv.style.position = 'absolute';
        tempDiv.style.visibility = 'hidden';
        tempDiv.style.fontSize = getComputedStyle(statusText).fontSize;
        tempDiv.style.fontFamily = getComputedStyle(statusText).fontFamily;
        document.body.appendChild(tempDiv);

        const availableWidth = statusText.offsetWidth - 40; // Account for spinner and padding

        for (let i = 0; i < words.length; i++) {
            const testLine = currentLine ? currentLine + ' ' + words[i] : words[i];
            tempDiv.textContent = testLine;
            const width = tempDiv.offsetWidth;

            if (width > availableWidth && currentLine) {
                lines.push(currentLine);
                currentLine = words[i];
            } else {
                currentLine = testLine;
            }
        }

        if (currentLine) {
            lines.push(currentLine);
        }

        if (tempDiv.parentNode) {
            tempDiv.parentNode.removeChild(tempDiv);
        }

        // Apply scratch lines to each line of text
        lines.forEach((lineText, index) => {
            setTimeout(() => {
                const scratchLine = document.createElement('div');
                scratchLine.className = 'scratch-line';
                scratchLine.style.cssText = `
                    --scratch-width: ${Math.random() * 20 + 80}%;
                    top: ${(index + 0.5) * 1.2}em;
                `;
                statusText.style.position = 'relative';
                statusText.appendChild(scratchLine);
            }, index * 50);
        });
    }

    updateStreamingStatusWithType(containerElement, newStatusText, statusType = 'active', taskName = '') {
        let prefix = '';
        let typeClass = statusType;

        switch (statusType) {
            case 'data_processing':
                prefix = '📊 ';
                typeClass = 'active';
                break;
            case 'analysis':
                prefix = '🔍 ';
                typeClass = 'active';
                break;
            case 'model_training':
                prefix = '🤖 ';
                typeClass = 'active';
                break;
            case 'processing':
                prefix = '⚙️ ';
                typeClass = 'active';
                break;
            case 'completed':
                prefix = '✅ ';
                typeClass = 'completed';
                break;
            case 'error':
                prefix = '❌ ';
                typeClass = 'error';
                break;
        }

        const fullStatusText = taskName ? `${taskName}: ${newStatusText}` : newStatusText;
        this.updateStreamingStatus(containerElement, `${prefix}${fullStatusText}`, typeClass);
    }

    addChatMessage(sender, responseData, plots = null) {
        const heroChatMessages = document.getElementById('heroChatMessages');
        if (!heroChatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content-wrapper';

        if (typeof responseData === 'string') {
            const content = document.createElement('div');
            content.className = 'message-content';
            content.innerHTML = `<p>${this.formatMessage(responseData)}</p>`;
            contentWrapper.appendChild(content);
        } else if (responseData && typeof responseData === 'object' && responseData.content) {
            // Handle structured response data
            responseData.content.forEach(item => {
                const content = document.createElement('div');
                content.className = 'message-content';

                switch (item.type) {
                    case 'text':
                        content.innerHTML = `<p>${this.formatMessage(item.text)}</p>`;
                        break;
                    case 'code':
                        content.innerHTML = `<pre><code class="${item.language || ''}">${item.code}</code></pre>`;
                        break;
                    case 'table':
                        content.innerHTML = this.formatTableContent(item.data);
                        break;
                    default:
                        content.innerHTML = `<p>${this.formatMessage(JSON.stringify(item))}</p>`;
                }

                contentWrapper.appendChild(content);
            });
        }

        // Handle plots if provided
        if (plots && Array.isArray(plots) && plots.length > 0) {
            plots.forEach(plotUrl => {
                const plotDiv = document.createElement('div');
                plotDiv.className = 'plot-container';

                if (plotUrl.endsWith('.html')) {
                    plotDiv.innerHTML = `
                        <iframe src="${plotUrl}" width="100%" height="400" frameborder="0">
                            Your browser does not support iframes.
                        </iframe>
                    `;
                } else {
                    plotDiv.innerHTML = `
                        <img src="${plotUrl}" alt="Generated Plot" style="max-width: 100%; height: auto;" />
                    `;
                }

                contentWrapper.appendChild(plotDiv);
            });
        }

        messageDiv.appendChild(contentWrapper);
        heroChatMessages.appendChild(messageDiv);

        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([messageDiv]).catch(err => console.warn('MathJax error:', err));
        }

        heroChatMessages.scrollTop = heroChatMessages.scrollHeight;
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

    convertMarkdownToHtml(message) {
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    convertMarkdownTableToHtml(message) {
        const lines = message.split('\n');
        let tableHTML = '<div class="table-responsive-container"><div class="table-scroll-wrapper"><table class="data-table">';
        let inTable = false;
        let otherHTML = '';

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            if (line.includes('|') && !inTable && line.includes('-')) {
                inTable = true;
                if (otherHTML.trim()) {
                    tableHTML = this.convertMarkdownToHtml(otherHTML) + tableHTML;
                    otherHTML = '';
                }

                const headers = line.split('|').map(h => h.trim()).filter(h => h);
                tableHTML += '<thead><tr>';
                headers.forEach(header => {
                    tableHTML += `<th>${header}</th>`;
                });
                tableHTML += '</tr></thead><tbody>';
                i++;
                continue;
            }

            if (inTable) {
                if (line.includes('|')) {
                    const cells = line.split('|').map(c => c.trim()).filter(c => c);
                    tableHTML += '<tr>';
                    cells.forEach(cell => {
                        tableHTML += `<td>${cell}</td>`;
                    });
                    tableHTML += '</tr>';
                } else if (line) {
                    otherHTML += line + '\n';
                } else {
                    inTable = false;
                    tableHTML += '</tbody></table></div></div>';
                }
            } else {
                if (line) {
                    otherHTML += line + '\n';
                }
            }
        }

        if (inTable) {
            tableHTML += '</tbody></table></div></div>';
        }

        if (otherHTML.trim()) {
            tableHTML += '<br>' + this.convertMarkdownToHtml(otherHTML);
        }

        return tableHTML;
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
        const dialog = document.createElement('div');
        dialog.className = 'pii-consent-dialog';
        const detectedTypes = piiData.detected_types && Array.isArray(piiData.detected_types)
            ? piiData.detected_types.map(type => `<li>${type}</li>`).join('')
            : '<li>Sensitive data detected</li>';

        dialog.innerHTML = `
            <div class="dialog-content">
                <h3>🔒 Privacy Notice</h3>
                <p>We detected potentially sensitive information in your dataset:</p>
                <ul>
                    ${detectedTypes}
                </ul>
                <p>Would you like us to apply privacy protection before analysis?</p>
                <div class="dialog-buttons">
                    <button id="applyProtection" class="btn-primary">Apply Protection</button>
                    <button id="continueWithoutProtection" class="btn-secondary">Continue Without Protection</button>
                </div>
            </div>
        `;

        document.body.appendChild(dialog);

        document.getElementById('applyProtection').addEventListener('click', () => {
            this.handlePIIConsent(true);
            dialog.remove();
        });

        document.getElementById('continueWithoutProtection').addEventListener('click', () => {
            this.handlePIIConsent(false);
            dialog.remove();
        });
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
}
