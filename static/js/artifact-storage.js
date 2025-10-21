class ArtifactStorage {
    constructor() {
        this.sessionId = null;
        this.artifacts = [];
        this.lastCheckTime = new Date().toISOString();
        this.checkInterval = null;
        this.isOpen = false;
        this.hasUnseenArtifacts = false;
        this.previewTooltip = null;
        this.init();
        this.createPreviewTooltip();
    }

    init() {
        const storageBtn = document.getElementById('artifactStorageBtn');
        const dropdown = document.getElementById('artifactStorageDropdown');

        if (dropdown) {
            dropdown.classList.add('hidden');
            dropdown.classList.remove('show');
        }

        if (storageBtn) {
            storageBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleDropdown();
            });
        }

        document.addEventListener('click', (e) => {
            const dropdown = document.getElementById('artifactStorageDropdown');
            const storageBtn = document.getElementById('artifactStorageBtn');

            if (dropdown && storageBtn &&
                !dropdown.contains(e.target) &&
                !storageBtn.contains(e.target) &&
                this.isOpen) {
                this.closeDropdown();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.closeDropdown();
            }
        });

        this.startPolling();
    }

    setSessionId(sessionId) {
        if (this.sessionId !== sessionId) {
            console.log(`[ArtifactStorage] Session ID changed: ${this.sessionId} → ${sessionId}`);
            this.sessionId = sessionId;
            this.artifacts = [];
            this.lastCheckTime = new Date().toISOString();
            this.hasUnseenArtifacts = false;
            this.loadArtifacts();
        }
    }

    async loadArtifacts() {
        if (!this.sessionId) {
            console.log('[ArtifactStorage] No session ID set, skipping load');
            return;
        }

        console.log(`[ArtifactStorage] Loading artifacts for session: ${this.sessionId}`);
        try {
            const response = await fetch(`/api/data/${this.sessionId}/artifacts`);
            const data = await response.json();

            if (data.status === 'success') {
                console.log(`[ArtifactStorage] Loaded ${data.total_count || 0} artifacts`);
                this.artifacts = data.artifacts || [];
                this.updateBadge(data.total_count || 0);
                this.renderArtifacts(data.categories || {});
            }
        } catch (error) {
            console.error('Failed to load artifacts:', error);
        }
    }

    async checkNewArtifacts() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(
                `/api/data/${this.sessionId}/artifacts/new?since=${encodeURIComponent(this.lastCheckTime)}`
            );
            const data = await response.json();

            if (data.status === 'success' && data.count > 0) {
                console.log(`[ArtifactStorage] ${data.count} new artifacts detected`);
                if (!this.hasUnseenArtifacts) {
                    this.notifyNewArtifacts(data.count);
                    this.hasUnseenArtifacts = true;
                }
                this.lastCheckTime = new Date().toISOString();

                if (!this.isOpen) {
                    await this.loadArtifacts();
                } else {
                    console.log('[ArtifactStorage] Dropdown is open - skipping reload to prevent flicker');
                }
            }
        } catch (error) {
            console.error('Failed to check new artifacts:', error);
        }
    }

    notifyNewArtifacts(count) {
        const storageBtn = document.getElementById('artifactStorageBtn');

        if (storageBtn) {
            console.log('[ArtifactStorage] Triggering pulse animation for new artifacts');
            storageBtn.classList.add('artifact-pulse');

            setTimeout(() => {
                storageBtn.classList.remove('artifact-pulse');
            }, 2000);
        }
    }

    updateBadge(count) {
        const badge = document.getElementById('artifactBadge');
        if (badge) {
            badge.textContent = count;
            if (count > 0) {
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }
        }
    }

    renderArtifacts(categories) {
        const categoriesContainer = document.getElementById('artifactCategories');
        const emptyState = document.getElementById('artifactEmptyState');

        if (!categoriesContainer || !emptyState) return;

        const hasArtifacts = Object.keys(categories).length > 0;

        if (hasArtifacts) {
            emptyState.classList.add('hidden');
            categoriesContainer.classList.remove('hidden');
        } else {
            emptyState.classList.remove('hidden');
            categoriesContainer.classList.add('hidden');
            return;
        }

        categoriesContainer.innerHTML = '';

        for (const [categoryKey, categoryData] of Object.entries(categories)) {
            const categorySection = this.createCategorySection(categoryKey, categoryData);
            categoriesContainer.appendChild(categorySection);
        }
    }

    createCategorySection(categoryKey, categoryData) {
        const section = document.createElement('div');
        section.className = 'artifact-category';

        const header = document.createElement('div');
        header.className = 'artifact-category-header';
        header.innerHTML = `
            <div class="category-title">
                <span class="category-icon">${categoryData.icon}</span>
                <span class="category-label">${categoryData.label}</span>
                <span class="category-count">${categoryData.count}</span>
            </div>
            <button class="category-toggle">
                <i class="fa-solid fa-chevron-down"></i>
            </button>
        `;

        const content = document.createElement('div');
        content.className = 'artifact-category-content';

        for (const artifact of categoryData.artifacts) {
            const artifactItem = this.createArtifactItem(artifact);
            content.appendChild(artifactItem);
        }

        header.addEventListener('click', () => {
            section.classList.toggle('collapsed');
        });

        section.appendChild(header);
        section.appendChild(content);

        return section;
    }

    createArtifactItem(artifact) {
        const item = document.createElement('div');
        item.className = 'artifact-item';

        const createdDate = new Date(artifact.created_at);
        const timeAgo = this.getTimeAgo(createdDate);

        const isImage = /\.(png|jpg|jpeg|gif|svg|webp)$/i.test(artifact.filename);
        const isModel = /\.(pkl|joblib|h5|pt|pth|onnx)$/i.test(artifact.filename);

        const previewUrl = artifact.file_path || artifact.url;

        item.innerHTML = `
            <div class="artifact-info">
                <div class="artifact-name">
                    <i class="fa-solid fa-file"></i>
                    <span>${artifact.filename}</span>
                </div>
                <div class="artifact-meta">
                    <span class="artifact-size">${artifact.size}</span>
                    <span class="artifact-date">${timeAgo}</span>
                </div>
                ${artifact.description ? `<div class="artifact-description">${artifact.description}</div>` : ''}
            </div>
            <div class="artifact-actions">
                <button class="btn-artifact-download" data-artifact-id="${artifact.artifact_id}" title="Download">
                    <i class="fa-solid fa-download"></i>
                </button>
                <button class="btn-artifact-delete" data-artifact-id="${artifact.artifact_id}" title="Delete">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        `;

        if (isImage && previewUrl) {
            item.setAttribute('data-preview', 'image');

            item.addEventListener('mouseenter', () => {
                this.showPreview(item, previewUrl, false);
            });

            item.addEventListener('mouseleave', () => {
                this.hidePreview();
            });
        } else if (isModel) {
            item.setAttribute('data-preview', 'model');
            const modelInfo = this.formatModelInfo(artifact);

            item.addEventListener('mouseenter', () => {
                this.showPreview(item, null, true, modelInfo);
            });

            item.addEventListener('mouseleave', () => {
                this.hidePreview();
            });
        }

        const downloadBtn = item.querySelector('.btn-artifact-download');
        const deleteBtn = item.querySelector('.btn-artifact-delete');

        downloadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.downloadArtifact(artifact.artifact_id);
        });

        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteArtifact(artifact.artifact_id);
        });

        return item;
    }

    getTimeAgo(date) {
        const seconds = Math.floor((new Date() - date) / 1000);

        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        if (seconds < 2592000) return `${Math.floor(seconds / 86400)}d ago`;

        return date.toLocaleDateString();
    }

    formatModelInfo(artifact) {
        const lines = [];
        lines.push(`📦 ${artifact.filename}`);
        lines.push(`📊 Size: ${artifact.size}`);

        if (artifact.metadata) {
            if (artifact.metadata.model_type) {
                lines.push(`🔧 Type: ${artifact.metadata.model_type}`);
            }
            if (artifact.metadata.accuracy) {
                lines.push(`✓ Accuracy: ${(artifact.metadata.accuracy * 100).toFixed(1)}%`);
            }
            if (artifact.metadata.r2_score) {
                lines.push(`✓ R²: ${artifact.metadata.r2_score.toFixed(3)}`);
            }
        }

        if (artifact.description && artifact.description !== artifact.filename) {
            lines.push(`📝 ${artifact.description}`);
        }

        return lines.join('\n');
    }

    async downloadArtifact(artifactId) {
        if (!this.sessionId) return;

        try {
            window.open(`/api/data/${this.sessionId}/artifacts/download/${artifactId}`, '_blank');
        } catch (error) {
            console.error('Failed to download artifact:', error);
        }
    }

    async deleteArtifact(artifactId) {
        if (!this.sessionId) return;

        if (!confirm('Are you sure you want to delete this artifact?')) {
            return;
        }

        try {
            const response = await fetch(
                `/api/data/${this.sessionId}/artifacts/${artifactId}`,
                { method: 'DELETE' }
            );

            const data = await response.json();

            if (data.status === 'success') {
                await this.loadArtifacts();
            } else {
                console.error('Failed to delete artifact:', data);
            }
        } catch (error) {
            console.error('Failed to delete artifact:', error);
        }
    }

    toggleDropdown() {
        if (this.isOpen) {
            this.closeDropdown();
        } else {
            this.openDropdown();
        }
    }

    openDropdown() {
        const dropdown = document.getElementById('artifactStorageDropdown');
        const storageBtn = document.getElementById('artifactStorageBtn');

        if (dropdown && storageBtn) {
            dropdown.classList.remove('hidden');
            setTimeout(() => {
                dropdown.classList.add('show');
            }, 10);
            storageBtn.classList.add('active');
            storageBtn.classList.remove('artifact-pulse');
            this.isOpen = true;
            this.hasUnseenArtifacts = false;
            console.log('[ArtifactStorage] Dropdown opened - clearing unseen artifacts flag');
            this.loadArtifacts();
        }
    }

    closeDropdown() {
        const dropdown = document.getElementById('artifactStorageDropdown');
        const storageBtn = document.getElementById('artifactStorageBtn');

        if (dropdown && storageBtn) {
            dropdown.classList.remove('show');
            setTimeout(() => {
                dropdown.classList.add('hidden');
            }, 300);
            storageBtn.classList.remove('active');
            this.isOpen = false;
            this.hidePreview();
        }
    }

    startPolling() {
        this.checkInterval = setInterval(() => {
            this.checkNewArtifacts();
        }, 3000);
    }

    stopPolling() {
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
            this.checkInterval = null;
        }
    }

    createPreviewTooltip() {
        this.previewTooltip = document.createElement('div');
        this.previewTooltip.id = 'artifact-preview-tooltip';
        this.previewTooltip.style.cssText = `
            position: fixed;
            width: 250px;
            height: 250px;
            background-color: var(--color-bg, #111);
            border: none;
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.8);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            z-index: 999999;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            display: none;
        `;
        document.body.appendChild(this.previewTooltip);
        console.log('[ArtifactStorage] Preview tooltip created and appended to body');
    }

    showPreview(element, previewUrl, isModel = false, modelInfo = '') {
        if (!this.previewTooltip) return;

        const rect = element.getBoundingClientRect();
        const tooltipLeft = rect.left - 260;
        const tooltipTop = rect.top + (rect.height / 2) - 125;

        this.previewTooltip.style.display = 'block';
        this.previewTooltip.style.left = `${tooltipLeft}px`;
        this.previewTooltip.style.top = `${tooltipTop}px`;

        if (isModel) {
            this.previewTooltip.style.backgroundImage = 'none';
            this.previewTooltip.style.width = '200px';
            this.previewTooltip.style.height = 'auto';
            this.previewTooltip.style.padding = '1rem';
            this.previewTooltip.style.whiteSpace = 'pre-line';
            this.previewTooltip.style.fontSize = '0.85rem';
            this.previewTooltip.style.color = 'var(--color-light, white)';
            this.previewTooltip.style.lineHeight = '1.5';
            this.previewTooltip.textContent = modelInfo;
        } else {
            this.previewTooltip.style.backgroundImage = `url(${previewUrl})`;
            this.previewTooltip.style.width = '250px';
            this.previewTooltip.style.height = '250px';
            this.previewTooltip.style.padding = '0';
            this.previewTooltip.textContent = '';
        }

        setTimeout(() => {
            this.previewTooltip.style.opacity = '1';
        }, 10);
    }

    hidePreview() {
        if (!this.previewTooltip) return;

        this.previewTooltip.style.opacity = '0';
        setTimeout(() => {
            this.previewTooltip.style.display = 'none';
        }, 200);
    }
}

const artifactStorage = new ArtifactStorage();
window.artifactStorage = artifactStorage;
