class ThemeManager {
    constructor() {
        this.app = null;
        this.currentTheme = 'dark';
        this.init();
    }

    setApp(app) {
        this.app = app;
    }

    init() {
        // Initialize theme after a short delay to ensure DOM is ready
        setTimeout(() => {
            this.initializeTheme();
        }, 100);
    }

    initializeTheme() {
        const savedTheme = localStorage.getItem('datainsight-theme') || 'dark';
        this.setTheme(savedTheme);
    }

    setTheme(theme) {
        this.currentTheme = theme;
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('datainsight-theme', theme);

        this.updateThemeIcon(theme);
        this.updateThreeJSMaterials(theme);
        this.updateChatMessages(theme);
    }

    updateThemeIcon(theme) {
        const themeIcon = document.querySelector('.icon-theme');
        if (themeIcon) {
            themeIcon.className = 'icon-theme fa-solid ' + (theme === 'dark' ? 'fa-moon' : 'fa-sun');
        }
    }

    updateThreeJSMaterials(theme) {
        // Update 3D renderer materials if available
        if (this.app && this.app.threeRenderer) {
            this.app.threeRenderer.updateTheme(theme);
        }

        // Fallback for direct material references (legacy compatibility)
        if (this.app && this.app.lineMaterial && this.app.connectingLineMaterial) {
            const isDark = theme === 'dark';
            this.app.lineMaterial.color.set(isDark ? 0x444444 : 0xdddddd);
            this.app.connectingLineMaterial.color.set(isDark ? 0xffffff : 0x000000);
        }

        if (this.app && this.app.waveMaterial) {
            if (theme === 'dark') {
                this.app.waveMaterial.uniforms.u_color1.value.set(0x222222);
                this.app.waveMaterial.uniforms.u_color2.value.set(0x181818);
            } else {
                this.app.waveMaterial.uniforms.u_color1.value.set(0xf0f0f0);
                this.app.waveMaterial.uniforms.u_color2.value.set(0xe8e8e8);
            }
        }
    }

    updateChatMessages(theme) {
        // Update existing chat messages to match new theme
        const chatMessages = document.getElementById('heroChatMessages');
        if (chatMessages) {
            const messages = chatMessages.querySelectorAll('.chat-message');
            messages.forEach(message => {
                // Force CSS recalculation for theme changes
                message.style.transition = 'all 0.3s ease';
            });
        }
    }

    setupThemeToggle() {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);

        // Provide visual feedback
        if (this.app && this.app.animationManager) {
            const themeToggle = document.getElementById('themeToggle');
            if (themeToggle) {
                this.app.animationManager.animateButton(themeToggle, { ripple: true });
            }
        }

        // Optional: Show toast notification
        if (this.app && this.app.uiManager) {
            this.app.uiManager.showElegantToast(
                `Switched to ${newTheme} theme`,
                'info'
            );
        }
    }

    getCurrentTheme() {
        return this.currentTheme;
    }

    isDarkTheme() {
        return this.currentTheme === 'dark';
    }

    isLightTheme() {
        return this.currentTheme === 'light';
    }

    // System theme detection
    detectSystemTheme() {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }
        return 'light';
    }

    // Auto theme switching based on system preference
    enableAutoTheme() {
        if (window.matchMedia) {
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

            // Set initial theme based on system preference
            const systemTheme = this.detectSystemTheme();
            this.setTheme(systemTheme);

            // Listen for changes in system theme
            mediaQuery.addListener((e) => {
                const newTheme = e.matches ? 'dark' : 'light';
                this.setTheme(newTheme);
            });
        }
    }

    // Manual theme override (disables auto theme)
    disableAutoTheme() {
        if (window.matchMedia) {
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            mediaQuery.removeListener();
        }
    }

    // Theme persistence utilities
    saveThemePreference(theme) {
        localStorage.setItem('datainsight-theme', theme);
        localStorage.setItem('datainsight-theme-manual', 'true'); // Mark as manually set
    }

    loadThemePreference() {
        const manualTheme = localStorage.getItem('datainsight-theme-manual');
        if (manualTheme === 'true') {
            // User has manually set theme, use it
            return localStorage.getItem('datainsight-theme') || 'dark';
        } else {
            // No manual preference, use system theme
            return this.detectSystemTheme();
        }
    }

    // CSS custom property updates for advanced theming
    updateCustomProperties(theme) {
        const root = document.documentElement;

        if (theme === 'dark') {
            root.style.setProperty('--theme-bg-primary', '#111111');
            root.style.setProperty('--theme-bg-secondary', '#0d0d0d');
            root.style.setProperty('--theme-text-primary', '#ffffff');
            root.style.setProperty('--theme-text-secondary', '#aaaaaa');
            root.style.setProperty('--theme-border', '#333333');
        } else {
            root.style.setProperty('--theme-bg-primary', '#ffffff');
            root.style.setProperty('--theme-bg-secondary', '#f5f5f5');
            root.style.setProperty('--theme-text-primary', '#000000');
            root.style.setProperty('--theme-text-secondary', '#555555');
            root.style.setProperty('--theme-border', '#e0e0e0');
        }
    }

    // Theme transition animations
    animateThemeTransition() {
        document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';

        // Remove transition after animation completes
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }

    // Accessibility considerations
    updateAccessibilityFeatures(theme) {
        // Update contrast ratios for better accessibility
        const highContrastMode = localStorage.getItem('datainsight-high-contrast') === 'true';

        if (highContrastMode) {
            document.body.classList.add('high-contrast');
            if (theme === 'dark') {
                document.body.classList.add('high-contrast-dark');
                document.body.classList.remove('high-contrast-light');
            } else {
                document.body.classList.add('high-contrast-light');
                document.body.classList.remove('high-contrast-dark');
            }
        }
    }

    // Theme cleanup
    cleanup() {
        // Remove event listeners and clear references
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.removeEventListener('click', this.toggleTheme);
        }

        this.app = null;
    }
}