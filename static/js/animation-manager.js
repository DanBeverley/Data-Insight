class AnimationManager {
    constructor() {
        this.addGlobalTransitions();
        this.setupScrollAnimations();
        this.setupLoadingAnimations();
    }

    // Consolidated button effect (replaces addButtonEffect + addDownloadEffect)
    animateButton(button, options = {}) {
        const {
            scale = 0.95,
            duration = 100,
            showProgress = false,
            ripple = true
        } = options;

        button.style.transform = `scale(${scale})`;
        button.style.transition = 'transform 0.1s ease';

        setTimeout(() => {
            button.style.transform = 'scale(1)';
            button.style.transition = 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)';
        }, duration);

        if (ripple) {
            this.createRippleEffect(null, button);
        }

        if (showProgress) {
            this.addProgressBar(button);
        }
    }

    // Consolidated input effect (replaces addInputEffect + addInputFocus + removeInputFocus)
    animateInput(input, type = 'interact') {
        const styles = DataInsightApp.STYLES;

        switch (type) {
            case 'interact':
                input.style.transform = 'translateY(-2px)';
                input.style.boxShadow = styles.BOX_SHADOW_LIGHT;
                input.style.transition = styles.OPACITY_ANIMATION;

                setTimeout(() => {
                    input.style.transform = 'translateY(0)';
                    input.style.boxShadow = 'none';
                }, 200);
                break;

            case 'focus':
                input.style.borderColor = styles.ACCENT_GREEN;
                input.style.boxShadow = '0 0 0 2px rgba(220, 220, 220, 0.2)';
                input.style.transition = styles.OPACITY_ANIMATION;
                break;

            case 'blur':
                input.style.borderColor = 'var(--color-light-grey)';
                input.style.boxShadow = 'none';
                break;
        }
    }

    // Enhanced success glow with more options
    addSuccessGlow(element, options = {}) {
        const {
            duration = 1000,
            borderColor = DataInsightApp.STYLES.ACCENT_GREEN,
            boxShadow = DataInsightApp.STYLES.BOX_SHADOW_MEDIUM
        } = options;

        element.style.boxShadow = boxShadow;
        element.style.borderColor = borderColor;
        element.style.transition = 'all 0.3s ease';

        setTimeout(() => {
            element.style.boxShadow = 'none';
            element.style.borderColor = 'var(--color-light-grey)';
        }, duration);
    }

    // Checkbox animation (kept as specialized since it's unique)
    animateCheckbox(checkbox) {
        const checkmark = checkbox.nextElementSibling;
        if (checkmark) {
            checkmark.style.transform = 'scale(1.1)';
            checkmark.style.transition = 'transform 0.2s cubic-bezier(0.68, -0.55, 0.265, 1.55)';

            setTimeout(() => {
                checkmark.style.transform = 'scale(1)';
            }, 200);
        }
    }

    // Enhanced ripple effect
    createRippleEffect(event, element) {
        const ripple = document.createElement('div');
        ripple.className = 'ripple-effect';

        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event ? event.clientX - rect.left - size / 2 : rect.width / 2 - size / 2;
        const y = event ? event.clientY - rect.top - size / 2 : rect.height / 2 - size / 2;

        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: ${DataInsightApp.STYLES.RIPPLE_COLOR};
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
            z-index: 10;
        `;

        element.style.position = element.style.position || 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    }

    // Progress bar for downloads (extracted from addDownloadEffect)
    addProgressBar(button) {
        const progressBar = document.createElement('div');
        progressBar.className = 'download-progress';
        progressBar.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            height: 2px;
            background: var(--color-accent-green);
            transition: width 1s ease;
            width: 0%;
        `;
        button.style.position = 'relative';
        button.appendChild(progressBar);
        setTimeout(() => progressBar.style.width = '100%', 50);
        setTimeout(() => progressBar.remove(), 1500);
    }

    // Message fade in animation
    animateMessageFadeIn(message) {
        message.style.opacity = '0';
        message.style.transform = 'scale(0.95) translateY(10px)';
        message.style.animation = 'messageFadeIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards';
    }

    // Value change animation
    animateValueChange(elementId, newValue, className = '') {
        const element = document.getElementById(elementId);
        if (!element) return;

        element.style.opacity = '0.5';
        element.style.transform = 'scale(0.95)';

        setTimeout(() => {
            element.textContent = newValue;
            if (className) {
                element.className = className;
            }
            element.style.opacity = '1';
            element.style.transform = 'scale(1)';
        }, 150);
    }

    addGlobalTransitions() {
        // Animation styles are now in styles.css - no dynamic injection needed
    }

    setupScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-transition');
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        });

        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            observer.observe(el);
        });
    }

    setupLoadingAnimations() {
        const processingCircle = document.querySelector('.processing-circle');
        if (processingCircle) {
            processingCircle.style.animation = 'spin 1s linear infinite';
        }
    }

    // Convenience methods for common use cases
    success(element) {
        element.classList.add('success-state');
        this.addSuccessGlow(element);
        setTimeout(() => element.classList.remove('success-state'), 2000);
    }

    error(element) {
        element.classList.add('error-state');
        setTimeout(() => element.classList.remove('error-state'), 2000);
    }

    loading(element, enabled = true) {
        if (enabled) {
            element.classList.add('loading-state');
        } else {
            element.classList.remove('loading-state');
        }
    }
}