class BlackHole {
    constructor(element) {
        this.container = document.querySelector(element);
        this.h = this.container.offsetHeight;
        this.w = this.container.offsetWidth;
        this.cw = this.w;
        this.ch = this.h;
        this.maxorbit = 255;
        this.centery = this.ch / 2;
        this.centerx = this.cw / 2;

        this.startTime = new Date().getTime();
        this.currentTime = 0;

        this.stars = [];
        this.collapse = false;
        this.expanse = false;
        this.returning = false;
        this.hasExpanded = false;

        this.init();
    }

    setDPI(canvas, dpi) {
        if (!canvas.style.width)
            canvas.style.width = canvas.width + 'px';
        if (!canvas.style.height)
            canvas.style.height = canvas.height + 'px';

        const scaleFactor = dpi / 96;
        canvas.width = Math.ceil(canvas.width * scaleFactor);
        canvas.height = Math.ceil(canvas.height * scaleFactor);
        const ctx = canvas.getContext('2d');
        ctx.scale(scaleFactor, scaleFactor);
    }

    rotate(cx, cy, x, y, angle) {
        const radians = angle;
        const cos = Math.cos(radians);
        const sin = Math.sin(radians);
        const nx = (cos * (x - cx)) + (sin * (y - cy)) + cx;
        const ny = (cos * (y - cy)) - (sin * (x - cx)) + cy;
        return [nx, ny];
    }

    createStar() {
        const rands = [];
        rands.push(Math.random() * (this.maxorbit / 2) + 1);
        rands.push(Math.random() * (this.maxorbit / 2) + this.maxorbit);

        const orbital = (rands.reduce((p, c) => p + c, 0) / rands.length);

        const star = {
            orbital: orbital,
            x: this.centerx,
            y: this.centery + orbital,
            yOrigin: this.centery + orbital,
            speed: (Math.floor(Math.random() * 2.5) + 1.5) * Math.PI / 180 * 0.3,
            rotation: 0,
            startRotation: (Math.floor(Math.random() * 360) + 1) * Math.PI / 180,
            id: this.stars.length,
            collapseBonus: 0,
            color: '',
            hoverPos: 0,
            expansePos: 0,
            prevR: 0,
            prevX: 0,
            prevY: 0,
            originalY: 0
        };

        star.collapseBonus = star.orbital - (this.maxorbit * 0.7);
        if (star.collapseBonus < 0) {
            star.collapseBonus = 0;
        }

        const baseOpacity = 1 - ((star.orbital) / 255);
        const enhancedOpacity = Math.min(baseOpacity * 1.2, 1.0);
        star.color = 'rgba(255,255,255,' + enhancedOpacity + ')';
        star.hoverPos = this.centery + (this.maxorbit / 2) + star.collapseBonus;
        star.expansePos = this.centery + (star.id % 100) * -10 + (Math.floor(Math.random() * 20) + 1);

        star.prevR = star.startRotation;
        star.prevX = star.x;
        star.prevY = star.y;
        star.originalY = star.yOrigin;

        this.stars.push(star);
    }

    drawStar(star) {
        if (!this.expanse && !this.returning) {
            star.rotation = star.startRotation + (this.currentTime * star.speed);
            if (!this.collapse) {
                if (star.y > star.yOrigin) {
                    star.y -= 2.5;
                }
                if (star.y < star.yOrigin - 4) {
                    star.y += (star.yOrigin - star.y) / 10;
                }
            } else {
                star.trail = 1;
                if (star.y > star.hoverPos) {
                    star.y -= (star.hoverPos - star.y) / -5;
                }
                if (star.y < star.hoverPos - 4) {
                    star.y += 2.5;
                }
            }
        } else if (this.expanse && !this.returning) {
            star.rotation = star.startRotation + (this.currentTime * (star.speed / 2));
            if (star.y > star.expansePos) {
                star.y -= Math.floor(star.expansePos - star.y) / -80;
            }
        } else if (this.returning) {
            star.rotation = star.startRotation + (this.currentTime * star.speed);
            if (Math.abs(star.y - star.originalY) > 2) {
                star.y += (star.originalY - star.y) / 50;
            } else {
                star.y = star.originalY;
                star.yOrigin = star.originalY;
            }
        }

        this.context.save();
        this.context.fillStyle = star.color;
        this.context.strokeStyle = star.color;
        this.context.beginPath();
        const oldPos = this.rotate(this.centerx, this.centery, star.prevX, star.prevY, -star.prevR);
        this.context.moveTo(oldPos[0], oldPos[1]);
        this.context.translate(this.centerx, this.centery);
        this.context.rotate(star.rotation);
        this.context.translate(-this.centerx, -this.centery);
        this.context.lineTo(star.x, star.y);
        this.context.stroke();
        this.context.restore();

        star.prevR = star.rotation;
        star.prevX = star.x;
        star.prevY = star.y;
    }

    triggerExpansion() {
        if (this.hasExpanded) return;

        this.collapse = false;
        this.expanse = true;
        this.returning = false;
        this.hasExpanded = true;

        const centerHover = document.querySelector('.centerHover');
        if (centerHover) {
            centerHover.classList.add('open');
        }
    }

    resetBlackHole() {
        this.hasExpanded = false;
        this.collapse = false;
        this.expanse = false;
        this.returning = true;

        const centerHover = document.querySelector('.centerHover');
        if (centerHover) {
            centerHover.classList.remove('open');
        }

        setTimeout(() => {
            this.returning = false;
        }, 3000);
    }

    loop() {
        const now = new Date().getTime();
        this.currentTime = (now - this.startTime) / 50;

        this.context.clearRect(0, 0, this.cw, this.ch);
        this.context.fillStyle = 'rgba(0,0,0,0.15)';
        this.context.fillRect(0, 0, this.cw, this.ch);

        for (let i = 0; i < this.stars.length; i++) {
            if (this.stars[i] !== undefined) {
                this.drawStar(this.stars[i]);
            }
        }

        requestAnimationFrame(() => this.loop());
    }

    init() {
        const canvas = document.createElement('canvas');
        canvas.width = this.cw;
        canvas.height = this.ch;
        this.container.appendChild(canvas);
        this.context = canvas.getContext("2d");

        this.context.globalCompositeOperation = "source-over";
        this.setDPI(canvas, 192);

        this.context.clearRect(0, 0, this.cw, this.ch);

        for (let i = 0; i < 2500; i++) {
            this.createStar();
        }

        this.loop();
    }
}

window.BlackHole = BlackHole;
