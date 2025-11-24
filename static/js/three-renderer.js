class ThreeRenderer {
    constructor() {
        this.donuts = [];
        this.donutRotations = [];
        this.font = null;
        this.lineMaterial = null;
        this.connectingLineMaterial = null;
        this.waveMaterial = null;
        this.cleanup = null;
        this.resize = this.resize.bind(this);
    }

    init() {
        this.setupInteractiveCube();
        this.setupBackgroundParticles();
        window.addEventListener('resize', this.resize);
    }
    resize() {
        if (!this.camera || !this.renderer) return;
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    createDonutC3D(donutIndex) {
        const donutGroup = new THREE.Group();

        // Classic donut.c parameters
        const R1 = 1; // Inner radius
        const R2 = 2; // Outer radius
        const K2 = 5; // Distance to screen
        const screenWidth = 80;
        const screenHeight = 24;
        const K1 = screenWidth * K2 * 3 / (8 * (R1 + R2)); // Screen scale

        // ASCII characters for different luminance levels (like original)
        const luminanceChars = ".,-~:;=!*#$@";

        // Create sparse donut like the classic ASCII version (reduced for performance)
        const thetaSpacing = 0.1;
        const phiSpacing = 0.1;

        // Collect all character positions and properties for instancing
        const characterData = [];
        let instanceCount = 0;

        for (let theta = 0; theta < 2 * Math.PI; theta += thetaSpacing) {
            const cosTheta = Math.cos(theta);
            const sinTheta = Math.sin(theta);

            for (let phi = 0; phi < 2 * Math.PI; phi += phiSpacing) {
                const cosPhi = Math.cos(phi);
                const sinPhi = Math.sin(phi);

                // Calculate 3D coordinates using donut.c formulas
                const circlex = R2 + R1 * cosTheta;
                const circley = R1 * sinTheta;

                // 3D coordinates after rotation
                const A = 1 + donutIndex * 0.1;
                const B = 1 + donutIndex * 0.05;
                const cosA = Math.cos(A);
                const sinA = Math.sin(A);
                const cosB = Math.cos(B);
                const sinB = Math.sin(B);

                const x = circlex * (cosB * cosPhi + sinA * sinB * sinPhi) - circley * cosA * sinB;
                const y = circlex * (sinB * cosPhi - sinA * cosB * sinPhi) + circley * cosA * cosB;
                const z = K2 + cosA * circlex * sinPhi + circley * sinA;
                const ooz = 1 / z;

                // Calculate luminance for ASCII character selection
                const L = cosPhi * cosTheta * sinB - cosA * cosTheta * sinPhi - sinA * sinTheta +
                         cosB * (cosA * sinTheta - cosTheta * sinA * sinPhi);

                // Only render if surface is facing us (L > 0)
                if (L > 0) {
                    // 3D position in our scene with better scale for visibility
                    const scale = 5; // Larger scale for clearer donut shape
                    const posX = x * scale;
                    const posY = y * scale;
                    const posZ = z * scale - 30; // Offset to be visible

                    // Select ASCII character based on luminance
                    const N = Math.floor(L * 8);
                    const char = luminanceChars[Math.min(N, luminanceChars.length - 1)];

                    // Store character data for instancing
                    characterData.push({
                        char: char,
                        position: [posX, posY, posZ],
                        luminance: L,
                        theta: theta,
                        phi: phi
                    });

                    instanceCount++;
                }
            }
        }

        // Create instanced rendering for maximum performance
        if (instanceCount > 0) {
            this.createInstancedDonut(donutGroup, characterData, donutIndex);
        }

        return donutGroup;
    }

    // Fallback method for when font loading fails
    createSimpleDonut(donutIndex) {
        const donutGroup = new THREE.Group();

        // Create a simple torus geometry as fallback
        const geometry = new THREE.TorusGeometry(10, 4, 8, 16);
        const material = new THREE.MeshBasicMaterial({
            color: 0x888888,
            wireframe: true,
            transparent: true,
            opacity: 0.7
        });

        const torus = new THREE.Mesh(geometry, material);
        donutGroup.add(torus);

        return donutGroup;
    }

    // Create instanced donut for maximum performance
    createInstancedDonut(donutGroup, characterData, donutIndex) {
        const instanceCount = characterData.length;

        if (instanceCount === 0) return;

        // Create geometry with vertex colors for gradient effect
        const geometry = this.createColoredPlaneGeometry();

        // Create instanced mesh with maximum instances
        const instancedMesh = new THREE.InstancedMesh(geometry, null, instanceCount);
        donutGroup.add(instancedMesh);

        // Create a dummy object for matrix transformations
        const dummy = new THREE.Object3D();
        const color = new THREE.Color();

        // Set up each instance
        characterData.forEach((charData, i) => {
            // Position the character
            dummy.position.set(charData.position[0], charData.position[1], charData.position[2]);

            // Orient the character to face outward from the donut center
            const angle = Math.atan2(charData.position[2], charData.position[0]);
            dummy.rotation.y = angle + Math.PI / 2;

            // Update the instance matrix
            dummy.updateMatrix();
            instancedMesh.setMatrixAt(i, dummy.matrix);

            // Set vertex colors for gradient effect
            const intensity = Math.max(0, Math.min(1, charData.luminance / Math.sqrt(2)));
            const greyValue = intensity; // 0 to 1
            color.setRGB(greyValue, greyValue, greyValue);
            instancedMesh.setColorAt(i, color);
        });

        // Create the final material
        const material = this.createSimpleInstancedMaterial();
        instancedMesh.material = material;
        instancedMesh.instanceMatrix.needsUpdate = true;
        if (instancedMesh.instanceColor) {
            instancedMesh.instanceColor.needsUpdate = true;
        }

        // Add reference to character data for cleanup
        instancedMesh.userData.characterData = characterData;
    }

    // Create geometry with vertex colors for gradient effects
    createColoredPlaneGeometry() {
        const geometry = new THREE.PlaneGeometry(1.5, 1.5);

        // Add vertex colors for gradient effects
        const colors = [];
        const color = new THREE.Color();

        // All vertices start with white color (will be modified per instance)
        for (let i = 0; i < geometry.attributes.position.count; i++) {
            color.setRGB(1, 1, 1);
            colors.push(color.r, color.g, color.b);
        }

        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        return geometry;
    }

    createSimpleInstancedMaterial() {
        return new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide,
            vertexColors: true
        });
    }

    setupInteractiveCube() {
        const canvas = document.getElementById('interactiveCube');
        if (!canvas || typeof THREE === 'undefined') return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
        camera.position.z = 250;
        let zoomLevel = 1.0;
        const minZoom = 0.3;
        const maxZoom = 20.0;

        const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        const checkResize = () => {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            if (canvas.width !== width || canvas.height !== height) {
                renderer.setSize(width, height, false);
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
            }
        };

        const geometry = new THREE.BoxGeometry(150, 150, 150);
        const edges = new THREE.EdgesGeometry(geometry);
        this.lineMaterial = new THREE.LineBasicMaterial({ color: 0x444444 });
        const cube = new THREE.LineSegments(edges, this.lineMaterial);
        scene.add(cube);

        const donutCount = 5;
        this.donuts = [];
        this.donutRotations = [];
        let fontLoaded = false;
        let fontLoadTimeout;
        const fontLoader = new THREE.FontLoader();
        this.font = null;

        // Load font with timeout fallback
        fontLoader.load(
            'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json',
            (font) => {
                // Font loaded successfully
                this.font = font;
                fontLoaded = true;

                // Clear the timeout since font loaded
                if (fontLoadTimeout) {
                    clearTimeout(fontLoadTimeout);
                    fontLoadTimeout = null;
                }
                createDonuts();
            },
            (progress) => {
                // Loading progress
            },
            (error) => {
                console.error('Font loading failed:', error);
                fontLoaded = true; // Mark as "loaded" (failed) to prevent infinite waiting
                createDonuts(); // Create fallback donuts
            }
        );

        // Fallback timeout in case font loading hangs
        fontLoadTimeout = setTimeout(() => {
            if (!fontLoaded) {
                fontLoaded = true;
                createDonuts();
            }
        }, 3000); // 3 second timeout

        const createDonuts = () => {
            if (!fontLoaded) return;
            for (let i = 0; i < donutCount; i++) {
                const donut = this.font ? this.createDonutC3D(i) : this.createSimpleDonut(i);

                // Position randomly within the cube
                const x = (Math.random() - 0.5) * 120;
                const y = (Math.random() - 0.5) * 120;
                const z = (Math.random() - 0.5) * 120;
                donut.position.set(x, y, z);

                // Random initial rotation
                donut.rotation.x = Math.random() * Math.PI * 2;
                donut.rotation.y = Math.random() * Math.PI * 2;
                donut.rotation.z = Math.random() * Math.PI * 2;

                // Store rotation speeds for animation
                this.donutRotations.push({
                    x: (Math.random() - 0.5) * 0.02,
                    y: (Math.random() - 0.5) * 0.02,
                    z: (Math.random() - 0.5) * 0.02
                });

                scene.add(donut);
                this.donuts.push(donut);
            }
        };

        // Add memory cleanup method
        this.cleanup = () => {
            // Dispose of geometries and materials
            this.donuts.forEach(donut => {
                donut.traverse((child) => {
                    if (child.geometry) {
                        child.geometry.dispose();
                    }
                    if (child.material) {
                        if (Array.isArray(child.material)) {
                            child.material.forEach(material => material.dispose());
                        } else {
                            child.material.dispose();
                        }
                    }
                    // Dispose instanced mesh attributes
                    if (child.instanceMatrix) {
                        child.instanceMatrix.dispose();
                    }
                    if (child.instanceColor) {
                        child.instanceColor.dispose();
                    }
                });
            });

            // Clear arrays
            this.donuts.length = 0;
            this.donutRotations.length = 0;

            // Clear font reference
            this.font = null;
        };

        const maxLines = 20;
        const lineGeometry = new THREE.BufferGeometry();
        const linePositions = new Float32Array(maxLines * 2 * 3);
        lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        this.connectingLineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2 });
        const lineMesh = new THREE.LineSegments(lineGeometry, this.connectingLineMaterial);
        scene.add(lineMesh);

        let isMouseDown = false;
        let isRightClick = false;
        let mouseX = 0;
        let mouseY = 0;
        let lastMouseX = 0;
        let lastMouseY = 0;
        let rotationVelocity = { x: 0, y: 0 };
        let damping = 0.95;

        // Mouse event handlers
        const handleMouseDown = (event) => {
            isMouseDown = true;
            isRightClick = event.button === 2; // Right mouse button

            if (isRightClick) {
                canvas.style.cursor = 'grabbing';
            } else {
                canvas.style.cursor = 'grabbing';
            }

            lastMouseX = event.clientX;
            lastMouseY = event.clientY;
        };

        const handleMouseMove = (event) => {
            mouseX = event.clientX;
            mouseY = event.clientY;

            if (isMouseDown) {
                const deltaX = mouseX - lastMouseX;
                const deltaY = mouseY - lastMouseY;

                if (isRightClick) {
                    // Right-click: Drag the entire scene (camera movement)
                    const dragSpeed = 0.01;
                    rotationVelocity.x = deltaX * dragSpeed;
                    rotationVelocity.y = deltaY * dragSpeed;
                } else {
                    // Left-click: Rotate objects directly
                    const rotationSpeed = 0.005;
                    cube.rotation.y += deltaX * rotationSpeed;
                    cube.rotation.x += deltaY * rotationSpeed;

                    // Rotate lineMesh in sync with cube
                    lineMesh.rotation.y += deltaX * rotationSpeed;
                    lineMesh.rotation.x += deltaY * rotationSpeed;

                    // Rotate donuts with mouse movement
                    this.donuts.forEach((donut) => {
                        donut.rotation.y += deltaX * rotationSpeed;
                        donut.rotation.x += deltaY * rotationSpeed;

                        // Rotate all ASCII characters in the donut group
                        donut.children.forEach((charMesh) => {
                            charMesh.rotation.y += deltaX * rotationSpeed;
                            charMesh.rotation.x += deltaY * rotationSpeed;
                        });
                    });
                }

                lastMouseX = mouseX;
                lastMouseY = mouseY;
            }
        };

        const handleMouseUp = () => {
            isMouseDown = false;
            isRightClick = false;
            canvas.style.cursor = 'grab';
        };

        // Prevent context menu on right-click for dragging
        const handleContextMenu = (event) => {
            event.preventDefault();
        };

        const handleMouseLeave = () => {
            isMouseDown = false;
            canvas.style.cursor = 'grab';
        };

        const handleDoubleClick = () => {
            // Reset rotation to initial state
            cube.rotation.set(0, 0, 0);
            lineMesh.rotation.set(0, 0, 0);
            rotationVelocity.x = 0;
            rotationVelocity.y = 0;

            // Reset all donuts to random initial rotations
            this.donuts.forEach((donut, index) => {
                donut.rotation.x = Math.random() * Math.PI * 2;
                donut.rotation.y = Math.random() * Math.PI * 2;
                donut.rotation.z = Math.random() * Math.PI * 2;

                // Reset all ASCII characters in the donut group
                donut.children.forEach((charMesh) => {
                    charMesh.rotation.x = Math.random() * Math.PI * 2;
                    charMesh.rotation.y = Math.random() * Math.PI * 2;
                    charMesh.rotation.z = Math.random() * Math.PI * 2;
                });
            });
        };

        // Add event listeners
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseup', handleMouseUp);
        canvas.addEventListener('mouseleave', handleMouseLeave);
        canvas.addEventListener('dblclick', handleDoubleClick);
        canvas.addEventListener('contextmenu', handleContextMenu);

        // Add zoom functionality
        const handleWheel = (event) => {
            event.preventDefault();
            const zoomSpeed = 0.1;
            const newZoom = zoomLevel + (event.deltaY > 0 ? -zoomSpeed : zoomSpeed);
            zoomLevel = Math.max(minZoom, Math.min(maxZoom, newZoom));

            // Update camera position based on zoom level
            camera.position.z = 250 / zoomLevel;
            camera.updateProjectionMatrix();
        };
        canvas.addEventListener('wheel', handleWheel);
        canvas.style.cursor = 'grab';

        const animate = () => {
            requestAnimationFrame(animate);
            checkResize();

            // Apply velocity damping only when not dragging for smooth deceleration
            if (!isMouseDown) {
                rotationVelocity.x *= damping;
                rotationVelocity.y *= damping;

                cube.rotation.x += rotationVelocity.y;
                cube.rotation.y += rotationVelocity.x;
                lineMesh.rotation.x += rotationVelocity.y;
                lineMesh.rotation.y += rotationVelocity.x;

                // Add idle spinning animation
                const idleSpeed = 0.002;
                cube.rotation.y += idleSpeed;
                lineMesh.rotation.y += idleSpeed;
            }

            // Continue rotating donuts independently and add floating movement
            this.donuts.forEach((donut, index) => {
                const rotation = this.donutRotations[index];
                donut.rotation.x += rotation.x;
                donut.rotation.y += rotation.y;
                donut.rotation.z += rotation.z;

                const time = Date.now() * 0.001; // Convert to seconds
                const floatSpeed = 0.05; // floating speed
                const flySpeed = 0.1; // flying speed

                donut.position.x += Math.sin(time + index * 0.5) * 0.01 * floatSpeed;
                donut.position.y += Math.cos(time * 0.7 + index * 0.3) * 0.01 * floatSpeed;
                donut.position.z += Math.sin(time * 0.9 + index * 0.7) * 0.01 * flySpeed;

                // Keep donuts within bounds (adjusted for larger donut size)
                const maxDistance = 50;
                if (Math.abs(donut.position.x) > maxDistance) donut.position.x *= 0.99;
                if (Math.abs(donut.position.y) > maxDistance) donut.position.y *= 0.99;
                if (Math.abs(donut.position.z) > maxDistance) donut.position.z *= 0.99;
            });
            renderer.render(scene, camera);
        };
        animate();
    }

    setupBackgroundParticles() {
        const canvas = document.getElementById('bgParticles');
        if (!canvas || typeof THREE === 'undefined') return;

        const scene = new THREE.Scene();
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const renderer = new THREE.WebGLRenderer({ canvas: canvas });

        const material = new THREE.ShaderMaterial({
            uniforms: {
                u_time: { value: 0.0 },
                u_resolution: { value: new THREE.Vector2() },
                u_mouse: { value: new THREE.Vector2(0.5, 0.5) }, // Normalized mouse position
                u_color1: { value: new THREE.Color(0x222222) }, // Lighter Grey Waves
                u_color2: { value: new THREE.Color(0x181818) }  // Darker Grey Background
            },
            vertexShader: `void main() { gl_Position = vec4(position, 1.0); }`,
            fragmentShader: `
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform vec2 u_mouse;
                uniform vec3 u_color1;
                uniform vec3 u_color2;

                vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
                vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
                vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
                float snoise(vec2 v) {
                    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
                    vec2 i  = floor(v + dot(v, C.yy) );
                    vec2 x0 = v - i + dot(i, C.xx);
                    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                    vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1;
                    i = mod289(i);
                    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0 ));
                    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
                    m = m*m; m = m*m;
                    vec3 x = 2.0 * fract(p * C.www) - 1.0; vec3 h = abs(x) - 0.5;
                    vec3 ox = floor(x + 0.5); vec3 a0 = x - ox;
                    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
                    vec3 g; g.x  = a0.x  * x0.x  + h.x  * x0.y; g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                    return 130.0 * dot(m, g);
                }

                float fbm(vec2 p) {
                    float value = 0.0; float amplitude = 0.5;
                    for (int i = 0; i < 6; i++) {
                        value += amplitude * snoise(p); p *= 2.0; amplitude *= 0.5;
                    }
                    return value;
                }

                void main() {
                    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
                    uv.x *= u_resolution.x / u_resolution.y;

                    float time = u_time * 0.02;

                    vec2 p = uv * 3.0;

                    vec2 mouseInfluence = (u_mouse - 0.5) * 2.0;
                    mouseInfluence *= 0.3;

                    p += mouseInfluence;

                    vec2 q = vec2(fbm(p + time), fbm(p + vec2(2.3, 8.5) + time));
                    float r = fbm(p + q * 0.8);

                    r += mouseInfluence.x * 0.1 + mouseInfluence.y * 0.1;

                    float lines = fract(r * 18.0);
                    lines = smoothstep(0.48, 0.5, lines);

                    vec3 color = mix(u_color2, u_color1, lines);
                    gl_FragColor = vec4(color, 1.0);
                }
            `
        });

        const plane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
        scene.add(plane);
        this.waveMaterial = material;

        const handleMouseMove = (event) => {
            const rect = canvas.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width;
            const y = 1.0 - (event.clientY - rect.top) / rect.height; // Flip Y coordinate
            material.uniforms.u_mouse.value.set(x, y);
        };

        const handleMouseLeave = () => {
            // Reset mouse position when cursor leaves canvas
            material.uniforms.u_mouse.value.set(0.5, 0.5);
        };

        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseleave', handleMouseLeave);

        const resize = () => {
            renderer.setSize(window.innerWidth, window.innerHeight);
            material.uniforms.u_resolution.value.set(window.innerWidth, window.innerHeight);
        };
        window.addEventListener('resize', resize);
        resize();

        const animate = (time) => {
            material.uniforms.u_time.value = time * 0.001;
            renderer.render(scene, camera);
            requestAnimationFrame(animate);
        };
        animate(0);
    }

    // Theme update method for materials
    updateTheme(theme) {
        if (this.lineMaterial && this.connectingLineMaterial) {
            const lineColor = theme === 'dark' ? 0x444444 : 0xcccccc;
            this.lineMaterial.color.setHex(lineColor);
            this.connectingLineMaterial.color.setHex(0xffffff);
        }

        if (this.waveMaterial) {
            if (theme === 'dark') {
                this.waveMaterial.uniforms.u_color1.value.setHex(0x222222);
                this.waveMaterial.uniforms.u_color2.value.setHex(0x181818);
            } else {
                this.waveMaterial.uniforms.u_color1.value.setHex(0xf0f0f0);
                this.waveMaterial.uniforms.u_color2.value.setHex(0xe8e8e8);
            }
        }
    }
}