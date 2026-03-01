
/**
 * SAPG Promo Animation Engine
 * A lightweight, dependency-free animation framework tailored for this specific visualization.
 */

// --- Constants & Config ---
const CONFIG = {
    width: 1920,
    height: 1080,
    fps: 60,
    totalDuration: 55000, // 55 seconds total
    sceneOffsetY: -70,
};

// --- State ---
const state = {
    time: 0, // Current time in ms
    isPlaying: false,
    lastFrameTime: 0,
    sceneRoot: document.getElementById('scene-root'),
    labelsLayer: document.getElementById('labels-layer'),
    ui: {
        subtitleEn: document.getElementById('subtitle-en'),
        subtitleCn: document.getElementById('subtitle-cn'),
        progressBar: document.getElementById('progress-fill'),
        timelineContainer: document.getElementById('timeline-container')
    },
    outro: {
        active: false,
        startTime: 0,
        duration: 900
    }
};

state.sceneRoot.setAttribute('transform', `translate(0 ${CONFIG.sceneOffsetY})`);
state.labelsLayer.setAttribute('transform', `translate(0 ${CONFIG.sceneOffsetY})`);

// --- Easing Functions ---
const Easing = {
    linear: t => t,
    easeInQuad: t => t * t,
    easeOutQuad: t => t * (2 - t),
    easeInOutQuad: t => t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeOutCubic: t => (--t) * t * t + 1,
    easeOutBack: t => {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }
};

// --- Utils ---
function setSubtitle(en, cn) {
    if (state.ui.subtitleEn.textContent !== en) {
        state.ui.subtitleEn.textContent = en;
        state.ui.subtitleCn.textContent = cn;
        
        // Fade in/out container based on content
        const container = document.getElementById('subtitle-container');
        if (!en && !cn) {
            container.style.opacity = '0';
        } else {
            container.style.opacity = '1';
        }
    }
}

const _subtitleTimelineCache = {};

function _countSubtitleChars(s) {
    if (!s) return 0;
    return String(s).replace(/\s+/g, '').length;
}

function _buildSubtitleTimeline(items) {
    const MIN_MS = 4200;
    const BASE_MS = 800;
    const MS_PER_CHAR = 90;
    let t = 0;
    return items.map(it => {
        const len = _countSubtitleChars(it.en) + _countSubtitleChars(it.cn);
        const autoDur = Math.max(MIN_MS, BASE_MS + len * MS_PER_CHAR);
        const dur = (typeof it.durationMs === 'number' && Number.isFinite(it.durationMs)) ? it.durationMs : autoDur;
        const start = t;
        const end = t + dur;
        t = end;
        return { start, end, en: it.en, cn: it.cn };
    });
}

function setSubtitleAuto(key, relativeTime, items) {
    const signature = JSON.stringify(items.map(it => ({
        en: it.en,
        cn: it.cn,
        durationMs: it.durationMs
    })));

    const cached = _subtitleTimelineCache[key];
    if (!cached || cached.signature !== signature) {
        _subtitleTimelineCache[key] = {
            signature,
            timeline: _buildSubtitleTimeline(items)
        };
    }
    const tl = _subtitleTimelineCache[key].timeline;
    const t = Math.max(0, relativeTime);
    let chosen = tl[tl.length - 1];
    for (let i = 0; i < tl.length; i++) {
        if (t < tl[i].end) {
            chosen = tl[i];
            break;
        }
    }
    setSubtitle(chosen.en, chosen.cn);
}

function createSVG(type, attrs = {}) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", type);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'text') {
            el.textContent = v;
        } else if (k === 'className') {
            el.setAttribute('class', v);
        } else {
            el.setAttribute(k, v);
        }
    }
    return el;
}

function clearGroup(group) {
    while (group.firstChild) {
        group.removeChild(group.firstChild);
    }
}

function createAnnotation(x, y, textEn, textCn, direction = 'up') {
    const g = createSVG('g', { class: 'annotation-group', opacity: 0 });
    
    // Offset for text
    const dy = direction === 'up' ? -60 : 60;
    const lineY = direction === 'up' ? -10 : 10;
    
    const line = createSVG('line', {
        x1: -80, y1: dy - 20,
        x2: 80,  y2: dy - 20,
        stroke: 'var(--accent)',
        'stroke-width': 2
    });
    
    const tEn = createSVG('text', {
        x: 0, y: dy,
        class: 'label-annotation',
        style: 'font-family: var(--font-en); font-size: 16px; fill: var(--text-primary); font-weight: 600;',
        text: textEn
    });
    
    const tCn = createSVG('text', {
        x: 0, y: dy + 20,
        class: 'label-annotation',
        style: 'font-family: var(--font-cn); font-size: 14px; fill: var(--text-secondary);',
        text: textCn
    });
    
    g.appendChild(line);
    g.appendChild(tEn);
    g.appendChild(tCn);
    
    g.setAttribute('transform', `translate(${x}, ${y})`);
    
    return g;
}

// --- Animation Primitives ---

class Animator {
    constructor() {
        this.tweens = [];
        this.delayedCalls = [];
    }

    _getSvgPropStart(target, key) {
        if (!(target instanceof SVGElement)) return undefined;

        // For style-driven SVG properties, computed style is the reliable source.
        if (key === 'fill' || key === 'stroke' || key === 'font-size') {
            const cs = window.getComputedStyle(target);
            if (key === 'font-size') return cs.fontSize;
            if (key === 'fill') return cs.fill;
            if (key === 'stroke') return cs.stroke;
        }

        return target.getAttribute(key);
    }

    _setSvgProp(target, key, val) {
        if (!(target instanceof SVGElement)) return;

        if (key === 'fill') {
            target.style.fill = String(val);
            return;
        }
        if (key === 'stroke') {
            target.style.stroke = String(val);
            return;
        }
        if (key === 'font-size') {
            // Allow passing numbers (treated as px) or strings like '14px'
            if (typeof val === 'number') {
                target.style.fontSize = `${val}px`;
            } else {
                target.style.fontSize = String(val);
            }
            return;
        }

        target.setAttribute(key, val);
    }

    // Add a tween to be executed
    // target: object to animate
    // duration: ms
    // props: { x: [start, end], opacity: [0, 1] }
    // startTime: relative to scene start
    to(target, duration, props, startTime, ease = Easing.easeOutQuad) {
        const normalizedProps = {};
        for (const [key, val] of Object.entries(props)) {
            if (Array.isArray(val)) {
                normalizedProps[key] = val;
                continue;
            }

            let start;
            if (target instanceof SVGElement) {
                start = this._getSvgPropStart(target, key);
            } else {
                start = target[key];
            }

            // If numeric, normalize to number for interpolation.
            const startNum = (start !== null && start !== undefined && start !== '') ? Number(start) : NaN;
            if (!Number.isNaN(startNum) && Number.isFinite(startNum) && typeof val === 'number') {
                normalizedProps[key] = [startNum, val];
            } else {
                normalizedProps[key] = [start, val];
            }
        }

        this.tweens.push({
            target,
            duration,
            props: normalizedProps,
            startTime,
            endTime: startTime + duration,
            ease
        });
    }

    // Execute tweens based on current scene time
    update(sceneTime) {
        for (const tween of this.tweens) {
            if (sceneTime >= tween.startTime && sceneTime <= tween.endTime) {
                const progress = (sceneTime - tween.startTime) / tween.duration;
                const eased = tween.ease(progress);

                for (const [key, [start, end]] of Object.entries(tween.props)) {
                    const startNum = Number(start);
                    const endNum = Number(end);
                    const isNumeric = !Number.isNaN(startNum) && Number.isFinite(startNum) && !Number.isNaN(endNum) && Number.isFinite(endNum);

                    const val = isNumeric ? (startNum + (endNum - startNum) * eased) : (eased < 1 ? start : end);
                    if (tween.target instanceof SVGElement) {
                        this._setSvgProp(tween.target, key, val);
                    } else {
                        tween.target[key] = val;
                    }
                }
            } else if (sceneTime > tween.endTime) {
                // Ensure final state
                 for (const [key, [start, end]] of Object.entries(tween.props)) {
                    if (tween.target instanceof SVGElement) {
                        this._setSvgProp(tween.target, key, end);
                    } else {
                        tween.target[key] = end;
                    }
                }
            }
        }
    }
}

function createArrow(x1, y1, x2, y2, className = 'connector-line', showMarker = true) {
    const path = createSVG('path', {
        d: `M ${x1} ${y1} L ${x2} ${y2}`,
        class: className,
        fill: 'none'
    });
    if (!showMarker) {
        path.style.markerEnd = 'none';
    }
    return path;
}

function updateArrowPath(path, x1, y1, x2, y2) {
    path.setAttribute('d', `M ${x1} ${y1} L ${x2} ${y2}`);

    const dx = x2 - x1;
    const dy = y2 - y1;
    const len2 = dx * dx + dy * dy;
    if (len2 > 1) {
        if (path.style.markerEnd === 'none') {
            path.style.markerEnd = '';
        }
    } else {
        path.style.markerEnd = 'none';
    }
}

// --- Scenes ---

const SCENES = [
    {
        id: 'intro',
        duration: 6000,
        setup: setupIntro,
        update: updateIntro,
        cleanup: cleanupIntro
    },
    {
        id: 'ar_viz',
        duration: 12000,
        setup: setupAR,
        update: updateAR,
        cleanup: cleanupAR
    },
    {
        id: 'dag_viz',
        duration: 14000,
        setup: setupDAG,
        update: updateDAG,
        cleanup: cleanupDAG
    },
    {
        id: 'stitched_ar',
        duration: 6000,
        setup: setupStitchedAR,
        update: updateStitchedAR,
        cleanup: cleanupStitchedAR
    },
    {
        id: 'verification',
        duration: 10000,
        setup: setupVerification,
        update: updateVerification,
        cleanup: cleanupVerification
    },
    {
        id: 'speed_compare',
        duration: 7000,
        setup: setupSpeedCompare,
        update: updateSpeedCompare,
        cleanup: cleanupSpeedCompare
    }
];

let currentSceneIndex = -1;
let currentSceneObj = null; // Object to hold scene specific state

// --- Scene 1: Intro (SAPG Morph) ---

function setupIntro(scene) {
    clearGroup(state.sceneRoot);
    const sceneObj = {
        animator: new Animator(),
        elements: []
    };

    // Transition Fade In
    const fadeRect = createSVG('rect', {
        x: 0, y: -CONFIG.sceneOffsetY, width: CONFIG.width, height: CONFIG.height - CONFIG.sceneOffsetY,
        fill: 'black', opacity: 1
    });
    state.sceneRoot.appendChild(fadeRect);
    sceneObj.animator.to(fadeRect, 1000, { opacity: [1, 0] }, 0);

    // Text: "Structure-Aware Parallel Generation"
    // Target: "S A P G"
    // Indices: S(0), A(10), P(16), G(25)
    
    const fullText = "Structure-Aware Parallel Generation";
    const targetIndices = [0, 10, 16, 25];
    const letterEls = [];
    
    const group = createSVG('g', { id: 'intro-text-group' });
    state.sceneRoot.appendChild(group);

    // Initial Layout: Centered
    const startY = CONFIG.height / 2;
    const charWidth = 40; // Optimized for spacing
    const totalWidth = fullText.length * charWidth;
    const startXOffset = (CONFIG.width - totalWidth) / 2;
    
    for (let i = 0; i < fullText.length; i++) {
        const char = fullText[i];
        const el = createSVG('text', {
            x: startXOffset + (i * charWidth),
            y: startY,
            class: 'title-text',
            'font-size': '40px', // Optimized size
            'opacity': 1,
            text: char
        });
        
        if (targetIndices.includes(i)) {
            el.dataset.target = 'true';
            el.dataset.targetIndex = targetIndices.indexOf(i); 
        } else {
            el.dataset.target = 'false';
        }
        
        group.appendChild(el);
        letterEls.push(el);
    }
    
    sceneObj.elements = letterEls;

    // Animation Sequence
    // 0-1s: Hold
    // 1-3s: Fade out non-targets, Move targets to left-center
    
    // Target Layout: "S A P G" [Translation]
    // S A P G starts at x=600?
    const finalLetterSpacing = 120;
    const finalGroupStartX = 600;
    
    // Translation Text
    const transText = createSVG('text', {
        x: finalGroupStartX + (4 * finalLetterSpacing) + 240, // To the right of G
        y: startY,
        class: 'title-text',
        'font-size': '52px',
        opacity: 0,
        text: "结构感知并行生成", // "Structure-Aware Parallel Generation" in Chinese
        style: 'font-family: var(--font-cn); fill: var(--text-secondary); font-weight: 400; letter-spacing: 2px;'
    });
    group.appendChild(transText);

    letterEls.forEach((el, i) => {
        if (el.dataset.target === 'false') {
            sceneObj.animator.to(el, 1000, { opacity: [1, 0] }, 1800);
        } else {
            const targetIdx = parseInt(el.dataset.targetIndex);
            const targetX = finalGroupStartX + (targetIdx * finalLetterSpacing);
            const startX = parseFloat(el.getAttribute('x'));
            
            // Move and Scale Up
            sceneObj.animator.to(el, 2000, { x: [startX, targetX], 'font-size': [40, 150] }, 1800, Easing.easeOutBack);
        }
    });
    
    // Show Translation
    sceneObj.animator.to(transText, 1000, { opacity: [0, 1], x: [finalGroupStartX + (4 * finalLetterSpacing) + 290, finalGroupStartX + (4 * finalLetterSpacing) + 240] }, 3300, Easing.easeOutQuad);

    // Transition Fade Out
    const fadeOutRect = createSVG('rect', {
        x: 0, y: -CONFIG.sceneOffsetY, width: CONFIG.width, height: CONFIG.height - CONFIG.sceneOffsetY,
        fill: 'black', opacity: 0
    });
    group.appendChild(fadeOutRect);
    sceneObj.animator.to(fadeOutRect, 700, { opacity: [0, 1] }, 5200);

    sceneObj.onUpdate = (t) => {
        if (t > 1.9 && !sceneObj.colorSwitched) {
            sceneObj.elements.forEach(el => {
                if (el.dataset.target === 'true') {
                    el.classList.add('highlight-char');
                }
            });
            sceneObj.colorSwitched = true;
            transText.style.fill = 'var(--text-primary)'; 
        }

    };
    
    return sceneObj;
}

function updateIntro(sceneObj, relativeTime) {
    sceneObj.animator.update(relativeTime);
    if (sceneObj.onUpdate) sceneObj.onUpdate(relativeTime / 1000);

    setSubtitleAuto('intro', relativeTime, [
        { en: "Structure-Aware Parallel Generation", cn: "结构感知并行生成", durationMs: 3000 },
        { en: "SAPG: Efficient and Precise", cn: "SAPG：高效且精准", durationMs: 3000 }
    ]);
}

function cleanupIntro() {
    clearGroup(state.sceneRoot);
}

// --- Scene 2: AR Visualization ---

function setupAR() {
    clearGroup(state.sceneRoot);
    const scene = {
        animator: new Animator(),
        elements: [],
        arrows: []
    };

    // Transition Fade In
    const fadeRect = createSVG('rect', {
        x: 0, y: -CONFIG.sceneOffsetY, width: CONFIG.width, height: CONFIG.height - CONFIG.sceneOffsetY,
        fill: 'black', opacity: 1
    });
    state.sceneRoot.appendChild(fadeRect);
    scene.animator.to(fadeRect, 1000, { opacity: [1, 0] }, 0);
    
    // Container for everything that moves with camera
    const worldGroup = createSVG('g', { id: 'ar-world' });
    state.sceneRoot.appendChild(worldGroup);

    const tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."];
    const y = CONFIG.height / 2;
    const tokenWidth = 140;
    const padding = 60; // Space for arrows
    const startX = (CONFIG.width - (tokens.length * (tokenWidth + padding))) / 2 + 500; // Start right-ish so we pan
    
    const tokenEls = [];
    
    const focusIndex = 3; // "fox"
    const stepTime = 600;
    const focusTime = focusIndex * stepTime; // 1800ms
    const zoomDuration = 1000;
    const probDuration = 3000; // Time to show probabilities
    const resumeTime = focusTime + zoomDuration + probDuration; // 5800ms

    // Create Tokens
    tokens.forEach((text, i) => {
        const x = startX + i * (tokenWidth + padding);
        
        const g = createSVG('g', { opacity: 0 });
        const rect = createSVG('rect', {
            x: x, y: y - 40, width: tokenWidth, height: 80,
            class: 'token-rect'
        });
        const txt = createSVG('text', {
            x: x + tokenWidth/2, y: y,
            class: 'token-text',
            'font-size': '20px',
            text: text
        });
        
        g.appendChild(rect);
        g.appendChild(txt);
        worldGroup.appendChild(g);
        tokenEls.push({ g, x, y, rect, txt });
        
        let appearTime;
        
        if (i < focusIndex) {
            // Normal sequential appearance for first few
            appearTime = i * stepTime;
            scene.animator.to(g, 300, { opacity: [0, 1] }, appearTime);
        } else if (i === focusIndex) {
            // "fox" - Focus Token
            appearTime = focusTime; // 1800ms
            
            // 1. Show empty box first
            scene.animator.to(g, 300, { opacity: [0, 1] }, appearTime);
            txt.setAttribute('opacity', 0); // Hide text initially
            
            // 2. Show Probability Distribution (After zoom in)
            const probStartTime = appearTime + zoomDuration; // 2800ms
            
            const probGroup = createSVG('g', { opacity: 0, transform: `translate(${x}, ${y + 60})` });
            const probs = [
                { t: "fox", p: "65%", correct: true },
                { t: "dog", p: "8%" },
                { t: "wolf", p: "12%" },
                { t: "rabbit", p: "4%" },
                { t: "raccoon", p: "3%" },
                { t: "cat", p: "5%" }
            ];
            
            probs.forEach((p, pIdx) => {
                const pText = createSVG('text', {
                    x: tokenWidth/2, y: pIdx * 25,
                    class: 'label-annotation',
                    'font-size': '14px',
                    fill: 'var(--text-secondary)',
                    'text-anchor': 'middle',
                    text: `${p.t} (${p.p})`
                });
                probGroup.appendChild(pText);
            });

            // Bottom-to-top selection animation: light up words until the selected one, then stop.
            const correctIdx = probs.findIndex(p => p.correct);
            const step = 180;
            const startScan = probStartTime + 350;
            for (let pIdx = probs.length - 1; pIdx >= correctIdx; pIdx--) {
                const order = (probs.length - 1) - pIdx;
                const scanTime = startScan + order * step;
                const pText = probGroup.childNodes[pIdx];

                // Instant highlight (non-numeric fill needs near-zero duration to switch)
                scene.animator.to(pText, 1, { fill: 'var(--accent)' }, scanTime);
                scene.animator.to(pText, 120, { 'font-size': [14, 18] }, scanTime);

                if (pIdx !== correctIdx) {
                    // Quickly revert if not selected
                    scene.animator.to(pText, 1, { fill: 'var(--text-secondary)' }, scanTime + 140);
                    scene.animator.to(pText, 120, { 'font-size': [18, 14] }, scanTime + 140);
                }
            }
            worldGroup.appendChild(probGroup);
            
            // Fade in probs
            scene.animator.to(probGroup, 500, { opacity: [0, 1] }, probStartTime);
            
            // Fill Text
            const fillTime = resumeTime - 500;
            scene.animator.to(txt, 500, { opacity: [0, 1] }, fillTime);
            
            // Hide Probs
            scene.animator.to(probGroup, 500, { opacity: [1, 0] }, fillTime + 200);
            
        } else {
            // Resume generation for rest
            // i > focusIndex
            appearTime = resumeTime + (i - focusIndex - 1) * stepTime;
            scene.animator.to(g, 300, { opacity: [0, 1] }, appearTime);
        }
        
        // Arrow from prev
        if (i > 0) {
            const prev = tokenEls[i-1];
            // Use createArrow
            const arrowStartX = prev.x + tokenWidth;
            const arrowEndX = x;
            
            // Start with zero length arrow
            const line = createArrow(arrowStartX, y, arrowStartX, y, 'connector-line', false);
            line.style.opacity = 1; 
            
            worldGroup.insertBefore(line, g);
            
            // Store for animation
            const arrowObj = {
                el: line,
                sx: arrowStartX, sy: y,
                ex: arrowEndX, ey: y,
                progress: 0
            };
            scene.arrows.push(arrowObj);
            
            // Animate progress
            // For normal tokens, arrow precedes text. For focus token, arrow appears with box.
            const arrowTime = (i === focusIndex) ? appearTime : appearTime - 200;
            scene.animator.to(arrowObj, 400, { progress: [0, 1] }, arrowTime);
        }
    });
    
    // Annotations
    const annotAR = createAnnotation(startX + tokenWidth/2, y - 60, "Traditional AR", "传统自回归", "up");
    worldGroup.appendChild(annotAR);
    scene.animator.to(annotAR, 500, { opacity: [0, 1] }, 500);
    scene.animator.to(annotAR, 500, { opacity: [1, 0] }, 10000); // Extended visibility

    // Camera Logic
    // 0-1.8s: Pan to "fox"
    // 1.8s-2.8s: Zoom In
    // 2.8s-5.8s: Hold (Probs animation)
    // 5.8s+: Zoom Out & Continue
    
    const focusToken = tokenEls[focusIndex];
    const focusX = focusToken.x + tokenWidth/2;
    
    scene.camera = { x: 0, scale: 1 };
    
    // Initial Camera: Center on first token
    const initialCamX = CONFIG.width/2 - (startX + tokenWidth/2);
    // At t=1.8s, center on "fox"
    const focusCamX = CONFIG.width/2 - focusX;
    
    // Pan to "fox"
    scene.animator.to(scene.camera, focusTime, { x: [initialCamX, focusCamX] }, 0, Easing.easeInOutQuad);

    // Zoom In
    const zoomScale = 2.5;
    const zoomedCamX = (CONFIG.width/2) - (focusX * zoomScale);
    
    scene.animator.to(scene.camera, zoomDuration, { x: [focusCamX, zoomedCamX], scale: [1, zoomScale] }, focusTime, Easing.easeInOutQuad);
    
    // Zoom Out / Pull Back
    // Start pulling back when generation resumes
    const pullBackStart = resumeTime;
    const pullBackDuration = 2500;
    const finalCamX = (CONFIG.width/2) - (focusX * 0.6); // Zoom out to 0.6
    
    scene.animator.to(scene.camera, pullBackDuration, { x: [zoomedCamX, finalCamX], scale: [zoomScale, 0.6] }, pullBackStart, Easing.easeOutCubic);

    // Outro: accelerate everything to the right before fading out
    const flyOutStart = 10800;
    const flyOutDuration = 900;
    scene.animator.to(scene.camera, flyOutDuration, { x: [finalCamX, finalCamX + 2600] }, flyOutStart, Easing.easeInQuad);
    
    scene.onUpdate = (t) => {
        worldGroup.setAttribute('transform', `translate(${scene.camera.x} ${CONFIG.height/2 * (1-scene.camera.scale)}) scale(${scene.camera.scale})`);
        
        // Update arrows
        scene.arrows.forEach(arr => {
            const curX = arr.sx + (arr.ex - arr.sx) * arr.progress;
            const curY = arr.sy + (arr.ey - arr.sy) * arr.progress;
            updateArrowPath(arr.el, arr.sx, arr.sy, curX, curY);
        });
        
        // Highlight logic at pull back
        if (t > 10.5 && t < 11.5) { // Adjusted time
            // Flash arrows red/highlight
             const arrows = worldGroup.querySelectorAll('.connector-line');
             arrows.forEach(a => a.style.stroke = 'var(--accent)');
        }
    };

    // Transition Fade Out
    // No black fade between AR and DAG (DAG will draw in immediately after AR fly-out)
    
    return scene;
}

function updateAR(scene, relativeTime) {
    scene.animator.update(relativeTime);
    if (scene.onUpdate) scene.onUpdate(relativeTime / 1000);

    setSubtitleAuto('ar_viz', relativeTime, [
        { en: "Traditional LLMs generate tokens sequentially.", cn: "传统大模型按顺序逐个生成词元。", durationMs: 6000 },
        { en: "Strict dependencies slow down decoding.", cn: "严格依赖会显著拖慢推理速度。", durationMs: 6000 }
    ]);
}

function cleanupAR() {
    clearGroup(state.sceneRoot);
}

function setupSpeedCompare() {
    clearGroup(state.sceneRoot);
    const scene = {
        animator: new Animator(),
        elements: [],
        arrowsLeft: [],
        arrowsRight: []
    };

    const group = createSVG('g', { id: 'speed-compare-group' });
    state.sceneRoot.appendChild(group);

    // Soft entry to avoid a hard cut.
    scene.enter = { y: 60, opacity: 0 };
    group.setAttribute('opacity', 0);

    const topY = 110;
    const baseY = (CONFIG.height / 2) + 40;
    const leftX0 = 80;
    const rightX0 = CONFIG.width / 2 + 80;
    const laneW = CONFIG.width / 2 - 160;

    const title = (centerX, labelEn, labelCn) => {
        const g = createSVG('g', {});
        const t1 = createSVG('text', {
            x: centerX,
            y: topY,
            class: 'label-annotation',
            'font-size': '20px',
            fill: 'var(--text-primary)',
            'text-anchor': 'middle',
            text: labelEn
        });
        const t2 = createSVG('text', {
            x: centerX,
            y: topY + 24,
            class: 'label-annotation',
            'font-size': '16px',
            fill: 'var(--text-secondary)',
            'text-anchor': 'middle',
            text: labelCn
        });
        g.appendChild(t1);
        g.appendChild(t2);
        return g;
    };

    const leftCenterX = leftX0 + laneW / 2;
    const rightCenterX = rightX0 + laneW / 2;
    group.appendChild(title(leftCenterX, 'AR (token-by-token)', '自回归（逐词元）'));
    group.appendChild(title(rightCenterX, 'SAPG (DAG, layer-by-layer)', 'SAPG（DAG 分层并行）'));

    const structCallout = createAnnotation(rightCenterX, topY + 80, 'Structure prediction (fast)', '推理前预测结构（很快）', 'down');
    group.appendChild(structCallout);
    scene.animator.to(structCallout, 220, { opacity: [0, 1] }, 260, Easing.easeOutQuad);
    scene.animator.to(structCallout, 260, { opacity: [1, 0] }, 1950, Easing.easeOutQuad);

    const tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."];
    const tokenWidth = 110;
    const padding = 22;
    const tokenH = 56;
    const perRow = 5;
    const rowGap = 120;
    const leftRowW = perRow * tokenWidth + (perRow - 1) * padding;
    const leftStartX = leftX0 + (laneW - leftRowW) / 2;
    const rowY0 = baseY - rowGap / 2;
    const rowY1 = baseY + rowGap / 2;

    const leftTokens = [];
    for (let i = 0; i < tokens.length; i++) {
        const row = Math.floor(i / perRow);
        const col = i % perRow;
        const x = row === 0
            ? (leftStartX + col * (tokenWidth + padding))
            : (leftStartX + (perRow - 1 - col) * (tokenWidth + padding));
        const y = row === 0 ? rowY0 : rowY1;

        const rect = createSVG('rect', {
            x,
            y: y - tokenH / 2,
            width: tokenWidth,
            height: tokenH,
            class: 'token-rect',
            opacity: 0
        });
        const txt = createSVG('text', {
            x: x + tokenWidth / 2,
            y: y + 2,
            class: 'token-text',
            'font-size': '18px',
            opacity: 0,
            text: tokens[i]
        });
        group.appendChild(rect);
        group.appendChild(txt);
        leftTokens.push({ i, row, col, x, y, rect, txt });
    }

    // Normal-sized arrows (snake layout, with a vertical turn between rows)
    for (let i = 1; i < leftTokens.length; i++) {
        const prev = leftTokens[i - 1];
        const cur = leftTokens[i];
        let sx, sy, ex, ey;
        if (prev.row === cur.row) {
            // Same row
            if (prev.x < cur.x) {
                sx = prev.x + tokenWidth;
                ex = cur.x;
            } else {
                sx = prev.x;
                ex = cur.x + tokenWidth;
            }
            sy = prev.y;
            ey = cur.y;
        } else {
            // Turn: go down from center of previous token to center of next token
            sx = prev.x + tokenWidth / 2;
            sy = prev.y + tokenH / 2;
            ex = cur.x + tokenWidth / 2;
            ey = cur.y - tokenH / 2;
        }

        const line = createArrow(sx, sy, sx, sy, 'connector-line', true);
        line.style.opacity = 1;
        group.insertBefore(line, group.firstChild);
        scene.arrowsLeft.push({ el: line, sx, sy, ex, ey, progress: 0 });
    }

    const leftStartBase = 2050;
    const leftStep = 500;
    leftTokens.forEach((t, i) => {
        const st = leftStartBase + i * leftStep;
        scene.animator.to(t.rect, 200, { opacity: [0, 1] }, st);
        scene.animator.to(t.txt, 200, { opacity: [0, 1] }, st);
    });
    scene.arrowsLeft.forEach((a, i) => {
        scene.animator.to(a, 280, { progress: [0, 1] }, (leftStartBase + 60) + i * leftStep);
    });

    const bDefs = [
        { idx: [0, 1] },
        { idx: [2, 3, 4] },
        { idx: [5, 6, 7] },
        { idx: [8, 9] }
    ];

    const innerTokenW = 96;
    const innerTokenH = 44;
    const innerPad = 14;
    const blockPaddingX = 18;
    const blockPaddingY = 18;

    const bW = (3 * innerTokenW) + (2 * innerPad) + (2 * blockPaddingX);
    const bH = innerTokenH + (2 * blockPaddingY);
    const layerDy = 150;
    const positions = [
        { x: rightCenterX - bW / 2, y: baseY - layerDy },
        { x: rightCenterX - (bW * 2 + 60) / 2, y: baseY },
        { x: rightCenterX - (bW * 2 + 60) / 2 + bW + 60, y: baseY },
        { x: rightCenterX - bW / 2, y: baseY + layerDy }
    ];

    const blocks = [];
    bDefs.forEach((b, i) => {
        const p = positions[i];
        const rect = createSVG('rect', {
            x: p.x,
            y: p.y,
            width: bW,
            height: bH,
            rx: 8,
            class: 'block-rect',
            opacity: 0
        });
        group.appendChild(rect);

        // Internal token boxes (full DAG-style, not simplified text)
        const inner = [];
        const n = b.idx.length;
        const innerRowW = n * innerTokenW + (n - 1) * innerPad;
        const innerStartX = p.x + (bW - innerRowW) / 2;
        const innerY = p.y + bH / 2;
        for (let j = 0; j < n; j++) {
            const tokIdx = b.idx[j];
            const tx = innerStartX + j * (innerTokenW + innerPad);
            const tRect = createSVG('rect', {
                x: tx,
                y: innerY - innerTokenH / 2,
                width: innerTokenW,
                height: innerTokenH,
                class: 'token-rect',
                opacity: 0
            });
            const tTxt = createSVG('text', {
                x: tx + innerTokenW / 2,
                y: innerY + 2,
                class: 'token-text',
                'font-size': '16px',
                opacity: 0,
                text: tokens[tokIdx]
            });
            group.appendChild(tRect);
            group.appendChild(tTxt);
            inner.push({ rect: tRect, txt: tTxt });
        }

        blocks.push({ rect, inner, x: p.x, y: p.y });
    });

    const edges = [
        { s: 0, e: 1 },
        { s: 0, e: 2 },
        { s: 1, e: 3 },
        { s: 2, e: 3 }
    ];

    const skeletonEdges = [];
    edges.forEach((ed) => {
        const sb = blocks[ed.s];
        const eb = blocks[ed.e];
        const x1 = sb.x + bW / 2;
        const y1 = sb.y + bH;
        const x2 = eb.x + bW / 2;
        const y2 = eb.y;
        const line = createArrow(x1, y1, x2, y2, 'connector-line', true);
        line.style.opacity = 0;
        group.insertBefore(line, group.firstChild);
        skeletonEdges.push(line);
    });

    blocks.forEach((b, i) => {
        scene.animator.to(b.rect, 220, { opacity: [0, 1] }, 260 + i * 40, Easing.easeOutQuad);
    });
    skeletonEdges.forEach((l, i) => {
        scene.animator.to(l, 220, { opacity: [0, 0.55] }, 360 + i * 35, Easing.easeOutQuad);
        scene.animator.to(l, 260, { opacity: [0.55, 0] }, 1850, Easing.easeOutQuad);
    });

    edges.forEach((ed, i) => {
        const sb = blocks[ed.s];
        const eb = blocks[ed.e];
        const x1 = sb.x + bW / 2;
        const y1 = sb.y + bH;
        const x2 = eb.x + bW / 2;
        const y2 = eb.y;
        const line = createArrow(x1, y1, x1, y1, 'connector-line highlight', false);
        line.setAttribute('stroke-width', 3);
        line.style.opacity = 1;
        group.insertBefore(line, group.firstChild);
        scene.arrowsRight.push({ el: line, sx: x1, sy: y1, ex: x2, ey: y2, progress: 0 });
    });

    const layerStarts = [2050, 3150, 4150];
    blocks[0].inner.forEach((t, j) => {
        scene.animator.to(t.rect, 220, { opacity: [0, 1] }, layerStarts[0] + 120 + j * 80);
        scene.animator.to(t.txt, 220, { opacity: [0, 1] }, layerStarts[0] + 120 + j * 80);
    });
    scene.animator.to(scene.arrowsRight[0], 500, { progress: [0, 1] }, layerStarts[0] + 250);
    scene.animator.to(scene.arrowsRight[1], 500, { progress: [0, 1] }, layerStarts[0] + 350);

    [1, 2].forEach((bi, j) => {
        const st = layerStarts[1];
        blocks[bi].inner.forEach((t, k) => {
            scene.animator.to(t.rect, 220, { opacity: [0, 1] }, st + 120 + k * 80);
            scene.animator.to(t.txt, 220, { opacity: [0, 1] }, st + 120 + k * 80);
        });
    });
    scene.animator.to(scene.arrowsRight[2], 500, { progress: [0, 1] }, layerStarts[1] + 350);
    scene.animator.to(scene.arrowsRight[3], 500, { progress: [0, 1] }, layerStarts[1] + 450);

    blocks[3].inner.forEach((t, j) => {
        scene.animator.to(t.rect, 220, { opacity: [0, 1] }, layerStarts[2] + 120 + j * 80);
        scene.animator.to(t.txt, 220, { opacity: [0, 1] }, layerStarts[2] + 120 + j * 80);
    });

    const countY = CONFIG.height - 190;
    const countLeft = createSVG('text', {
        x: leftCenterX,
        y: countY,
        class: 'label-annotation',
        'font-size': '18px',
        fill: 'var(--text-secondary)',
        'text-anchor': 'middle',
        opacity: 0,
        text: '10 steps'
    });
    const countRight = createSVG('text', {
        x: rightCenterX,
        y: countY,
        class: 'label-annotation',
        'font-size': '18px',
        fill: 'var(--text-secondary)',
        'text-anchor': 'middle',
        opacity: 0,
        text: '3 layers'
    });
    group.appendChild(countLeft);
    group.appendChild(countRight);
    scene.animator.to(countLeft, 300, { opacity: [0, 1] }, 3400);
    scene.animator.to(countRight, 300, { opacity: [0, 1] }, 3400);

    scene.onUpdate = () => {
        group.setAttribute('opacity', scene.enter.opacity);
        group.setAttribute('transform', `translate(0 ${scene.enter.y})`);
        scene.arrowsLeft.forEach(a => {
            const curX = a.sx + (a.ex - a.sx) * a.progress;
            const curY = a.sy + (a.ey - a.sy) * a.progress;
            updateArrowPath(a.el, a.sx, a.sy, curX, curY);
        });
        scene.arrowsRight.forEach(a => {
            const curX = a.sx + (a.ex - a.sx) * a.progress;
            const curY = a.sy + (a.ey - a.sy) * a.progress;
            updateArrowPath(a.el, a.sx, a.sy, curX, curY);
        });
    };

    scene.animator.to(scene.enter, 800, { y: [60, 0], opacity: [0, 1] }, 0, Easing.easeOutCubic);

    return scene;
}

function updateSpeedCompare(scene, relativeTime) {
    scene.animator.update(relativeTime);
    if (scene.onUpdate) scene.onUpdate(relativeTime / 1000);
    setSubtitleAuto('speed_compare', relativeTime, [
        { en: "AR needs one step per token.", cn: "自回归每个词元都要一步。", durationMs: 2200 },
        { en: "SAPG first predicts the structure (very fast).", cn: "SAPG 推理前先预测结构（非常快）。", durationMs: 2800 },
        { en: "Then it generates layers in parallel — significant speed advantage.", cn: "随后分层并行生成，因此有显著的速度优势。", durationMs: 2000 }
    ]);
}

function cleanupSpeedCompare() {
    clearGroup(state.sceneRoot);
}

// --- Scene 3: DAG Transition ---

function setupDAG() {
    clearGroup(state.sceneRoot);
    const scene = {
        animator: new Animator(),
        elements: [],
        arrows: []
    };
    
    const group = createSVG('g', { id: 'dag-group' });
    state.sceneRoot.appendChild(group);
    
    // Initial State: Recreate end of AR (Linear chain, Zoomed out)
    const tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."];
    const tokenWidth = 140;
    const padding = 60;
    const y = CONFIG.height / 2;
    // Calculate total width
    const totalW = tokens.length * tokenWidth + (tokens.length-1)*padding;
    const startX = (CONFIG.width - totalW) / 2; // Centered
    
    const tokenEls = [];
    const arrowEls = [];
    
    tokens.forEach((text, i) => {
        const x = startX + i * (tokenWidth + padding);
        
        // Token
        const rect = createSVG('rect', {
            x: x, y: y - 40, width: tokenWidth, height: 80,
            class: 'token-rect',
            'data-token-index': i,
            opacity: 0
        });
        const txt = createSVG('text', {
            x: x + tokenWidth/2, y: y,
            class: 'token-text',
            'font-size': '20px',
            text: text,
            'data-token-index': i,
            opacity: 0
        });
        
        group.appendChild(rect);
        group.appendChild(txt);
        tokenEls.push({ x, y, rect, txt, text });
        
        // Arrow (Legacy AR arrows, fading out)
        if (i > 0) {
            const prevX = tokenEls[i-1].x;
            // Use createArrow for consistency
            const line = createArrow(prevX + tokenWidth, y, x, y);
            // FIX: Use setAttribute for compatibility with Animator if targeted, though here it's static initially
            // But we animate opacity later. 
            // Important: Use setAttribute 'opacity' so Animator can tween it.
            line.setAttribute('opacity', 0);
            
            group.insertBefore(line, rect);
            arrowEls.push(line);
        }
    });

    // Quick "draw on" of initial chain (no black fade): tokens + arrows appear fast.
    tokenEls.forEach((t, i) => {
        scene.animator.to(t.rect, 220, { opacity: [0, 1] }, 0);
        scene.animator.to(t.txt, 220, { opacity: [0, 1] }, 0);
    });
    arrowEls.forEach(line => {
        scene.animator.to(line, 220, { opacity: [0, 1] }, 0);
    });

    const extraEdges = [];
    for (let i = 0; i < tokenEls.length; i++) {
        for (let j = i + 1; j < Math.min(i + 4, tokenEls.length); j++) {
            if (j === i + 1) continue;
            const a = tokenEls[i];
            const b = tokenEls[j];
            const x1 = a.x + tokenWidth;
            const x2 = b.x;
            const line = createArrow(x1, y, x2, y, 'connector-line', true);
            line.style.opacity = 0;
            line.style.stroke = 'var(--text-secondary)';
            line.style.strokeWidth = '2';
            group.insertBefore(line, group.firstChild);
            extraEdges.push(line);
        }
    }
    extraEdges.forEach((line, i) => {
        scene.animator.to(line, 180, { opacity: [0, 0.35] }, 240 + i * 20);
        scene.animator.to(line, 600, { opacity: [0.35, 0] }, 1200);
    });

    // Animation 1: Disconnect (0-2s)
    arrowEls.forEach(line => {
        scene.animator.to(line, 1000, { opacity: [1, 0] }, 500);
    });
    
    const annotBreak = createAnnotation(CONFIG.width/2, y - 100, "Break Dependencies", "断开连接", "up");
    group.appendChild(annotBreak);
    scene.animator.to(annotBreak, 500, { opacity: [0, 1] }, 500);
    scene.animator.to(annotBreak, 500, { opacity: [1, 0] }, 2000);
    
    // Animation 2: Form Blocks (2-4s)
    const blockDefs = [
        { indices: [0, 1] },
        { indices: [2, 3, 4] },
        { indices: [5, 6, 7] },
        { indices: [8, 9] }
    ];
    
    const blockEls = [];
    
    blockDefs.forEach((def, bIdx) => {
        const firstT = tokenEls[def.indices[0]];
        const lastT = tokenEls[def.indices[def.indices.length-1]];
        
        const bx = firstT.x - 10;
        const by = firstT.y - 50;
        const bw = (lastT.x + tokenWidth) - firstT.x + 20;
        const bh = 100;
        
        const bRect = createSVG('rect', {
            x: bx, y: by, width: bw, height: bh,
            class: 'block-rect',
            opacity: 0
        });
        
        group.insertBefore(bRect, group.firstChild);
        scene.animator.to(bRect, 800, { opacity: [0, 1] }, 2500 + bIdx*200);
        
        blockEls.push({ rect: bRect, indices: def.indices, x: bx, y: by, w: bw, h: bh });
    });
    
    const annotBlock = createAnnotation(CONFIG.width/2, y + 100, "Pack into Blocks", "打包成块", "down");
    group.appendChild(annotBlock);
    scene.animator.to(annotBlock, 500, { opacity: [0, 1] }, 2500);
    scene.animator.to(annotBlock, 500, { opacity: [1, 0] }, 4000);

    // Animation 3: Move to DAG (4-8s)
    const centerX = CONFIG.width / 2;
    const layerH = 200;
    
    const layouts = [
        { x: centerX - blockEls[0].w/2, y: y - layerH }, // L0
        { x: centerX - (blockEls[1].w + 20 + blockEls[2].w)/2, y: y }, // L1 (left)
        { x: centerX - (blockEls[1].w + 20 + blockEls[2].w)/2 + blockEls[1].w + 20, y: y }, // L1 (right)
        { x: centerX - blockEls[3].w/2, y: y + layerH } // L2
    ];
    
    blockEls.forEach((b, i) => {
        const target = layouts[i];
        const dx = target.x - b.x;
        const dy = target.y - b.y;
        
        scene.animator.to(b.rect, 2000, { x: [b.x, target.x], y: [b.y, target.y] }, 5000, Easing.easeInOutQuad);
        
        b.indices.forEach(tIdx => {
            const t = tokenEls[tIdx];
            scene.animator.to(t.rect, 2000, { x: [t.rect.getAttribute('x')*1, t.rect.getAttribute('x')*1 + dx], y: [y - 40, y - 40 + dy] }, 5000, Easing.easeInOutQuad);
            scene.animator.to(t.txt, 2000, { x: [t.txt.getAttribute('x')*1, t.txt.getAttribute('x')*1 + dx], y: [y, y + dy] }, 5000, Easing.easeInOutQuad);
        });
    });

    // Animation 4: New Arrows (8-10s)
    const dagEdges = [
        {s: 0, e: 1}, {s: 0, e: 2},
        {s: 1, e: 3}, {s: 2, e: 3}
    ];
    
    dagEdges.forEach((edge, i) => {
        const startB = layouts[edge.s];
        const endB = layouts[edge.e];
        const startW = blockEls[edge.s].w;
        const endW = blockEls[edge.e].w;
        const startH = blockEls[edge.s].h;
        
        const x1 = startB.x + startW/2;
        const y1 = startB.y + startH;
        const x2 = endB.x + endW/2;
        const y2 = endB.y;
        
        // Use createArrow
        const line = createArrow(x1, y1, x1, y1, 'connector-line highlight', false);
        line.setAttribute('stroke-width', 3);
        line.style.opacity = 1;
        
        group.insertBefore(line, group.firstChild); // Behind
        
        const arrowObj = {
            el: line,
            sx: x1, sy: y1,
            ex: x2, ey: y2,
            progress: 0
        };
        scene.arrows.push(arrowObj);
        
        scene.animator.to(arrowObj, 500, { progress: [0, 1] }, 7500 + i*100);
    });

    // Annotation "Parallel"
    const annotPara = createAnnotation(centerX, y - 170, "Parallel Generation", "并行生成", "up");
    group.appendChild(annotPara);
    scene.animator.to(annotPara, 500, { opacity: [0, 1] }, 8500);

    // Post-layout highlight: layer-by-layer to show intra-layer independence
    // Layout is stable after blocks finish moving (~7000ms) and edges are drawn (~7900ms)
    const highlightOverlays = [];
    blockEls.forEach((b, i) => {
        const target = layouts[i];
        const overlay = createSVG('rect', {
            x: target.x,
            y: target.y,
            width: b.w,
            height: b.h,
            rx: 6,
            fill: 'rgba(56, 189, 248, 0.20)',
            stroke: 'var(--accent)',
            'stroke-width': 3,
            opacity: 0
        });
        group.appendChild(overlay);
        highlightOverlays.push(overlay);
    });

    const pulse = (overlay, start) => {
        scene.animator.to(overlay, 300, { opacity: [0, 1] }, start, Easing.easeOutQuad);
        scene.animator.to(overlay, 500, { opacity: [1, 0] }, start + 450, Easing.easeOutQuad);
    };

    // L0
    pulse(highlightOverlays[0], 9000);
    // L1 (two blocks in parallel)
    pulse(highlightOverlays[1], 9800);
    pulse(highlightOverlays[2], 9800);
    // L2
    pulse(highlightOverlays[3], 10600);

    scene.onUpdate = (t) => {
        // Update arrows
        scene.arrows.forEach(arr => {
            const curX = arr.sx + (arr.ex - arr.sx) * arr.progress;
            const curY = arr.sy + (arr.ey - arr.sy) * arr.progress;
            updateArrowPath(arr.el, arr.sx, arr.sy, curX, curY);
        });
    };

    // Transition Fade Out
    // No black fade between DAG and Verification (verification reuses this scene)

    return scene;
}

function updateDAG(scene, relativeTime) {
    scene.animator.update(relativeTime);
    if (scene.onUpdate) scene.onUpdate(relativeTime / 1000);
    setSubtitleAuto('dag_viz', relativeTime, [
        { en: "We prune unnecessary dependencies.", cn: "我们剪掉不必要的依赖。", durationMs: 4600 },
        { en: "Then we pack tokens into independent blocks.", cn: "再把词元打包成相互独立的块。", durationMs: 4600 },
        { en: "Blocks in the same layer can be generated in parallel.", cn: "同一层的块可以并行生成。", durationMs: 4800 }
    ]);

}

function cleanupDAG() {
}

function setupStitchedAR() {
    const prev = document.getElementById('stitched-ar-group');
    if (prev && prev.parentNode) prev.parentNode.removeChild(prev);
    const scene = {
        animator: new Animator(),
        elements: []
    };

    // Reuse the DAG visuals and move the camera into one large block.
    const dagGroup = document.getElementById('dag-group');
    if (!dagGroup) return scene;

    if (dagGroup.dataset.prevTransform === undefined) {
        dagGroup.dataset.prevTransform = dagGroup.getAttribute('transform') || '';
    }

    // Overlay group for annotations during this zoom-in.
    const group = createSVG('g', { id: 'stitched-ar-group' });
    state.sceneRoot.appendChild(group);
    group.setAttribute('opacity', 0);
    scene.overlayEnter = { y: 20, opacity: 0 };

    // Pick a block to zoom into (prefer a wide middle-layer block).
    const blockRects = Array.from(dagGroup.querySelectorAll('rect.block-rect'));
    let targetBlock = null;
    if (blockRects.length) {
        targetBlock = blockRects.reduce((best, r) => {
            const w = parseFloat(r.getAttribute('width')) || 0;
            const bw = best ? (parseFloat(best.getAttribute('width')) || 0) : -1;
            return w > bw ? r : best;
        }, null);
    }
    if (!targetBlock) return scene;

    const bx = parseFloat(targetBlock.getAttribute('x')) || 0;
    const by = parseFloat(targetBlock.getAttribute('y')) || 0;
    const bw = parseFloat(targetBlock.getAttribute('width')) || 1;
    const bh = parseFloat(targetBlock.getAttribute('height')) || 1;
    const blockCx = bx + bw / 2;
    const blockCy = by + bh / 2;

    // Camera state applied as a transform to the DAG group.
    scene.camera = { x: 0, y: 0, scale: 1 };

    const zoomScale = 2.2;
    const targetX = (CONFIG.width / 2) - (blockCx * zoomScale);
    const targetY = (CONFIG.height / 2) - (blockCy * zoomScale);

    // Pan + zoom in, hold, then zoom out.
    scene.animator.to(scene.camera, 900, { x: [0, targetX], y: [0, targetY], scale: [1, zoomScale] }, 0, Easing.easeInOutQuad);
    scene.animator.to(scene.camera, 900, { x: [targetX, 0], y: [targetY, 0], scale: [zoomScale, 1] }, 4400, Easing.easeInOutQuad);

    // Find tokens inside this block (by bounding box) and highlight sequentially.
    const tokenRects = Array.from(dagGroup.querySelectorAll('rect.token-rect[data-token-index]'));
    const tokenTexts = Array.from(dagGroup.querySelectorAll('text.token-text[data-token-index]'));
    const textByIdx = {};
    tokenTexts.forEach(t => {
        const idx = parseInt(t.getAttribute('data-token-index'), 10);
        if (!Number.isNaN(idx)) textByIdx[idx] = t;
    });
    const tokensInBlock = tokenRects
        .map(r => {
            const idx = parseInt(r.getAttribute('data-token-index'), 10);
            const rx = parseFloat(r.getAttribute('x')) || 0;
            const ry = parseFloat(r.getAttribute('y')) || 0;
            const rw = parseFloat(r.getAttribute('width')) || 0;
            const rh = parseFloat(r.getAttribute('height')) || 0;
            return { idx, rect: r, txt: textByIdx[idx], x: rx, y: ry, w: rw, h: rh };
        })
        .filter(t => !Number.isNaN(t.idx))
        .filter(t => {
            const cx = t.x + t.w / 2;
            const cy = t.y + t.h / 2;
            return cx >= bx && cx <= (bx + bw) && cy >= by && cy <= (by + bh);
        })
        .sort((a, b) => a.x - b.x);

    scene.tokensInBlock = tokensInBlock;

    // During zoom-in: fade out in-block token boxes/texts.
    tokensInBlock.forEach(t => {
        scene.animator.to(t.rect, 450, { opacity: [1, 0] }, 150, Easing.easeOutQuad);
        if (t.txt) scene.animator.to(t.txt, 450, { opacity: [1, 0] }, 150, Easing.easeOutQuad);
    });

    // After camera stops: sequentially reveal + bounce to show stitched AR.
    tokensInBlock.forEach((t, i) => {
        const st = 1050 + i * 520;
        scene.animator.to(t.rect, 220, { opacity: [0, 1] }, st, Easing.easeOutQuad);
        if (t.txt) scene.animator.to(t.txt, 220, { opacity: [0, 1] }, st, Easing.easeOutQuad);
        scene.animator.to(t.rect, 220, { stroke: '#22c55e', 'stroke-width': 3 }, st + 40);
        scene.animator.to(t.rect, 260, { stroke: '#94a3b8', 'stroke-width': 1 }, st + 300);
        if (t.txt) {
            scene.animator.to(t.txt, 180, { 'font-size': [20, 26] }, st + 80, Easing.easeOutQuad);
            scene.animator.to(t.txt, 220, { 'font-size': [26, 20] }, st + 300, Easing.easeInOutQuad);
        }
    });

    const annot = createAnnotation(CONFIG.width / 2, CONFIG.height - 240, "Inside a block: stitched AR", "块内：拼接式自回归", "down");
    group.appendChild(annot);
    scene.animator.to(annot, 500, { opacity: [0, 1] }, 300);
    scene.animator.to(annot, 500, { opacity: [1, 0] }, 5200);

    const kv = createSVG('text', {
        x: CONFIG.width - 200,
        y: 120,
        class: 'label-annotation',
        'font-size': '18px',
        fill: 'var(--text-secondary)',
        'text-anchor': 'middle',
        opacity: 0,
        text: "KV-cache reuse"
    });
    group.appendChild(kv);
    scene.animator.to(kv, 300, { opacity: [0, 1] }, 900);
    scene.animator.to(kv, 300, { opacity: [1, 0] }, 5200);

    scene.animator.to(scene.overlayEnter, 450, { y: [20, 0], opacity: [0, 1] }, 0, Easing.easeOutCubic);

    scene.onUpdate = () => {
        group.setAttribute('opacity', scene.overlayEnter.opacity);
        group.setAttribute('transform', `translate(0 ${scene.overlayEnter.y})`);
        dagGroup.setAttribute('transform', `translate(${scene.camera.x} ${scene.camera.y}) scale(${scene.camera.scale})`);
    };

    return scene;
}

function updateStitchedAR(scene, relativeTime) {
    scene.animator.update(relativeTime);
    if (scene.onUpdate) scene.onUpdate(relativeTime / 1000);
    setSubtitleAuto('stitched_ar', relativeTime, [
        { en: "Blocks are parallel, but tokens inside a block are locally sequential.", cn: "块之间并行，但块内仍是局部串行。", durationMs: 3000 },
        { en: "This keeps quality and reduces steps — KV-cache further accelerates.", cn: "既保证质量，又减少步数——KV-cache 进一步加速。", durationMs: 3000 }
    ]);
}

function cleanupStitchedAR() {
    const g = document.getElementById('stitched-ar-group');
    if (g && g.parentNode) g.parentNode.removeChild(g);
    const dagGroup = document.getElementById('dag-group');
    if (dagGroup) {
        const prevTransform = dagGroup.dataset.prevTransform;
        if (prevTransform !== undefined) {
            dagGroup.setAttribute('transform', prevTransform);
            delete dagGroup.dataset.prevTransform;
        } else {
            dagGroup.setAttribute('transform', '');
        }
    }

    // Ensure any temporarily hidden in-block tokens are restored.
    if (currentSceneObj && currentSceneObj.tokensInBlock) {
        currentSceneObj.tokensInBlock.forEach(t => {
            if (t.rect) t.rect.setAttribute('opacity', 1);
            if (t.txt) t.txt.setAttribute('opacity', 1);
        });
    }
}

// --- Scene 4: Verification ---

function setupVerification() {
    const scene = {
        animator: new Animator(),
        elements: []
    };

    // Reuse previous DAG scene visuals
    let group = document.getElementById('dag-group');
    if (!group) {
        // If user seeks/jumps directly here, DAG may not exist. Build it to a stable final state.
        const dagScene = setupDAG();
        dagScene.animator.update(14000);
        if (dagScene.onUpdate) dagScene.onUpdate(14);
        group = document.getElementById('dag-group');
    }
    if (!group) {
        return scene;
    }

    // Ensure visibility when entering (important for seek/replay).
    group.setAttribute('opacity', 1);
    const tokenEls = [];
    const tokenRects = group.querySelectorAll('rect.token-rect[data-token-index]');
    tokenRects.forEach(r => {
        const idx = parseInt(r.getAttribute('data-token-index'), 10);
        tokenEls[idx] = tokenEls[idx] || {};
        tokenEls[idx].rect = r;
        tokenEls[idx].x = parseFloat(r.getAttribute('x'));
        tokenEls[idx].y = parseFloat(r.getAttribute('y')) + 40;
    });
    const tokenTexts = group.querySelectorAll('text.token-text[data-token-index]');
    tokenTexts.forEach(t => {
        const idx = parseInt(t.getAttribute('data-token-index'), 10);
        tokenEls[idx] = tokenEls[idx] || {};
        tokenEls[idx].txt = t;
    });

    // Inject errors into existing text (3: fox -> cat, 7: lazy -> blue)
    scene.animator.delayedCalls.push({
        time: 0,
        callback: () => {
            if (tokenEls[3]?.txt) tokenEls[3].txt.textContent = 'cat';
            if (tokenEls[7]?.txt) tokenEls[7].txt.textContent = 'blue';
        }
    });

    const y = CONFIG.height / 2;
    const centerX = CONFIG.width / 2;
    const tokenWidth = 140;

    // Animation Sequence
    
    // 1. First pass output (0-3s): a couple tokens may be off
    const errorIndices = [3, 7];
    const errorEls = errorIndices.map(i => tokenEls[i]);
    
    errorEls.forEach(el => {
        scene.animator.to(el.rect, 500, { stroke: '#f59e0b', 'stroke-width': 3 }, 500);
        scene.animator.to(el.rect, 500, { stroke: '#94a3b8', 'stroke-width': 1 }, 2200);
    });
    
    const annotRound1 = createAnnotation(centerX, y + 80, "Round 1 (parallel)", "第1轮（并行）", "down");
    const errorTarget = tokenEls[7];
    const errorAnchorX = errorTarget?.x !== undefined ? (errorTarget.x + tokenWidth / 2) : centerX;
    const errorAnchorY = errorTarget?.y !== undefined ? (errorTarget.y + 60) : (y + 80);
    annotRound1.setAttribute('transform', `translate(${errorAnchorX}, ${errorAnchorY})`);
    
    group.appendChild(annotRound1);
    scene.animator.to(annotRound1, 500, { opacity: [0, 1] }, 700);
    scene.animator.to(annotRound1, 500, { opacity: [1, 0] }, 2800);
    
    // 2. Refinement round (3-6s): overwrite a few tokens
    const correctTexts = { 3: "fox", 7: "lazy" };
    scene.animator.delayedCalls.push({
        time: 3300,
        callback: () => {
            errorIndices.forEach(i => {
                if (tokenEls[i]?.txt) tokenEls[i].txt.textContent = correctTexts[i];
            });
        }
    });

    errorEls.forEach(el => {
        scene.animator.to(el.txt, 180, { 'font-size': [20, 26] }, 3350, Easing.easeOutQuad);
        scene.animator.to(el.txt, 220, { 'font-size': [26, 20] }, 3550, Easing.easeInOutQuad);
    });
    
    errorEls.forEach(el => {
        scene.animator.to(el.rect, 600, { stroke: '#22c55e', 'stroke-width': 3 }, 3300);
        scene.animator.to(el.rect, 600, { stroke: '#94a3b8', 'stroke-width': 1 }, 5200);
    });
    
    const annotRound2 = createAnnotation(centerX, y + 80, "Round 2 (refine)", "第2轮（修正）", "down");
    annotRound2.setAttribute('transform', `translate(${errorAnchorX}, ${errorAnchorY})`);
    
    group.appendChild(annotRound2);
    scene.animator.to(annotRound2, 500, { opacity: [0, 1] }, 3400);
    scene.animator.to(annotRound2, 500, { opacity: [1, 0] }, 6000);

    const blockRects = Array.from(group.querySelectorAll('rect.block-rect'));
    blockRects.forEach(b => {
        scene.animator.to(b, 500, { stroke: '#22c55e', 'stroke-width': 3 }, 6200);
        scene.animator.to(b, 600, { stroke: '#94a3b8', 'stroke-width': 2 }, 7600);
    });

    const annotRound3 = createAnnotation(centerX, y + 80, "Round 3 (stable)", "第3轮（稳定）", "down");
    annotRound3.setAttribute('transform', `translate(${errorAnchorX}, ${errorAnchorY})`);
    group.appendChild(annotRound3);
    scene.animator.to(annotRound3, 500, { opacity: [0, 1] }, 6400);
    scene.animator.to(annotRound3, 500, { opacity: [1, 0] }, 8200);

    // Exit fade-out to avoid a hard cut into the next scene.
    // Verification duration is 10000ms; start fading close to the end.
    scene.animator.to(group, 800, { opacity: [1, 0] }, 8800, Easing.easeOutQuad);

    return scene;
}

function updateVerification(scene, relativeTime) {
    scene.animator.update(relativeTime);
    
    // Execute delayed calls
    if (scene.animator.delayedCalls) {
        scene.animator.delayedCalls = scene.animator.delayedCalls.filter(call => {
            if (relativeTime >= call.time) {
                call.callback();
                return false;
            }
            return true;
        });
    }

    if (scene.onUpdate) scene.onUpdate(relativeTime / 1000);
    
    setSubtitleAuto('verification', relativeTime, [
        { en: "Round 1 predicts blocks in parallel.", cn: "第1轮并行预测各个块。", durationMs: 5000 },
        { en: "Later rounds refine a few tokens to stabilize quality.", cn: "后续轮次只修正少量词元，很快稳定质量。", durationMs: 5000 }
    ]);
}

function cleanupVerification() {
    clearGroup(state.sceneRoot);
}



// --- Main Loop ---

function loadScene(index) {
    if (
        currentSceneObj &&
        currentSceneIndex >= 0 &&
        currentSceneIndex < SCENES.length &&
        SCENES[currentSceneIndex].cleanup
    ) {
        SCENES[currentSceneIndex].cleanup();
    }
    
    currentSceneIndex = index;
    const sceneDef = SCENES[index];
    currentSceneObj = sceneDef.setup();
    
    // Update markers?
}

function tick(timestamp) {
    if (!state.lastFrameTime) state.lastFrameTime = timestamp;
    const dt = timestamp - state.lastFrameTime;
    state.lastFrameTime = timestamp;

    if (state.isPlaying) {
        state.time += dt;
        if (state.time >= CONFIG.totalDuration) {
            state.time = CONFIG.totalDuration;
            state.isPlaying = false;
            state.outro.active = true;
            state.outro.startTime = timestamp;
        }
    }

    // Global outro: fade everything out, then remain stopped.
    if (state.outro.active) {
        const p = Math.max(0, Math.min(1, (timestamp - state.outro.startTime) / state.outro.duration));
        const alpha = 1 - p;
        state.sceneRoot.setAttribute('opacity', alpha);
        state.labelsLayer.setAttribute('opacity', alpha);
        const uiLayer = document.getElementById('ui-layer');
        if (uiLayer) uiLayer.style.opacity = String(alpha);
        if (p >= 1) {
            state.outro.active = false;
        }
    }

    // Determine current scene
    let accumulatedTime = 0;
    let activeSceneIdx = -1;
    let sceneRelativeTime = 0;

    for (let i = 0; i < SCENES.length; i++) {
        const s = SCENES[i];
        if (state.time >= accumulatedTime && state.time < accumulatedTime + s.duration) {
            activeSceneIdx = i;
            sceneRelativeTime = state.time - accumulatedTime;
            break;
        }
        accumulatedTime += s.duration;
    }

    if (activeSceneIdx !== -1) {
        if (activeSceneIdx !== currentSceneIndex) {
            loadScene(activeSceneIdx);
        }
        SCENES[activeSceneIdx].update(currentSceneObj, sceneRelativeTime);
    }

    // Update UI
    const progress = state.time / CONFIG.totalDuration;
    if (state.ui.progressBar) {
        state.ui.progressBar.style.width = `${progress * 100}%`;
    }

    requestAnimationFrame(tick);
}

// --- Init ---

function init() {
    // Generate timeline markers
    let acc = 0;
    SCENES.forEach((s, i) => {
        const marker = document.createElement('div');
        marker.style.position = 'absolute';
        marker.style.left = `${(acc / CONFIG.totalDuration) * 100}%`;
        marker.style.height = '100%';
        marker.style.width = '2px';
        marker.style.background = 'rgba(255,255,255,0.2)';
        state.ui.timelineContainer.appendChild(marker);
        acc += s.duration;
    });

    // Interaction: Progress Bar Seek
    state.ui.timelineContainer.addEventListener('click', (e) => {
        const rect = state.ui.timelineContainer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const p = x / rect.width;
        const clamped = Math.max(0, Math.min(1, p));
        state.time = clamped * CONFIG.totalDuration;
        if (state.ui.progressBar) {
            state.ui.progressBar.style.width = `${clamped * 100}%`;
        }
        // Force reload of current scene to ensure clean state
        currentSceneIndex = -1;
        currentSceneObj = null;
        state.lastFrameTime = 0;
    });

    // Space: toggle play/pause (start on demand)
    window.addEventListener('keydown', (e) => {
        if (e.code !== 'Space') return;
        e.preventDefault();
        if (state.time >= CONFIG.totalDuration) return;
        state.isPlaying = !state.isPlaying;
        state.lastFrameTime = 0;
    });

    requestAnimationFrame(tick);
}

init();
