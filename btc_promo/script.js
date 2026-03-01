
/**
 * BTC Promo Animation Engine
 * Based on SAPG Promo Engine
 */

// --- Constants & Config ---
const CONFIG = {
    width: 1920,
    height: 1080,
    fps: 60,
    sceneOffsetY: 0, 
};

// --- State ---
const state = {
    time: 0,
    isPlaying: false,
    lastFrameTime: 0,
    sceneRoot: document.getElementById('scene-root'),
    labelsLayer: document.getElementById('labels-layer'),       
    ui: {
        subtitleEn: document.getElementById('subtitle-en'),
        subtitleCn: document.getElementById('subtitle-cn'),
        progressBar: document.getElementById('progress-fill'),
        timelineContainer: document.getElementById('timeline-container')
    }
};

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

// Simple Seeded Random
class Random {
    constructor(seed) {
        this.seed = seed || Date.now();
    }
    
    // Returns 0...1
    next() {
        this.seed = (this.seed * 9301 + 49297) % 233280;
        return this.seed / 233280;
    }
    
    // Returns min...max
    range(min, max) {
        return min + this.next() * (max - min);
    }
}

function setupLogo() {
    clearGroup(state.sceneRoot);
    clearGroup(state.labelsLayer);

    const sceneObj = { animator: new Animator(), camera: { x: 0, y: 0, scale: 1 } };
    const cameraGroup = createSVG('g', { id: 'logo-camera-group' });
    state.sceneRoot.appendChild(cameraGroup);

    const g = createSVG('g', { id: 'logo-group', opacity: 0 });
    cameraGroup.appendChild(g);

    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;

    const imgW = 1200;
    const imgH = 220;
    const img = createSVG('image', {
        href: 'logo.png',
        x: cx - imgW / 2,
        y: cy - imgH / 2,
        width: imgW,
        height: imgH,
        opacity: 1
    });
    g.appendChild(img);

    sceneObj.animator.to(g, 500, { opacity: [0, 1] }, 0, Easing.easeOutQuad);
    sceneObj.animator.to(sceneObj.camera, 1400, { scale: [0.98, 1.02] }, 200, Easing.easeInOutQuad);
    sceneObj.animator.to(g, 500, { opacity: [1, 0] }, 2000, Easing.easeOutQuad);

    sceneObj.onUpdate = () => {
        const s = sceneObj.camera.scale;
        cameraGroup.setAttribute('transform', `translate(${cx}, ${cy}) scale(${s}) translate(${-cx}, ${-cy})`);
    };

    return sceneObj;
}

function updateLogo(sceneObj, t) {
    sceneObj.animator.update(t);
    if (sceneObj.onUpdate) sceneObj.onUpdate(t);
    setSubtitle('', '');
}

function setupEmailScene() {
    clearGroup(state.sceneRoot);
    clearGroup(state.labelsLayer);

    const sceneObj = { animator: new Animator() };
    const g = createSVG('g', { id: 'email-scene-group', opacity: 0 });
    state.sceneRoot.appendChild(g);

    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;

    const cardW = 980;
    const cardH = 720;
    const x0 = cx - cardW / 2;
    const y0 = cy - cardH / 2;

    const container = createSVG('g', { opacity: 1 });
    g.appendChild(container);

    const outer = createSVG('rect', { x: x0, y: y0, width: cardW, height: cardH, rx: 18, fill: '#ffffff', opacity: 0.98 });
    container.appendChild(outer);

    const headerH = 110;
    const header = createSVG('g', { opacity: 0 });
    container.appendChild(header);
    header.appendChild(createSVG('rect', { x: x0, y: y0, width: cardW, height: headerH, rx: 18, fill: '#f8f9fa' }));
    header.appendChild(createSVG('rect', { x: x0, y: y0 + headerH - 1, width: cardW, height: 1, fill: '#dee2e6' }));
    header.appendChild(createSVG('text', {
        x: x0 + 28,
        y: y0 + 46,
        fill: '#495057',
        'font-size': '28px',
        'font-weight': '700',
        text: 'BTC 期货智能决策系统',
        'dominant-baseline': 'middle'
    }));
    header.appendChild(createSVG('text', {
        x: x0 + 28,
        y: y0 + 80,
        fill: '#6c757d',
        'font-size': '18px',
        'font-weight': '500',
        text: '自动交易决策通知',
        'dominant-baseline': 'middle'
    }));

    const contentY = y0 + headerH + 26;
    const content = createSVG('g', { opacity: 1 });
    container.appendChild(content);

    const decisionCard = createSVG('g', { opacity: 0 });
    content.appendChild(decisionCard);
    decisionCard.appendChild(createSVG('rect', {
        x: x0 + 28,
        y: contentY,
        width: cardW - 56,
        height: 120,
        rx: 14,
        fill: '#ffffff',
        stroke: '#e9ecef',
        'stroke-width': 2
    }));
    decisionCard.appendChild(createSVG('rect', {
        x: x0 + 50,
        y: contentY + 34,
        width: 120,
        height: 44,
        rx: 22,
        fill: '#28a745'
    }));
    decisionCard.appendChild(createSVG('text', {
        x: x0 + 110,
        y: contentY + 56,
        fill: '#ffffff',
        'font-size': '20px',
        'font-weight': '800',
        'text-anchor': 'middle',
        'dominant-baseline': 'middle',
        text: 'LONG'
    }));
    decisionCard.appendChild(createSVG('text', {
        x: x0 + cardW - 70,
        y: contentY + 56,
        fill: '#6c757d',
        'font-size': '18px',
        'font-weight': '600',
        'text-anchor': 'end',
        'dominant-baseline': 'middle',
        text: '✅ 执行成功'
    }));
    decisionCard.appendChild(createSVG('text', {
        x: x0 + 50,
        y: contentY + 96,
        fill: '#adb5bd',
        'font-size': '16px',
        'font-weight': '500',
        text: '2026-02-07 15:18:00 UTC',
        'dominant-baseline': 'middle'
    }));

    const grid = createSVG('g', { opacity: 0 });
    content.appendChild(grid);
    const gridY = contentY + 140;
    const cellW = (cardW - 56 - 16) / 2;
    const cellH = 92;
    function infoCard(x, y, title, value) {
        const gg = createSVG('g', {});
        gg.appendChild(createSVG('rect', {
            x, y, width: cellW, height: cellH, rx: 12,
            fill: '#f8f9fa', stroke: '#e9ecef', 'stroke-width': 2
        }));
        gg.appendChild(createSVG('text', {
            x: x + 18, y: y + 30,
            fill: '#495057', 'font-size': '18px', 'font-weight': '700',
            text: title, 'dominant-baseline': 'middle'
        }));
        gg.appendChild(createSVG('text', {
            x: x + 18, y: y + 64,
            fill: '#6c757d', 'font-size': '18px', 'font-weight': '600',
            text: value, 'dominant-baseline': 'middle'
        }));
        return gg;
    }
    grid.appendChild(infoCard(x0 + 28, gridY, 'Entry Price', '$45,230.50'));
    grid.appendChild(infoCard(x0 + 28 + cellW + 16, gridY, 'Leverage', '5x Cross'));
    grid.appendChild(infoCard(x0 + 28, gridY + cellH + 16, 'Stop Loss', '-1.2%'));
    grid.appendChild(infoCard(x0 + 28 + cellW + 16, gridY + cellH + 16, 'Take Profit', '+2.8%'));

    const reasoning = createSVG('g', { opacity: 0 });
    content.appendChild(reasoning);
    const rY = gridY + cellH * 2 + 16 * 2 + 12;
    reasoning.appendChild(createSVG('text', {
        x: x0 + 28,
        y: rY,
        fill: '#495057',
        'font-size': '20px',
        'font-weight': '800',
        text: 'Reasoning / 决策依据',
        'dominant-baseline': 'middle'
    }));
    reasoning.appendChild(createSVG('rect', {
        x: x0 + 28,
        y: rY + 18,
        width: cardW - 56,
        height: 130,
        rx: 12,
        fill: '#ffffff',
        stroke: '#e9ecef',
        'stroke-width': 2
    }));
    reasoning.appendChild(createSVG('text', {
        x: x0 + 48,
        y: rY + 52,
        fill: '#6c757d',
        'font-size': '18px',
        'font-weight': '500',
        text: 'Quant signals align with momentum + Bollinger expansion; news sentiment confirms trend.',
        'dominant-baseline': 'middle'
    }));
    reasoning.appendChild(createSVG('text', {
        x: x0 + 48,
        y: rY + 84,
        fill: '#6c757d',
        'font-size': '18px',
        'font-weight': '500',
        text: '量化信号与波动突破一致，新闻情绪同步增强。',
        'dominant-baseline': 'middle',
        'font-family': 'var(--font-cn)'
    }));

    const footer = createSVG('g', { opacity: 0 });
    container.appendChild(footer);
    footer.appendChild(createSVG('rect', {
        x: x0,
        y: y0 + cardH - 72,
        width: cardW,
        height: 72,
        rx: 18,
        fill: '#f8f9fa'
    }));
    footer.appendChild(createSVG('rect', {
        x: x0,
        y: y0 + cardH - 72,
        width: cardW,
        height: 1,
        fill: '#dee2e6'
    }));
    footer.appendChild(createSVG('text', {
        x: cx,
        y: y0 + cardH - 36,
        fill: '#6c757d',
        'font-size': '16px',
        'font-weight': '600',
        'text-anchor': 'middle',
        'dominant-baseline': 'middle',
        text: 'Generated by BTC Trading Bot • Do not reply'
    }));

    sceneObj.animator.to(g, 800, { opacity: [0, 1] }, 0, Easing.easeOutQuad);
    sceneObj.animator.to(header, 600, { opacity: [0, 1] }, 400, Easing.easeOutQuad);
    sceneObj.animator.to(decisionCard, 600, { opacity: [0, 1] }, 1100, Easing.easeOutQuad);
    sceneObj.animator.to(grid, 600, { opacity: [0, 1] }, 2000, Easing.easeOutQuad);
    sceneObj.animator.to(reasoning, 600, { opacity: [0, 1] }, 2900, Easing.easeOutQuad);
    sceneObj.animator.to(footer, 600, { opacity: [0, 1] }, 3800, Easing.easeOutQuad);
    sceneObj.animator.to(g, 1000, { opacity: [1, 0] }, 13000, Easing.easeOutQuad);

    return sceneObj;
}

function updateEmailScene(sceneObj, t) {
    sceneObj.animator.update(t);
    setSubtitle("Instant notification via Email.", "决策结果可通过邮件实时推送。");
}

// Catmull-Rom Spline implementation for smooth curves
function catmullRomSpline(data, k = 0.5) {
    if (data.length < 2) return "";
    
    // Filter invalid points first
    const validData = data.filter(p => 
        p && Number.isFinite(p.x) && Number.isFinite(p.y)
    );
    
    if (validData.length < 2) return "";
    
    // Flatten data to [x, y, x, y...]
    const pts = [];
    validData.forEach(p => { pts.push(p.x); pts.push(p.y); });
    
    const size = pts.length;
    let last = size - 2;
    let path = `M ${pts[0]} ${pts[1]}`;

    for (let i = 0; i < size - 2; i += 2) {
        const x0 = i ? pts[i - 2] : pts[0];
        const y0 = i ? pts[i - 1] : pts[1];

        const x1 = pts[i + 0];
        const y1 = pts[i + 1];

        const x2 = pts[i + 2];
        const y2 = pts[i + 3];

        const x3 = i !== last ? pts[i + 4] : x2;
        const y3 = i !== last ? pts[i + 5] : y2;

        const cp1x = x1 + (x2 - x0) / 6 * k;
        const cp1y = y1 + (y2 - y0) / 6 * k;

        const cp2x = x2 - (x3 - x1) / 6 * k;
        const cp2y = y2 - (y3 - y1) / 6 * k;

        path += ` C ${cp1x} ${cp1y} ${cp2x} ${cp2y} ${x2} ${y2}`;
    }
    return path;
}

// Technical Indicators
function calculateSMA(data, period) {
    const results = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            results.push(null);
            continue;
        }
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += data[i - j].close;
        }
        results.push(sum / period);
    }
    return results;
}

function calculateBollinger(data, period = 20, multiplier = 2) {
    const sma = calculateSMA(data, period);
    const bands = [];
    
    for (let i = 0; i < data.length; i++) {
        if (sma[i] === null) {
            bands.push({ upper: null, lower: null, middle: null });
            continue;
        }
        
        let sumSqDiff = 0;
        const avg = sma[i];
        for (let j = 0; j < period; j++) {
            const diff = data[i - j].close - avg;
            sumSqDiff += diff * diff;
        }
        const stdDev = Math.sqrt(sumSqDiff / period);
        
        bands.push({
            middle: avg,
            upper: avg + stdDev * multiplier,
            lower: avg - stdDev * multiplier
        });
    }
    return bands;
}

function calculateRSI(data, period = 14) {
    const results = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period) {
            results.push(null);
            continue;
        }
        
        let gains = 0;
        let losses = 0;
        
        for (let j = 0; j < period; j++) {
            const change = data[i - j].close - data[i - j].open; // Or close - prevClose? Standard is close - prevClose.
            // But our data doesn't guarantee prevClose = currentOpen. 
            // Let's use (close - open) for simplicity or calculate close-prevClose.
            // Better: use close - data[i-j-1].close.
            // We need i-j-1.
        }
        
        // Simplified RSI for Promo (Smoother):
        // Calculate based on close price changes over period
        let avgGain = 0;
        let avgLoss = 0;
        
        for (let j = 0; j < period; j++) {
            const curr = data[i - j].close;
            const prev = data[i - j - 1] ? data[i - j - 1].close : data[i - j].open;
            const change = curr - prev;
            
            if (change > 0) avgGain += change;
            else avgLoss += Math.abs(change);
        }
        
        avgGain /= period;
        avgLoss /= period;
        
        if (avgLoss === 0) {
            results.push(100);
        } else {
            const rs = avgGain / avgLoss;
            results.push(100 - (100 / (1 + rs)));
        }
    }
    return results;
}

function setSubtitle(en, cn) {
    if (state.ui.subtitleEn.textContent !== en) {
        state.ui.subtitleEn.textContent = en;
        state.ui.subtitleCn.textContent = cn;
        
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
    const MIN_MS = 2000;
    const BASE_MS = 500;
    const MS_PER_CHAR = 60;
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
    let chosen = { en: "", cn: "" }; // Default empty
    
    // Find active subtitle
    for (let i = 0; i < tl.length; i++) {
        if (t >= tl[i].start && t < tl[i].end) {
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

// --- Animation Primitives ---

class Animator {
    constructor() {
        this.tweens = [];
    }

    _getSvgPropStart(target, key) {
        if (!(target instanceof SVGElement)) return undefined;
        if (key === 'fill' || key === 'stroke' || key === 'font-size' || key === 'opacity') {
            const cs = window.getComputedStyle(target);
            if (key === 'font-size') return parseFloat(cs.fontSize); // Always work with numbers for font-size if possible
            if (key === 'opacity') return parseFloat(cs.opacity);
            if (key === 'fill') return cs.fill;
            if (key === 'stroke') return cs.stroke;
        }
        return target.getAttribute(key);
    }

    _setSvgProp(target, key, val) {
        if (!(target instanceof SVGElement)) return;

        if (key === 'fill') { target.style.fill = String(val); return; }
        if (key === 'stroke') { target.style.stroke = String(val); return; }
        if (key === 'opacity') { target.style.opacity = String(val); return; }
        if (key === 'font-size') {
            target.style.fontSize = (typeof val === 'number') ? `${val}px` : String(val);
            return;
        }
        if (key === 'translateX') {
            target.setAttribute('transform', `translate(${val}, 0)`);
            return;
        }
        target.setAttribute(key, val);
    }

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

            const startNum = (start !== null && start !== undefined && start !== '') ? Number(start) : NaN;
            if (!Number.isNaN(startNum) && Number.isFinite(startNum) && typeof val === 'number') {
                normalizedProps[key] = [startNum, val];
            } else {
                normalizedProps[key] = [start, val];
            }
        }

        this.tweens.push({
            target, duration, props: normalizedProps,
            startTime, endTime: startTime + duration, ease
        });
    }

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

// --- Data Generators ---

function generateKLineData(count = 200, seed = 12345) {
    // Seed with fixed number for deterministic output
    const rng = new Random(seed); 
    let price = 45000;
    const data = [];
    let minP = Infinity;
    let maxP = -Infinity;
    
    // Dynamic Volatility & Trend state
    let currentVol = 0.01;
    let trendBias = 0.0; // 0 = Neutral, >0 = Bullish, <0 = Bearish

    let currentVolBase = 80;
    let currentVolMult = 3000;

    for (let i = 0; i < count; i++) {
        // Create Market Phases for "Promo" look
        // Adjusted for 30 points warmup slicing (Visible starts at i=30)
        
        let volBaseTarget, volMultTarget;

        if (i < 75) { 
            // Phase 1: Squeeze / Consolidation (Visible: 0 to ~45)
            // JAGGY VOLATILITY: High enough to be clearly visible on screen
            const targetVol = 0.05; 
            currentVol += (targetVol - currentVol) * 0.1;
            trendBias = 0.0; 
            
            // VISIBLE VOLUME: Make sure bars are tall enough to be seen
            volBaseTarget = 80;
            volMultTarget = 3000;
        } else if (i < 120) { 
            // Phase 2: Breakout (Visible: ~45 to ~90)
            // Volatility expands
            const targetVol = 0.06;
            currentVol += (targetVol - currentVol) * 0.05;
            
            // FLATTEN THE CURVE: Low trend bias prevents Y-axis compression
            // This ensures the start of the chart doesn't look like a flat line
            trendBias = 0.03; 
            
            // High VOLUME for breakout
            volBaseTarget = 180;
            volMultTarget = 6000;
        } else {
            // Phase 3: Volatile Uptrend / Choppy
            const targetVol = 0.04;
            currentVol += (targetVol - currentVol) * 0.1;
            trendBias = 0.05;
            
            // Moderate/High Volume
            volBaseTarget = 80;
            volMultTarget = 3000;
        }

        currentVolBase += (volBaseTarget - currentVolBase) * 0.10;
        currentVolMult += (volMultTarget - currentVolMult) * 0.10;

        // Add varying noise to volatility to keep it organic
        currentVol += (rng.next() - 0.5) * 0.002;
        currentVol = Math.max(0.002, Math.min(0.012, currentVol)); // Further reduced max vol to 1.2%

        // Calculate Price Change
        const direction = rng.next() - 0.5 + trendBias; 
        const change = direction * currentVol;
        
        const close = price * (1 + change);
        
        // High/Low
        // Reduced wick length to be very conservative
        const wickLen = currentVol * price * (0.05 + rng.next() * 0.25); 
        
        const high = Math.max(price, close) + wickLen;
        const low = Math.min(price, close) - wickLen;
        
        // Random variation for Volume
        const vol = rng.range(currentVolBase, currentVolBase + currentVol * currentVolMult);

        if (high > maxP) maxP = high;
        if (low < minP) minP = low;

        data.push({
            open: price,
            close: close,
            high: high,
            low: low,
            vol: vol,
            isUp: close >= price
        });
        price = close;
    }
    data.minPrice = minP;
    data.maxPrice = maxP;
    return data;
}

// --- Scenes ---

// --- Scene 1: Architecture ---
function setupArch() {
    clearGroup(state.sceneRoot);
    clearGroup(state.labelsLayer);
    
    const sceneObj = { animator: new Animator(), camera: { x: 0, y: 0, scale: 1 } };
    const g = createSVG('g', { id: 'arch-group' });
    state.sceneRoot.appendChild(g);

    // Positions
    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;
    const leftX = cx - 500;
    const rightX = cx + 500;
    
    // Nodes
    const nodes = [
        { id: 'quant', label: 'Quant Models\n量化模型', x: leftX, y: cy - 250, color: '#E6CFAA' },
        { id: 'news', label: 'News Intelligence\n新闻情报', x: leftX, y: cy, color: '#B5B0D6' },
        { id: 'kline', label: 'K-Line Analysis\nK线图表', x: leftX, y: cy + 250, color: '#A4B0BE' },
        { id: 'exec', label: 'Execution\n自动交易', x: rightX, y: cy, color: '#FF9F43' }
    ];
    
    sceneObj.nodes = nodes; // Save for reference

    // Draw Nodes
    nodes.forEach(n => {
        const box = createSVG('rect', {
            x: n.x - 170, y: n.y - 60, width: 340, height: 120, // Widened to 340 to fit "News Intelligence"
            fill: 'transparent', stroke: n.color, 'stroke-width': 4, rx: 10,
            opacity: 0
        });
        n.boxEl = box;
        g.appendChild(box);
        
        const lines = n.label.split('\n');
        lines.forEach((line, i) => {
            const text = createSVG('text', {
                x: n.x, y: n.y + (i * 35) - 10,
                'text-anchor': 'middle', 'dominant-baseline': 'middle',
                fill: 'var(--text-primary)', 'font-size': i===0 ? '32px' : '24px', 'font-weight': i===0 ? '700' : '400',
                opacity: 0,
                text: line // Fixed: Added text content
            });
            n[`textEl${i}`] = text;
            g.appendChild(text);
        });
    });

    // Gemini Center (Chip Style)
    const chipW = 340;
    const chipH = 100;
    const geminiGroup = createSVG('g', {
        transform: `translate(${cx}, ${cy})`,
        opacity: 0 // Start hidden
    });
    
    // Pill Background
    geminiGroup.appendChild(createSVG('rect', {
        x: -chipW / 2, y: -chipH / 2, width: chipW, height: chipH, rx: chipH / 2,
        fill: '#1e1e1e', stroke: '#333', 'stroke-width': 2,
        'fill-opacity': 1.0
    }));
    
    // Icon (Left)
    geminiGroup.appendChild(createSVG('image', {
        x: -chipW / 2 + 10, y: -chipH / 2 + 10, width: 80, height: 80,
        href: 'gemini-color.png'
    }));
    
    // Text (Right)
    geminiGroup.appendChild(createSVG('text', {
        x: -40, y: 0, // Adjusted X to fit inside chip (centered relative to remaining space)
        'font-family': 'var(--font-mono)', 'font-weight': 'bold', 'font-size': '40px',
        fill: '#ffffff', 'dominant-baseline': 'middle', 'text-anchor': 'start',
        text: 'Gemini 3'
    }));
    
    g.appendChild(geminiGroup);
    
    // Animations
    // Appear left nodes
    nodes.slice(0, 3).forEach((n, i) => {
        const delay = 500 + i * 500;
        sceneObj.animator.to(n.boxEl, 600, { opacity: [0, 1], translateX: [-50, 0] }, delay);
        sceneObj.animator.to(n.textEl0, 600, { opacity: [0, 1] }, delay + 200);
        if (n.textEl1) sceneObj.animator.to(n.textEl1, 600, { opacity: [0, 1] }, delay + 200);
    });

    // Appear Gemini
    sceneObj.animator.to(geminiGroup, 800, { opacity: [0, 1], scale: [0.5, 1] }, 2500, Easing.easeOutBack);
    
    // Connect Lines (Left to Center)
    const connLines = [];
    nodes.slice(0, 3).forEach((n, i) => {
        const line = createSVG('line', {
            x1: n.x + 170, y1: n.y, x2: cx - chipW/2, y2: cy, // Connect to Chip Edge
            stroke: 'var(--text-secondary)', 'stroke-width': 2, 'stroke-dasharray': '10 10', opacity: 0
        });
        connLines.push(line);
        g.insertBefore(line, geminiGroup);
        sceneObj.animator.to(line, 600, { opacity: [0, 0.5], strokeDashoffset: [200, 0] }, 3000 + i * 200);
    });

    // Connect Center to Right
    const outLine = createSVG('line', {
        x1: cx + chipW/2, y1: cy, x2: rightX - 170, y2: cy, // Connect from Chip Edge
        stroke: 'var(--text-secondary)', 'stroke-width': 2, 'stroke-dasharray': '10 10', opacity: 0
    });
    connLines.push(outLine);
    g.insertBefore(outLine, geminiGroup);
    sceneObj.animator.to(outLine, 600, { opacity: [0, 0.5] }, 4000);

    // Appear Execution
    const exec = nodes[3];
    sceneObj.animator.to(exec.boxEl, 600, { opacity: [0, 1], translateX: [50, 0] }, 4500);
    sceneObj.animator.to(exec.textEl0, 600, { opacity: [0, 1] }, 4700);
    if (exec.textEl1) sceneObj.animator.to(exec.textEl1, 600, { opacity: [0, 1] }, 4700);

    // --- Transition to Next Scene ---
    // User Request: Zoom entire scene into "Quant Models" node.
    // "Zoom together" -> Do not hide other elements aggressively.
    
    const quantNode = nodes[0]; // Target node
    
    // Zoom Animation (Start at 8000ms)
    // Target Scale: 12x (Zoom deep into the node so borders go off-screen)
    // Target Position: Keep Quant Node centered
    
    const targetScale = 12;
    const targetX = cx - (quantNode.x * targetScale);
    const targetY = cy - (quantNode.y * targetScale);
    
    // Camera Zoom
    sceneObj.animator.to(sceneObj.camera, 1500, {
        scale: [1, targetScale],
        x: [0, targetX],
        y: [0, targetY]
    }, 8500, Easing.easeInExpo); // easeInExpo for dramatic "warp speed" effect

    // Fade out text as we zoom in (so it doesn't get huge and blurry)
    nodes.forEach(n => {
        [n.textEl0, n.textEl1].forEach(t => {
            if (t) sceneObj.animator.to(t, 500, { opacity: [1, 0] }, 8800);
        });
    });
    
    // Fade out lines and Gemini and OTHER nodes
    const fadeList = [
        geminiGroup,
        ...connLines,
        nodes[3].boxEl, // Exec box
        nodes[1].boxEl, // News
        nodes[2].boxEl  // Kline
    ];

    fadeList.forEach(el => {
        if (el) sceneObj.animator.to(el, 500, { opacity: [1, 0] }, 9000);
    });
    
    // Note: quantNode.boxEl remains visible. 
    // Since we zoom to 12x, the 120px height becomes 1440px > 1080px screen height.
    // The borders will be off-screen, effectively showing a "black void" inside the box.
    // This matches the start of Scene 2 (darkness).

    return sceneObj;
}

function updateArch(sceneObj, t) {
    sceneObj.animator.update(t);
    // Apply Camera
    const g = document.getElementById('arch-group');
    if (g) {
        g.setAttribute('transform', `translate(${sceneObj.camera.x}, ${sceneObj.camera.y}) scale(${sceneObj.camera.scale})`);
    }
    setSubtitle("Multi-source data drives intelligent decisions.", "多源数据驱动的智能决策系统。");
}

// --- Scene 2: Quant Models ---
function setupModels() {
    clearGroup(state.sceneRoot);
    clearGroup(state.labelsLayer);

    const sceneObj = { animator: new Animator() };
    const g = createSVG('g', { id: 'models-group' });
    state.sceneRoot.appendChild(g);
    
    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;
    
    // 1. Transition from Arch (Simulate Zoom)
    // Removed yellow zoomBox overlay to avoid "yellow flash"
    // The previous scene ends with a zoom into the Quant node (beige border).
    // We can start this scene with the chart elements appearing.
    
    // Models Config
    const models = [
        { name: 'MA Crossover', cn: '均线交叉策略', color: '#E6CFAA', desc: 'Capture trends with dual moving averages.\n双均线金叉死叉，捕捉趋势反转点。' },
        { name: 'Bollinger Breakout', cn: '布林带突破', color: '#B5B0D6', desc: 'Detect volatility breakouts.\n基于波动率突破上轨，跟随趋势爆发。' },
        { name: 'RF4 Divergence', cn: 'RF4背离检测', color: '#A4B0BE', desc: 'Identify trend reversals via RSI divergence.\n利用RSI顶底背离，提前识别反转信号。' },
        { name: 'Trend Pullback', cn: '趋势回调', color: '#FF9F43', desc: 'Enter on trend pullbacks.\n上升趋势中的缩量回调，低风险入场。' }
    ];
    
    let timeCursor = 200; // Start almost immediately (reduced from 500)
    const durationPerModel = 7000;
    
    // Generate Shared Data for Consistency
    // Use a fixed seed so all models analyze the SAME market data
    // Add Warmup to ensure valid indicators at start
    const visibleCount = 100;
    const warmupCount = 40;
    const sharedFullData = generateKLineData(visibleCount + warmupCount, 67890); 
    
    const signals = [];

    models.forEach((m, i) => {
        const group = createSVG('g', { opacity: 0 });
        g.appendChild(group);
        
        // Composition: Chart Left, Text Right
        const chartW = 700; // Reduced from 800
        const chartH = 500;
        const chartX = cx - 550; // Shifted left
        const chartY = cy - 250;
        
        // Chart Frame
        const frame = createSVG('rect', {
            x: chartX, y: chartY, width: chartW, height: chartH,
            fill: 'rgba(255,255,255,0.05)', rx: 8, stroke: '#333', 'stroke-width': 1
        });
        group.appendChild(frame);
        
        // Draw K-Lines Background (Removed, handled by drawModelChart)
        // drawKLines(group, chartX + 20, chartY + 20, chartW - 40, chartH - 40);

        // Text Area (Right)
        // Ensure opacity is explicitly set to avoid inheritance issues
        const textX = cx + 200; 
        const title = createSVG('text', {
            x: textX, y: cy - 100, 'text-anchor': 'start',
            fill: m.color, 'font-size': '48px', 'font-weight': '700',
            opacity: 1,
            text: m.name
        });
        group.appendChild(title);
        
        const sub = createSVG('text', {
            x: textX, y: cy - 40, 'text-anchor': 'start',
            fill: 'var(--text-secondary)', 'font-size': '28px',
            opacity: 1,
            text: m.cn
        });
        group.appendChild(sub);
        
        const lines = m.desc.split('\n');
        lines.forEach((l, li) => {
             const t = createSVG('text', {
                x: textX, y: cy + 40 + li*40, 'text-anchor': 'start',
                fill: 'var(--text-primary)', 'font-size': '20px',
                opacity: 1,
                text: l
            });
            group.appendChild(t);
        });

        // Schematic Chart Content
        // Real K-Line Chart + Indicators
        // Pass warmupCount to slice data correctly
        drawModelChart(i, group, sceneObj, chartX + 20, chartY + 20, chartW - 40, chartH - 40, timeCursor, sharedFullData, warmupCount);
        
        // Animations
        sceneObj.animator.to(group, 500, { opacity: [0, 1] }, timeCursor);
        
        // Fade out group but Keep Signal
        sceneObj.animator.to(group, 500, { opacity: [1, 0] }, timeCursor + durationPerModel - 500);
        
        // Leave Signal
        const sigX = cx - 300 + i * 200;
        // Moved badges UP to avoid subtitle overlap (CONFIG.height - 300)
        const badge = createSVG('g', { opacity: 0 });
        const bg = createSVG('rect', {
            x: sigX - 80, y: CONFIG.height - 300, width: 160, height: 60,
            fill: 'transparent', stroke: m.color, 'stroke-width': 2, rx: 8
        });
        const txt = createSVG('text', {
            x: sigX, y: CONFIG.height - 270, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
            fill: m.color, 'font-size': '18px', 'font-weight': 'bold',
            text: 'SIGNAL ' + (i+1)
        });
        badge.appendChild(bg);
        badge.appendChild(txt);
        g.appendChild(badge);
        
        // Appear when main group fades
        sceneObj.animator.to(badge, 500, { opacity: [0, 1] }, timeCursor + durationPerModel - 500);
        
        timeCursor += durationPerModel;
    });

    // Fade Out Scene at End
    const sceneEndTime = timeCursor;
    sceneObj.animator.to(g, 800, { opacity: [1, 0] }, sceneEndTime);

    return sceneObj;
}

function drawModelChart(type, group, sceneObj, x, y, w, h, delay, fullData, warmupCount = 0) {
    // 1. Prepare Data (Slice Warmup)
    // We calculate indicators on fullData first, then slice both data and indicators.
    
    const visibleData = fullData.slice(warmupCount);
    
    // Recalculate Min/Max for visible range to ensure correct scaling
    let minP = Infinity, maxP = -Infinity;
    visibleData.forEach(d => {
        if (d.high > maxP) maxP = d.high;
        if (d.low < minP) minP = d.low;
    });
    visibleData.minPrice = minP;
    visibleData.maxPrice = maxP;

    // Calculate layout
    let mainH = h;
    let mainY = y;
    let subH = 0;
    let subY = 0;
    
    // If RSI (type 2), split screen
    if (type === 2) {
        mainH = h * 0.7;
        subH = h * 0.25;
        subY = y + h - subH;
    }

    // 1. Draw K-Lines (Main Chart)
    drawKLines(group, x, mainY, w, mainH, visibleData);
    
    // 2. Calculate & Draw Indicators
    if (type === 0) {
        // MA Crossover (MA7 & MA20)
        const ma7Full = calculateSMA(fullData, 7);
        const ma20Full = calculateSMA(fullData, 20);
        
        const ma7 = ma7Full.slice(warmupCount);
        const ma20 = ma20Full.slice(warmupCount);
        
        drawIndicatorLine(group, sceneObj, ma7, visibleData, x, mainY, w, mainH, '#E6CFAA', delay + 500); // MA7 (Sand)
        drawIndicatorLine(group, sceneObj, ma20, visibleData, x, mainY, w, mainH, '#B5B0D6', delay + 500); // MA20 (Lavender)
    } 
    else if (type === 1) {
        // Bollinger Bands
        const bbFull = calculateBollinger(fullData, 20, 2);
        const bb = bbFull.slice(warmupCount);
        
        // Draw Upper/Lower/Middle
        const upper = bb.map(b => b.upper);
        const lower = bb.map(b => b.lower);
        const middle = bb.map(b => b.middle);
        
        drawIndicatorLine(group, sceneObj, upper, visibleData, x, mainY, w, mainH, '#A4B0BE', delay + 500); // Upper (Steel Blue)
        drawIndicatorLine(group, sceneObj, lower, visibleData, x, mainY, w, mainH, '#A4B0BE', delay + 500); // Lower (Steel Blue)
        drawIndicatorLine(group, sceneObj, middle, visibleData, x, mainY, w, mainH, 'rgba(255,255,255,0.5)', delay + 500, 1);
    }
    else if (type === 2) {
        // RSI Divergence
        const rsiFull = calculateRSI(fullData, 14);
        const rsi = rsiFull.slice(warmupCount);
        
        // Draw RSI in sub-window
        // RSI is 0-100
        drawSubChartLine(group, sceneObj, rsi, x, subY, w, subH, 0, 100, '#FF9F43', delay + 500); // RSI (Orange)
        
        // Draw 70/30 Levels
        const level70 = subY + subH * 0.3; // 100 is top (0), 0 is bottom (h). 70 is 30% from top.
        const level30 = subY + subH * 0.7;
        
        group.appendChild(createSVG('line', {
            x1: x, y1: level70, x2: x + w, y2: level70,
            stroke: 'rgba(255,255,255,0.3)', 'stroke-dasharray': '4 4'
        }));
        group.appendChild(createSVG('line', {
            x1: x, y1: level30, x2: x + w, y2: level30,
            stroke: 'rgba(255,255,255,0.3)', 'stroke-dasharray': '4 4'
        }));
    }
    else if (type === 3) {
        // Trend Pullback (MA20 Support)
        const ma20Full = calculateSMA(fullData, 20);
        const ma20 = ma20Full.slice(warmupCount);
        
        drawIndicatorLine(group, sceneObj, ma20, visibleData, x, mainY, w, mainH, '#FF9F43', delay + 500);
    }
}

function drawIndicatorLine(group, sceneObj, values, sourceData, x, y, w, h, color, delay, width=3) {
    const minP = sourceData.minPrice;
    const maxP = sourceData.maxPrice;
    const range = maxP - minP;
    
    const step = w / sourceData.length;
    
    // Build points
    const points = [];
    values.forEach((val, i) => {
        if (val !== null && val !== undefined) {
            const cx = x + i * step + step/2;
            const cy = y + h - ((val - minP) / range) * h;
            points.push({ x: cx, y: cy });
        }
    });
    
    if (points.length < 2) return;
    
    const d = catmullRomSpline(points);
    const path = createSVG('path', {
        d: d, fill: 'none', stroke: color, 'stroke-width': width
    });
    group.appendChild(path);
    
    // Use actual path length for consistent speed
    const len = path.getTotalLength() || 1000;
    path.setAttribute('stroke-dasharray', len);
    path.setAttribute('stroke-dashoffset', len);
    
    sceneObj.animator.to(path, 2500, { 'stroke-dashoffset': [len, 0] }, delay);
}

function drawSubChartLine(group, sceneObj, values, x, y, w, h, minVal, maxVal, color, delay) {
    const range = maxVal - minVal;
    const step = w / values.length;
    
    const points = [];
    values.forEach((val, i) => {
        if (val !== null && val !== undefined) {
            const cx = x + i * step + step/2;
            const cy = y + h - ((val - minVal) / range) * h;
            points.push({ x: cx, y: cy });
        }
    });
    
    if (points.length < 2) return;
    
    const d = catmullRomSpline(points);
    const path = createSVG('path', {
        d: d, fill: 'none', stroke: color, 'stroke-width': 3
    });
    group.appendChild(path);
    
    // Use actual path length for consistent speed
    const len = path.getTotalLength() || 1000;
    path.setAttribute('stroke-dasharray', len);
    path.setAttribute('stroke-dashoffset', len);
    
    sceneObj.animator.to(path, 2500, { 'stroke-dashoffset': [len, 0] }, delay);
}

function updateModels(sceneObj, t) {
    sceneObj.animator.update(t);
    
    const start = 200;
    const per = 7000;
    const idx = Math.floor(Math.max(0, t - start) / per);

    if (idx <= 0) {
        setSubtitle("Dual MA Crossover Strategy.", "双均线交叉策略，捕捉趋势。");
    } else if (idx === 1) {
        setSubtitle("Bollinger Bands Breakout.", "布林带突破，跟随波动。");
    } else if (idx === 2) {
        setSubtitle("RF4 Divergence Detection.", "RF4背离检测，预警反转。");
    } else {
        setSubtitle("Trend Pullback Entry.", "趋势回调策略，低吸入场。");
    }
}

// Helper: Draw K-Lines in a box
function drawKLines(group, x, y, w, h, dataOverride = null) {
    const data = dataOverride || generateKLineData(50);
    const minP = data.minPrice;
    const maxP = data.maxPrice;
    const range = maxP - minP;
    
    const candleW = (w / data.length) * 0.7; // Thicker candles
    const step = w / data.length;
    
    data.forEach((d, i) => {
        const cx = x + i * step + step/2;
        
        // Normalize Y (SVG Y=0 is top, so we invert)
        const normalize = (val) => y + h - ((val - minP) / range) * h;
        
        const openY = normalize(d.open);
        const closeY = normalize(d.close);
        const highY = normalize(d.high);
        const lowY = normalize(d.low);
        
        const color = d.close >= d.open ? '#0ECB81' : '#F6465D'; // Binance Green/Red
        
        // Wick
        group.appendChild(createSVG('line', {
            x1: cx, y1: highY, x2: cx, y2: lowY,
            stroke: color, 'stroke-width': 1
        }));
        
        // Body
        const bodyH = Math.max(1, Math.abs(closeY - openY));
        const bodyY = Math.min(openY, closeY);
        group.appendChild(createSVG('rect', {
            x: cx - candleW/2, y: bodyY, width: candleW, height: bodyH,
            fill: color
        }));
    });
}

// Helper for Architecture Nodes
function drawArchNodes(g, cx, cy, filled = false, showSignals = false) {
    const leftX = cx - 500;
    const rightX = cx + 500;
    const nodes = [
        { id: 'quant', label: 'Quant Models\n量化模型', x: leftX, y: cy - 250, color: '#E6CFAA' },
        { id: 'news', label: 'News Intelligence\n新闻情报', x: leftX, y: cy, color: '#B5B0D6' },
        { id: 'kline', label: 'K-Line Analysis\nK线图表', x: leftX, y: cy + 250, color: '#A4B0BE' },
        { id: 'exec', label: 'Execution\n自动交易', x: rightX, y: cy, color: '#FF9F43' }
    ];
    
    const elements = [];

    nodes.forEach(n => {
        // Box
        const box = createSVG('rect', {
            x: n.x - 170, y: n.y - 60, width: 340, height: 120, // Widened to 340
            fill: filled ? n.color : 'transparent',
            'fill-opacity': filled ? 0.15 : 0,
            stroke: n.color, 'stroke-width': 4, rx: 10
        });
        n.boxEl = box;
        g.appendChild(box);
        elements.push(box);
        
        // Text
        const lines = n.label.split('\n');
        lines.forEach((line, i) => {
            const text = createSVG('text', {
                x: n.x, y: n.y + (i * 35) - 10,
                'text-anchor': 'middle', 'dominant-baseline': 'middle',
                fill: 'var(--text-primary)', 'font-size': i===0 ? '32px' : '24px', 'font-weight': i===0 ? '700' : '400',
                text: line // Fixed: Added text content
            });
            text.textContent = line;
            g.appendChild(text);
            n[`textEl${i}`] = text;
            elements.push(text);
        });
        
        // Signals on Quant (Removed to clean up UI)
        // (Badges removed)
    });
    
    // Gemini Center Group (Chip Style)
    // Box: Left Icon + Right Text "Gemini 3"
    const chipW = 340;
    const chipH = 100;
    const geminiGroup = createSVG('g', {
        transform: `translate(${cx}, ${cy})` // Centered using transform for easier manipulation
    });
    
    // Pill Background
    geminiGroup.appendChild(createSVG('rect', {
        x: -chipW / 2, y: -chipH / 2, width: chipW, height: chipH, rx: chipH / 2,
        fill: '#1e1e1e', stroke: '#333', 'stroke-width': 2,
        'fill-opacity': 1.0
    }));
    
    // Icon (Left)
    geminiGroup.appendChild(createSVG('image', {
        x: -chipW / 2 + 10, y: -chipH / 2 + 10, width: 80, height: 80,
        href: 'gemini-color.png'
    }));
    
    // Text (Right)
    geminiGroup.appendChild(createSVG('text', {
        x: -40, y: 0, 
        'font-family': 'var(--font-mono)', 'font-weight': 'bold', 'font-size': '40px',
        fill: '#ffffff', 'dominant-baseline': 'middle', 'text-anchor': 'start',
        text: 'Gemini 3'
    }));

    g.appendChild(geminiGroup);
    elements.push(geminiGroup);
    
    return { nodes, elements, geminiImg: geminiGroup };
}

function setupNews() {
    clearGroup(state.sceneRoot);
    clearGroup(state.labelsLayer);

    const sceneObj = { animator: new Animator(), camera: { x: 0, y: 0, scale: 1 } };
    const g = createSVG('g', { id: 'news-group', opacity: 0 }); // Start hidden for fade-in
    state.sceneRoot.appendChild(g);
    
    // Fade In Scene
    sceneObj.animator.to(g, 800, { opacity: [0, 1] }, 0);

    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;
    
    // 1. Draw Architecture
    const arch = drawArchNodes(g, cx, cy, false, true); 
    
    // 2. Camera Zoom to News - Removed conflicting animation
    // The main zoom is handled below by the "Deep Dive" sequence

    
    // 3. Browser Frame (Image 1: Raw News)
    // Place inside the News Node (340x120 box). 
    // We want the image to be approx 16:9. 
    // Node is at (cx-500, cy).
    const newsNodeX = cx - 500;
    
    // Size it small to fit "inside" the node conceptually
    // Box is 340x120. 
    // Using 180x100 (Zoom 10x -> 1800x1000). Fits safely inside 1920x1080.
    const imgW = 180;
    const imgH = 100; 
    const bx = newsNodeX - imgW/2;
    const by = cy - imgH/2;
    
    const rawNewsGroup = createSVG('g', { opacity: 0 }); 
    g.appendChild(rawNewsGroup);

    rawNewsGroup.appendChild(createSVG('rect', {
        x: bx, y: by, width: imgW, height: imgH,
        fill: '#0c0d0f', opacity: 1
    }));
    
    const rawImage = createSVG('image', {
        x: bx, y: by, width: imgW, height: imgH,
        href: 'image.png',
        preserveAspectRatio: 'xMidYMid slice'
    });
    rawNewsGroup.appendChild(rawImage);

    // 4. Structured Data View (Image 2: Structured)
    const structuredGroup = createSVG('g', { opacity: 0 });
    g.appendChild(structuredGroup);

    structuredGroup.appendChild(createSVG('rect', {
        x: bx, y: by, width: imgW, height: imgH,
        fill: '#0c0d0f', opacity: 1
    }));
    
    const structImage = createSVG('image', {
        x: bx, y: by, width: imgW, height: imgH,
        href: 'image2.png',
        preserveAspectRatio: 'xMidYMid slice'
    });
    structuredGroup.appendChild(structImage);

    // Label for Structured Version (Small font size because we will zoom in 10x)
    // 16px / 10 ~= 1.6px. Let's use 2px.
    const labelBox = createSVG('rect', {
        x: bx + imgW - 65, y: by + imgH - 8, width: 60, height: 6, rx: 1,
        fill: '#000', 'fill-opacity': 0.7
    });
    structuredGroup.appendChild(labelBox);
    
    const labelText = createSVG('text', {
        x: bx + imgW - 35, y: by + imgH - 4,
        'text-anchor': 'middle', fill: '#fff', 'font-size': '3px', 'font-family': 'sans-serif',
        'dominant-baseline': 'middle',
        text: 'Structured Version / 结构化数据'
    });
    structuredGroup.appendChild(labelText);
    
    // Animation Sequence
    
    // 1. Initial State: Camera at Overview.
    
    // 2. Zoom In Sequence (Deep Dive)
    // Target Scale: ScreenWidth (1920) / imgW (200) = 9.6 -> Use 10x
    const targetScale = 10;
    const targetX = cx - (newsNodeX * targetScale);
    const targetY = cy - (cy * targetScale); // Node is at cy
    
    // Zoom in (1.5s)
    // Delayed to 3500ms to let the user view the structure first
    sceneObj.animator.to(sceneObj.camera, 1500, { 
        x: [0, targetX], 
        y: [0, targetY], 
        scale: [1, targetScale] 
    }, 3500, Easing.easeInOutQuad);
    
    // Fade in Raw News as we zoom (so it looks like we are entering the node)
    sceneObj.animator.to(rawNewsGroup, 500, { opacity: [0, 1] }, 3800);
    
    // 3. Switch to Structured (image2.png) with overlapping crossfade (avoid flash of underlying node text)
    sceneObj.animator.to(structuredGroup, 500, { opacity: [0, 1] }, 6300);
    sceneObj.animator.to(rawNewsGroup, 500, { opacity: [1, 0] }, 6600);
    
    // Fade Out Scene at End (approx 14s)
    sceneObj.animator.to(g, 800, { opacity: [1, 0] }, 14000);
    
    // Scene update handles camera
    sceneObj.onUpdate = (t) => {
        g.setAttribute('transform', `translate(${sceneObj.camera.x}, ${sceneObj.camera.y}) scale(${sceneObj.camera.scale})`);
    };
    
    return sceneObj;
}

function updateNews(sceneObj, t) {
    sceneObj.animator.update(t);
    if (sceneObj.onUpdate) sceneObj.onUpdate(t);

    if (t < 5000) {
        setSubtitle("Hourly polling CoinDesk & TruthSocial feeds.", "每小时轮询 CoinDesk 与 TruthSocial 新闻源。");
    } else if (t < 10000) {
        setSubtitle("RSS parsed; key headlines passed to LLM reviewer.", "解析RSS后将要点交给LLM复核。");
    } else {
        setSubtitle("Signals reviewed alongside quant models.", "新闻要点与量化信号一并决策。");
    }
}

function setupFusion() {
    clearGroup(state.sceneRoot);
    clearGroup(state.labelsLayer);

    const sceneObj = { animator: new Animator(), camera: { x: 0, y: 0, scale: 1 } };
    const g = createSVG('g', { id: 'fusion-group', opacity: 0 }); // Start hidden for fade-in
    state.sceneRoot.appendChild(g);
    
    // Fade In Scene
    sceneObj.animator.to(g, 800, { opacity: [0, 1] }, 0);
    
    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;
    
    // 1. Draw Architecture (Filled)
    const arch = drawArchNodes(g, cx, cy, true, true);
    sceneObj.gemini = arch.geminiImg;
    
    // 2. Animate "Black Hole" - Suck Left Nodes into Gemini
    // Nodes: Quant(0), News(1), Kline(2)
    const targetNodes = [arch.nodes[0], arch.nodes[1], arch.nodes[2]];
    
    targetNodes.forEach((n, i) => {
        // Animate Box: Suck INTO Gemini Center
        // Start: n.x - 170. End: cx - 10 (Tiny box in center).
        // Delayed to 4000ms to follow particles
        
        sceneObj.animator.to(n.boxEl, 1500, {
            x: [n.x - 170, cx - 10],
            y: [n.y - 60, cy - 5],
            width: [340, 20],
            height: [120, 10],
            opacity: [1, 1] // Keep opacity, hide behind Gemini
        }, 4000 + i*300, Easing.easeInExpo);
        
        // Hide Text
        [n.textEl0, n.textEl1].forEach(t => {
            if (t) sceneObj.animator.to(t, 500, { opacity: [1, 0] }, 4000 + i*300);
        });
    });
    
    // Particles
    const particleCount = 45;
    const words = ["Price", "Vol", "RSI", "News", "Sentiment", "Trend", "Whale", "Flow", "MA20", "BOLL", "LSTM", "Transformer"];
    const sources = [arch.nodes[0], arch.nodes[1], arch.nodes[2]]; // Use original positions
    
    for(let i=0; i<particleCount; i++) {
        const src = sources[i % 3];
        const startX = src.x + (Math.random() - 0.5) * 200;
        const startY = src.y + (Math.random() - 0.5) * 100;
        const word = words[Math.floor(Math.random() * words.length)];
        
        const p = createSVG('text', {
            x: startX, y: startY, fill: 'var(--text-secondary)', 'font-size': '24px', opacity: 0,
            text: word, 'text-anchor': 'middle'
        });
        g.appendChild(p);
        
        const duration = 2000 + Math.random() * 2500;
        const delay = Math.random() * 3000;
        
        sceneObj.animator.to(p, duration, {
            x: [startX, cx], y: [startY, cy],
            opacity: [0, 1, 0],
            'font-size': [24, 8]
        }, delay, Easing.easeInCubic); 
    }
    
    // Connection to Execution
    const rightX = cx + 500;
    const execLine = createSVG('line', {
        x1: cx, y1: cy, x2: cx + 170, y2: cy, // Start from center (masked by Gemini), initially end at edge
        stroke: '#FF9F43', 'stroke-width': 6, opacity: 1
    });
    g.appendChild(execLine);
    
    // Move Gemini to top layer (so sucked nodes, particles AND start of execLine go BEHIND it)
    g.appendChild(sceneObj.gemini);
    
    // Animate x2 to grow the line deep into the Decision Box (to center rightX)
    // The end will be covered by the Decision Box
    sceneObj.animator.to(execLine, 800, { x2: [cx + 170, rightX - 95] }, 7000, Easing.easeInOutQuad);
    
    // Decision Result
    const decisionBox = createSVG('g', { opacity: 0, transform: `translate(${rightX-100}, ${cy-50})` });
    g.appendChild(decisionBox);
    decisionBox.appendChild(createSVG('rect', { x: 0, y: 0, width: 200, height: 100, rx: 10, fill: '#28a745' }));
    decisionBox.appendChild(createSVG('text', { x: 100, y: 50, 'text-anchor': 'middle', 'dominant-baseline': 'middle', fill: '#fff', 'font-size': '36px', 'font-weight': 'bold', text: 'LONG' }));
    // Show Decision Box AFTER line arrives (7800ms) - strictly sequential
    sceneObj.animator.to(decisionBox, 500, { opacity: [0, 1], scale: [0.5, 1] }, 7800, Easing.easeOutBack);
    
    // --- Camera Movement ---
    // 1. Pan to Connection (Left 1/3)
    sceneObj.animator.to(sceneObj.camera, 1500, { x: [0, -570] }, 7000, Easing.easeInOutQuad);

    // Fade Out Scene shortly after LONG appears (email is its own scene now)
    sceneObj.animator.to(g, 800, { opacity: [1, 0] }, 9800);
    
    // Scene update handles camera
    sceneObj.onUpdate = (t) => {
        g.setAttribute('transform', `translate(${sceneObj.camera.x}, ${sceneObj.camera.y}) scale(${sceneObj.camera.scale})`);
    };
    
    return sceneObj;
}

function updateFusion(sceneObj, t) {
    sceneObj.animator.update(t);
    
    // Glitch Effect
    const isGlitching = (t > 5000 && t < 7000);
    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;
    
    if (isGlitching && sceneObj.gemini) {
        const jitterX = cx + (Math.random() - 0.5) * 15;
        const jitterY = cy + (Math.random() - 0.5) * 15;
        const scale = 1 + (Math.random() - 0.5) * 0.15;
        sceneObj.gemini.setAttribute('transform', `translate(${jitterX}, ${jitterY}) scale(${scale})`);
        
        const hue = Math.floor(Math.random() * 360);
        sceneObj.gemini.style.filter = `hue-rotate(${hue}deg) contrast(1.5)`;
        sceneObj.gemini.style.opacity = 0.8 + Math.random() * 0.2;
    } else if (sceneObj.gemini) {
        sceneObj.gemini.setAttribute('transform', `translate(${cx}, ${cy})`);
        sceneObj.gemini.style.filter = 'none';
        sceneObj.gemini.style.opacity = 1;
    }

    if (t < 5000) {
        setSubtitle("Synthesizing multi-modal data...", "多模态数据实时融合分析。");
    } else if (t < 10000) {
        setSubtitle("Generating high-confidence trading signal.", "生成高置信度交易决策。");
    } else {
        setSubtitle("Instant notification via Email/Telegram.", "决策结果实时推送至用户终端。");
    }
}

// 1. Intro (restored)
function setupIntro() {
    clearGroup(state.sceneRoot);
    clearGroup(state.labelsLayer);

    const sceneObj = { animator: new Animator() };
    const g = createSVG('g', { id: 'intro-group' });
    state.sceneRoot.appendChild(g);

    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;

    const titleEn = createSVG('text', {
        x: cx, y: cy - 20,
        'text-anchor': 'middle',
        'font-size': '48px',
        'font-weight': '700',
        fill: 'var(--text-primary)', opacity: 0,
        text: 'Bitcoin (BTC) is a decentralized digital currency.'
    });
    const titleCn = createSVG('text', {
        x: cx, y: cy + 40,
        'text-anchor': 'middle',
        'font-size': '40px',
        'font-weight': '600',
        fill: 'var(--text-secondary)', opacity: 0,
        text: '比特币（BTC）是一种去中心化的数字货币。'
    });
    g.appendChild(titleEn);
    g.appendChild(titleCn);

    sceneObj.animator.to(titleEn, 800, { opacity: [0, 1], y: [cy - 40, cy - 20] }, 200, Easing.easeOutCubic);
    sceneObj.animator.to(titleCn, 800, { opacity: [0, 1], y: [cy + 60, cy + 40] }, 500, Easing.easeOutCubic);

    // Fade out near end of intro duration (8s)
    sceneObj.animator.to(g, 800, { opacity: [1, 0] }, 7000, Easing.easeOutQuad);

    return sceneObj;
}

function updateIntro(sceneObj, relativeTime) {
    sceneObj.animator.update(relativeTime);
    setSubtitle('Bitcoin (BTC) is a decentralized digital currency.', '比特币（BTC）是一种去中心化的数字货币。');
}

function setupKline(scene) {
    clearGroup(state.sceneRoot);
    
    // --- Warmup Data Logic ---
    // Generate extra data so indicators are valid from index 0 of visible chart
    const totalPoints = 240;
    const warmup = 30; // 30 points history for MA20/Boll
    const rawData = generateKLineData(totalPoints);
    
    // Calculate indicators on FULL history
    const rawMA5 = calculateSMA(rawData, 5);
    const rawBoll = calculateBollinger(rawData, 20, 2);
    const rawRSI = calculateRSI(rawData, 14);
    
    // Slice to Visible
    const visibleData = rawData.slice(warmup);
    const visibleMA5 = rawMA5.slice(warmup);
    const visibleBoll = rawBoll.slice(warmup);
    const visibleRSI = rawRSI.slice(warmup);
    
    // Recalc Min/Max for scaling based on Visible Data
    let minP = Infinity, maxP = -Infinity;
    visibleData.forEach(d => {
        if (d.high > maxP) maxP = d.high;
        if (d.low < minP) minP = d.low;
    });
    visibleData.minPrice = minP;
    visibleData.maxPrice = maxP;

    const sceneObj = {
        animator: new Animator(),
        data: visibleData,
        ma5Data: visibleMA5,
        bollData: visibleBoll,
        rsiData: visibleRSI,
        camera: { x: 0, y: 0, scale: 1 }
    };
    
    // Camera Group to pan the whole scene
    const cameraGroup = createSVG('g', { id: 'camera-group' });
    state.sceneRoot.appendChild(cameraGroup);
    
    const g = createSVG('g', { id: 'kline-group' });
    cameraGroup.appendChild(g);

    // --- Phase 1: Single Candle Formation (Centered) ---
    const cx = CONFIG.width / 2;
    const cy = CONFIG.height / 2;
    
    // Define Demo Candle Points relative to center
    // Make it a nice big Up candle
    const demoCandle = {
        open: cy + 50,
        close: cy - 150, // Up candle
        high: cy - 250,
        low: cy + 150
    };
    
    // 1. Points
    const points = [
        { label: 'Open', y: demoCandle.open, labelEn: 'Open', labelCn: '开盘价', color: 'var(--text-secondary)' },
        { label: 'High', y: demoCandle.high, labelEn: 'High', labelCn: '最高价', color: 'var(--text-secondary)' },
        { label: 'Low', y: demoCandle.low, labelEn: 'Low', labelCn: '最低价', color: 'var(--text-secondary)' },
        { label: 'Close', y: demoCandle.close, labelEn: 'Close', labelCn: '收盘价', color: 'var(--text-secondary)' } // Unified color
    ];

    const pointGroup = createSVG('g', {});
    g.appendChild(pointGroup);

    points.forEach((p, i) => {
        // Dot
        const dot = createSVG('circle', {
            cx: cx, cy: p.y, r: 6,
            fill: p.color, opacity: 0
        });
        p.dotEl = dot;
        pointGroup.appendChild(dot);
        
        // Label Line
        const line = createSVG('line', {
            x1: cx, y1: p.y, x2: cx + 120, y2: p.y,
            stroke: 'var(--text-secondary)', 'stroke-width': 1, 'stroke-dasharray': '4 4', opacity: 0
        });
        p.lineEl = line;
        pointGroup.appendChild(line);

        // Text
        const text = createSVG('text', {
            x: cx + 130, y: p.y + 5,
            fill: 'var(--text-primary)', 'font-size': '20px', opacity: 0,
            text: `${p.labelEn} ${p.labelCn}`
        });
        p.textEl = text;
        pointGroup.appendChild(text);

        // Animate in - Sync with "Four points..." subtitle (~5.5s)
        const delay = 5500 + i * 500;
        sceneObj.animator.to(dot, 400, { opacity: [0, 1], r: [0, 6] }, delay, Easing.easeOutBack);
        sceneObj.animator.to(line, 400, { opacity: [0, 1], x2: [cx, cx + 120] }, delay + 200);
        sceneObj.animator.to(text, 400, { opacity: [0, 1], x: [cx + 140, cx + 130] }, delay + 300);
    });

    // 2. Connect Lines (Wick first, then Body)
    const candleGroup = createSVG('g', { opacity: 1 }); // Visible immediately, but children hidden/scaled
    g.insertBefore(candleGroup, pointGroup);

    // Wick (Line)
    const wick = createSVG('line', {
        x1: cx, y1: demoCandle.high, x2: cx, y2: demoCandle.high, // Start with 0 length (y2 = y1)
        class: 'wick', stroke: 'var(--candle-up)', 'stroke-width': 4
    });
    
    // Body (Rect)
    const bodyWidth = 40;
    const bodyHeight = Math.abs(demoCandle.open - demoCandle.close);
    const bodyY = Math.min(demoCandle.open, demoCandle.close);
    // Center of body for expansion
    const bodyCenterY = bodyY + bodyHeight / 2;
    
    const body = createSVG('rect', {
        x: cx - bodyWidth/2, y: bodyCenterY,
        width: bodyWidth, height: 0,
        fill: 'var(--candle-up)'
    });

    candleGroup.appendChild(wick);
    candleGroup.appendChild(body);

    // Sync with "Connect them..." subtitle (~8s)
    const drawTime = 8000;
    // Animate Wick (Grow top to bottom)
    sceneObj.animator.to(wick, 800, { y2: [demoCandle.high, demoCandle.low] }, drawTime, Easing.easeInOutQuad);
    
    // Animate Body (Expand height from center)
    sceneObj.animator.to(body, 600, { y: [bodyCenterY, bodyY], height: [0, bodyHeight] }, drawTime + 800, Easing.easeOutBack);
    
    // Hide points after connecting
    points.forEach((p, i) => {
        sceneObj.animator.to(p.dotEl, 500, { opacity: [1, 0] }, drawTime + 1500);
        sceneObj.animator.to(p.lineEl, 500, { opacity: [1, 0] }, drawTime + 1500);
        sceneObj.animator.to(p.textEl, 500, { opacity: [1, 0] }, drawTime + 1500);
    });

    // --- Phase 2: Move & Chart Generation ---
    // Sync with "Multiple K-lines..." (~11s)
    const transformTime = 11000;
    
    // Prepare Chart Data
    // Dynamic Scaling to fit screen
    const dataMin = sceneObj.data.minPrice;
    const dataMax = sceneObj.data.maxPrice;
    const priceRange = dataMax - dataMin;
    const availableHeight = CONFIG.height * 0.45; // Reduced to 45% for safety
    const paddingTop = 250; // Increased top padding to prevent overflow

    // basePrice is d0.open
    const basePrice = sceneObj.data[0].open;
    
    // Scale to fit range into availableHeight
    const scaleY = availableHeight / priceRange;
    
    // Calculate baseY such that maxPrice is at paddingTop
    // y = baseY - (price - basePrice) * scaleY
    // paddingTop = baseY - (dataMax - basePrice) * scaleY
    // baseY = paddingTop + (dataMax - basePrice) * scaleY
    const baseY = paddingTop + (dataMax - basePrice) * scaleY;

    const chartConfig = {
        baseY: baseY, 
        scaleY: scaleY,
        candleWidth: 12,
        spacing: 12 // wider spacing for better visibility
    };
    
    const klineEls = [];
    
    // We want the Big Candle to morph into chart index 0.
    // Let's create the chart candles.
    
    // Calculate props for index 0 to match Big Candle logic
    const d0 = sceneObj.data[0];
    // basePrice already defined above
    
    const stepX = chartConfig.candleWidth + chartConfig.spacing;
    
    // Offset so index 0 is at cx
    const chartOffsetX = cx;
    
    sceneObj.data.forEach((d, i) => {
        const x = chartOffsetX + i * stepX; 
        
        const openY = chartConfig.baseY - (d.open - basePrice) * chartConfig.scaleY;
        const closeY = chartConfig.baseY - (d.close - basePrice) * chartConfig.scaleY;
        const highY = chartConfig.baseY - (d.high - basePrice) * chartConfig.scaleY;
        const lowY = chartConfig.baseY - (d.low - basePrice) * chartConfig.scaleY;
        
        const color = d.isUp ? 'var(--candle-up)' : 'var(--candle-down)';
        
        const candleG = createSVG('g', { opacity: 0 }); // Hidden initially
        if (i === 0) candleG.setAttribute('id', 'chart-candle-0');
        
        const wickEl = createSVG('line', {
            x1: x, y1: highY, x2: x, y2: lowY,
            stroke: color, 'stroke-width': 2
        });
        
        const bodyEl = createSVG('rect', {
            x: x - chartConfig.candleWidth/2, y: Math.min(openY, closeY),
            width: chartConfig.candleWidth, height: Math.max(2, Math.abs(openY - closeY)),
            fill: color
        });
        
        candleG.appendChild(wickEl);
        candleG.appendChild(bodyEl);
        g.appendChild(candleG);
        
        klineEls.push({ g: candleG, x, y: closeY, openY, closeY, highY, lowY, isUp: d.isUp });
    });

    // --- TRANSITION ANIMATION ---
    // Target: Index 0
    const target0 = klineEls[0];
    const targetWick = target0.g.firstChild;
    const targetBody = target0.g.lastChild;
    
    // Wick Target
    const ty1 = parseFloat(targetWick.getAttribute('y1'));
    const ty2 = parseFloat(targetWick.getAttribute('y2'));
    
    // Body Target
    const tbx = parseFloat(targetBody.getAttribute('x'));
    const tby = parseFloat(targetBody.getAttribute('y'));
    const tbw = parseFloat(targetBody.getAttribute('width'));
    const tbh = parseFloat(targetBody.getAttribute('height'));
    
    // Animate Big Candle to match Chart Index 0
    sceneObj.animator.to(wick, 1000, { 
        y1: [demoCandle.high, ty1], 
        y2: [demoCandle.low, ty2],
        'stroke-width': [4, 2]
    }, transformTime, Easing.easeInOutQuad);
    
    sceneObj.animator.to(body, 1000, {
        x: [cx - bodyWidth/2, tbx],
        y: [bodyY, tby],
        width: [bodyWidth, tbw],
        height: [bodyHeight, tbh]
    }, transformTime, Easing.easeInOutQuad);
    
    // Swap Opacity
    sceneObj.animator.to(target0.g, 100, { opacity: [0, 1] }, transformTime + 900);
    sceneObj.animator.to(candleGroup, 100, { opacity: [1, 0] }, transformTime + 900);

    // --- Phase 3: Chart Reveal & Camera Pan ---
    const revealStart = transformTime + 1500; // Start reveal after transform finishes
    
    // We want to pan so the "head" (latest candle) is always somewhat centered.
    // Initially head is at x = cx (index 0). Camera x = 0.
    // At index i, head is at cx + i*step.
    // To keep head at cx, Camera x should be -i*step.
    
    // Reveal Candles one by one
    // Also animate indicators drawing along with it
    
    const msPerCandle = 100; // Faster drawing as requested
    
    klineEls.forEach((el, i) => {
        if (i === 0) return; // Already there
        
        const delay = revealStart + i * msPerCandle;
        sceneObj.animator.to(el.g, 300, { opacity: [0, 1] }, delay);
        
        // Camera Pan Keyframes
        // We can't make infinite keyframes efficiently in this simple engine.
        // Instead, we just animate camera X linearly from 0 to end position over total duration.
        // Or update it in onUpdate?
        // Let's use a single long tween for camera.
    });
    
    const totalDrawTime = klineEls.length * msPerCandle;
    const totalDistance = (klineEls.length - 1) * stepX;
    
    // Pan camera to keep head at 60% of screen width (Leaving 1/3 space on right)
    const targetHeadX = CONFIG.width * 0.6;
    const finalCameraX = targetHeadX - (cx + totalDistance);

    sceneObj.animator.to(sceneObj.camera, totalDrawTime, { x: [0, finalCameraX] }, revealStart, Easing.linear);

    const drawEndTime = revealStart + totalDrawTime;

    // --- Phase 4: Indicators (Curves) ---
    // We want them to draw WITH the chart.
    // So we need to reveal them progressively or draw them using stroke-dashoffset.
    
    const indicatorsTime = revealStart; // Start drawing immediately with chart
    
    // Use pre-calculated indicators (valid from index 0 due to warmup)
    const ma5Data = sceneObj.ma5Data;
    const bollData = sceneObj.bollData;
    const rsiData = sceneObj.rsiData;
    
    // Helper to map price to screen Y
    const getScreenY = (price) => chartConfig.baseY - (price - basePrice) * chartConfig.scaleY;

    // Helper to map RSI to screen Y (Top padding area: 80px to 230px)
    const getRsiY = (val) => {
        const topY = 80;
        const bottomY = 230;
        // RSI 0 -> bottomY, RSI 100 -> topY
        return bottomY - (val / 100) * (bottomY - topY);
    };

    // Helper to make curve path
    function createCurvePath(pts, color, dashArray) {
        const d = catmullRomSpline(pts, 0.4);
        const path = createSVG('path', {
            d: d, fill: 'none', stroke: color, 'stroke-width': 2, opacity: 1
        });
        if (dashArray) path.setAttribute('stroke-dasharray', dashArray);
        
        // Get length for dashoffset animation
        // Since we can't easily get length before insertion without layout, 
        // we'll approximate or use a masking rect that moves with the head.
        // Masking is easier given our engine.
        return path;
    }
    
    // Generate full paths
    const ma5Pts = [];
    const ma20Pts = [];
    const bollUPts = [];
    const bollLPts = [];
    const rsiPts = [];

    klineEls.forEach((k, i) => {
        // MA5
        // Sanitize data to prevent NaN in SVG paths
        if (ma5Data[i] !== null && Number.isFinite(ma5Data[i])) {
            const y = getScreenY(ma5Data[i]);
            if (Number.isFinite(y)) {
                ma5Pts.push({ x: k.x, y: y });
            }
        }
        
        // Bollinger (and MA20 which is middle band)
        const b = bollData[i];
        if (b && b.middle !== null && Number.isFinite(b.middle)) {
            const ym = getScreenY(b.middle);
            const yu = getScreenY(b.upper);
            const yl = getScreenY(b.lower);
            
            if (Number.isFinite(ym)) ma20Pts.push({ x: k.x, y: ym });
            if (Number.isFinite(yu)) bollUPts.push({ x: k.x, y: yu });
            if (Number.isFinite(yl)) bollLPts.push({ x: k.x, y: yl });
        }

        // RSI
        if (rsiData[i] !== null && Number.isFinite(rsiData[i])) {
            const yr = getRsiY(rsiData[i]);
            if (Number.isFinite(yr)) {
                rsiPts.push({ x: k.x, y: yr });
            }
        }
    });
    
    const ma5 = createCurvePath(ma5Pts, '#E6CFAA'); // Morandi Sand
    const ma20 = createCurvePath(ma20Pts, '#B5B0D6'); // Morandi Lavender
    const bollU = createCurvePath(bollUPts, '#A4B0BE', '5 5'); // Morandi Steel Blue
    const bollL = createCurvePath(bollLPts, '#A4B0BE', '5 5');
    const rsiLine = createCurvePath(rsiPts, '#FF9F43'); // RSI Orange

    // Group for indicators
    const indicatorGroup = createSVG('g', {});
    indicatorGroup.appendChild(rsiLine); // Draw RSI first (behind?) or separate?
    indicatorGroup.appendChild(ma5);
    indicatorGroup.appendChild(ma20);
    indicatorGroup.appendChild(bollU);
    indicatorGroup.appendChild(bollL);
    g.appendChild(indicatorGroup);
    
    // Mask for indicators (Reveal as we go)
    // A rect that expands from left to right covers the indicators?
    // Actually we need a clipPath.
    
    const clipId = 'indicator-clip-' + Math.random().toString(36).substr(2, 9);
    const clipPath = createSVG('clipPath', { id: clipId });
    const clipRect = createSVG('rect', {
        x: cx - 100, y: 0, // Start slightly before cx
        width: 0, height: CONFIG.height * 2
    });
    clipPath.appendChild(clipRect);
    // Append clipPath to defs or sceneRoot (must be in defs usually, but inline works in some browsers)
    // Let's put in sceneRoot
    state.sceneRoot.appendChild(clipPath);
    
    indicatorGroup.setAttribute('clip-path', `url(#${clipId})`);
    
    // Animate Clip Rect Width
    // It needs to cover from startX to current head X.
    // StartX is cx. Head goes to cx + totalDistance.
    // So width goes from 0 to totalDistance + padding.
    sceneObj.animator.to(clipRect, totalDrawTime, { width: [0, totalDistance + 200] }, revealStart, Easing.linear);
    
    // Volume Bars (Also reveal with mask? Or reusing existing reveal loop)
    // Reuse existing kline loop for opacity is fine, but let's add volume bars there too.
    
    const volGroup = createSVG('g', {});
    g.appendChild(volGroup);

    let volBottomY = CONFIG.height - 5;
    const stageEl = document.getElementById('stage');
    const subtitleContainerEl = document.getElementById('subtitle-container');
    if (stageEl && subtitleContainerEl) {
        const stageRect = stageEl.getBoundingClientRect();
        const subRect = subtitleContainerEl.getBoundingClientRect();
        if (stageRect.height > 0) {
            const bottomInStagePx = subRect.bottom - stageRect.top;
            const bottomRatio = bottomInStagePx / stageRect.height;
            const bottomInSvg = bottomRatio * CONFIG.height;
            if (Number.isFinite(bottomInSvg)) {
                volBottomY = Math.max(0, Math.min(CONFIG.height - 2, bottomInSvg + 8));
            }
        }
    }
    
    // Calculate Max Volume for Scaling (Prevent overlap)
    let maxVol = 0;
    sceneObj.data.forEach(d => {
        if (d.vol > maxVol) maxVol = d.vol;
    });
    const maxVolHeight = CONFIG.height * 0.15; // Max 15% of screen height
    const volScale = maxVol > 0 ? maxVolHeight / maxVol : 0;

    klineEls.forEach((k, i) => {
        if (i===0) return;
        
        const rawVol = sceneObj.data[i].vol; 
        const h = Math.max(2, rawVol * volScale); // Ensure at least 2px visible
        
        const color = k.isUp ? 'var(--candle-up)' : 'var(--candle-down)';
        const volY = volBottomY - h;
        
        const volBar = createSVG('rect', {
            x: k.x - 6, y: volY,
            width: 12, height: h,
            fill: color, opacity: 0
        });
        volGroup.appendChild(volBar);
        
        // Reveal with candle
        const delay = revealStart + i * msPerCandle;
        sceneObj.animator.to(volBar, 300, { opacity: [0, 0.6] }, delay);
    });

    // --- Legend Animation ---
    // Integrated Legend in the right 1/3 space (No box)
    
    // Start after subtitles finish (approx 20s)
    // Subtitles end at: 3+2.5+2.5+3+4+5 = 20s
    const legendStart = 20000;

    const legendConfig = [
        { key: 'MA5', color: '#E6CFAA', descEn: 'Short-term Trend (5 Days)', descCn: '短期趋势线 (5日均线)' },
        { key: 'MA20', color: '#B5B0D6', descEn: 'Medium-term Trend (20 Days)', descCn: '中期趋势线 (20日均线)' },
        { key: 'BOLL', color: '#A4B0BE', descEn: 'Volatility Bands', descCn: '布林带 (波动范围)', dashed: true },
        { key: 'RSI', color: '#FF9F43', descEn: 'Relative Strength Index', descCn: '相对强弱指数' }
    ];

    // Position: Just to the right of the 60% chart area
    const legendX = CONFIG.width * 0.62; 
    const legendY = CONFIG.height * 0.35; // Start a bit higher
    const rowHeight = 140; // Spacing between items

    // 2. Items
    legendConfig.forEach((item, i) => {
        const rowY = legendY + i * rowHeight;
        const itemDelay = legendStart + i * 1500; // Sequence

        // Group for this item
        const g = createSVG('g', { opacity: 0 });
        state.labelsLayer.appendChild(g);

        // Color Marker (Vertical Bar style for modern look)
        if (item.dashed) {
            // "Five bars, hide even ones" -> 5 segments total, show 1, 3, 5
            const totalHeight = 60;
            const segmentCount = 5;
            const segmentH = totalHeight / segmentCount;
            
            for (let k = 0; k < segmentCount; k++) {
                // Show only 0, 2, 4 (1st, 3rd, 5th)
                if (k % 2 === 0) {
                    const seg = createSVG('rect', {
                        x: legendX, y: rowY + 5 + k * segmentH, 
                        width: 8, height: segmentH - 2, // -2 for small gap visual if needed, or exact? 
                        // User said "hide even ones". If we have 5 slots:
                        // Slot 0: Draw
                        // Slot 1: Gap
                        // Slot 2: Draw
                        // ...
                        // If we fill the slot fully (height 12), and gap is empty slot (height 12), the spacing is naturally 12.
                        // So height should be segmentH (12).
                        // Let's make it slightly smaller than 12 if we want a cleaner look? 
                        // "Hide even ones" implies the space is there.
                        // Let's use full height of the slot but maybe -1 for pixel crispness? 
                        // Actually, if we just skip k=1, gap is 12px.
                        height: segmentH, 
                        fill: item.color, rx: 2
                    });
                    g.appendChild(seg);
                }
            }
        } else {
            // Standard Solid Bar
            const marker = createSVG('rect', {
                x: legendX, y: rowY + 5, width: 8, height: 60,
                fill: item.color, rx: 2
            });
            g.appendChild(marker);
        }

        // Key (e.g., MA5)
        const label = createSVG('text', {
            x: legendX + 30, y: rowY + 30,
            fill: item.color, 'font-size': '48px', 'font-weight': '800',
            text: item.key, 'font-family': 'var(--font-mono)', 'dominant-baseline': 'middle'
        });
        g.appendChild(label);

        // Description English
        const descEn = createSVG('text', {
            x: legendX + 30, y: rowY + 70,
            fill: 'var(--text-primary)', 'font-size': '28px', 'font-weight': '500',
            text: item.descEn, 'dominant-baseline': 'middle'
        });
        g.appendChild(descEn);
        
        // Description Chinese
        const descCn = createSVG('text', {
            x: legendX + 30, y: rowY + 105,
            fill: 'var(--text-secondary)', 'font-size': '22px', 'font-weight': '400',
            text: item.descCn, 'dominant-baseline': 'middle', 'font-family': 'var(--font-cn)'
        });
        g.appendChild(descCn);

        // Animation Sequence
        // Appear all together nicely
        // Use translateX for smooth numeric interpolation (supported by our Animator update)
        sceneObj.animator.to(g, 800, { opacity: [0, 1], translateX: [50, 0] }, itemDelay, Easing.easeOutQuad);
    });
    
    // Fade out while still moving
    // Move disappearance earlier (fade begins well before draw ends).
    const fadeOutTime = Math.max(revealStart, drawEndTime - 2500);
    const fadeDuration = 1500;

    // Camera already pans linearly from revealStart -> drawEndTime, so it continues moving during fade.
    sceneObj.animator.to(g, fadeDuration, { opacity: [1, 0] }, fadeOutTime);
    sceneObj.animator.to(state.labelsLayer, fadeDuration, { opacity: [1, 0] }, fadeOutTime);

    // Scene Update Callback
    sceneObj.onUpdate = (t) => {
        cameraGroup.setAttribute('transform', `translate(${sceneObj.camera.x}, ${sceneObj.camera.y}) scale(${sceneObj.camera.scale})`);
        
        // Apply transform animation if any (handling generic props in animator is tricky if not direct, 
        // but our Animator handles object props. SVG transform attribute needs manual string building if animating x/y/scale separately on object)
        // For the legend group 'g', we animated 'opacity' (direct attribute) and 'transform' (needs special handling in Animator? 
        // Our simple Animator might not handle 'transform' string interpolation perfectly unless it's a single numeric value.
        // Let's assume standard CSS-like opacity works on SVG elements via style or attribute.
        // If our Animator is simple numeric interpolation, 'transform' string wont work.
        // Let's check Animator. 
        // If it doesn't support strings, we should remove transform animation or handle it manually.
        // Given the simple engine, let's stick to Opacity only to be safe, or assume the user wants it simple.
        // I will remove the transform animation from the 'to' call above to avoid breaking it if Animator is simple.
    };

    return sceneObj;
}

function updateKline(sceneObj, relativeTime) {
    sceneObj.animator.update(relativeTime);
    if (sceneObj.onUpdate) sceneObj.onUpdate(relativeTime);
    
    // Subtitles
    setSubtitleAuto('kline', relativeTime, [
        { 
            en: "BTC price changes in real-time.", 
            cn: "BTC的价格随买卖实时变化。",
            durationMs: 3000
        },
        {
            en: "K-lines record this process.",
            cn: "而K线记录了变化的过程。",
            durationMs: 2500
        },
        {
            en: "Four points define a candle.",
            cn: "开盘、收盘、最高、最低。",
            durationMs: 2500
        },
        {
            en: "Connect them to form a complete K-line.",
            cn: "连接四个价格点，形成一根完整的K线。",
            durationMs: 3000
        },
        {
            en: "Multiple K-lines form a trend.",
            cn: "多根K线连在一起，形成价格走势图。",
            durationMs: 4000
        },
        {
            en: "K-lines + Indicators = Technical Analysis.",
            cn: "K线与各类技术指标，构成了技术分析的基础。",
            durationMs: 5000
        }
    ]);
}

// ... (rest of the code)

// --- Main Loop ---
const SCENES = [
    { id: 'intro', duration: 8000, setup: setupIntro, update: updateIntro, label: 'Intro / 简介' },
    { id: 'kline', duration: 32600, setup: setupKline, update: updateKline, label: 'K-Line Demo / K线演示' },
    { id: 'logo', duration: 2600, setup: setupLogo, update: updateLogo, label: 'Logo / 标识' },
    { id: 'arch', duration: 10000, setup: setupArch, update: updateArch, label: 'Architecture / 系统架构' },
    { id: 'models', duration: 29500, setup: setupModels, update: updateModels, label: 'Quant Models / 量化模型' },
    { id: 'news', duration: 15000, setup: setupNews, update: updateNews, label: 'News Intelligence / 新闻情报' },
    { id: 'fusion', duration: 11000, setup: setupFusion, update: updateFusion, label: 'Data Fusion / 数据融合' },
    { id: 'email', duration: 14000, setup: setupEmailScene, update: updateEmailScene, label: 'Email Notification / 邮件通知' }
];

let currentSceneIndex = -1;
let currentSceneObj = null;
let sceneStartTime = 0;

// Init Timeline Markers
function setupTimelineMarkers() {
    if (!state.ui.timelineContainer) return;
    
    // Remove existing markers if any
    const existing = state.ui.timelineContainer.querySelectorAll('.timeline-marker');
    existing.forEach(e => e.remove());
    
    const totalDuration = SCENES.reduce((acc, s) => acc + s.duration, 0);
    let accumulatedTime = 0;
    
    SCENES.forEach(scene => {
        // Place marker at start of scene
        const percent = (accumulatedTime / totalDuration) * 100;
        
        const marker = document.createElement('div');
        marker.className = 'timeline-marker';
        marker.style.left = `${percent}%`;
        marker.setAttribute('data-label', scene.label || scene.id);
        
        state.ui.timelineContainer.appendChild(marker);
        
        accumulatedTime += scene.duration;
    });
}

function loadScene(index) {
    if (index < 0 || index >= SCENES.length) return;
    
    // Cleanup prev
    if (currentSceneObj) {
        clearGroup(state.sceneRoot);
        clearGroup(state.labelsLayer);
    }
    
    currentSceneIndex = index;
    const sceneDef = SCENES[index];
    currentSceneObj = sceneDef.setup();
    sceneStartTime = state.time;
}

function loop(timestamp) {
    if (!state.lastFrameTime) state.lastFrameTime = timestamp;
    const dt = timestamp - state.lastFrameTime;
    state.lastFrameTime = timestamp;
    
    // Basic Time Management
    state.time += dt;
    
    if (currentSceneIndex === -1) {
        loadScene(0);
    }
    
    const sceneDef = SCENES[currentSceneIndex];
    const relativeTime = state.time - sceneStartTime;
    
    if (relativeTime > sceneDef.duration) {
        // Next scene
        if (currentSceneIndex < SCENES.length - 1) {
            loadScene(currentSceneIndex + 1);
        } else {
            // Stop at end (No Loop)
            state.isPlaying = false;
            return; // Stop the loop
        }
    } else {
        sceneDef.update(currentSceneObj, relativeTime);
    }
    
    // UI Updates
    const totalDuration = SCENES.reduce((acc, s) => acc + s.duration, 0);
    let elapsed = 0;
    for (let i = 0; i < currentSceneIndex; i++) {
        elapsed += SCENES[i].duration;
    }
    elapsed += relativeTime;
    
    const progress = Math.min(100, (elapsed / totalDuration) * 100);
    if (state.ui.progressBar) {
        state.ui.progressBar.style.width = `${progress}%`;
    }
    
    requestAnimationFrame(loop);
}

function resetPlaybackState() {
    state.time = 0;
    state.lastFrameTime = 0;
    state.isPlaying = true;
    currentSceneIndex = -1;
    currentSceneObj = null;
    sceneStartTime = 0;
    if (state.ui.progressBar) state.ui.progressBar.style.width = '0%';
}

function startPlayback() {
    if (state.isPlaying) return;
    resetPlaybackState();
    requestAnimationFrame(loop);
}

// --- Interaction ---
if (state.ui.timelineContainer) {
    state.ui.timelineContainer.addEventListener('click', (e) => {
        const rect = state.ui.timelineContainer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const clickRatio = Math.max(0, Math.min(1, x / rect.width));
        
        const totalDuration = SCENES.reduce((acc, s) => acc + s.duration, 0);
        const targetTime = clickRatio * totalDuration;
        
        let accumulatedTime = 0;
        let targetSceneIndex = 0;
        let targetRelativeTime = 0;
        
        for (let i = 0; i < SCENES.length; i++) {
            if (targetTime < accumulatedTime + SCENES[i].duration) {
                targetSceneIndex = i;
                targetRelativeTime = targetTime - accumulatedTime;
                break;
            }
            accumulatedTime += SCENES[i].duration;
        }
        
        // Load the target scene
        // If it's the same scene, loadScene will reset it, which is what we want for seeking usually
        // But maybe we can optimize? For now, full reload is safer to ensure state consistency.
        loadScene(targetSceneIndex);
        
        // Adjust sceneStartTime so that (state.time - sceneStartTime) equals targetRelativeTime
        sceneStartTime = state.time - targetRelativeTime;
        
        // Force immediate update
        const sceneDef = SCENES[currentSceneIndex];
        sceneDef.update(currentSceneObj, targetRelativeTime);
    });
}

// Start
setupTimelineMarkers();

window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        startPlayback();
    }
});
