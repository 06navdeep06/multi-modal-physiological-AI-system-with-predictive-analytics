"use strict";

/**
 * PhysioAI Monitor — Research-Grade Dashboard
 *
 * Architecture:
 *   Primary data: SSE /api/stream  (~4 Hz push)
 *   Fallback:     HTTP polling /api/metrics  (1 Hz)
 *   Session stats: /api/stats  (every 15 s)
 *
 * Visualisations:
 *   - Chart.js time-series charts (6 panels)
 *   - Canvas PPG waveform (scrolling, ~20 Hz repaint)
 *   - Canvas Poincaré plot (RR(n) vs RR(n+1))
 *   - Canvas ANS balance gauge (sympathetic / parasympathetic donut)
 */

// ============================================================
// CONFIG
// ============================================================
const CFG = {
    API:             '/api',
    POLL_MS:         1000,
    STATS_MS:        15000,
    CHART_PTS:       60,
    RECONNECT_MS:    4000,
    MAX_RETRIES:     10,
    MAX_ALERT_LOG:   100,
    PPG_SCROLL_MS:   50,      // repaint PPG canvas every 50 ms (20 fps)
};

// ============================================================
// GLOBAL STATE
// ============================================================
let charts         = {};
let chartData      = {};
let connState      = 'disconnected';
let retryCount     = 0;
let pollTimer      = null;
let statsTimer     = null;
let sessionStart   = Date.now();
let sessionTick    = null;
let alertLog       = [];
let sseSource      = null;
let sseAlive       = false;
let bannerTimer    = null;
let ppgScrollTimer = null;

// Waveform buffer — receives samples from SSE, rendered by Canvas loop
const PPG_BUF_SIZE = 300;
let ppgBuffer      = new Array(PPG_BUF_SIZE).fill(0);

// Poincaré data
let rrHistory      = [];

// ANS state
let ansState = { lf: 0, hf: 0, ratio: 0, si: 0 };

// ============================================================
// COLOUR PALETTE  (mirrors CSS variables)
// ============================================================
const C = {
    cardiac:   '#00d4ff',
    oxygen:    '#00e5a0',
    resp:      '#4f8ef7',
    hrv:       '#a78bfa',
    fatigue:   '#fb923c',
    stress:    '#f87171',
    emotion:   '#f472b6',
    coherence: '#34d399',
    success:   '#10b981',
    warning:   '#f59e0b',
    danger:    '#ef4444',
    muted:     '#4b5e7a',
};

const EMOTION_COLORS = {
    anger:    C.danger,    disgust: '#94a3b8', fear:      C.hrv,
    happiness:C.warning,   neutral: C.resp,    sadness:   '#6366f1',
    surprise: C.coherence,
};

// ============================================================
// CHART DEFINITIONS
// ============================================================
const CHART_DEFS = [
    { id: 'chartHR',      label: 'HR (BPM)',         color: C.cardiac,   min: 40,  max: 180, fill: true  },
    { id: 'chartSpO2',    label: 'SpO₂ (%)',          color: C.oxygen,    min: 88,  max: 100, fill: false },
    { id: 'chartResp',    label: 'Resp (BPM)',         color: C.resp,      min: 5,   max: 40,  fill: true  },
    { id: 'chartHRV',     label: 'RMSSD (ms)',         color: C.hrv,       min: 0,   max: 100, fill: false },
    { id: 'chartFatigue', label: 'Fatigue',            color: C.fatigue,   min: 0,   max: 1,   fill: true,
      extra: { label: 'PERCLOS (norm)', color: C.stress, normalize: 0.01 } },
    { id: 'chartStress',  label: 'Stress Score',       color: C.stress,    min: 0,   max: 1,   fill: false },
];

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    applyStoredTheme();
    setupCharts();
    initChartData();
    startPPGCanvas();
    startSessionTimer();
    connectSSE();
    statsTimer = setInterval(fetchStats, CFG.STATS_MS);
    fetchStats();
    drawANSGauge(0, 0);         // blank initial state
    drawPoincare([]);
});

window.addEventListener('beforeunload', () => {
    sseSource?.close();
    [pollTimer, statsTimer, sessionTick, ppgScrollTimer].forEach(t => clearInterval(t));
});

// ============================================================
// SSE / POLLING
// ============================================================
function connectSSE() {
    sseSource?.close();
    sseSource = null;
    try {
        sseSource = new EventSource(`${CFG.API}/stream`);

        sseSource.onopen = () => {
            sseAlive   = true;
            retryCount = 0;
            setConnStatus('ok');
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        };

        sseSource.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                retryCount = 0;
                setConnStatus('ok');
                onMetrics({ data });
            } catch (_) {}
        };

        sseSource.onerror = () => {
            sseAlive = false;
            setConnStatus('error');
            sseSource.close();
            sseSource = null;
            if (!pollTimer) pollTimer = setInterval(pollFallback, CFG.POLL_MS);
            setTimeout(connectSSE, CFG.RECONNECT_MS);
        };
    } catch (_) {
        if (!pollTimer) pollTimer = setInterval(pollFallback, CFG.POLL_MS);
    }
}

function pollFallback() {
    fetch(`${CFG.API}/metrics`)
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then(d => { retryCount = 0; setConnStatus('ok'); onMetrics(d);
                     if (sseAlive && pollTimer) { clearInterval(pollTimer); pollTimer = null; } })
        .catch(() => {
            retryCount++;
            setConnStatus('error');
            if (retryCount >= CFG.MAX_RETRIES)
                showBanner('Backend unreachable — retrying…', 'error');
        });
}

function fetchStats() {
    fetch(`${CFG.API}/stats`)
        .then(r => r.json())
        .then(d => updateSessionStats(d))
        .catch(() => {});
}

// ============================================================
// MAIN METRICS HANDLER
// ============================================================
function onMetrics(payload) {
    if (!payload?.data) return;
    const m = payload.data;

    // --- Warmup ---
    const wu = document.getElementById('warmup-notice');
    if (wu) wu.style.display = m.warmup ? 'flex' : 'none';
    const wuf = document.getElementById('warmup-fill');
    if (wuf) wuf.style.width = `${m.warmup_pct ?? 0}%`;
    const wup = document.getElementById('warmup-pct');
    if (wup) wup.textContent = m.warmup ? `${m.warmup_pct ?? 0}%` : '';

    // --- Face indicator ---
    const fi = document.getElementById('face-indicator');
    if (fi) {
        fi.textContent = m.face_detected ? '✔ Face' : '✖ No face';
        fi.className   = `face-indicator ${m.face_detected ? 'face-ok' : 'face-none'}`;
    }

    // --- Core scalar metrics ---
    setText('hr',          fmt(m.hr,         0));
    setText('spo2',        fmt(m.spo2,        1));
    setText('resp',        fmt(m.resp,        1));
    setText('rmssd',       fmt(m.rmssd,       1));
    setText('fatigue',     fmt(m.fatigue,     2));
    setText('perclos',     fmt(m.perclos,     1));
    setText('blink',       fmt(m.blink,       1));
    setText('microsleeps', m.microsleeps ?? '--');
    setText('timestamp',   m.timestamp ? new Date(m.timestamp).toLocaleTimeString() : '--');

    // --- Header vitals strip ---
    setText('hv-hr',    m.hr    != null ? `${fmt(m.hr, 0)} BPM` : '--');
    setText('hv-spo2',  m.spo2  != null ? `${fmt(m.spo2, 1)} %` : '--');
    setText('hv-rmssd', m.rmssd != null ? `${fmt(m.rmssd, 1)} ms` : '--');
    setText('hv-resp',  m.resp  != null ? `${fmt(m.resp, 1)} BPM` : '--');
    setText('hv-stress', m.stress ?? '--');

    // --- Stress & Drowsiness ---
    updateStress(m.stress, m.stress_score);
    updateDrowsiness(m.drowsiness);
    updateSQ(m.signal_quality);

    // --- Metric card colouring ---
    colourMetric('metric-hr',         m.hr,      40,  55, 100, 120);
    colourMetric('metric-spo2',        m.spo2,    90,  95, null, null);
    colourMetric('metric-resp',        m.resp,     8,  10,  20,  30);
    colourMetric('metric-rmssd',       m.rmssd,   10,  20, null, null);
    colourMetric('metric-fatigue',     m.fatigue, null, null, 0.6, 0.8);

    // --- Emotion ---
    if (m.emotion_scores) updateEmotionBars(m.emotion, m.emotion_scores);

    // --- PPG waveform buffer ---
    if (Array.isArray(m.ppg_signal) && m.ppg_signal.length > 0) {
        const sig = m.ppg_signal;
        // Push new samples into the ring buffer
        for (const v of sig) {
            ppgBuffer.push(v);
        }
        while (ppgBuffer.length > PPG_BUF_SIZE) ppgBuffer.shift();
    }

    // PPG overlay label
    const ppgHREl = document.getElementById('ppg-hr-label');
    if (ppgHREl) ppgHREl.textContent = m.hr != null ? `${fmt(m.hr, 0)} BPM` : '';

    // --- RR / Poincaré ---
    if (Array.isArray(m.rr_intervals) && m.rr_intervals.length > 3) {
        rrHistory = m.rr_intervals.slice(-50);
        drawPoincare(rrHistory);
    }

    // --- Frequency-domain HRV ---
    updateHRVBands(m.hrv_freq);
    updateANS(m.lf_power, m.hf_power, m.lf_hf_ratio, m.stress_index);
    updateCoherence(m.coherence);

    // --- Charts ---
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    pushChart('chartHR',      ts, m.hr);
    pushChart('chartSpO2',    ts, m.spo2);
    pushChart('chartResp',    ts, m.resp);
    pushChart('chartHRV',     ts, m.rmssd);
    pushChart('chartFatigue', ts, m.fatigue,
              m.perclos != null ? m.perclos * 0.01 : null);
    pushChart('chartStress',  ts, m.stress_score);

    // --- Alerts ---
    if (m.alerts?.length) {
        m.alerts.forEach(a => addAlert(a.message, a.severity, a.timestamp));
        showBanner(m.alerts[0].message,
                   m.alerts[0].severity === 'critical' ? 'error' : 'warning');
    }
}

// ============================================================
// PPG WAVEFORM CANVAS
// ============================================================
function startPPGCanvas() {
    ppgScrollTimer = setInterval(renderPPG, CFG.PPG_SCROLL_MS);
}

function renderPPG() {
    const canvas = document.getElementById('ppgWaveform');
    if (!canvas) return;

    // Size canvas to its CSS width
    const W = canvas.clientWidth;
    const H = 90;
    if (canvas.width !== W) canvas.width = W;
    canvas.height = H;

    const ctx   = canvas.getContext('2d');
    const dark  = document.documentElement.getAttribute('data-theme') !== 'light';
    const bgCol = dark ? '#0c1120' : '#f0f4f9';
    const gridC = dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)';
    const lineC = C.cardiac;

    ctx.fillStyle = bgCol;
    ctx.fillRect(0, 0, W, H);

    // Horizontal grid lines
    ctx.strokeStyle = gridC;
    ctx.lineWidth   = 1;
    for (let y = 0; y <= H; y += H / 4) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Zero line
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();

    const buf  = ppgBuffer;
    const n    = buf.length;
    const midY = H / 2;
    const amp  = H * 0.38;

    if (n < 2) return;

    // Gradient fill below waveform
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0,   hexAlpha(lineC, 0.25));
    grad.addColorStop(0.5, hexAlpha(lineC, 0.08));
    grad.addColorStop(1,   hexAlpha(lineC, 0.0));

    ctx.beginPath();
    ctx.moveTo(0, midY - buf[0] * amp);
    for (let i = 1; i < n; i++) {
        const x = (i / (n - 1)) * W;
        const y = midY - buf[i] * amp;
        ctx.lineTo(x, y);
    }
    // Close fill path
    ctx.lineTo(W, midY);
    ctx.lineTo(0, midY);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Waveform line
    ctx.beginPath();
    ctx.strokeStyle = lineC;
    ctx.lineWidth   = 1.8;
    ctx.shadowColor = lineC;
    ctx.shadowBlur  = dark ? 6 : 0;
    ctx.moveTo(0, midY - buf[0] * amp);
    for (let i = 1; i < n; i++) {
        const x = (i / (n - 1)) * W;
        const y = midY - buf[i] * amp;
        ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
}

// ============================================================
// POINCARÉ PLOT
// ============================================================
function drawPoincare(rrList) {
    const canvas = document.getElementById('poincareCanvas');
    if (!canvas) return;
    const W = canvas.clientWidth || 280;
    const H = 160;
    if (canvas.width !== W) canvas.width = W;
    canvas.height = H;

    const ctx  = canvas.getContext('2d');
    const dark = document.documentElement.getAttribute('data-theme') !== 'light';
    const bg   = dark ? '#0c1120' : '#f0f4f9';
    const grid = dark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)';

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    if (!rrList || rrList.length < 4) {
        ctx.fillStyle = dark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.18)';
        ctx.font      = '11px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Awaiting RR data…', W / 2, H / 2);
        return;
    }

    // Axis range
    const all  = rrList;
    const lo   = Math.max(300, Math.min(...all) - 30);
    const hi   = Math.min(2000, Math.max(...all) + 30);
    const span = hi - lo + 1e-6;

    const toX = v => ((v - lo) / span) * (W - 20) + 10;
    const toY = v => H - (((v - lo) / span) * (H - 20) + 10);

    // Grid lines
    ctx.strokeStyle = grid;
    ctx.lineWidth   = 1;
    for (let t = Math.ceil(lo / 100) * 100; t <= hi; t += 100) {
        const px = toX(t);
        const py = toY(t);
        ctx.beginPath(); ctx.moveTo(px, 0);  ctx.lineTo(px, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0,  py); ctx.lineTo(W,  py); ctx.stroke();
    }

    // Identity line  RR(n) = RR(n+1)
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.1)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(lo), toY(lo));
    ctx.lineTo(toX(hi), toY(hi));
    ctx.stroke();
    ctx.setLineDash([]);

    // SD1 / SD2 ellipse approximation
    const mean = all.reduce((a, b) => a + b, 0) / all.length;
    const diffs = [];
    for (let i = 0; i < all.length - 1; i++) diffs.push(all[i + 1] - all[i]);
    const sd1 = Math.sqrt(diffs.reduce((a, b) => a + b * b, 0) / diffs.length) / Math.sqrt(2);
    const sd2 = Math.sqrt(all.slice(0, -1).reduce((a, v, i) => {
        const d = (v + all[i + 1]) / 2 - mean; return a + d * d;
    }, 0) / (all.length - 1)) * Math.sqrt(2);

    const cx = toX(mean);
    const cy = toY(mean);
    const rx = (sd2 / span) * (W - 20);
    const ry = (sd1 / span) * (H - 20);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-Math.PI / 4);
    ctx.beginPath();
    ctx.ellipse(0, 0, Math.max(rx, 3), Math.max(ry, 2), 0, 0, 2 * Math.PI);
    ctx.strokeStyle = hexAlpha(C.hrv, 0.35);
    ctx.lineWidth   = 1.5;
    ctx.stroke();
    ctx.fillStyle   = hexAlpha(C.hrv, 0.06);
    ctx.fill();
    ctx.restore();

    // Scatter points
    const pairs = [];
    for (let i = 0; i < all.length - 1; i++) pairs.push([all[i], all[i + 1]]);

    for (const [rn, rn1] of pairs) {
        ctx.beginPath();
        ctx.arc(toX(rn), toY(rn1), 3, 0, 2 * Math.PI);
        ctx.fillStyle   = hexAlpha(C.cardiac, 0.75);
        ctx.shadowColor = C.cardiac;
        ctx.shadowBlur  = dark ? 4 : 0;
        ctx.fill();
    }
    ctx.shadowBlur = 0;

    // SD1 / SD2 annotation
    ctx.font      = '9px JetBrains Mono, monospace';
    ctx.fillStyle = dark ? 'rgba(167,139,250,0.7)' : 'rgba(100,80,200,0.75)';
    ctx.textAlign = 'left';
    ctx.fillText(`SD1 ${sd1.toFixed(1)} ms`, 8, H - 18);
    ctx.fillText(`SD2 ${sd2.toFixed(1)} ms`, 8, H - 8);
}

// ============================================================
// ANS BALANCE GAUGE  (canvas donut)
// ============================================================
function drawANSGauge(lfPct, hfPct) {
    const canvas = document.getElementById('ansGaugeCanvas');
    if (!canvas) return;
    const S  = 110;
    canvas.width = canvas.height = S;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, S, S);

    const cx = S / 2, cy = S / 2, r = 40, thick = 14;
    const dark = document.documentElement.getAttribute('data-theme') !== 'light';

    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)';
    ctx.lineWidth   = thick;
    ctx.stroke();

    const total = lfPct + hfPct + 1e-9;
    const lfAng = (lfPct / total) * 2 * Math.PI;
    const hfAng = (hfPct / total) * 2 * Math.PI;
    const start = -Math.PI / 2;

    // LF arc (sympathetic)
    if (lfAng > 0.01) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, start, start + lfAng);
        ctx.strokeStyle = C.stress;
        ctx.lineWidth   = thick;
        ctx.shadowColor = C.stress;
        ctx.shadowBlur  = dark ? 8 : 0;
        ctx.stroke();
        ctx.shadowBlur  = 0;
    }

    // HF arc (parasympathetic)
    if (hfAng > 0.01) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, start + lfAng, start + lfAng + hfAng);
        ctx.strokeStyle = C.coherence;
        ctx.lineWidth   = thick;
        ctx.shadowColor = C.coherence;
        ctx.shadowBlur  = dark ? 8 : 0;
        ctx.stroke();
        ctx.shadowBlur  = 0;
    }

    // Centre text
    const ratio = lfPct > 0 && hfPct > 0 ? (lfPct / hfPct).toFixed(1) : '--';
    ctx.fillStyle = dark ? 'rgba(226,232,240,0.9)' : '#0f172a';
    ctx.font      = 'bold 13px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(ratio, cx, cy - 6);
    ctx.font      = '9px Inter, sans-serif';
    ctx.fillStyle = dark ? 'rgba(148,163,184,0.7)' : '#64748b';
    ctx.fillText('LF/HF', cx, cy + 8);
}

function updateANS(lf, hf, ratio, si) {
    const lfPct  = lf != null ? Math.round(lf * 10000) : 0;
    const hfPct  = hf != null ? Math.round(hf * 10000) : 0;

    drawANSGauge(lfPct, hfPct);

    setText('ans-lf-val',    lf    != null ? `${(lf * 1000).toFixed(2)} ms²` : '--');
    setText('ans-hf-val',    hf    != null ? `${(hf * 1000).toFixed(2)} ms²` : '--');
    setText('ans-ratio-val', ratio != null ? ratio.toFixed(2)                : '--');
    setText('ans-si-val',    si    != null ? si.toFixed(2)                   : '--');
}

// ============================================================
// CARDIAC COHERENCE
// ============================================================
function updateCoherence(coh) {
    const fill = document.getElementById('coherence-fill');
    const val  = document.getElementById('coherence-value');
    if (fill) fill.style.width = coh != null ? `${Math.min(coh, 100)}%` : '0%';
    if (val)  val.textContent  = coh != null ? `${coh.toFixed(1)}%` : '--%';
}

// ============================================================
// HRV FREQUENCY BANDS
// ============================================================
function updateHRVBands(freq) {
    if (!freq) return;

    const { vlf_power = 0, lf_power = 0, hf_power = 0,
            lf_hf_ratio, lf_pct = 0, hf_pct = 0 } = freq;

    const total = vlf_power + lf_power + hf_power + 1e-12;

    // Band bars (normalise each to % of total)
    const vPct = Math.min(100, vlf_power / total * 100);
    const lPct = Math.min(100, lf_power  / total * 100);
    const hPct = Math.min(100, hf_power  / total * 100);

    const bv = document.getElementById('band-vlf');
    const bl = document.getElementById('band-lf');
    const bh = document.getElementById('band-hf');
    if (bv) bv.style.width = `${vPct.toFixed(1)}%`;
    if (bl) bl.style.width = `${lPct.toFixed(1)}%`;
    if (bh) bh.style.width = `${hPct.toFixed(1)}%`;

    setText('band-vlf-val', `${(vlf_power * 1e6).toFixed(0)} µs²`);
    setText('band-lf-val',  `${(lf_power  * 1e6).toFixed(0)} µs²`);
    setText('band-hf-val',  `${(hf_power  * 1e6).toFixed(0)} µs²`);

    // LF/HF ratio + interpretation
    if (lf_hf_ratio != null) {
        setText('lf-hf-ratio', lf_hf_ratio.toFixed(2));
        const interp = lf_hf_ratio < 0.5  ? 'Parasympathetic dominant'
                     : lf_hf_ratio < 1.5  ? 'Balanced ANS'
                     : lf_hf_ratio < 3.0  ? 'Sympathetic elevated'
                     :                      'High sympathetic load';
        setText('lf-hf-interp', interp);
    }
}

// ============================================================
// STRESS / DROWSINESS / SIGNAL QUALITY
// ============================================================
function updateStress(level, score) {
    const el = document.getElementById('stress');
    const sc = document.getElementById('stress-score');
    if (el) {
        el.textContent = level || '--';
        el.className   = `stress-badge${level ? ` stress-${level.toLowerCase()}` : ''}`;
    }
    if (sc) sc.textContent = score != null ? `Score: ${score.toFixed(3)}` : '';
}

function updateDrowsiness(level) {
    const el = document.getElementById('drowsiness');
    if (!el) return;
    el.textContent = level || '--';
    el.className   = `drowsiness-badge${level ? ` drowsy-${level.toLowerCase()}` : ''}`;
}

function updateSQ(sq) {
    const fill  = document.getElementById('sq-bar-fill');
    const text  = document.getElementById('sq-value-text');
    const badge = document.getElementById('sq-badge');
    const pct   = sq != null ? Math.round(sq * 100) : 0;
    const cls   = sqClass(sq);

    if (fill)  { fill.style.width = `${pct}%`;  fill.className  = `sq-bar-fill ${cls}`; }
    if (text)  text.textContent   = sq != null ? `${pct}%` : '--';
    if (badge) { badge.textContent = `SQ ${pct}%`; badge.className = `sq-badge ${cls}`; }
}

function sqClass(sq) {
    if (sq == null) return 'sq-unknown';
    if (sq >= 0.6)  return 'sq-good';
    if (sq >= 0.3)  return 'sq-fair';
    return 'sq-poor';
}

// ============================================================
// EMOTION BARS
// ============================================================
function updateEmotionBars(top, scores) {
    const lbl  = document.getElementById('emotion-primary');
    const conf = document.getElementById('emotion-confidence');
    const bars = document.getElementById('emotion-bars');
    if (!bars) return;

    if (lbl)  lbl.textContent  = top || '--';
    const topScore = top && scores[top] != null ? scores[top] : null;
    if (conf) conf.textContent = topScore != null ? `${Math.round(topScore * 100)}% conf.` : '';

    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    bars.innerHTML = sorted.map(([emo, p]) => {
        const pct = Math.round(p * 100);
        const col = EMOTION_COLORS[emo] || C.resp;
        return `<div class="emo-row">
            <span class="emo-name">${emo}</span>
            <div class="emo-track">
              <div class="emo-fill" style="width:${pct}%;background:${col}"></div>
            </div>
            <span class="emo-pct">${pct}%</span>
        </div>`;
    }).join('');
}

// ============================================================
// METRIC CARD COLOURING
// ============================================================
function colourMetric(id, val, critLo, warnLo, warnHi, critHi) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('metric-warn', 'metric-critical');
    if (val == null) return;
    if ((critLo != null && val < critLo) || (critHi != null && val > critHi)) {
        el.classList.add('metric-critical');
    } else if ((warnLo != null && val < warnLo) || (warnHi != null && val > warnHi)) {
        el.classList.add('metric-warn');
    }
}

// ============================================================
// CONNECTION STATUS
// ============================================================
function setConnStatus(state) {
    const el = document.getElementById('connection-status');
    if (!el) return;
    if (state === 'ok') {
        el.textContent = '● Live';
        el.className   = 'conn-badge conn-ok';
        connState      = 'ok';
    } else {
        el.textContent = '● Offline';
        el.className   = 'conn-badge conn-error';
        connState      = 'error';
    }
}

// ============================================================
// ALERT BANNER
// ============================================================
function showBanner(msg, type = 'warning') {
    const el = document.getElementById('alert-banner');
    if (!el) return;
    const icons = { warning: '⚠', error: '🚨', info: 'ℹ' };
    el.innerHTML     = `${icons[type] || '⚠'} ${msg}`;
    el.className     = `alert-banner alert-${type}`;
    el.style.display = 'flex';
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => { el.style.display = 'none'; }, 6000);
}

// ============================================================
// ALERT LOG
// ============================================================
function addAlert(msg, severity, ts) {
    // Deduplicate within last 5 entries
    if (alertLog.slice(-5).some(e => e.msg === msg)) return;
    alertLog.push({ msg, severity, ts });
    if (alertLog.length > CFG.MAX_ALERT_LOG) alertLog.shift();
    renderAlertLog();
}

function renderAlertLog() {
    const log = document.getElementById('alert-log');
    if (!log) return;
    if (!alertLog.length) {
        log.innerHTML = '<p class="no-alerts-msg">No alerts recorded.</p>';
        return;
    }
    log.innerHTML = [...alertLog].reverse().map(e => {
        const time = e.ts ? new Date(e.ts).toLocaleTimeString() : '';
        const cls  = e.severity === 'critical' ? 'log-critical' : 'log-warning';
        return `<div class="log-entry ${cls}">
            <span class="log-time">${time}</span>
            <span class="log-msg">${e.msg}</span>
        </div>`;
    }).join('');
}

function clearAlertLog() {
    alertLog = [];
    renderAlertLog();
}

// ============================================================
// SESSION STATS
// ============================================================
function updateSessionStats(data) {
    if (!data?.averages_last_hour) return;
    const a = data.averages_last_hour;
    setText('stat-avg-hr',   a.hr    != null ? `${fmt(a.hr,   0)} BPM` : '--');
    setText('stat-avg-spo2', a.spo2  != null ? `${fmt(a.spo2, 1)} %`   : '--');
    setText('stat-avg-resp', a.resp  != null ? `${fmt(a.resp, 1)} BPM` : '--');
    setText('stat-avg-rmssd',a.rmssd != null ? `${fmt(a.rmssd,1)} ms`  : '--');
    const hrRange = (a.hr_min != null && a.hr_max != null)
        ? `${fmt(a.hr_min, 0)}–${fmt(a.hr_max, 0)}`
        : '--';
    setText('stat-hr-range', hrRange);
    setText('stat-readings', a.readings ?? '--');
}

// ============================================================
// CHART.JS SETUP
// ============================================================
function setupCharts() {
    const dark = document.documentElement.getAttribute('data-theme') !== 'light';
    CHART_DEFS.forEach(def => {
        const canvas = document.getElementById(def.id);
        if (!canvas) return;

        const datasets = [{
            label:            def.label,
            data:             [],
            borderColor:      def.color,
            backgroundColor:  hexAlpha(def.color, 0.1),
            borderWidth:      2,
            tension:          0.35,
            fill:             def.fill,
            pointRadius:      0,
            pointHoverRadius: 4,
        }];

        if (def.extra) {
            datasets.push({
                label:           def.extra.label,
                data:            [],
                borderColor:     def.extra.color,
                backgroundColor: hexAlpha(def.extra.color, 0.06),
                borderWidth:     1.5,
                tension:         0.35,
                fill:            false,
                pointRadius:     0,
                borderDash:      [4, 3],
            });
        }

        charts[def.id] = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets },
            options: {
                responsive:          true,
                maintainAspectRatio: true,
                animation:           { duration: 0 },
                plugins: {
                    legend: {
                        display: !!def.extra,
                        labels:  { boxWidth: 10, font: { size: 10 },
                                   color: dark ? '#94a3b8' : '#64748b' },
                    },
                    tooltip: { mode: 'index', intersect: false },
                },
                scales: {
                    x: { display: false },
                    y: {
                        min: def.min, max: def.max,
                        ticks: { color: dark ? '#4b5e7a' : '#94a3b8', font: { size: 10 },
                                 maxTicksLimit: 5 },
                        grid:  { color: dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)' },
                    },
                },
            },
        });
    });
}

function initChartData() {
    CHART_DEFS.forEach(def => {
        chartData[def.id] = { primary: [], secondary: [], labels: [] };
    });
}

function pushChart(id, label, primary, secondary) {
    const d = chartData[id];
    if (!d) return;

    d.labels.push(label);
    d.primary.push(primary);
    if (secondary !== undefined) d.secondary.push(secondary);

    if (d.labels.length    > CFG.CHART_PTS) { d.labels.shift(); d.primary.shift(); }
    if (d.secondary.length > CFG.CHART_PTS) d.secondary.shift();

    const chart = charts[id];
    if (!chart) return;
    chart.data.labels           = d.labels;
    chart.data.datasets[0].data = d.primary;
    if (chart.data.datasets[1]) chart.data.datasets[1].data = d.secondary;
    chart.update('none');
}

// ============================================================
// SESSION TIMER
// ============================================================
function startSessionTimer() {
    sessionTick = setInterval(() => {
        const s   = Math.floor((Date.now() - sessionStart) / 1000);
        const h   = String(Math.floor(s / 3600)).padStart(2, '0');
        const m   = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
        const sec = String(s % 60).padStart(2, '0');
        setText('session-timer', `${h}:${m}:${sec}`);
    }, 1000);
}

// ============================================================
// THEME TOGGLE
// ============================================================
function toggleTheme() {
    const html   = document.documentElement;
    const theme  = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    document.getElementById('theme-toggle').textContent = theme === 'dark' ? '☀️' : '🌙';

    // Update chart grid/tick colours
    const dark = theme === 'dark';
    Object.values(charts).forEach(c => {
        c.options.scales.y.ticks.color = dark ? '#4b5e7a' : '#94a3b8';
        c.options.scales.y.grid.color  = dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)';
        if (c.data.datasets[0]?.label) {
            c.options.plugins.legend.labels.color = dark ? '#94a3b8' : '#64748b';
        }
        c.update('none');
    });

    // Redraw canvases with new theme
    renderPPG();
    drawPoincare(rrHistory);
    drawANSGauge(
        ansState.lf ? Math.round(ansState.lf * 10000) : 0,
        ansState.hf ? Math.round(ansState.hf * 10000) : 0,
    );
}

function applyStoredTheme() {
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = saved === 'dark' ? '☀️' : '🌙';
}

// ============================================================
// CSV EXPORT
// ============================================================
function exportCSV() {
    window.open(`${CFG.API}/export?limit=5000`, '_blank');
}

// ============================================================
// UTILITIES
// ============================================================
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? '--';
}

function fmt(v, dec = 1) {
    if (v == null || isNaN(v)) return '--';
    return Number(v).toFixed(dec);
}

function hexAlpha(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
}
