"use strict";

/**
 * Multi-Modal Physiological Monitoring Dashboard
 *
 * Features:
 *   - Real-time metric display (HR, SpO2, RMSSD, Resp, Fatigue, PERCLOS, Blink, Microsleeps)
 *   - 6 live trend charts (HR, SpO2, Resp, HRV, Fatigue+PERCLOS, Stress)
 *   - Emotion confidence bars
 *   - Signal quality progress bar
 *   - Alert history log (persistent during session)
 *   - CSV export via /api/export
 *   - Session timer
 *   - Stats panel (updated from /api/stats)
 *   - Dark / Light mode toggle
 */

// ===== CONFIG =====
const CONFIG = {
    API_URL:        'http://localhost:5000/api',
    POLL_INTERVAL:  1000,       // ms
    STATS_INTERVAL: 15000,      // ms – refresh session stats less often
    CHART_POINTS:   60,         // data points shown per chart
    RECONNECT_DELAY: 3000,
    MAX_RETRIES:    10,
    MAX_ALERT_LOG:  100,
};

// ===== STATE =====
let charts          = {};
let chartData       = {};
let connectionState = 'disconnected';
let retryCount      = 0;
let pollTimer       = null;
let statsTimer      = null;
let sessionStart    = Date.now();
let sessionTimerInterval = null;
let alertLogEntries = [];

// ===== CHART DEFINITIONS =====
// Each entry: { id, label, color, fill, min, max }
const CHART_DEFS = [
    { id: 'chartHR',     label: 'Heart Rate (BPM)',  color: '#ff4757', min: 40,  max: 180, fill: true },
    { id: 'chartSpO2',   label: 'SpO2 (%)',          color: '#2ed573', min: 90,  max: 100, fill: false },
    { id: 'chartResp',   label: 'Respiration (BPM)', color: '#1e90ff', min: 5,   max: 40,  fill: true },
    { id: 'chartHRV',    label: 'RMSSD (ms)',         color: '#a29bfe', min: 0,   max: 100, fill: false },
    { id: 'chartFatigue',label: 'Fatigue / PERCLOS',  color: '#fdcb6e', min: 0,   max: 1,   fill: true,
      extra: { label: 'PERCLOS %', color: '#e17055', normalize: 0.01 } }, // PERCLOS / 100
    { id: 'chartStress', label: 'Stress Score',       color: '#fd79a8', min: 0,   max: 1,   fill: false },
];

const EMOTION_COLORS = {
    anger:    '#ff4757', disgust: '#2d3436', fear:     '#6c5ce7',
    happiness:'#fdcb6e', neutral: '#74b9ff', sadness:  '#a29bfe',
    surprise: '#00cec9',
};


// ===== INITIALISATION =====
document.addEventListener('DOMContentLoaded', () => {
    setupCharts();
    initChartData();
    startWebcam();
    startSessionTimer();
    pollBackend();
    pollTimer  = setInterval(pollBackend,  CONFIG.POLL_INTERVAL);
    statsTimer = setInterval(fetchStats,   CONFIG.STATS_INTERVAL);
    fetchStats();
});

window.addEventListener('beforeunload', () => {
    clearInterval(pollTimer);
    clearInterval(statsTimer);
    clearInterval(sessionTimerInterval);
});


// ===== CHART SETUP =====
function setupCharts() {
    CHART_DEFS.forEach(def => {
        const canvas = document.getElementById(def.id);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const datasets = [{
            label: def.label,
            data: [],
            borderColor: def.color,
            backgroundColor: def.fill
                ? hexToRgba(def.color, 0.12)
                : 'transparent',
            borderWidth: 2,
            tension: 0.4,
            fill: def.fill,
            pointRadius: 0,
        }];

        if (def.extra) {
            datasets.push({
                label: def.extra.label,
                data: [],
                borderColor: def.extra.color,
                backgroundColor: hexToRgba(def.extra.color, 0.08),
                borderWidth: 1.5,
                tension: 0.4,
                fill: false,
                pointRadius: 0,
                borderDash: [4, 3],
            });
        }

        charts[def.id] = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                animation:  { duration: 0 },
                plugins: {
                    legend: { display: !!def.extra, labels: { boxWidth: 12, font: { size: 11 } } },
                },
                scales: {
                    x: { display: false },
                    y: {
                        min: def.min, max: def.max,
                        ticks: { color: '#888', font: { size: 11 } },
                        grid:  { color: 'rgba(128,128,128,0.1)' },
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


// ===== BACKEND POLLING =====
function pollBackend() {
    fetch(`${CONFIG.API_URL}/metrics`)
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then(data => {
            retryCount = 0;
            updateConnectionStatus('ok');
            updateMetrics(data);
        })
        .catch(err => {
            retryCount++;
            updateConnectionStatus('error');
            if (retryCount >= CONFIG.MAX_RETRIES) {
                showBanner(`Backend unreachable (retry ${retryCount})`, 'error');
            }
        });
}

function fetchStats() {
    fetch(`${CONFIG.API_URL}/stats`)
        .then(r => r.json())
        .then(data => updateSessionStats(data))
        .catch(() => {});
}


// ===== METRICS UPDATE =====
function updateMetrics(payload) {
    if (!payload?.data) return;
    const m = payload.data;

    // Scalar metrics
    setText('hr',         fmt(m.hr,        1));
    setText('spo2',       fmt(m.spo2,      1));
    setText('resp',       fmt(m.resp,       1));
    setText('rmssd',      fmt(m.rmssd,      1));
    setText('fatigue',    fmt(m.fatigue,    2));
    setText('perclos',    fmt(m.perclos,    1));
    setText('blink',      fmt(m.blink,      1));
    setText('microsleeps', m.microsleeps ?? '--');
    setText('timestamp',  m.timestamp ? new Date(m.timestamp).toLocaleTimeString() : '--');

    // Stress
    updateStress(m.stress, m.stress_score);

    // Drowsiness
    updateDrowsiness(m.drowsiness);

    // Signal quality
    updateSignalQuality(m.signal_quality);

    // Metric card colour coding
    colourMetric('metric-hr',    m.hr,       40,  60,  100, 120);
    colourMetric('metric-spo2',  m.spo2,     null, null, 95, null);
    colourMetric('metric-resp',  m.resp,     8,   10,   20,  30);
    colourMetric('metric-rmssd', m.rmssd,    null, null, 20, null, true);
    colourMetric('metric-fatigue', m.fatigue, null, null, 0.6, 0.8);

    // Emotion
    if (m.emotion_scores) updateEmotionBars(m.emotion, m.emotion_scores);

    // Charts
    const ts = new Date().toLocaleTimeString();
    pushChart('chartHR',      ts, m.hr);
    pushChart('chartSpO2',    ts, m.spo2);
    pushChart('chartResp',    ts, m.resp);
    pushChart('chartHRV',     ts, m.rmssd);
    pushChart('chartFatigue', ts, m.fatigue,
              m.perclos != null ? m.perclos * 0.01 : null);  // normalise PERCLOS to 0–1
    pushChart('chartStress',  ts, m.stress_score);

    // Alerts
    if (m.alerts && m.alerts.length > 0) {
        m.alerts.forEach(a => addAlertToLog(a.message, a.severity, a.timestamp));
        showBanner(m.alerts[0].message,
                   m.alerts[0].severity === 'critical' ? 'error' : 'warning');
    }
}


// ===== CHART PUSH =====
function pushChart(id, label, primary, secondary) {
    const d = chartData[id];
    if (!d) return;

    d.labels.push(label);
    d.primary.push(primary);
    if (secondary !== undefined) d.secondary.push(secondary);

    if (d.labels.length  > CONFIG.CHART_POINTS) { d.labels.shift(); d.primary.shift(); }
    if (d.secondary.length > CONFIG.CHART_POINTS) d.secondary.shift();

    const chart = charts[id];
    if (!chart) return;
    chart.data.labels             = d.labels;
    chart.data.datasets[0].data   = d.primary;
    if (chart.data.datasets[1]) chart.data.datasets[1].data = d.secondary;
    chart.update('none');
}


// ===== STRESS =====
function updateStress(level, score) {
    const el = document.getElementById('stress');
    const sc = document.getElementById('stress-score');
    if (!el) return;
    el.textContent = level || '--';
    el.className = 'stress-badge' + (level ? ` stress-${level.toLowerCase()}` : '');
    if (sc) sc.textContent = score != null ? `Score: ${score.toFixed(2)}` : '';
}


// ===== DROWSINESS =====
function updateDrowsiness(level) {
    const el = document.getElementById('drowsiness');
    if (!el) return;
    el.textContent = level || '--';
    el.className = 'drowsiness-badge' +
        (level ? ` drowsy-${level.toLowerCase()}` : '');
}


// ===== SIGNAL QUALITY =====
function updateSignalQuality(sq) {
    const bar  = document.getElementById('sq-bar');
    const text = document.getElementById('sq-value');
    const badge = document.getElementById('signal-quality-badge');
    const pct  = sq != null ? Math.round(sq * 100) : 0;

    if (bar)  { bar.style.width = `${pct}%`; bar.className = `sq-bar ${sqClass(sq)}`; }
    if (text) text.textContent = sq != null ? `${pct}%` : '--';
    if (badge) { badge.textContent = `SQ: ${pct}%`; badge.className = `sq-badge ${sqClass(sq)}`; }
}

function sqClass(sq) {
    if (sq == null)  return 'sq-unknown';
    if (sq >= 0.6)   return 'sq-good';
    if (sq >= 0.3)   return 'sq-fair';
    return 'sq-poor';
}


// ===== EMOTION BARS =====
function updateEmotionBars(topEmotion, scores) {
    const label = document.getElementById('emotion-label');
    const bars  = document.getElementById('emotion-bars');
    if (!bars) return;

    if (label) label.textContent = topEmotion || '--';

    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    bars.innerHTML = sorted.map(([emo, prob]) => {
        const pct = Math.round(prob * 100);
        const col = EMOTION_COLORS[emo] || '#74b9ff';
        return `
          <div class="emo-row">
            <span class="emo-name">${emo}</span>
            <div class="emo-bar-track">
              <div class="emo-bar-fill" style="width:${pct}%;background:${col}"></div>
            </div>
            <span class="emo-pct">${pct}%</span>
          </div>`;
    }).join('');
}


// ===== METRIC CARD COLOURING =====
// Thresholds: warn if < warnLo OR > warnHi; critical if < critLo OR > critHi
// invertLow: for metrics where low value is bad (e.g., HRV, SpO2)
function colourMetric(id, val, critLo, warnLo, warnHi, critHi, invertLow = false) {
    const el = document.getElementById(id);
    if (!el || val == null) { el && el.classList.remove('metric-warn', 'metric-critical'); return; }

    let cls = '';
    if ((critLo != null && val < critLo) || (critHi != null && val > critHi)) cls = 'metric-critical';
    else if ((warnLo != null && val < warnLo) || (warnHi != null && val > warnHi)) cls = 'metric-warn';

    el.classList.remove('metric-warn', 'metric-critical');
    if (cls) el.classList.add(cls);
}


// ===== CONNECTION STATUS =====
function updateConnectionStatus(state) {
    const el = document.getElementById('connection-status');
    if (!el) return;
    if (state === 'ok') {
        el.textContent = '🟢 Connected';
        el.className = 'status-badge status-ok';
        if (connectionState !== 'ok') connectionState = 'ok';
    } else {
        el.textContent = '🔴 Offline';
        el.className = 'status-badge status-error';
        connectionState = 'error';
    }
}


// ===== ALERT BANNER =====
let bannerTimer = null;
function showBanner(msg, type = 'warning') {
    const el = document.getElementById('alert-banner');
    if (!el) return;
    const icons = { warning: '⚠️', error: '🚨', info: 'ℹ️' };
    el.innerHTML = `${icons[type] || '⚠️'} ${msg}`;
    el.className = `alert-banner alert-${type}`;
    el.style.display = 'flex';
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => { el.style.display = 'none'; }, 6000);
}


// ===== ALERT LOG =====
function addAlertToLog(msg, severity, ts) {
    // Deduplicate: skip if same message within last 5 entries
    const recent = alertLogEntries.slice(-5);
    if (recent.some(e => e.msg === msg)) return;

    alertLogEntries.push({ msg, severity, ts });
    if (alertLogEntries.length > CONFIG.MAX_ALERT_LOG) alertLogEntries.shift();
    renderAlertLog();
}

function renderAlertLog() {
    const log = document.getElementById('alert-log');
    if (!log) return;
    if (alertLogEntries.length === 0) {
        log.innerHTML = '<p class="no-alerts-msg">No alerts yet.</p>';
        return;
    }
    log.innerHTML = [...alertLogEntries].reverse().map(e => {
        const time = e.ts ? new Date(e.ts).toLocaleTimeString() : '';
        const cls = e.severity === 'critical' ? 'log-critical' : 'log-warning';
        return `<div class="log-entry ${cls}">
                    <span class="log-time">${time}</span>
                    <span class="log-msg">${e.msg}</span>
                </div>`;
    }).join('');
}

function clearAlertLog() {
    alertLogEntries = [];
    renderAlertLog();
}


// ===== SESSION STATS =====
function updateSessionStats(data) {
    if (!data?.averages_last_hour) return;
    const a = data.averages_last_hour;
    setText('stat-avg-hr',   fmt(a.hr,    1) + (a.hr    ? ' BPM' : ''));
    setText('stat-avg-spo2', fmt(a.spo2,  1) + (a.spo2  ? '%'    : ''));
    setText('stat-avg-resp', fmt(a.resp,  1) + (a.resp   ? ' BPM' : ''));
    setText('stat-readings', a.readings ?? '--');
}


// ===== SESSION TIMER =====
function startSessionTimer() {
    sessionTimerInterval = setInterval(() => {
        const s = Math.floor((Date.now() - sessionStart) / 1000);
        const h = String(Math.floor(s / 3600)).padStart(2, '0');
        const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
        const sec = String(s % 60).padStart(2, '0');
        setText('session-timer', `Session: ${h}:${m}:${sec}`);
    }, 1000);
}


// ===== WEBCAM =====
function startWebcam() {
    const video = document.getElementById('webcam');
    if (!video || !navigator.mediaDevices?.getUserMedia) return;
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => { video.srcObject = stream; })
        .catch(() => { console.warn('Webcam unavailable'); });
}


// ===== DARK MODE =====
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    document.getElementById('theme-toggle').textContent = next === 'dark' ? '☀️' : '🌙';
    // Update chart grid colours
    Object.values(charts).forEach(c => {
        c.options.scales.y.ticks.color = next === 'dark' ? '#aaa' : '#666';
        c.options.scales.y.grid.color  = next === 'dark'
            ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.07)';
        c.update('none');
    });
}

// Restore saved theme on load
(function applySavedTheme() {
    const saved = localStorage.getItem('theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
        const btn = document.getElementById('theme-toggle');
        if (btn) btn.textContent = saved === 'dark' ? '☀️' : '🌙';
    }
})();


// ===== CSV EXPORT =====
function exportCSV() {
    window.open(`${CONFIG.API_URL}/export?limit=5000`, '_blank');
}


// ===== UTILITIES =====
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? '--';
}

function fmt(v, decimals = 1) {
    if (v == null || v === undefined) return '--';
    return Number(v).toFixed(decimals);
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}
