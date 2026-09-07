'use strict';
/**
 * app.js — Object Detection Lab  v1.6
 * Build: 2026-08-26
 *
 * Detection:    COCO-SSD (TensorFlow.js, fully in-browser)
 * Description:  Google Gemini 3.6 Flash (vision API)
 *
 * API key is loaded from config.js (generated from .env — never commit config.js).
 */

// ── Gemini API Key — loaded from config.js or localStorage ──────────────────
const GEMINI_API_KEY = (typeof window !== 'undefined' && (window.GEMINI_API_KEY || localStorage.getItem('GEMINI_API_KEY')))
  ? (window.GEMINI_API_KEY || localStorage.getItem('GEMINI_API_KEY'))
  : '';

// ── Config ──────────────────────────────────────────────────────────────────
const GEMINI_MODEL  = 'gemini-3.6-flash';
const GEMINI_URL    = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;
const MAX_IMG_PX    = 1920;
const CONF_IMG      = 0.08;   // lower threshold → detects small and low-contrast objects
const CONF_VIDEO    = 0.25;   // balanced — reduces false-positive flicker in video
const CONF_WEBCAM   = 0.20;   // slightly lower for live lighting variation

// ── Video accuracy tuning ─────────────────────────────────────────────────────
const VIDEO_MAX_BOXES     = 50;    // max detections per frame
const SMOOTH_ALPHA        = 0.35;  // box position smoothing (0=frozen, 1=instant)
const SMOOTH_SCORE_ALPHA  = 0.40;  // confidence score smoothing
const SMOOTH_IOU_THRESH   = 0.40;  // min IoU to consider two boxes the same object
const SMOOTH_MAX_MISS     = 6;     // frames an object can be missing before removed
const DETECT_INTERVAL_MS  = 80;    // run model every ~80ms max (~12fps detect rate)

const GEMINI_PROMPT = `You are an AI vision assistant. Analyse this image carefully and write a detailed description covering:
- What objects, people, animals, or structures are present
- What actions or activities are happening
- The setting, environment, or location
- Notable colours, lighting, or atmosphere
- Any text or signs visible

Write in clear, flowing paragraphs. Be specific and observational. Do not say "I see" — just describe directly.`;

// ── State ────────────────────────────────────────────────────────────────────
let cocoModel    = null;
let geminiKey    = GEMINI_API_KEY;   // loaded from constant above
let currentMode  = 'image';
let animFrameId  = null;
let webcamStream = null;
let isDetecting  = false;
let isScrubbing  = false;
let lastCaption  = '';
let geminiInterval    = null;   // periodic Gemini refresh for video/webcam
let isGeminiRunning   = false;  // prevent overlapping Gemini requests
let geminiLabelOverrides = {};  // e.g. { 'cell phone': 'remote' } — updated every 15s
let lastPreds         = [];     // most recent raw COCO-SSD predictions

// ── Temporal smoothing state (video / webcam) ────────────────────────────────
// Each tracked object: { id, class, score, bbox:[x,y,w,h], missCount, age }
let trackedObjects    = [];     // smoothed object list across frames
let trackIdCounter    = 0;      // incrementing unique ID for each tracked object
let isDetectRunning   = false;  // prevent concurrent model.detect() calls
let lastDetectTime    = 0;      // timestamp of last detect call

// ── DOM refs ─────────────────────────────────────────────────────────────────
const canvas        = document.getElementById('main-canvas');
const ctx           = canvas.getContext('2d');
const vidSrc        = document.getElementById('vid-src');
const placeholder   = document.getElementById('canvas-placeholder');
const manifestBody  = document.getElementById('manifest-body');
const modelStatus   = document.getElementById('model-status');
const actionBtn     = document.getElementById('action-btn');
const stopBtn       = document.getElementById('stop-btn');
const descPanel     = document.getElementById('desc-panel');
const descBody      = document.getElementById('desc-body');
const descBadge     = document.getElementById('desc-badge');
const copyBtn       = document.getElementById('copy-btn');
const badgeText     = document.getElementById('badge-text');
const vidControls   = document.getElementById('vid-controls');
const scrubberTrack = document.getElementById('scrubber-track');
const scrubberFill  = document.getElementById('scrubber-fill');
const scrubberThumb = document.getElementById('scrubber-thumb');
const vcTime        = document.getElementById('vc-time');
const iconPlay      = document.getElementById('icon-play');
const iconPause     = document.getElementById('icon-pause');
const iconVol       = document.getElementById('icon-vol');
const iconMuted     = document.getElementById('icon-muted');

// ── Color palette ────────────────────────────────────────────────────────────
const PALETTE = [
  '#e05a1e','#4fc3c3','#f0c040','#a78bfa','#34d399','#f472b6',
  '#60a5fa','#fb923c','#f87171','#a3e635','#38bdf8','#e879f9',
  '#fbbf24','#818cf8','#86efac','#fca5a1','#c084fc','#67e8f9',
  '#fdba74','#6ee7b7','#bef264','#f9a8d4'
];
const colorMap = {};
function colorFor(label) {
  const k = label.toLowerCase().trim();
  if (!colorMap[k]) colorMap[k] = PALETTE[Object.keys(colorMap).length % PALETTE.length];
  return colorMap[k];
}

function fmtTime(s) {
  if (!isFinite(s)) return '0:00';
  return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
}

// ── Gemini Key Management ───────────────────────────────────────────────────
function getEffectiveApiKey() {
  const winKey = (typeof window !== 'undefined' && window.GEMINI_API_KEY && window.GEMINI_API_KEY !== 'YOUR_GEMINI_API_KEY_HERE')
    ? window.GEMINI_API_KEY
    : '';
  const localKey = (typeof localStorage !== 'undefined' ? (localStorage.getItem('GEMINI_API_KEY') || '') : '');
  return localKey || winKey || '';
}

function updateApiKeyUI() {
  const key = getEffectiveApiKey();
  const label = document.getElementById('api-key-label');
  const input = document.getElementById('user-gemini-key');

  if (key) {
    geminiKey = key;
    if (label) label.innerHTML = `<span style="color:#34d399;font-weight:bold">● GEMINI ACTIVE</span> (${key.slice(0, 4)}…${key.slice(-4)})`;
    if (input) input.placeholder = 'Update API key…';
    if (badgeText) badgeText.textContent = 'MODEL: COCO-SSD + GEMINI FLASH · IN-BROWSER + VISION API';
    setStatus('MODEL READY', 'COCO + GEMINI', true);
  } else {
    geminiKey = '';
    if (label) label.innerHTML = `<span style="color:#f87171;font-weight:bold">○ GEMINI OFF</span> (Paste key below to activate full detection)`;
    if (badgeText) badgeText.textContent = 'MODEL: COCO-SSD · RUNNING IN-BROWSER';
    setStatus('MODEL READY', 'COCO-SSD', true);
  }
}

function saveUserApiKey() {
  const input = document.getElementById('user-gemini-key');
  const val = input ? input.value.trim() : '';
  if (val) {
    localStorage.setItem('GEMINI_API_KEY', val);
    if (typeof window !== 'undefined') window.GEMINI_API_KEY = val;
    geminiKey = val;
    input.value = '';
    updateApiKeyUI();
    alert('✓ Gemini API key saved! Full AI object detection is now active.');
  } else {
    alert('Please enter a valid Gemini API key.');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// COCO-SSD
// ═══════════════════════════════════════════════════════════════════════════
async function loadCocoModel() {
  setStatus('LOADING MODEL…', '', false);
  updateApiKeyUI();
  try {
    cocoModel = await cocoSsd.load({ base: 'mobilenet_v2' });
    updateApiKeyUI();
  } catch (e) {
    setStatus('LOAD FAILED', '', false);
    console.error(e);
  }
}

function setStatus(text, badge, ready) {
  modelStatus.innerHTML = badge
    ? `${text} <span class="engine-badge">${badge}</span>`
    : text;
  modelStatus.classList.toggle('ready', !!ready);
}

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function resizeImage(img, maxPx) {
  let w = img.naturalWidth  || img.width  || 640;
  let h = img.naturalHeight || img.height || 480;
  if (w > maxPx || h > maxPx) {
    const r = Math.min(maxPx / w, maxPx / h);
    w = Math.round(w * r);
    h = Math.round(h * r);
  }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(img, 0, 0, w, h);
  return c;
}

function canvasToBlob(c, quality) {
  return new Promise(res => c.toBlob(res, 'image/jpeg', quality || 0.90));
}

// ═══════════════════════════════════════════════════════════════════════════
// GEMINI API CALLER (Supports fallback models and safe part concatenation)
// ═══════════════════════════════════════════════════════════════════════════
async function callGeminiApi(payload) {
  const key = geminiKey || GEMINI_API_KEY || (typeof window !== 'undefined' && window.GEMINI_API_KEY) || (typeof localStorage !== 'undefined' && localStorage.getItem('GEMINI_API_KEY'));
  if (!key || key === 'YOUR_GEMINI_API_KEY_HERE') {
    throw new Error('Gemini API key is missing. Please set it in config.js or .env');
  }

  let lastError = null;
  for (const modelName of FALLBACK_MODELS) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${key}`;
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const errTxt = await resp.text();
        let msg = `HTTP ${resp.status}`;
        try { msg = JSON.parse(errTxt).error?.message || msg; } catch (_) {}
        lastError = new Error(msg);
        continue;
      }

      const data = await resp.json();
      let text = '';
      for (const p of data?.candidates?.[0]?.content?.parts || []) {
        if (p.text) text += p.text;
      }
      return text.trim();
    } catch (e) {
      lastError = e;
    }
  }

  throw lastError || new Error('All Gemini API endpoints failed.');
}

// ═══════════════════════════════════════════════════════════════════════════
// GEMINI VISION — scene description
// ═══════════════════════════════════════════════════════════════════════════
async function describeImage(resizedCanvas) {
  const hasKey = geminiKey || GEMINI_API_KEY || (typeof window !== 'undefined' && window.GEMINI_API_KEY) || (typeof localStorage !== 'undefined' && localStorage.getItem('GEMINI_API_KEY'));
  if (!hasKey || hasKey === 'YOUR_GEMINI_API_KEY_HERE') return null;

  descBadge.textContent = '…';
  descBody.innerHTML = `<p class="manifest-empty desc-loading">Asking Gemini to describe the scene…</p>`;

  // Resize to 768px for fast API response
  const apiCanvas = resizeImage(resizedCanvas, 768);
  const base64    = apiCanvas.toDataURL('image/jpeg', 0.88).split(',')[1];

  const payload = {
    contents: [{
      parts: [
        { text: GEMINI_PROMPT },
        { inline_data: { mime_type: 'image/jpeg', data: base64 } }
      ]
    }],
    generationConfig: { temperature: 0.4, maxOutputTokens: 1000 }
  };

  const text = await callGeminiApi(payload);
  if (!text) throw new Error('Empty response from Gemini.');
  return text.trim();
}

// ═══════════════════════════════════════════════════════════════════════════
// HARDCODED COCO-SSD SHAPE CONFUSIONS — instant correction lookup
// ═══════════════════════════════════════════════════════════════════════════
const SHAPE_CONFUSION_HINT = {
  'cell phone': 'mouse / remote / cell phone',
  'remote':     'remote / mouse / cell phone',
  'book':       'book / laptop / keyboard',
  'laptop':     'laptop / book',
  'vase':       'bottle / cup / vase',
  'cup':        'bottle / cup / mug',
  'bowl':       'cup / bowl / plate',
  'clock':      'clock / cell phone / remote',
  'keyboard':   'keyboard / mouse / remote'
};

// ═══════════════════════════════════════════════════════════════════════════
// GEMINI LABEL VERIFICATION & SMALL OBJECT DISCOVERY
// Corrects misclassifications (e.g. vase -> bottle) and detects small objects (caps, pens, etc.)
// ═══════════════════════════════════════════════════════════════════════════
async function verifyDetectionsWithGemini(resizedCanvas, preds) {
  const hasKey = geminiKey || GEMINI_API_KEY || (typeof window !== 'undefined' && window.GEMINI_API_KEY) || (typeof localStorage !== 'undefined' && localStorage.getItem('GEMINI_API_KEY'));
  if (!hasKey || hasKey === 'YOUR_GEMINI_API_KEY_HERE') return preds;

  const uniqueClasses = [...new Set((preds || []).map(p => p.class))];

  const verifyPrompt = `You are a world-class computer vision object detection system.
The local detector found only: ${uniqueClasses.length ? uniqueClasses.join(', ') : 'none'}.

Exhaustively detect ALL objects in this image (both prominent and small), including:
- Furniture & Seating: sofa, couch, armchair, chair, coffee table, side table, TV console, cabinets, desk, shelves
- Electronics & Appliances: television/TV, wall clock, floor lamp, chandelier, ceiling light, laptop, computer, mouse, remote, phone
- Decor & Small Items: potted plants, flowers, vase, cushions/pillows, rug/carpet, statues/sculptures, books, bottles, cups, glasses, bottle cap, keys
- Architectural elements: stairs/staircase, fireplace, door, window

Also verify & correct any misclassified labels (e.g. vase -> bottle, cell phone -> mouse or remote).

Return ONLY valid JSON:
{
  "corrections": [
    {"detected": "vase", "correct": "bottle"}
  ],
  "detected_objects": [
    {"label": "television", "box": [ymin, xmin, ymax, xmax]},
    {"label": "armchair", "box": [ymin, xmin, ymax, xmax]},
    {"label": "sofa", "box": [ymin, xmin, ymax, xmax]},
    {"label": "coffee table", "box": [ymin, xmin, ymax, xmax]},
    {"label": "chandelier", "box": [ymin, xmin, ymax, xmax]},
    {"label": "floor lamp", "box": [ymin, xmin, ymax, xmax]},
    {"label": "wall clock", "box": [ymin, xmin, ymax, xmax]}
  ]
}
Coordinates [ymin, xmin, ymax, xmax] must be normalized integers between 0 and 1000. Output pure JSON without markdown.`;

  const apiCanvas = resizeImage(resizedCanvas, 800);
  const base64 = apiCanvas.toDataURL('image/jpeg', 0.88).split(',')[1];

  const payload = {
    contents: [{ parts: [
      { text: verifyPrompt },
      { inline_data: { mime_type: 'image/jpeg', data: base64 } }
    ]}],
    generationConfig: {
      temperature: 0.1,
      maxOutputTokens: 2500
    }
  };

  try {
    let text = await callGeminiApi(payload);
    if (!text) return preds;

    if (text.startsWith('```')) {
      text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
    }

    let result = null;
    try {
      result = JSON.parse(text);
    } catch (_) {
      const fb = text.indexOf('{');
      const lb = text.lastIndexOf('}');
      if (fb !== -1 && lb !== -1) {
        result = JSON.parse(text.slice(fb, lb + 1));
      }
    }

    if (!result) return preds;

    const corrMap = {};
    const corrections = result.corrections || (Array.isArray(result) ? result : []);
    if (Array.isArray(corrections)) {
      corrections.forEach(c => {
        if (c.detected && c.correct) {
          corrMap[c.detected.toLowerCase()] = c.correct;
        }
      });
      Object.assign(geminiLabelOverrides, corrMap);
    }

    let corrected = [...(preds || [])].map(p => {
      const fix = corrMap[p.class.toLowerCase()];
      return fix ? { ...p, class: fix, geminiCorrected: true } : p;
    });

    // Add all detected objects from Gemini with smart IoU deduplication
    const geminiObjs = result.detected_objects || result.missed_objects || (Array.isArray(result) ? result : []);
    if (Array.isArray(geminiObjs) && geminiObjs.length) {
      const w = resizedCanvas.width;
      const h = resizedCanvas.height;
      for (const obj of geminiObjs) {
        if (obj.box && Array.isArray(obj.box) && obj.box.length === 4) {
          const [ymin, xmin, ymax, xmax] = obj.box;
          const px = Math.round((xmin / 1000) * w);
          const py = Math.round((ymin / 1000) * h);
          const pw = Math.round(((xmax - xmin) / 1000) * w);
          const ph = Math.round(((ymax - ymin) / 1000) * h);

          if (pw > 6 && ph > 6) {
            // Check overlap with existing detection
            const existingIdx = corrected.findIndex(p => {
              const [ex, ey, ew, eh] = p.bbox;
              const xOverlap = Math.max(0, Math.min(px + pw, ex + ew) - Math.max(px, ex));
              const yOverlap = Math.max(0, Math.min(py + ph, ey + eh) - Math.max(py, ey));
              const intersection = xOverlap * yOverlap;
              const union = (pw * ph) + (ew * eh) - intersection;
              const iou = union > 0 ? intersection / union : 0;
              return iou > 0.45;
            });

            if (existingIdx !== -1) {
              corrected[existingIdx].class = obj.label || corrected[existingIdx].class;
              corrected[existingIdx].geminiCorrected = true;
            } else {
              corrected.push({
                class: obj.label || 'object',
                score: 0.95,
                bbox: [px, py, pw, ph],
                geminiAdded: true
              });
            }
          }
        }
      }
    }

    const fixedLabels = (corrections || []).map(c => `${c.detected} → ${c.correct}`).join(', ');
    if (fixedLabels) console.info(`[Gemini verify] Corrections applied: ${fixedLabels}`);

    return corrected;
  } catch (e) {
    console.warn('[Gemini verify] failed:', e.message);
    return preds;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// DRAW DETECTIONS
// ═══════════════════════════════════════════════════════════════════════════
function drawDetections(source, preds, threshold) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

  const thr = threshold || CONF_IMG;
  preds.forEach(p => {
    if (p.score < thr) return;
    const [x, y, w, h] = p.bbox;
    const color = colorFor(p.class);
    const label = p.class.toUpperCase() + '  ' + (p.score * 100).toFixed(0) + '%';

    // Box with slight inner glow
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Label pill
    ctx.font = 'bold 12px "Courier New"';
    const tw = ctx.measureText(label).width + 14;
    const ly = y > 26 ? y - 24 : y + 2;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, 22);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 7, ly + 15);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// MANIFEST
// ═══════════════════════════════════════════════════════════════════════════
function updateManifest(preds, engine, threshold) {
  const thr = threshold || CONF_IMG;
  const filtered = (preds || []).filter(p => p.score >= thr);

  if (!filtered.length) {
    manifestBody.innerHTML = `<p class="manifest-empty">No objects detected in this frame.</p>`;
    return;
  }

  // Aggregate by class
  const map = {};
  filtered.forEach(p => {
    const k = p.class.toLowerCase();
    if (!map[k]) map[k] = { label: p.class, score: p.score, count: 1 };
    else { map[k].count++; if (p.score > map[k].score) map[k].score = p.score; }
  });

  const rows = Object.values(map).sort((a,b) => b.score - a.score);
  const badge = engine ? `<span class="engine-badge">${engine}</span>` : '';

  manifestBody.innerHTML =
    `<p class="manifest-count">${filtered.length} OBJECT${filtered.length!==1?'S':''} DETECTED ${badge}</p>` +
    `<ul class="manifest-list">` +
    rows.map(({label, score, count}) =>
      `<li class="manifest-item">
         <span class="label" style="color:${colorFor(label)}">${label}${count>1?` ×${count}`:''}</span>
         <span class="score">${(score*100).toFixed(0)}%</span>
       </li>`
    ).join('') + `</ul>`;
}

// ═══════════════════════════════════════════════════════════════════════════
// DESCRIPTION PANEL
// ═══════════════════════════════════════════════════════════════════════════
function showDescription(text) {
  lastCaption = text;
  descBadge.textContent = 'GEMINI';
  // Convert newlines to paragraph breaks for readability
  const html = text
    .split(/\n\n+/)
    .filter(p => p.trim())
    .map(p => `<p class="desc-text">${p.trim().replace(/\n/g, '<br>')}</p>`)
    .join('');
  descBody.innerHTML = html;
  copyBtn.style.display = 'inline-flex';
}

function showDescError(msg) {
  descBadge.textContent = 'ERR';
  descBody.innerHTML = `<p class="manifest-empty" style="color:#f87171">${msg}</p>`;
  copyBtn.style.display = 'none';
}

function copyDescription() {
  if (!lastCaption) return;
  navigator.clipboard.writeText(lastCaption).then(() => {
    copyBtn.textContent = 'Copied!';
    setTimeout(() => { copyBtn.textContent = 'Copy text'; }, 2000);
  });
}

// Manually refresh AI description from current canvas frame
async function refreshDescription() {
  if (isGeminiRunning) return;
  isGeminiRunning = true;
  try {
    // Snapshot the current canvas frame
    const snap = document.createElement('canvas');
    snap.width  = canvas.width;
    snap.height = canvas.height;
    snap.getContext('2d').drawImage(canvas, 0, 0);
    const caption = await describeImage(snap);
    if (caption) showDescription(caption);
    else showDescError('No description returned by Gemini.');
  } catch(err) {
    console.error('Gemini error:', err.message);
    showDescError('Gemini error: ' + err.message.slice(0, 120));
  } finally {
    isGeminiRunning = false;
  }
}

// Refresh Gemini label corrections from current live frame (webcam / video)
// Runs in background — updates geminiLabelOverrides so every subsequent detection frame benefits instantly
let isLabelVerifyRunning = false;
async function refreshLabelVerification() {
  if (isLabelVerifyRunning || !lastPreds.length) return;
  if (!geminiKey || geminiKey === 'YOUR_GEMINI_API_KEY_HERE') return;
  isLabelVerifyRunning = true;
  try {
    const snap = document.createElement('canvas');
    snap.width  = canvas.width;
    snap.height = canvas.height;
    snap.getContext('2d').drawImage(canvas, 0, 0);
    await verifyDetectionsWithGemini(snap, lastPreds);
    // geminiLabelOverrides is updated inside verifyDetectionsWithGemini
  } catch(e) {
    console.warn('[Label refresh] skipped:', e.message);
  } finally {
    isLabelVerifyRunning = false;
  }
}

// Start periodic Gemini scene description + label correction
// Label verification runs every 8s; description every 16s (alternating to stay within rate limits)
function startGeminiInterval() {
  if (geminiInterval) clearInterval(geminiInterval);
  let tick = 0;
  // Immediately verify labels and get description on start
  refreshLabelVerification();
  refreshDescription();
  geminiInterval = setInterval(() => {
    if (!isDetecting) { clearInterval(geminiInterval); geminiInterval = null; return; }
    tick++;
    // Label verification runs every cycle (~8s)
    refreshLabelVerification();
    // Full scene description runs every other cycle (~16s) to save quota
    if (tick % 2 === 0) refreshDescription();
  }, 8000);
}

// ═══════════════════════════════════════════════════════════════════════════
// MODE SELECTION
// ═══════════════════════════════════════════════════════════════════════════
function selectMode(mode) {
  stopDetection();
  currentMode = mode;
  document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
  document.getElementById(`card-${mode}`).classList.add('active');
  actionBtn.textContent = {
    image:  'Choose image',
    video:  'Choose video',
    webcam: 'Start webcam',
    bill:   'Upload Bill Video'
  }[mode];

  const manifestPanel = document.querySelector('.manifest-panel');
  const billPanel     = document.getElementById('bill-results-panel');

  if (mode === 'bill') {
    if (manifestPanel) manifestPanel.style.display = 'none';
    descPanel.classList.add('hidden');
    if (billPanel)     billPanel.style.display = 'flex';
  } else {
    if (manifestPanel) manifestPanel.style.display = '';
    descPanel.classList.remove('hidden');
    if (billPanel)     billPanel.style.display = 'none';
  }
}

function handleAction() {
  if (currentMode === 'image')  document.getElementById('image-input').click();
  if (currentMode === 'video')  document.getElementById('video-input').click();
  if (currentMode === 'webcam') startWebcam();
  if (currentMode === 'bill')   document.getElementById('bill-input').click();
}

// ═══════════════════════════════════════════════════════════════════════════
// BILL SCANNER — Multi-Bill Video OCR & Structured Data Extraction
// ═══════════════════════════════════════════════════════════════════════════
let allBills = [];
let billKeyframes = [];

const BILL_PROMPT = `You are a precise OCR engine specialised in Indian courier shipping labels and invoices.
Read EVERY piece of text visible in this image carefully — including small print, barcodes labels, and handwritten text.

Extract ALL of the following fields from the document (shipping label, Delhivery/Amazon/Flipkart/BlueDart tag, tax invoice, retail bill, or receipt):

FOR SHIPPING LABELS specifically look for:
- AWB / Tracking number (long numeric or alphanumeric code, often under a barcode)
- Recipient / Consignee name (labelled "To:", "Ship To:", "Deliver To:", or just a name near the address)
- Recipient full address including house/flat no, street, area, city, state
- PIN code (6-digit number near the address)
- Sender / From name and address
- Product description or SKU (what is being shipped)
- COD amount (Cash on Delivery — labelled "COD", "Amount", "Invoice Value")
- Payment mode (Pre-paid / COD)
- Date of label / shipment date
- Courier company name (DELHIVERY, Amazon, Flipkart, BlueDart, etc.)

Output ONLY pure JSON in this exact schema — no markdown, no explanation:
{
  "is_bill": true,
  "doc_type": "Shipping Label",
  "vendor_name": "Courier or seller name (e.g. DELHIVERY, Amazon)",
  "bill_number": "AWB / Tracking ID / Invoice No (exact number from label)",
  "date": "Date visible on label (e.g. 12-Jan-2024)",
  "customer_name": "Recipient full name",
  "customer_address": "Full delivery address including street, area, city, state",
  "pin_code": "6-digit PIN code",
  "sender_name": "Sender / From name",
  "sender_address": "Sender full address",
  "product_description": "Product name or description being shipped",
  "items": [
    {"name": "Product name", "qty": "1", "price": "unit price", "amount": "total price"}
  ],
  "subtotal": "Subtotal if shown, else null",
  "tax": "GST / Tax if shown, else null",
  "discount": "Discount if shown, else null",
  "total": "COD amount or Invoice value with currency symbol (e.g. ₹549.00)",
  "payment_method": "Pre-paid or COD",
  "notes": "Any other text on the label (e.g. fragile, instructions)"
}

If the image is completely blank, dark, blurry, or has no readable document/label text, output:
{"is_bill": false}`;

async function loadBillVideo(event) {
  const file = event.target.files[0];
  if (!file) return;
  event.target.value = '';

  allBills = [];
  billKeyframes = [];

  const billBody     = document.getElementById('bill-results-body');
  const billProgress = document.getElementById('bill-progress');
  const progressTxt  = document.getElementById('bill-progress-text');
  const progressBar  = document.getElementById('bill-progress-fill');
  const exportRow    = document.getElementById('bill-export-row');

  exportRow.style.display    = 'none';
  billBody.innerHTML         = '';
  billProgress.style.display = 'block';
  progressTxt.textContent    = '⟳ Loading video…';
  progressBar.style.width    = '5%';

  const url = URL.createObjectURL(file);
  vidSrc.src = url;
  vidSrc.muted = true;
  vidSrc.preload = 'auto';

  await new Promise((res, rej) => {
    vidSrc.onloadedmetadata = res;
    vidSrc.onerror = rej;
  });

  const duration = vidSrc.duration;
  resizeCanvas(vidSrc.videoWidth, vidSrc.videoHeight);
  showCanvas();
  hideVideoControls();

  // Draw initial frame
  ctx.drawImage(vidSrc, 0, 0, canvas.width, canvas.height);

  progressTxt.textContent = `⟳ Extracting keyframes from ${duration.toFixed(1)}s video…`;
  progressBar.style.width = '20%';

  // ── Step 1: Dense Keyframe Extraction (1 frame / 1.5s) ──────────────
  const keyframes = await fastExtractKeyframes(vidSrc, duration, (pct) => {
    progressBar.style.width = (20 + pct * 30) + '%';
  });

  if (!keyframes.length) {
    billProgress.style.display = 'none';
    billBody.innerHTML = `<p class="manifest-empty" style="color:#f87171">Could not read video frames. Please check your video file.</p>`;
    showVideoControls();
    return;
  }

  progressBar.style.width = '55%';
  progressTxt.textContent = `⚡ Running parallel OCR on ${keyframes.length} keyframe(s) with Gemini Flash…`;

  // Draw the clearest frame to canvas
  const previewImg = new Image();
  await new Promise(r => { previewImg.onload = r; previewImg.src = keyframes[0].dataUrl; });
  ctx.drawImage(previewImg, 0, 0, canvas.width, canvas.height);
  drawScanHUD(keyframes[0].t, duration, 1, keyframes.length, 0);

  // ── Step 2: Batched Gemini OCR (5 frames at a time to respect rate limits) ──
  let lastError = null;
  const results = [];
  const BATCH = 5;

  for (let i = 0; i < keyframes.length; i += BATCH) {
    const batch = keyframes.slice(i, i + BATCH);
    progressTxt.textContent = `⚡ OCR: processing frames ${i + 1}–${Math.min(i + BATCH, keyframes.length)} of ${keyframes.length}…`;
    progressBar.style.width = (55 + ((i / keyframes.length) * 35)) + '%';

    const batchResults = await Promise.all(batch.map(async (kf) => {
      try {
        const res = await analyzeBillFrame(kf.dataUrl);
        return { kf, res };
      } catch (err) {
        lastError = err.message;
        return { kf, res: null };
      }
    }));
    results.push(...batchResults);

    // Small pause between batches to avoid hitting Gemini rate limits
    if (i + BATCH < keyframes.length) {
      await new Promise(r => setTimeout(r, 300));
    }
  }
  progressBar.style.width = '90%';

  // ── Step 3: Parse and Deduplicate by AWB Number ─────────────────────
  // Show each unique bill once — deduplicate by exact AWB/tracking number
  console.log(`[OCR Results] Processing ${results.length} frames...`);
  
  for (let i = 0; i < results.length; i++) {
    const { kf, res } = results[i];
    if (!res) {
      console.log(`  Frame ${i + 1} @ ${kf.t.toFixed(2)}s: OCR failed (null response)`);
      continue;
    }

    // Very permissive check — if Gemini returned ANYTHING useful, process it
    const hasData = res.is_bill ||
                    res.bill_number ||
                    res.total ||
                    res.vendor_name ||
                    res.customer_name ||
                    res.customer_address ||
                    res.doc_type ||
                    (res.items && res.items.length);

    if (!hasData) {
      console.log(`  Frame ${i + 1} @ ${kf.t.toFixed(2)}s: Blank (is_bill=${res.is_bill})`);
      continue;
    }

    // Check if this is a duplicate bill (same AWB already extracted)
    const isDuplicate = checkDuplicateBill(res, allBills);
    
    if (isDuplicate) {
      console.log(`  Frame ${i + 1} @ ${kf.t.toFixed(2)}s: DUPLICATE of AWB ${res.bill_number || 'N/A'} — skipped`);
      continue;
    }

    console.log(`  Frame ${i + 1} @ ${kf.t.toFixed(2)}s: ✓ ${res.vendor_name || res.doc_type || 'Document'} | AWB: ${res.bill_number || 'N/A'} | Total: ${res.total || 'N/A'}`);

    res.bill_index = allBills.length + 1;
    res.timestamp  = kf.t;
    res.frameData  = kf.dataUrl;
    allBills.push(res);
    renderBillCard(res, billBody);
  }
  console.log(`[OCR Results] ${allBills.length} unique bills extracted from ${results.length} frames`);

  // ── Step 4: Finished ────────────────────────────────────────────────
  billProgress.style.display = 'none';
  progressBar.style.width    = '100%';
  showVideoControls();

  if (!allBills.length) {
    const errDetail = lastError ? `<br><span style="font-size:11px;color:#cbd5e1">Error: ${lastError}</span>` : '';
    billBody.innerHTML = `<p class="manifest-empty" style="color:#f87171">No bills/labels recognized in the video.${errDetail}</p>`;
  } else {
    flashDetectedHUD(allBills.length, allBills[0].vendor_name || allBills[0].bill_number);
    const gt = computeGrandTotal(allBills);
    billBody.insertAdjacentHTML('beforeend',
      `<div class="bill-summary-row">
         <span>📋 <strong>${allBills.length}</strong> bill${allBills.length !== 1 ? 's' : ''} recognized</span>
         ${gt ? `<span class="bill-grand-total">Grand Total: <strong>${gt}</strong></span>` : ''}
       </div>`
    );
    exportRow.style.display = 'flex';
  }
}

// Keyframe Extraction — samples every ~1.5s and skips near-duplicate frames
// For a 32s video this gives ~20 candidates, then deduplication cuts repeats
async function fastExtractKeyframes(vid, duration, onProgress) {
  const capCanvas  = document.createElement('canvas');
  const capCtx     = capCanvas.getContext('2d');
  const diffCanvas = document.createElement('canvas'); // small canvas for diff check
  const diffCtx    = diffCanvas.getContext('2d');

  // OCR resolution: max 900px keeps text sharp without bloating the base64 payload
  let w = vid.videoWidth  || 1280;
  let h = vid.videoHeight || 720;
  if (Math.max(w, h) > 900) {
    const scale = 900 / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }
  capCanvas.width  = w;
  capCanvas.height = h;

  // Tiny canvas for scene-change pixel diff (64×36 is enough)
  diffCanvas.width  = 64;
  diffCanvas.height = 36;

  // ── Sample interval: 1 frame per second (catches bills shown for as little as 2s)
  // Maximum 60 frames to avoid excessive API cost on very long videos
  const INTERVAL   = 1.0;   // seconds between candidate samples
  const MAX_FRAMES = 60;
  const MIN_FRAMES = 2;

  const totalSamples = Math.max(MIN_FRAMES, Math.min(MAX_FRAMES, Math.floor(duration / INTERVAL)));
  const step         = duration / totalSamples;

  // Build sample timestamps — evenly spaced, offset by half-step to avoid cut edges
  const sampleTimes = [];
  for (let i = 0; i < totalSamples; i++) {
    const t = Math.min(step * i + step * 0.4, duration - 0.1);
    sampleTimes.push(parseFloat(t.toFixed(2)));
  }

  // ── Per-frame pixel data for scene-change detection ──────────────────
  let lastPixels = null;

  // Returns average absolute diff between two Uint8ClampedArrays (0–255)
  function pixelDiff(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i += 4) sum += Math.abs(a[i] - b[i]);
    return sum / (a.length / 4);
  }

  const keyframes = [];

  for (let i = 0; i < sampleTimes.length; i++) {
    const t = sampleTimes[i];
    onProgress(i / sampleTimes.length);
    await seekVideo(vid, t);

    // Draw preview to main canvas so user sees progress
    ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);

    // Check scene change using tiny diff canvas
    diffCtx.drawImage(vid, 0, 0, 64, 36);
    const pixels = diffCtx.getImageData(0, 0, 64, 36).data;

    // Skip frame only if it looks nearly IDENTICAL (same frame, no movement at all)
    // Threshold raised to 4 — only skip genuine static duplicates, not different labels
    // Delhivery labels look visually similar but have different AWBs, so we keep them all
    if (lastPixels && pixelDiff(pixels, lastPixels) < 4) {
      lastPixels = pixels;
      continue;
    }
    lastPixels = pixels;

    // Capture full-res frame for OCR
    capCtx.drawImage(vid, 0, 0, w, h);
    keyframes.push({
      t,
      dataUrl: capCanvas.toDataURL('image/jpeg', 0.88)
    });
  }

  console.log(`[Keyframes] ${sampleTimes.length} sampled → ${keyframes.length} unique after dedup`);
  return keyframes;
}

function seekVideo(vid, t) {
  return new Promise(res => {
    vid.onseeked = res;
    vid.currentTime = Math.min(t, vid.duration);
  });
}

// Strict duplicate detection: only exact AWB match counts.
// If no AWB, never auto-dedupe (treat as unique).
function checkDuplicateBill(newBill, existingBills) {
  if (!existingBills.length) return false;

  // Clean and normalize the new bill's AWB number
  const newNum = (newBill.bill_number || '').trim().replace(/[\s\-_]/g, '').toUpperCase();
  
  // If no AWB or too short, treat as unique (don't dedupe)
  if (!newNum || newNum.length < 8) return false;

  for (const b of existingBills) {
    const exNum = (b.bill_number || '').trim().replace(/[\s\-_]/g, '').toUpperCase();
    if (!exNum || exNum.length < 8) continue;

    // Only exact full match counts as duplicate
    if (newNum === exNum) {
      return true;
    }
  }
  
  return false;
}

const FALLBACK_MODELS = [
  'gemini-3.6-flash',       // primary — Gemini 3.6 Flash, optimized for code & reasoning
  'gemini-3.5-flash-lite',  // fallback 1 — cheaper, faster for high volume
  'gemini-2.0-flash-exp',   // fallback 2 — experimental 2.0
  'gemini-1.5-flash',       // fallback 3 — stable 1.5 production
  'gemini-1.5-pro'          // fallback 4 — most capable but slower
];

async function analyzeBillFrame(dataUrl) {
  const key = GEMINI_API_KEY || (typeof window !== 'undefined' && window.GEMINI_API_KEY);
  if (!key || key === 'YOUR_GEMINI_API_KEY_HERE') {
    throw new Error('Gemini API key is missing. Please set it in config.js');
  }

  const base64Data = dataUrl.split(',')[1];

  // NOTE: Do NOT use responseMimeType:"application/json" or thinkingConfig here —
  // those fields cause HTTP 400 "invalid argument" on gemini-2.0-flash and 1.5-flash.
  // We ask for JSON in the prompt and parse it manually instead.
  const payload = {
    contents: [{
      parts: [
        { text: BILL_PROMPT },
        { inline_data: { mime_type: 'image/jpeg', data: base64Data } }
      ]
    }],
    generationConfig: {
      temperature:     0.1,
      maxOutputTokens: 2500
    }
  };

  let lastError = null;

  for (const modelName of FALLBACK_MODELS) {
    try {
      const url  = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${key}`;
      const resp = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload)
      });

      if (!resp.ok) {
        const errTxt = await resp.text();
        let msg = `HTTP ${resp.status}`;
        try { const j = JSON.parse(errTxt); msg = j.error?.message || msg; } catch (_) {}
        console.warn(`[Gemini] ${modelName} → ${msg}. Trying next model…`);
        lastError = new Error(msg);
        await new Promise(r => setTimeout(r, 500));
        continue;
      }

      const data = await resp.json();
      let text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || '';

      // Strip markdown code fences if Gemini wraps the JSON
      if (text.startsWith('```')) {
        text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
      }

      // Extract the outermost JSON object
      const firstBrace = text.indexOf('{');
      const lastBrace  = text.lastIndexOf('}');
      if (firstBrace !== -1 && lastBrace !== -1) {
        text = text.slice(firstBrace, lastBrace + 1);
      }

      const parsed = JSON.parse(text);
      console.log(`[Gemini] ${modelName} succeeded`);
      return parsed;

    } catch (err) {
      lastError = err;
      console.warn(`[Gemini] ${modelName} failed:`, err.message);
    }
  }

  throw lastError || new Error('Gemini OCR failed on all model endpoints.');
}


// ── Canvas HUD Graphics ──────────────────────────────────────────────────
function drawScanLaser(t, duration) {
  const pct = t / (duration || 1);
  const y = (pct * canvas.height) % canvas.height;

  // Scanner laser line
  ctx.strokeStyle = 'rgba(240, 192, 64, 0.85)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(canvas.width, y);
  ctx.stroke();

  // Top info HUD
  ctx.fillStyle = 'rgba(11, 25, 41, 0.8)';
  ctx.fillRect(16, 16, 260, 32);
  ctx.strokeStyle = 'rgba(240, 192, 64, 0.4)';
  ctx.lineWidth = 1;
  ctx.strokeRect(16, 16, 260, 32);

  ctx.fillStyle = '#f0c040';
  ctx.font = 'bold 12px "SFMono-Regular", Consolas, monospace';
  ctx.fillText(`⟳ SCANNING VIDEO: ${t.toFixed(1)}s / ${duration.toFixed(1)}s`, 26, 36);
}

function drawScanHUD(t, duration, curFrame, totalFrames, billsFound) {
  ctx.fillStyle = 'rgba(11, 25, 41, 0.85)';
  ctx.fillRect(16, 16, 320, 48);
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(16, 16, 320, 48);

  ctx.fillStyle = '#38bdf8';
  ctx.font = 'bold 12px "SFMono-Regular", Consolas, monospace';
  ctx.fillText(`GEMINI OCR · FRAME ${curFrame}/${totalFrames}`, 26, 35);
  ctx.fillStyle = '#f0c040';
  ctx.fillText(`BILLS EXTRACTED: ${billsFound} FOUND`, 26, 52);
}

function flashDetectedHUD(billIndex, title) {
  ctx.fillStyle = 'rgba(52, 211, 153, 0.85)';
  ctx.fillRect(16, canvas.height - 54, 340, 38);
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 13px "SFMono-Regular", Consolas, monospace';
  ctx.fillText(`✓ BILL #${billIndex} CAPTURED: ${String(title || '').slice(0, 22)}`, 26, canvas.height - 30);
}

// ── Render Formatted High-Contrast Bill Card ─────────────────────────────
function renderBillCard(bill, container) {
  const itemsHtml = (bill.items && bill.items.length)
    ? `<table class="bill-items-table">
        <thead>
          <tr>
            <th>Item / SKU</th>
            <th style="text-align:center;width:40px">Qty</th>
            <th style="text-align:right">Price</th>
            <th style="text-align:right">Total</th>
          </tr>
        </thead>
        <tbody>
          ${bill.items.map(it => `
            <tr>
              <td><strong>${it.name || 'Item'}</strong></td>
              <td style="text-align:center">${it.qty || '1'}</td>
              <td style="text-align:right">${it.price || '—'}</td>
              <td style="text-align:right;color:#f0c040;font-weight:600">${it.amount || it.price || '—'}</td>
            </tr>`).join('')}
        </tbody>
       </table>`
    : '';

  // Always render — shows "—" if value is missing
  const fld = (label, val, isMono = false) =>
    `<div class="bill-field">
       <span class="bill-label">${label}</span>
       <span class="bill-val ${isMono ? 'mono' : ''}">${val || '—'}</span>
     </div>`;

  // Only renders if value exists (for multi-line address blocks)
  const infoBox = (label, val) => val
    ? `<div class="bill-info-box">
         <span style="color:#94a3b8;font-weight:600;font-size:10px;text-transform:uppercase">${label}:</span><br>
         ${String(val).replace(/\n/g, '<br>')}
       </div>`
    : '';

  // Timeline display — sits between header and scrollable body
  const timelineHtml = bill.timestamp !== undefined
    ? `<div class="bill-timeline" style="background:rgba(56,189,248,0.08);border-bottom:1px solid rgba(56,189,248,0.2);border-left:3px solid #38bdf8;padding:7px 14px;font-size:11px;color:#94a3b8;flex-shrink:0">
         <span style="color:#38bdf8;font-weight:bold">⏱ TIMELINE:</span>
         This bill appears at <strong style="color:#f0c040">${bill.timestamp.toFixed(2)}s</strong> in the video
       </div>`
    : '';

  // Unrecognized badge — kept for safety but should never appear now
  const isUnrecognized = false;
  const unrecognizedBadge = '';

  // Build the full address string combining address + pin code
  const fullAddress = [bill.customer_address, bill.pin_code ? `PIN: ${bill.pin_code}` : ''].filter(Boolean).join(' · ') || null;
  const senderFull  = [bill.sender_name, bill.sender_address].filter(Boolean).join(' — ') || null;

  const cardHtml = `
    <div class="bill-card ${isUnrecognized ? 'unrecognized' : ''}"
         id="bill-card-${bill.bill_index}"
         onclick="inspectBillFrame(${bill.bill_index})"
         style="cursor:pointer;${isUnrecognized ? 'border-color:#f87171;opacity:0.85;' : ''}"
         title="Click to view this frame on canvas">

      <div class="bill-card-header">
        <span class="bill-card-num">BILL #${bill.bill_index}</span>
        <span class="bill-card-vendor">${bill.vendor_name || bill.doc_type || 'Document'}</span>
      </div>

      ${timelineHtml}

      <div class="bill-card-body">
        <div class="bill-fields-grid">
          ${fld('Doc Type',       bill.doc_type)}
          ${fld('Bill / AWB No',  bill.bill_number, true)}
          ${fld('Date',           bill.date)}
          ${fld('Payment',        bill.payment_method)}
        </div>

        <div class="bill-section-label">RECIPIENT</div>
        <div class="bill-fields-grid">
          ${fld('Customer Name',  bill.customer_name)}
          ${fld('PIN Code',       bill.pin_code)}
        </div>
        ${infoBox('Delivery Address', fullAddress)}

        ${senderFull ? `<div class="bill-section-label">SENDER</div>${infoBox('From', senderFull)}` : ''}

        ${bill.product_description
          ? `<div class="bill-section-label">PRODUCT</div>
             <div style="padding:8px 10px;background:rgba(255,255,255,0.04);border-radius:6px;font-size:12px;color:var(--text-bright);margin-bottom:10px">
               ${bill.product_description}
             </div>`
          : ''}

        ${itemsHtml
          ? `<div class="bill-section-label">ITEMS</div>${itemsHtml}`
          : ''}

        <div class="bill-section-label">AMOUNTS</div>
        <div class="bill-fields-grid">
          ${fld('Subtotal',  bill.subtotal)}
          ${fld('Tax / GST', bill.tax)}
          ${fld('Discount',  bill.discount)}
        </div>

        ${infoBox('Notes', bill.notes)}

        ${!isUnrecognized
          ? `<div class="bill-total-row">
               <span>TOTAL / COD AMOUNT</span>
               <strong>${bill.total || '—'}</strong>
             </div>`
          : ''}      </div>
    </div>`;

  container.insertAdjacentHTML('beforeend', cardHtml);
}

// Click bill card to inspect its frame on canvas
function inspectBillFrame(billIndex) {
  const bill = allBills.find(b => b.bill_index === billIndex);
  if (!bill) return;

  if (bill.timestamp !== undefined && vidSrc.duration) {
    vidSrc.currentTime = bill.timestamp;
    renderCanvasVideoFrame();
  } else if (bill.frameData) {
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      flashDetectedHUD(bill.bill_index, bill.vendor_name || bill.bill_number);
    };
    img.src = bill.frameData;
  }

  // Highlight card
  document.querySelectorAll('.bill-card').forEach(c => c.style.borderColor = '');
  const activeCard = document.getElementById(`bill-card-${billIndex}`);
  if (activeCard) activeCard.style.borderColor = '#f0c040';
}

function computeGrandTotal(bills) {
  let sum = 0, cur = '';
  for (const b of bills) {
    const m = String(b.total || '').match(/([₹$€£]?)([0-9,.]+)/);
    if (m) {
      if (!cur && m[1]) cur = m[1];
      sum += parseFloat(m[2].replace(/,/g, ''));
    }
  }
  return sum > 0 ? `${cur || '₹'}${sum.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : null;
}

function exportBillsJSON() {
  const clean = allBills.map(({ frameData, ...rest }) => rest);
  const blob  = new Blob([JSON.stringify({ total_bills: allBills.length, bills: clean }, null, 2)], { type: 'application/json' });
  const a     = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'bills_extracted.json' });
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportBillsCSV() {
  const cols = ['bill_index', 'doc_type', 'vendor_name', 'bill_number', 'date', 'customer_name', 'customer_address', 'subtotal', 'tax', 'discount', 'total', 'payment_method', 'notes'];
  const rows = [cols.join(',')];
  for (const b of allBills) {
    rows.push(cols.map(c => JSON.stringify(b[c] ?? '')).join(','));
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const a    = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'bills_extracted.csv' });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ═══════════════════════════════════════════════════════════════════════════
// EXCEL EXPORT  (SheetJS / xlsx)
// Generates a formatted .xlsx with two sheets:
//   Sheet 1 — "Bills Summary"  : one row per recognized bill (all key fields)
//   Sheet 2 — "Items Detail"   : one row per line-item across all bills
// ═══════════════════════════════════════════════════════════════════════════
function exportBillsExcel() {
  if (!allBills.length) { alert('No bills to export yet.'); return; }

  if (typeof XLSX === 'undefined') {
    alert('Excel library not loaded. Please refresh the page and try again.');
    return;
  }

  // ── Sheet 1: Bills Summary ───────────────────────────────────────────
  const summaryHeaders = [
    'Bill #', 'Doc Type', 'Vendor / Courier', 'AWB / Bill No',
    'Date', 'Customer Name', 'Delivery Address', 'PIN Code',
    'Sender Name', 'Sender Address', 'Product Description',
    'Subtotal', 'Tax / GST', 'Discount', 'Total / COD Amount',
    'Payment Method', 'Notes', 'Appears At (sec)', 'Status'
  ];

  const summaryRows = allBills.map(b => [
    b.bill_index,
    b.doc_type             || '—',
    b.vendor_name          || '—',
    b.bill_number          || '—',
    b.date                 || '—',
    b.customer_name        || '—',
    b.customer_address     || '—',
    b.pin_code             || '—',
    b.sender_name          || '—',
    b.sender_address       || '—',
    b.product_description  || '—',
    b.subtotal             || '—',
    b.tax                  || '—',
    b.discount             || '—',
    b.total                || '—',
    b.payment_method       || '—',
    b.notes                || '—',
    b.timestamp !== undefined ? parseFloat(b.timestamp.toFixed(2)) : '—',
    b.is_unrecognized ? 'NOT RECOGNIZED' : 'RECOGNIZED'
  ]);

  const summarySheet = XLSX.utils.aoa_to_sheet([summaryHeaders, ...summaryRows]);

  // Column widths for readability
  summarySheet['!cols'] = [
    { wch: 7  },  // Bill #
    { wch: 18 },  // Doc Type
    { wch: 20 },  // Vendor
    { wch: 22 },  // AWB
    { wch: 14 },  // Date
    { wch: 22 },  // Customer Name
    { wch: 35 },  // Address
    { wch: 10 },  // PIN
    { wch: 20 },  // Sender Name
    { wch: 30 },  // Sender Address
    { wch: 30 },  // Product
    { wch: 12 },  // Subtotal
    { wch: 12 },  // Tax
    { wch: 12 },  // Discount
    { wch: 18 },  // Total
    { wch: 14 },  // Payment
    { wch: 30 },  // Notes
    { wch: 14 },  // Timestamp
    { wch: 16 },  // Status
  ];

  // ── Sheet 2: Items Detail ────────────────────────────────────────────
  const itemHeaders = ['Bill #', 'Vendor / Courier', 'AWB / Bill No', 'Item Name', 'Qty', 'Unit Price', 'Line Total'];
  const itemRows = [];
  for (const b of allBills) {
    if (b.is_unrecognized) continue;
    const items = b.items || [];
    if (items.length) {
      for (const it of items) {
        itemRows.push([
          b.bill_index,
          b.vendor_name  || '—',
          b.bill_number  || '—',
          it.name        || '—',
          it.qty         || '1',
          it.price       || '—',
          it.amount      || it.price || '—'
        ]);
      }
    } else {
      // No line items — still add a summary row
      itemRows.push([
        b.bill_index,
        b.vendor_name         || '—',
        b.bill_number         || '—',
        b.product_description || '—',
        '1', '—',
        b.total               || '—'
      ]);
    }
  }

  const itemSheet = XLSX.utils.aoa_to_sheet(
    itemRows.length ? [itemHeaders, ...itemRows] : [itemHeaders, ['No line-item data extracted']]
  );
  itemSheet['!cols'] = [
    { wch: 7  }, { wch: 20 }, { wch: 22 },
    { wch: 35 }, { wch: 6  }, { wch: 14 }, { wch: 14 }
  ];

  // ── Build Workbook ────────────────────────────────────────────────────
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, summarySheet, 'Bills Summary');
  XLSX.utils.book_append_sheet(wb, itemSheet,    'Items Detail');

  // Filename: bills_YYYY-MM-DD.xlsx
  const dateStr = new Date().toISOString().slice(0, 10);
  XLSX.writeFile(wb, `bills_${dateStr}.xlsx`);
}

// ═══════════════════════════════════════════════════════════════════════════
// CANVAS HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function showCanvas()      { placeholder.style.display='none'; canvas.style.display='block'; }
function hideCanvas()      { canvas.style.display='none'; placeholder.style.display='flex'; }
function resizeCanvas(w,h) { canvas.width=w; canvas.height=h; }

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE MODE
// ═══════════════════════════════════════════════════════════════════════════
function loadImage(event) {
  const file = event.target.files[0];
  if (!file) return;
  event.target.value = '';

  if (!cocoModel) {
    manifestBody.innerHTML = `<p class="manifest-empty" style="color:#f87171">Model still loading, wait a moment…</p>`;
    return;
  }

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onerror = () => { URL.revokeObjectURL(url); };

  img.onload = async () => {
    const rc = resizeImage(img, MAX_IMG_PX);
    resizeCanvas(rc.width, rc.height);
    showCanvas();
    hideVideoControls();
    ctx.drawImage(rc, 0, 0);
    URL.revokeObjectURL(url);

    manifestBody.innerHTML = `<p class="manifest-empty">Detecting objects…</p>`;
    copyBtn.style.display = 'none';

    // ── Detection (always runs, no key needed) ────────────
    let preds = await cocoModel.detect(rc, 100);   // up to 100 boxes
    drawDetections(rc, preds, CONF_IMG);
    updateManifest(preds, 'COCO-SSD', CONF_IMG);

    // ── Gemini label verification & small object enhancement ──
    const hasKey = geminiKey || GEMINI_API_KEY || (typeof window !== 'undefined' && window.GEMINI_API_KEY) || (typeof localStorage !== 'undefined' && localStorage.getItem('GEMINI_API_KEY'));
    if (hasKey && hasKey !== 'YOUR_GEMINI_API_KEY_HERE') {
      manifestBody.innerHTML += `<p class="manifest-empty" style="font-size:10px;margin-top:6px;color:var(--text-dim)">⟳ Verifying labels with Gemini…</p>`;
      try {
        const verified = await verifyDetectionsWithGemini(rc, preds);
        preds = verified;
        drawDetections(rc, preds, CONF_IMG);
        updateManifest(preds, 'COCO + GEMINI ✓', CONF_IMG);
      } catch(e) {
        console.warn('Label verify skipped:', e.message);
        drawDetections(rc, preds, CONF_IMG);
        updateManifest(preds, 'COCO-SSD', CONF_IMG);
      }
    }

    // ── Scene description (needs Gemini key) ─────────────
    if (geminiKey) {
      try {
        const caption = await describeImage(rc);
        if (caption) showDescription(caption);
        else showDescError('No description returned by Gemini.');
      } catch(err) {
        console.error('Gemini error:', err.message);
        showDescError('Gemini error: ' + err.message.slice(0, 120));
      }
    } else {
      descBody.innerHTML = `<p class="manifest-empty">Add a Gemini API key above to generate an AI scene description.</p>`;
      descBadge.textContent = 'OFF';
    }
  };

  img.src = url;
}

// ═══════════════════════════════════════════════════════════════════════════
// VIDEO CONTROLS & LIVE CANVAS SCRUBBING
// ═══════════════════════════════════════════════════════════════════════════
function showVideoControls() { vidControls.classList.add('visible'); }
function hideVideoControls() { vidControls.classList.remove('visible'); }

// Draws the current video frame directly to the visible canvas
function renderCanvasVideoFrame() {
  if (vidSrc.readyState >= 2 && canvas.width && canvas.height) {
    ctx.drawImage(vidSrc, 0, 0, canvas.width, canvas.height);
  }
}

function updateScrubber() {
  if (!vidSrc.duration || isScrubbing) return;
  const pct = (vidSrc.currentTime / vidSrc.duration) * 100;
  scrubberFill.style.width = pct + '%';
  scrubberThumb.style.left = pct + '%';
  vcTime.textContent = `${fmtTime(vidSrc.currentTime)} / ${fmtTime(vidSrc.duration)}`;
}

function updatePlayIcon() {
  iconPlay.style.display  = vidSrc.paused ? 'block' : 'none';
  iconPause.style.display = vidSrc.paused ? 'none'  : 'block';
}

function playVideoFrameLoop() {
  if (vidSrc.paused || vidSrc.ended) {
    updatePlayIcon();
    return;
  }
  if (!isDetecting) {
    renderCanvasVideoFrame();
  }
  requestAnimationFrame(playVideoFrameLoop);
}

function togglePlay() {
  if (vidSrc.paused || vidSrc.ended) {
    if (vidSrc.ended || (vidSrc.duration && vidSrc.currentTime >= vidSrc.duration - 0.1)) {
      vidSrc.currentTime = 0;
    }
    
    // Play with fallback for muted autoplay policy
    vidSrc.play().then(() => {
      updatePlayIcon();
      if (currentMode === 'video' && !isDetecting) {
        isDetecting = true;
        detectVideoFrame();
      } else {
        playVideoFrameLoop();
      }
    }).catch(() => {
      vidSrc.muted = true;
      iconVol.style.display   = 'none';
      iconMuted.style.display = 'block';
      vidSrc.play().then(() => {
        updatePlayIcon();
        if (currentMode === 'video' && !isDetecting) {
          isDetecting = true;
          detectVideoFrame();
        } else {
          playVideoFrameLoop();
        }
      });
    });
  } else {
    vidSrc.pause();
    updatePlayIcon();
  }
}

function toggleMute() {
  vidSrc.muted = !vidSrc.muted;
  iconVol.style.display   = vidSrc.muted ? 'none'  : 'block';
  iconMuted.style.display = vidSrc.muted ? 'block' : 'none';
}

function scrubTo(cx) {
  const rect = scrubberTrack.getBoundingClientRect();
  const pct  = Math.max(0, Math.min(1, (cx - rect.left) / rect.width));
  vidSrc.currentTime = pct * vidSrc.duration;
  scrubberFill.style.width = (pct * 100) + '%';
  scrubberThumb.style.left = (pct * 100) + '%';
  vcTime.textContent = `${fmtTime(vidSrc.currentTime)} / ${fmtTime(vidSrc.duration)}`;
  renderCanvasVideoFrame();
}

scrubberTrack.addEventListener('mousedown', e => { isScrubbing = true; scrubTo(e.clientX); });
document.addEventListener('mousemove', e => { if (isScrubbing) scrubTo(e.clientX); });
document.addEventListener('mouseup',   () => { isScrubbing = false; });
scrubberTrack.addEventListener('touchstart', e => { isScrubbing = true; scrubTo(e.touches[0].clientX); }, { passive: true });
document.addEventListener('touchmove', e => { if (isScrubbing) scrubTo(e.touches[0].clientX); }, { passive: true });
document.addEventListener('touchend',  () => { isScrubbing = false; });

// Live video events -> keep canvas in sync with video at all times
vidSrc.addEventListener('seeked', () => {
  if (!isDetecting) renderCanvasVideoFrame();
});

vidSrc.addEventListener('timeupdate', () => {
  updateScrubber();
  if (!isDetecting && !vidSrc.paused) renderCanvasVideoFrame();
});

vidSrc.addEventListener('play', () => {
  updatePlayIcon();
  if (!isDetecting) playVideoFrameLoop();
});

vidSrc.addEventListener('pause', updatePlayIcon);
vidSrc.addEventListener('ended', () => {
  updatePlayIcon();
  stopDetection();
});


// ═══════════════════════════════════════════════════════════════════════════
// VIDEO MODE
// ═══════════════════════════════════════════════════════════════════════════
function loadVideo(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (!cocoModel) { alert('Model still loading, please wait.'); return; }
  stopDetection();
  resetTracking();   // clear any tracked objects from previous video
  const url = URL.createObjectURL(file);
  vidSrc.src=url; vidSrc.muted=false; vidSrc.style.display='none'; vidSrc.load();
  iconVol.style.display='block'; iconMuted.style.display='none';
  vidSrc.onloadedmetadata = () => {
    resizeCanvas(vidSrc.videoWidth, vidSrc.videoHeight);
    showCanvas(); showVideoControls();
    vcTime.textContent = `0:00 / ${fmtTime(vidSrc.duration)}`;
    vidSrc.play();
    isDetecting=true; stopBtn.style.display='inline-block';
    detectVideoFrame();
    startGeminiInterval();
  };
  event.target.value='';
}

// Apply cached Gemini label overrides to a raw pred array
function applyLabelOverrides(preds) {
  if (!Object.keys(geminiLabelOverrides).length) return preds;
  return preds.map(p => {
    const fix = geminiLabelOverrides[p.class.toLowerCase()];
    return fix ? { ...p, class: fix } : p;
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// TEMPORAL SMOOTHING ENGINE
// Tracks objects across frames using IoU matching.
// Smooths bounding box positions and confidence scores over time.
// Removes objects that disappear for more than SMOOTH_MAX_MISS frames.
// Result: stable, jitter-free boxes that stay locked onto moving objects.
// ═══════════════════════════════════════════════════════════════════════════

function boxIoU(a, b) {
  // a, b = [x, y, w, h]
  const ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const bx2 = b[0] + b[2], by2 = b[1] + b[3];
  const ix  = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
  const iy  = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  if (inter === 0) return 0;
  return inter / (a[2]*a[3] + b[2]*b[3] - inter);
}

function lerpBox(old, cur, alpha) {
  return [
    old[0] + (cur[0] - old[0]) * alpha,
    old[1] + (cur[1] - old[1]) * alpha,
    old[2] + (cur[2] - old[2]) * alpha,
    old[3] + (cur[3] - old[3]) * alpha
  ];
}

function updateTrackedObjects(newPreds) {
  // Mark all existing tracks as unmatched
  const matched = new Set();

  // Match each new detection to the closest existing tracked object
  for (const pred of newPreds) {
    if (pred.score < CONF_VIDEO) continue;

    let bestIdx   = -1;
    let bestIoU   = SMOOTH_IOU_THRESH;

    for (let i = 0; i < trackedObjects.length; i++) {
      if (matched.has(i)) continue;
      if (trackedObjects[i].class !== pred.class) continue;
      const iou = boxIoU(trackedObjects[i].bbox, pred.bbox);
      if (iou > bestIoU) { bestIoU = iou; bestIdx = i; }
    }

    if (bestIdx >= 0) {
      // Update existing track — smooth position and score
      const t = trackedObjects[bestIdx];
      t.bbox      = lerpBox(t.bbox, pred.bbox, SMOOTH_ALPHA);
      t.score     = t.score + (pred.score - t.score) * SMOOTH_SCORE_ALPHA;
      t.missCount = 0;
      t.age++;
      matched.add(bestIdx);
    } else {
      // New object — create fresh track
      trackedObjects.push({
        id:        ++trackIdCounter,
        class:     pred.class,
        score:     pred.score,
        bbox:      [...pred.bbox],
        missCount: 0,
        age:       1
      });
    }
  }

  // Age out unmatched tracks
  for (let i = trackedObjects.length - 1; i >= 0; i--) {
    if (!matched.has(i)) {
      trackedObjects[i].missCount++;
      if (trackedObjects[i].missCount > SMOOTH_MAX_MISS) {
        trackedObjects.splice(i, 1);
      }
    }
  }

  return trackedObjects;
}

function resetTracking() {
  trackedObjects  = [];
  trackIdCounter  = 0;
  isDetectRunning = false;
  lastDetectTime  = 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// VIDEO DETECTION LOOP  (throttled async — no rAF queue buildup)
// Instead of requestAnimationFrame which fires every 16ms and queues faster
// than the model can respond, we use a self-scheduling async loop that waits
// for each detect() call to finish before scheduling the next frame.
// ═══════════════════════════════════════════════════════════════════════════
async function detectVideoFrame() {
  if (!isDetecting || vidSrc.paused || vidSrc.ended) return;

  const now = performance.now();
  const elapsed = now - lastDetectTime;

  // Throttle: if less than DETECT_INTERVAL_MS since last detect, draw current
  // tracked objects without running the model again (saves CPU, keeps display smooth)
  if (elapsed < DETECT_INTERVAL_MS || isDetectRunning) {
    // Draw current smoothed boxes without re-detecting
    drawSmoothedBoxes();
    animFrameId = requestAnimationFrame(detectVideoFrame);
    return;
  }

  isDetectRunning = true;
  lastDetectTime  = now;

  // Snapshot current video frame to offscreen canvas — prevents tearing
  // when the video advances while we're still drawing
  const offscreen = document.createElement('canvas');
  offscreen.width  = canvas.width;
  offscreen.height = canvas.height;
  offscreen.getContext('2d').drawImage(vidSrc, 0, 0, canvas.width, canvas.height);

  try {
    const raw   = await cocoModel.detect(offscreen, VIDEO_MAX_BOXES);
    lastPreds   = raw;
    const fixed = applyLabelOverrides(raw);
    updateTrackedObjects(fixed);
  } catch(e) {
    console.warn('detect error:', e.message);
  }

  isDetectRunning = false;

  // Draw the offscreen snapshot with current smoothed boxes on top
  ctx.drawImage(offscreen, 0, 0);
  drawSmoothedBoxes();

  const engine = Object.keys(geminiLabelOverrides).length ? 'COCO + GEMINI ✓' : 'COCO-SSD';
  updateManifest(trackedObjects, engine, CONF_VIDEO);

  animFrameId = requestAnimationFrame(detectVideoFrame);
}

function drawSmoothedBoxes() {
  trackedObjects.forEach(t => {
    if (t.score < CONF_VIDEO) return;
    // Fade boxes that are going stale (missed frames)
    const alpha = t.missCount > 0 ? Math.max(0.3, 1 - t.missCount / SMOOTH_MAX_MISS) : 1;
    const [x, y, w, h] = t.bbox;
    const color = colorFor(t.class);
    const label = t.class.toUpperCase() + '  ' + (t.score * 100).toFixed(0) + '%';

    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    ctx.font = 'bold 12px "Courier New"';
    const tw = ctx.measureText(label).width + 14;
    const ly = y > 26 ? y - 24 : y + 2;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, 22);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 7, ly + 15);
    ctx.globalAlpha = 1;
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// WEBCAM MODE
// ═══════════════════════════════════════════════════════════════════════════
async function startWebcam() {
  if (!cocoModel) { alert('Model still loading.'); return; }
  stopDetection();
  resetTracking();   // clear tracked objects from any previous session
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
  } catch(e) { alert('Camera denied: ' + e.message); return; }

  vidSrc.srcObject = webcamStream;
  vidSrc.style.display = 'none';
  vidSrc.muted = true;

  await new Promise(resolve => {
    vidSrc.onloadedmetadata = resolve;
  });

  resizeCanvas(vidSrc.videoWidth, vidSrc.videoHeight);
  showCanvas();
  hideVideoControls();

  await vidSrc.play();

  isDetecting = true;
  stopBtn.style.display = 'inline-block';
  detectWebcamFrame();
  startGeminiInterval();
}

async function detectWebcamFrame() {
  if (!isDetecting) return;

  const now     = performance.now();
  const elapsed = now - lastDetectTime;

  // Always draw the live frame so canvas never freezes
  if (vidSrc.readyState >= 2) {
    ctx.drawImage(vidSrc, 0, 0, canvas.width, canvas.height);
  }

  if (elapsed >= DETECT_INTERVAL_MS && !isDetectRunning) {
    isDetectRunning = true;
    lastDetectTime  = now;

    // Snapshot to offscreen so detection is on a frozen frame (no mid-inference tearing)
    const offscreen = document.createElement('canvas');
    offscreen.width  = canvas.width;
    offscreen.height = canvas.height;
    offscreen.getContext('2d').drawImage(vidSrc, 0, 0, canvas.width, canvas.height);

    try {
      const raw  = await cocoModel.detect(offscreen, VIDEO_MAX_BOXES);
      lastPreds  = raw;
      const fixed = applyLabelOverrides(raw);
      updateTrackedObjects(fixed);
    } catch(e) {
      console.warn('webcam detect error:', e.message);
    }

    isDetectRunning = false;
  }

  // Draw smoothed boxes on top of live frame
  const engine = Object.keys(geminiLabelOverrides).length ? 'COCO + GEMINI ✓' : 'COCO-SSD';
  drawSmoothedBoxes();
  updateManifest(trackedObjects, engine, CONF_WEBCAM);

  animFrameId = requestAnimationFrame(detectWebcamFrame);
}

// ═══════════════════════════════════════════════════════════════════════════
// STOP
// ═══════════════════════════════════════════════════════════════════════════
function stopDetection() {
  isDetecting     = false;
  isDetectRunning = false;
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  if (geminiInterval) { clearInterval(geminiInterval); geminiInterval = null; }
  resetTracking();

  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
    vidSrc.srcObject = null;
  }

  try { if (!vidSrc.paused) vidSrc.pause(); } catch(_) {}

  // Clear the canvas back to blank — prevents stale bill/image showing through
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.width  = canvas.width;   // force repaint
  hideCanvas();

  stopBtn.style.display = 'none';
  hideVideoControls();

  manifestBody.innerHTML = `<p class="manifest-empty">Detection stopped.</p>`;
}

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════
loadCocoModel();
