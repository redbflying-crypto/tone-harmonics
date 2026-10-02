/* Voice Harmonics Analyzer — browser side.
 *
 * 1. Microphone: permission, device list (with rescan / hot-plug), level meter.
 * 2. Recording: raw PCM captured by an AudioWorklet, encoded as 16-bit WAV.
 * 3. Takes: stored in IndexedDB (this browser only), playable, renamable, deletable.
 * 4. Analysis: the WAV is POSTed to /api/analyze; results are drawn here.
 */
'use strict';

// Vercel functions accept request bodies up to 4.5 MB: 40 s of 48 kHz 16-bit mono
// WAV is ~3.84 MB, and takes recorded above 48 kHz are resampled to 48 kHz.
const MAX_REC_SEC = 40;
const MAX_SAMPLE_RATE = 48000;
const MAX_UPLOAD_BYTES = 4400000;
const MIN_REC_SEC = 0.3;
const SILENCE_CHECK_SEC = 1.5;
const NAMES = {
  en: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
  fr: ['Do', 'Do#', 'Ré', 'Ré#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si'],
};

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (k in node && typeof v !== 'string') node[k] = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const prefs = {
  get(key, fallback) {
    try { const v = localStorage.getItem('vh.' + key); return v === null ? fallback : v; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem('vh.' + key, value); } catch { /* storage unavailable */ }
  },
};

let msgTimer = 0;
function showMsg(text, kind = 'error', sticky = false) {
  const box = $('global-msg');
  box.textContent = text;
  box.className = 'msg ' + kind;
  clearTimeout(msgTimer);
  if (!sticky) msgTimer = setTimeout(() => box.classList.add('hidden'), kind === 'error' ? 12000 : 6000);
}

const fmtDb = (v) => (Number.isFinite(v) ? v.toFixed(1) : '—');
const fmtSec = (s) => `${s.toFixed(1)} s`;
const dbOf = (amp) => 20 * Math.log10(Math.max(amp, 1e-9));

function newId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------- storage
const store = (() => {
  let dbPromise = null;
  let memoryOnly = false;
  const memory = new Map();

  function open() {
    if (!dbPromise) {
      dbPromise = new Promise((resolve, reject) => {
        if (!window.indexedDB) { reject(new Error('IndexedDB unavailable')); return; }
        const req = indexedDB.open('voice-harmonics', 1);
        req.onupgradeneeded = () => req.result.createObjectStore('takes', { keyPath: 'id' });
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
        req.onblocked = () => reject(new Error('IndexedDB blocked by another tab'));
      });
    }
    return dbPromise;
  }

  async function tx(mode, fn) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const t = db.transaction('takes', mode);
      const req = fn(t.objectStore('takes'));
      t.oncomplete = () => resolve(req ? req.result : undefined);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error || new Error('Transaction aborted'));
    });
  }

  function fallBack(err) {
    if (!memoryOnly) {
      memoryOnly = true;
      console.warn('IndexedDB unavailable, keeping takes in memory:', err);
      showMsg('This browser does not allow saving recordings (private mode?). Takes are kept until the page is closed.', 'warn', true);
    }
  }

  async function run(mode, fn, memFn) {
    if (!memoryOnly) {
      try { return await tx(mode, fn); } catch (err) {
        if (err && err.name === 'QuotaExceededError') throw new Error('Browser storage is full. Delete some recordings first.');
        fallBack(err);
      }
    }
    return memFn();
  }

  return {
    all: () => run('readonly', (s) => s.getAll(), () => [...memory.values()]),
    put: (take) => run('readwrite', (s) => s.put(take), () => { memory.set(take.id, take); }),
    del: (id) => run('readwrite', (s) => s.delete(id), () => { memory.delete(id); }),
    clear: () => run('readwrite', (s) => s.clear(), () => { memory.clear(); }),
  };
})();

// ---------------------------------------------------------------- audio context
const mic = {
  ctx: null, workletCtx: null, stream: null, track: null, source: null, analyser: null,
  worklet: null, sink: null, deviceId: null, label: '', openToken: 0, raf: 0,
  silence: { until: 0, maxPeak: 0, reported: false }, peakHold: 0, clipUntil: 0,
};

async function ensureContext(sampleRate) {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) throw new Error('This browser does not support the Web Audio API.');
  if (mic.ctx && sampleRate && mic.ctx.sampleRate !== sampleRate) {
    await mic.ctx.close().catch(() => {});
    mic.ctx = null;
  }
  if (!mic.ctx) mic.ctx = sampleRate ? new AC({ sampleRate }) : new AC();
  if (mic.ctx.state === 'suspended') await mic.ctx.resume();
  return mic.ctx;
}

function describeGumError(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access was refused. Allow it in the browser (icon in the address bar) and, on Windows, in ' +
        'Settings → Privacy & security → Microphone ("Let apps access your microphone" and "Let desktop apps access your microphone").';
    case 'NotFoundError':
      return 'No microphone was found. Plug one in, then press “Rescan”.';
    case 'NotReadableError':
    case 'AbortError':
      return 'The microphone could not be started — another application may be using it exclusively, or the driver failed. ' +
        'Close other audio apps or pick another device.';
    case 'OverconstrainedError':
      return 'The selected microphone is no longer available. Pick another device.';
    default:
      return `Could not open the microphone: ${err && err.message ? err.message : err}`;
  }
}

function closeMic() {
  cancelAnimationFrame(mic.raf);
  mic.raf = 0;
  for (const node of [mic.source, mic.analyser, mic.worklet, mic.sink]) {
    try { node && node.disconnect(); } catch { /* already disconnected */ }
  }
  if (mic.worklet) mic.worklet.port.onmessage = null;
  if (mic.stream) mic.stream.getTracks().forEach((t) => { t.onended = null; t.stop(); });
  Object.assign(mic, { stream: null, track: null, source: null, analyser: null, worklet: null, sink: null });
  updateControls();
  drawMeter(-Infinity);
}

async function openMic(deviceId) {
  if (rec.active) return;
  if (!window.isSecureContext) {
    showMsg('The microphone only works on https:// pages or on http://localhost — open the app through one of those.', 'error', true);
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showMsg('This browser cannot access microphones. Use a recent Chrome, Edge, Firefox or Safari.', 'error', true);
    return;
  }
  const token = ++mic.openToken;
  closeMic();
  setMicInfo('Opening microphone…');

  let ctx;
  try { ctx = await ensureContext(); } catch (err) { showMsg(err.message); return; }

  const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: { ideal: 1 } };
  if (deviceId) audio.deviceId = { exact: deviceId };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio });
  } catch (err) {
    if (token !== mic.openToken) return;
    if (deviceId && (err.name === 'OverconstrainedError' || err.name === 'NotFoundError')) {
      prefs.set('device', '');
      showMsg('The previously used microphone is not available; switching to the default one.', 'warn');
      return openMic(null);
    }
    setMicInfo(describeGumError(err), true);
    showMsg(describeGumError(err), 'error', true);
    await refreshDevices();
    return;
  }
  if (token !== mic.openToken) { stream.getTracks().forEach((t) => t.stop()); return; }

  const track = stream.getAudioTracks()[0];
  const settings = track.getSettings ? track.getSettings() : {};

  try {
    if (mic.workletCtx !== ctx) {
      await ctx.audioWorklet.addModule('/static/recorder-worklet.js');
      mic.workletCtx = ctx;
    }
    let source;
    try {
      source = ctx.createMediaStreamSource(stream);
    } catch (err) {
      // Older Firefox refuses to resample: rebuild the context at the device rate.
      if (err.name !== 'NotSupportedError' || !settings.sampleRate) throw err;
      ctx = await ensureContext(settings.sampleRate);
      await ctx.audioWorklet.addModule('/static/recorder-worklet.js');
      mic.workletCtx = ctx;
      source = ctx.createMediaStreamSource(stream);
    }
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    const worklet = new AudioWorkletNode(ctx, 'recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const sink = ctx.createGain();
    sink.gain.value = 0; // keeps the worklet pulled by the graph without audible monitoring
    source.connect(analyser);
    source.connect(worklet);
    worklet.connect(sink).connect(ctx.destination);
    worklet.port.onmessage = onWorkletMessage;
    if (token !== mic.openToken) { stream.getTracks().forEach((t) => t.stop()); return; }
    Object.assign(mic, { stream, track, source, analyser, worklet, sink });
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    showMsg(`Audio engine error: ${err.message}`);
    setMicInfo('Audio engine error.', true);
    return;
  }

  mic.deviceId = settings.deviceId || deviceId || '';
  mic.label = track.label || 'Microphone';
  prefs.set('device', mic.deviceId);
  track.onended = onTrackEnded;
  mic.silence = { until: performance.now() + SILENCE_CHECK_SEC * 1000, maxPeak: 0, reported: false };

  const processing = ['echoCancellation', 'noiseSuppression', 'autoGainControl'].filter((k) => settings[k] === true);
  let info = `Using “${mic.label}” — recording at ${ctx.sampleRate} Hz.`;
  if (processing.length) info += ` Note: the browser kept ${processing.join(', ')} on, which can colour the harmonics.`;
  setMicInfo(info);

  await refreshDevices();
  updateControls();
  meterLoop();
}

async function onTrackEnded() {
  const wasRecording = rec.active;
  if (wasRecording) await stopRecording();
  closeMic();
  setMicInfo('The microphone was disconnected. Select a device to continue.', true);
  showMsg(wasRecording ? 'Microphone disconnected — the take was saved up to that point.' : 'Microphone disconnected.', 'warn');
  await refreshDevices();
}

async function refreshDevices() {
  const select = $('mic-select');
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  let devices = [];
  try { devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput'); } catch { /* ignore */ }
  select.replaceChildren();
  if (!devices.length) {
    select.append(el('option', { value: '', text: '— no microphone found —' }));
  }
  devices.forEach((d, i) => {
    select.append(el('option', { value: d.deviceId, text: d.label || `Microphone ${i + 1} (label hidden until permission is granted)` }));
  });
  if (mic.deviceId && devices.some((d) => d.deviceId === mic.deviceId)) select.value = mic.deviceId;
  updateControls();
}

function setMicInfo(text, isError = false) {
  const info = $('mic-info');
  info.textContent = text;
  info.classList.toggle('error-text', isError);
}

// ---------------------------------------------------------------- level meter
const meterBuf = new Float32Array(2048);

function meterLoop() {
  if (!mic.analyser) return;
  mic.analyser.getFloatTimeDomainData(meterBuf);
  let peak = 0;
  for (let i = 0; i < meterBuf.length; i++) { const a = Math.abs(meterBuf[i]); if (a > peak) peak = a; }
  const now = performance.now();
  if (peak >= 0.99) mic.clipUntil = now + 1000;

  if (!mic.silence.reported) {
    mic.silence.maxPeak = Math.max(mic.silence.maxPeak, peak);
    if (now > mic.silence.until) {
      mic.silence.reported = true;
      if (mic.silence.maxPeak === 0) {
        setMicInfo(`“${mic.label}” delivers pure silence. Check that it is not muted, its input level, and Windows ` +
          'Settings → Privacy & security → Microphone; or choose another device.', true);
      }
    }
  }
  drawMeter(dbOf(peak), now < mic.clipUntil);
  mic.raf = requestAnimationFrame(meterLoop);
}

function drawMeter(db, clipping = false) {
  const pct = (v) => `${Math.max(0, Math.min(100, ((v + 60) / 60) * 100))}%`;
  const valid = Number.isFinite(db);
  mic.peakHold = valid ? Math.max(db, mic.peakHold - 0.4) : -Infinity;
  $('meter-bar').style.width = valid ? pct(db) : '0%';
  $('meter-peak').style.left = Number.isFinite(mic.peakHold) ? pct(mic.peakHold) : '0%';
  $('meter-db').textContent = valid ? `${db.toFixed(0).padStart(3)} dBFS` : '— dBFS';
  $('clip-led').classList.toggle('on', clipping);
}

// ---------------------------------------------------------------- recording
const rec = { active: false, stopping: false, chunks: [], length: 0, timer: 0, resolveStop: null };

function onWorkletMessage(e) {
  const msg = e.data;
  if (msg.type === 'data' && rec.active) {
    rec.chunks.push(msg.samples);
    rec.length += msg.samples.length;
  } else if (msg.type === 'stopped' && rec.resolveStop) {
    rec.resolveStop();
  }
}

function startRecording() {
  if (rec.active || !mic.worklet) return;
  Object.assign(rec, { active: true, chunks: [], length: 0 });
  mic.worklet.port.postMessage('start');
  const sr = mic.ctx.sampleRate;
  rec.timer = setInterval(() => {
    const sec = rec.length / sr;
    $('rec-timer').textContent = fmtSec(sec);
    if (sec >= MAX_REC_SEC) stopRecording();
  }, 100);
  updateControls();
}

async function stopRecording() {
  if (!rec.active || rec.stopping) return;
  rec.stopping = true;
  clearInterval(rec.timer);
  const stopped = new Promise((resolve) => {
    rec.resolveStop = resolve;
    setTimeout(resolve, 1500); // the worklet may be gone if the device vanished
  });
  try { mic.worklet && mic.worklet.port.postMessage('stop'); } catch { /* port closed */ }
  await stopped;
  rec.active = false; // after the final flush has arrived
  rec.stopping = false;
  rec.resolveStop = null;
  updateControls();

  const sr = mic.ctx.sampleRate;
  const samples = new Float32Array(rec.length);
  let offset = 0;
  for (const c of rec.chunks) { samples.set(c, offset); offset += c.length; }
  rec.chunks = [];
  $('rec-timer').textContent = fmtSec(samples.length / sr);

  if (samples.length < MIN_REC_SEC * sr) {
    showMsg(`Take too short (${fmtSec(samples.length / sr)}) — not saved.`, 'warn');
    return;
  }
  const name = $('take-name').value.trim();
  $('take-name').value = '';
  await saveTake(samples, sr, name || `Take ${new Date().toLocaleTimeString()}`, mic.label);
}

// ---------------------------------------------------------------- WAV
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); writeStr(8, 'WAVE');
  writeStr(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  writeStr(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0, off = 44; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, Math.round(s < 0 ? s * 32768 : s * 32767), true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function levelStats(samples, buckets = 240) {
  let peak = 0, sum = 0;
  const peaks = new Array(buckets).fill(0);
  const per = samples.length / buckets;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    sum += samples[i] * samples[i];
    if (a > peak) peak = a;
    const b = Math.min(buckets - 1, Math.floor(i / per));
    if (a > peaks[b]) peaks[b] = a;
  }
  return { peak, rms: Math.sqrt(sum / Math.max(1, samples.length)), peaks: peaks.map((p) => Math.round(p * 1000) / 1000) };
}

async function resample(samples, fromRate, toRate) {
  const length = Math.round((samples.length * toRate) / fromRate);
  const offline = new OfflineAudioContext(1, length, toRate);
  const buffer = offline.createBuffer(1, samples.length, fromRate);
  buffer.copyToChannel(samples, 0);
  const src = offline.createBufferSource();
  src.buffer = buffer;
  src.connect(offline.destination);
  src.start();
  return (await offline.startRendering()).getChannelData(0);
}

async function saveTake(samples, sampleRate, name, origin) {
  if (sampleRate > MAX_SAMPLE_RATE) {
    try {
      samples = await resample(samples, sampleRate, MAX_SAMPLE_RATE);
      sampleRate = MAX_SAMPLE_RATE;
    } catch (err) {
      showMsg(`Could not resample the take to ${MAX_SAMPLE_RATE} Hz: ${err.message}`);
      return null;
    }
  }
  const stats = levelStats(samples);
  const take = {
    id: newId(), name, origin, createdAt: Date.now(), sampleRate,
    duration: samples.length / sampleRate, peakDb: dbOf(stats.peak), rmsDb: dbOf(stats.rms),
    peaks: stats.peaks, wav: encodeWav(samples, sampleRate),
  };
  try {
    await store.put(take);
  } catch (err) {
    showMsg(`Could not save the take: ${err.message}`);
    return null;
  }
  takes.unshift(take);
  renderList();
  if (take.peakDb < -50) showMsg(`“${name}” is almost silent (peak ${fmtDb(take.peakDb)} dBFS). Check the microphone and its level.`, 'warn');
  else if (stats.peak >= 0.999) showMsg(`“${name}” clips. Lower the input level or sing further from the microphone.`, 'warn');
  return take;
}

async function importFile(file) {
  if (!file) return;
  let ctx;
  try { ctx = await ensureContext(); } catch (err) { showMsg(err.message); return; }
  let buffer;
  try {
    buffer = await ctx.decodeAudioData(await file.arrayBuffer());
  } catch {
    showMsg(`“${file.name}” could not be decoded by this browser. Try a WAV file.`);
    return;
  }
  const len = Math.min(buffer.length, Math.floor(MAX_REC_SEC * buffer.sampleRate));
  const mono = new Float32Array(len);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < len; i++) mono[i] += data[i] / buffer.numberOfChannels;
  }
  if (buffer.length > len) showMsg(`Only the first ${MAX_REC_SEC} s of “${file.name}” were kept.`, 'warn');
  if (len < MIN_REC_SEC * buffer.sampleRate) { showMsg('That file is too short to analyse.', 'warn'); return; }
  await saveTake(mono, buffer.sampleRate, file.name.replace(/\.[^.]+$/, ''), `Imported file ${file.name}`);
}

// ---------------------------------------------------------------- takes list
let takes = [];
const items = new Map(); // id -> { li, url }

function drawWave(canvas, peaks) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = cssVar('--accent');
  const bw = w / peaks.length;
  peaks.forEach((p, i) => {
    const bh = Math.max(1, p * (h - 2));
    ctx.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 0.5), bh);
  });
}

function buildItem(take) {
  const url = URL.createObjectURL(take.wav);
  const nameInput = el('input', {
    class: 'take-name', type: 'text', value: take.name, maxlength: '60', 'aria-label': 'Recording name',
    onchange: async (e) => {
      const v = e.target.value.trim() || take.name;
      e.target.value = v;
      if (v === take.name) return;
      take.name = v;
      try { await store.put(take); } catch (err) { showMsg(`Rename failed: ${err.message}`); }
      download.download = `${safeFileName(v)}.wav`;
      if (current && current.take.id === take.id) $('result-title').textContent = v;
    },
  });
  const download = el('a', { class: 'btn small', href: url, download: `${safeFileName(take.name)}.wav`, text: 'Download' });
  const analyseBtn = el('button', { class: 'primary small', text: 'Analyse', onclick: () => analyseTake(take, analyseBtn) });
  const deleteBtn = el('button', { class: 'danger small', text: 'Delete', onclick: () => deleteTake(take) });
  const wave = el('canvas', { class: 'wave', height: '44' });
  const meta = [
    fmtSec(take.duration), `${take.sampleRate} Hz`, `peak ${fmtDb(take.peakDb)} dBFS`,
    new Date(take.createdAt).toLocaleString(), take.origin,
  ].filter(Boolean).join(' · ');

  const li = el('li', { class: 'take', 'data-id': take.id }, [
    el('div', { class: 'take-head' }, [nameInput, el('div', { class: 'take-actions' }, [analyseBtn, download, deleteBtn])]),
    el('div', { class: 'take-meta', text: meta }),
    wave,
    el('audio', { controls: true, preload: 'metadata', src: url }),
  ]);
  items.set(take.id, { li, url, wave });
  return li;
}

function renderList() {
  const list = $('rec-list');
  const ids = new Set(takes.map((t) => t.id));
  for (const [id, item] of items) {
    if (!ids.has(id)) { item.li.remove(); URL.revokeObjectURL(item.url); items.delete(id); }
  }
  takes.forEach((take, i) => {
    const li = items.has(take.id) ? items.get(take.id).li : buildItem(take);
    if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);
  });
  for (const take of takes) drawWave(items.get(take.id).wave, take.peaks || []);
  $('rec-count').textContent = String(takes.length);
  $('rec-empty').classList.toggle('hidden', takes.length > 0);
  $('delete-all').disabled = takes.length === 0;
}

function safeFileName(s) {
  return s.replace(/[\\/:*?"<>|]+/g, '_').slice(0, 60) || 'take';
}

async function deleteTake(take) {
  if (!confirm(`Delete “${take.name}”? This cannot be undone.`)) return;
  try { await store.del(take.id); } catch (err) { showMsg(`Delete failed: ${err.message}`); return; }
  takes = takes.filter((t) => t.id !== take.id);
  if (current && current.take.id === take.id) hideResult();
  renderList();
}

async function deleteAll() {
  if (!takes.length || !confirm(`Delete all ${takes.length} recordings? This cannot be undone.`)) return;
  try { await store.clear(); } catch (err) { showMsg(`Delete failed: ${err.message}`); return; }
  takes = [];
  hideResult();
  renderList();
}

// Only one take plays at a time.
document.addEventListener('play', (e) => {
  for (const a of document.querySelectorAll('audio')) if (a !== e.target) a.pause();
}, true);

// ---------------------------------------------------------------- analysis
let current = null; // { take, data, selected }
let analysing = false;

function readSettings() {
  const a4 = Number($('a4').value);
  const gate = Number($('gate').value);
  if (!(a4 >= 400 && a4 <= 480)) throw new Error('A4 must be between 400 and 480 Hz.');
  if (!(gate >= -90 && gate <= -10)) throw new Error('The noise gate must be between -90 and -10 dBFS.');
  return { a4, gate };
}

async function analyseTake(take, button) {
  if (analysing) return;
  let settings;
  try { settings = readSettings(); } catch (err) { showMsg(err.message); return; }
  if (take.wav.size > MAX_UPLOAD_BYTES) {
    showMsg(`“${take.name}” is too large to analyse online (${(take.wav.size / 1e6).toFixed(1)} MB; ` +
      `limit ${(MAX_UPLOAD_BYTES / 1e6).toFixed(1)} MB, about ${MAX_REC_SEC} s).`);
    return;
  }
  analysing = true;
  const label = button ? button.textContent : '';
  if (button) { button.textContent = 'Analysing…'; button.disabled = true; }
  $('reanalyse').disabled = true;
  try {
    const resp = await fetch(`/api/analyze?a4=${settings.a4}&gate=${settings.gate}`, {
      method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: take.wav,
    });
    let body = null;
    try { body = await resp.json(); } catch { /* non-JSON error page */ }
    if (!resp.ok) throw new Error((body && body.error) || `Server error ${resp.status}`);
    const longest = body.segments.reduce((best, s, i) => (best < 0 || s.duration > body.segments[best].duration ? i : best), -1);
    current = { take, data: body, selected: longest };
    $('reanalyse').classList.remove('attention');
    renderResult();
    $('result-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    showMsg(err instanceof TypeError ? 'Cannot reach the analysis server. Is it running?' : `Analysis failed: ${err.message}`);
  } finally {
    analysing = false;
    if (button) { button.textContent = label; button.disabled = false; }
    $('reanalyse').disabled = false;
  }
}

function hideResult() {
  current = null;
  $('result-card').classList.add('hidden');
}

const notation = () => $('notation').value;
const noteLabel = (n) => (n ? (notation() === 'fr' ? n.label_fr : n.label) : '—');
const fmtCents = (c) => (c === null || c === undefined ? '—' : `${c > 0 ? '+' : ''}${c.toFixed(0)}¢`);

function renderResult() {
  if (!current) return;
  const { take, data } = current;
  $('result-card').classList.remove('hidden');
  $('result-title').textContent = take.name;

  const warn = $('result-warnings');
  warn.replaceChildren(...data.warnings.map((w) => el('div', { class: 'msg warn', text: w })));

  const s = data.summary;
  const tile = (label, value, sub) => el('div', { class: 'tile' }, [
    el('div', { class: 'tile-label', text: label }), el('div', { class: 'tile-value', text: value }),
    sub ? el('div', { class: 'tile-sub', text: sub }) : null,
  ]);
  $('summary').replaceChildren(
    tile('Median pitch', noteLabel(s.median_note), s.median_f0 ? `${s.median_f0.toFixed(1)} Hz · ${fmtCents(s.median_note.cents)}` : 'no pitch'),
    tile('Range', s.lowest ? `${noteLabel(s.lowest)} – ${noteLabel(s.highest)}` : '—', `${s.segment_count} note(s) found`),
    tile('Voiced', `${Math.round(s.voiced_ratio * 100)}%`, `of ${data.duration.toFixed(1)} s`),
    tile('Level', `${fmtDb(s.peak_dbfs)} dBFS`, `peak · ${data.sample_rate} Hz`),
  );

  renderSegments();
  drawPitch();
  renderDetail();
}

function renderSegments() {
  const { data, selected } = current;
  const table = $('seg-table');
  const head = el('tr', {}, ['#', 'Time', 'Note', 'f0 (Hz)', 'Tuning', 'Vibrato', 'Harmonics', 'Strongest', 'Confidence'].map((h) => el('th', { text: h })));
  const rows = data.segments.map((seg, i) => el('tr', {
    class: i === selected ? 'selected clickable' : 'clickable',
    tabindex: '0',
    onclick: () => selectSegment(i),
    onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectSegment(i); } },
  }, [
    el('td', { text: String(i + 1) }),
    el('td', { text: `${seg.start.toFixed(2)}–${seg.end.toFixed(2)} s` }),
    el('td', { class: 'note', text: noteLabel(seg.note) }),
    el('td', { class: 'num', text: seg.f0.toFixed(2) }),
    el('td', { class: 'num', text: fmtCents(seg.note.cents) }),
    el('td', { text: seg.vibrato ? `${seg.vibrato.rate_hz} Hz, ±${seg.vibrato.extent_cents.toFixed(0)}¢` : '—' }),
    el('td', { class: 'num', text: String(seg.harmonics_present) }),
    el('td', { text: seg.strongest_harmonic ? `H${seg.strongest_harmonic}` : '—' }),
    el('td', { class: 'num', text: `${Math.round(seg.confidence * 100)}%` }),
  ]));
  if (!rows.length) rows.push(el('tr', {}, el('td', { colspan: '9', class: 'empty', text: 'No stable note found in this recording.' })));
  table.replaceChildren(el('thead', {}, head), el('tbody', {}, rows));
}

function selectSegment(i) {
  current.selected = i;
  renderSegments();
  drawPitch();
  renderDetail();
}

function renderDetail() {
  const seg = current.data.segments[current.selected];
  $('seg-detail').classList.toggle('hidden', !seg);
  if (!seg) return;
  $('seg-title').textContent = `note ${current.selected + 1}: ${noteLabel(seg.note)} (${seg.f0.toFixed(2)} Hz)`;
  drawSpectrum(seg);

  const head = el('tr', {}, ['n', 'Ideal n·f0 (Hz)', 'Measured (Hz)', 'Note', '¢ vs n·f0', 'Level', '', 'dBFS'].map((h) => el('th', { text: h })));
  const rows = seg.harmonics.map((h) => {
    const rel = h.level_rel_db;
    const bar = el('div', { class: 'lvl' }, el('div', { class: 'lvl-bar' }));
    bar.firstChild.style.width = `${Math.max(0, Math.min(100, ((rel ?? -60) + 60) / 60 * 100))}%`;
    return el('tr', { class: h.present ? '' : 'absent' }, [
      el('td', { class: 'num', text: `H${h.n}` }),
      el('td', { class: 'num', text: h.ideal_freq.toFixed(1) }),
      el('td', { class: 'num', text: h.present ? h.freq.toFixed(1) : '—' }),
      el('td', { class: 'note', text: h.present ? (notation() === 'fr' ? h.note_fr : h.note) : '—' }),
      el('td', { class: 'num', text: h.present ? fmtCents(h.cents_dev) : '—' }),
      el('td', { class: 'num', text: rel === null ? '—' : `${rel.toFixed(1)} dB` }),
      el('td', {}, bar),
      el('td', { class: 'num', text: fmtDb(h.level_dbfs) }),
    ]);
  });
  $('harm-table').replaceChildren(el('thead', {}, head), el('tbody', {}, rows));
}

// ---------------------------------------------------------------- plots
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
  const h = Number(canvas.getAttribute('height')) || 200;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function noteName(midi) {
  return `${NAMES[notation()][((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

function drawPitch() {
  const canvas = $('pitch-canvas');
  const { ctx, w, h } = setupCanvas(canvas);
  const { data, selected } = current;
  const a4 = data.a4;
  const pad = { l: 52, r: 10, t: 10, b: 24 };
  const voiced = data.track.f0.filter((f) => f);
  const lo = voiced.length ? Math.min(...voiced) / 1.09 : 100;
  const hi = voiced.length ? Math.max(...voiced) * 1.09 : 400;
  const toMidi = (f) => 69 + 12 * Math.log2(f / a4);
  const mLo = toMidi(lo), mHi = Math.max(toMidi(hi), mLo + 4);
  const X = (t) => pad.l + (t / Math.max(data.duration, 1e-3)) * (w - pad.l - pad.r);
  const Y = (m) => pad.t + (1 - (m - mLo) / (mHi - mLo)) * (h - pad.t - pad.b);
  const fg = cssVar('--fg'), muted = cssVar('--muted'), grid = cssVar('--grid');

  const seg = data.segments[selected];
  if (seg) {
    ctx.fillStyle = cssVar('--highlight');
    ctx.fillRect(X(seg.start), pad.t, X(seg.end) - X(seg.start), h - pad.t - pad.b);
  }

  const span = mHi - mLo;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  for (let m = Math.ceil(mLo); m <= Math.floor(mHi); m++) {
    const pc = ((m % 12) + 12) % 12;
    const natural = [0, 2, 4, 5, 7, 9, 11].includes(pc);
    const labelled = span <= 14 || (span <= 30 ? natural : pc === 0);
    ctx.strokeStyle = grid;
    ctx.globalAlpha = pc === 0 ? 1 : natural ? 0.7 : 0.35;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(m)); ctx.lineTo(w - pad.r, Y(m)); ctx.stroke();
    ctx.globalAlpha = 1;
    if (labelled) { ctx.fillStyle = muted; ctx.fillText(noteName(m), 6, Y(m)); }
  }

  ctx.fillStyle = muted;
  ctx.textBaseline = 'top';
  const step = data.duration > 20 ? 5 : data.duration > 6 ? 1 : 0.5;
  for (let t = 0; t <= data.duration + 1e-9; t += step) ctx.fillText(`${+t.toFixed(1)}s`, X(t) - 8, h - pad.b + 6);

  ctx.strokeStyle = cssVar('--accent');
  ctx.lineWidth = 2;
  ctx.beginPath();
  let pen = false;
  data.track.t.forEach((t, i) => {
    const f = data.track.f0[i];
    if (!f) { pen = false; return; }
    const x = X(t), y = Y(toMidi(f));
    if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    pen = true;
  });
  ctx.stroke();
  ctx.lineWidth = 1;
  if (!voiced.length) { ctx.fillStyle = fg; ctx.textBaseline = 'middle'; ctx.fillText('No pitch detected', w / 2 - 50, h / 2); }
}

function drawSpectrum(seg) {
  const canvas = $('spec-canvas');
  const { ctx, w, h } = setupCanvas(canvas);
  const { freq, db } = seg.spectrum;
  if (!freq.length) return;
  const pad = { l: 44, r: 10, t: 22, b: 24 };
  const fLo = freq[0], fHi = freq[freq.length - 1];
  const top = Math.ceil((Math.max(...db) + 5) / 10) * 10, bottom = top - 90;
  const X = (f) => pad.l + (Math.log(f / fLo) / Math.log(fHi / fLo)) * (w - pad.l - pad.r);
  const Y = (d) => pad.t + ((top - Math.max(bottom, Math.min(top, d))) / (top - bottom)) * (h - pad.t - pad.b);
  const muted = cssVar('--muted'), grid = cssVar('--grid');

  ctx.font = '11px system-ui, sans-serif';
  ctx.strokeStyle = grid; ctx.fillStyle = muted;
  ctx.textBaseline = 'middle';
  for (let d = bottom; d <= top; d += 10) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(d)); ctx.lineTo(w - pad.r, Y(d)); ctx.stroke();
    ctx.fillText(`${d}`, 4, Y(d));
  }
  ctx.textBaseline = 'top';
  for (const f of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
    if (f < fLo || f > fHi) continue;
    ctx.beginPath(); ctx.moveTo(X(f), pad.t); ctx.lineTo(X(f), h - pad.b); ctx.stroke();
    ctx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, X(f) - 8, h - pad.b + 6);
  }

  ctx.beginPath();
  freq.forEach((f, i) => (i ? ctx.lineTo(X(f), Y(db[i])) : ctx.moveTo(X(f), Y(db[i]))));
  ctx.lineTo(X(fHi), Y(bottom)); ctx.lineTo(X(fLo), Y(bottom)); ctx.closePath();
  ctx.fillStyle = cssVar('--fill'); ctx.fill();
  ctx.strokeStyle = cssVar('--accent'); ctx.stroke();

  ctx.textBaseline = 'bottom';
  ctx.textAlign = 'center';
  for (const hm of seg.harmonics) {
    if (!hm.present || hm.freq < fLo || hm.freq > fHi) continue;
    const x = X(hm.freq);
    ctx.strokeStyle = cssVar('--accent2'); ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    if (hm.n <= 12 || hm.n % 2 === 0) { ctx.fillStyle = cssVar('--accent2'); ctx.fillText(`${hm.n}`, x, pad.t - 4); }
  }
  ctx.textAlign = 'start';
  ctx.fillStyle = muted; ctx.textBaseline = 'top';
  ctx.fillText('dB', 4, 2);
}

// ---------------------------------------------------------------- controls
function updateControls() {
  const hasMic = !!mic.worklet;
  $('mic-enable').textContent = hasMic ? 'Microphone on' : 'Enable microphone';
  $('mic-enable').disabled = hasMic || rec.active;
  $('mic-select').disabled = rec.active || $('mic-select').options.length === 0 || !$('mic-select').options[0].value;
  $('mic-refresh').disabled = rec.active;
  $('rec-start').disabled = !hasMic || rec.active;
  $('rec-stop').disabled = !rec.active;
  $('rec-start').classList.toggle('live', rec.active);
  $('file-input').disabled = rec.active;
}

function init() {
  $('max-sec').textContent = String(MAX_REC_SEC);
  $('a4').value = prefs.get('a4', '440');
  $('gate').value = prefs.get('gate', '-50');
  $('notation').value = prefs.get('notation', 'en') === 'fr' ? 'fr' : 'en';

  $('mic-enable').addEventListener('click', () => openMic(prefs.get('device', '') || null));
  $('mic-select').addEventListener('change', (e) => { if (e.target.value) openMic(e.target.value); });
  $('mic-refresh').addEventListener('click', refreshDevices);
  $('rec-start').addEventListener('click', startRecording);
  $('rec-stop').addEventListener('click', stopRecording);
  $('file-input').addEventListener('change', async (e) => { await importFile(e.target.files[0]); e.target.value = ''; });
  $('delete-all').addEventListener('click', deleteAll);
  $('reanalyse').addEventListener('click', () => current && analyseTake(current.take, null));
  $('notation').addEventListener('change', (e) => { prefs.set('notation', e.target.value); renderResult(); });
  for (const id of ['a4', 'gate']) {
    $(id).addEventListener('change', (e) => {
      prefs.set(id, e.target.value);
      if (current) $('reanalyse').classList.add('attention');
    });
  }
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', refreshDevices);
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && rec.active) stopRecording();
  });
  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      for (const take of takes) drawWave(items.get(take.id).wave, take.peaks || []);
      if (current) { drawPitch(); renderDetail(); }
    }, 150);
  });
  window.addEventListener('beforeunload', (e) => { if (rec.active) { e.preventDefault(); e.returnValue = ''; } });

  if (!window.isSecureContext) {
    showMsg('This page is not served over HTTPS or localhost, so browsers will block the microphone.', 'error', true);
  }

  refreshDevices();
  updateControls();
  store.all().then((all) => {
    takes = all.sort((a, b) => b.createdAt - a.createdAt);
    renderList();
  }).catch((err) => showMsg(`Could not load saved recordings: ${err.message}`));
}

// Hook for automated end-to-end tests (adds a synthetic take without a microphone).
window.VoiceHarmonics = { addSamples: (samples, sampleRate, name) => saveTake(Float32Array.from(samples), sampleRate, name, 'test') };

document.addEventListener('DOMContentLoaded', init);
