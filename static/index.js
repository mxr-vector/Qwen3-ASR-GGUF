// Qwen3-ASR 语音转写服务 - Client Logic
(function() {
  'use strict';

  const AUTH_TOKEN = 'Bearer qwen3-asr-token';
  const API_PREFIX = '/qwen3-asr/api/v1';

  // ── Elements ───────────────────────────────
  const $ = id => document.getElementById(id);
  const docsLink     = $('docsLink');
  const healthStatus = $('healthStatus');
  const dropzone     = $('dropzone');
  const fileInput    = $('fileInput');
  const fileInfo     = $('fileInfo');
  const fileName     = $('fileName');
  const fileSize     = $('fileSize');
  const removeFile   = $('removeFile');
  const configToggle = $('configToggle');
  const configPanel  = $('configPanel');
  const cfgTemp      = $('cfgTemp');
  const cfgTempVal   = $('cfgTempVal');
  const btnStart     = $('btnStart');
  const resultArea   = $('resultArea');
  const resultLog    = $('resultLog');
  const liveIndicator= $('liveIndicator');
  const heartbeatBadge=$('heartbeatBadge');
  const summaryBar   = $('summaryBar');
  const btnCopy      = $('btnCopy');
  const btnClear     = $('btnClear');

  let selectedFile = null;
  let isTranscribing = false;
  let allTexts = [];

  // ── Docs link ──────────────────────────────
  const baseUrl = location.origin;
  docsLink.href = baseUrl + '/docs';
  docsLink.textContent = baseUrl + '/docs';

  // ── Health check ───────────────────────────
  async function checkHealth() {
    try {
      const res = await fetch(API_PREFIX + '/transcribe/health', {
        headers: { 'Authorization': AUTH_TOKEN }
      });
      const json = await res.json();
      const dot = healthStatus.querySelector('.status-dot');
      if (json.code !== 200 || !json.data) {
        dot.className = 'status-dot fail';
        healthStatus.innerHTML = '';
        healthStatus.appendChild(dot);
        healthStatus.append(' 服务异常');
        return;
      }
      const d = json.data;
      if (d.status === 'ok') {
        dot.className = 'status-dot ok';
        healthStatus.innerHTML = '';
        healthStatus.appendChild(dot);
        healthStatus.append(' 服务正常');
        if (d.gpu_enabled) healthStatus.append(' · GPU');
      } else {
        dot.className = 'status-dot fail';
        healthStatus.innerHTML = '';
        healthStatus.appendChild(dot);
        healthStatus.append(' 服务不可用');
      }
    } catch {
      const dot = healthStatus.querySelector('.status-dot');
      dot.className = 'status-dot fail';
      healthStatus.innerHTML = '';
      healthStatus.appendChild(dot);
      healthStatus.append(' 连接失败');
    }
  }
  checkHealth();
  setInterval(checkHealth, 30000);

  // ── File helpers ───────────────────────────
  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(2) + ' MB';
  }

  function setFile(file) {
    if (!file) return;
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    fileInfo.classList.add('show');
    btnStart.disabled = false;
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.remove('show');
    btnStart.disabled = true;
  }

  // ── Dropzone ───────────────────────────────
  dropzone.addEventListener('click', () => { if (!isTranscribing) fileInput.click(); });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
  removeFile.addEventListener('click', (e) => { e.stopPropagation(); clearFile(); });

  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('drag-over'); });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });

  // ── Config toggle ──────────────────────────
  configToggle.addEventListener('click', () => {
    configToggle.classList.toggle('open');
    configPanel.classList.toggle('open');
  });
  cfgTemp.addEventListener('input', () => {
    cfgTempVal.textContent = parseFloat(cfgTemp.value).toFixed(2);
  });

  // ── Log helpers ────────────────────────────
  function appendLog(html, cls) {
    const div = document.createElement('div');
    div.className = 'log-entry ' + (cls || '');
    div.innerHTML = html;
    resultLog.appendChild(div);
    resultLog.scrollTop = resultLog.scrollHeight;
  }

  function clearLog() {
    resultLog.innerHTML = '';
    allTexts = [];
    summaryBar.classList.remove('show');
    btnCopy.style.display = 'none';
    btnClear.style.display = 'none';
  }

  // ── SSE Parser ─────────────────────────────
  function parseSSEEvent(raw) {
    const lines = raw.split('\n');
    let event = 'message', data = '', id = '';
    for (const line of lines) {
      if (line.startsWith(':')) continue;            // comment / heartbeat
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data = line.slice(5).trim();
      else if (line.startsWith('id:')) id = line.slice(3).trim();
    }
    return { event, data, id };
  }

  // ── SSE Event Dispatcher ──────────────────
  function processSSEEvent(evt) {
    if (evt.event === 'start') {
      try {
        const d = JSON.parse(evt.data);
        appendLog('● ' + escHtml(d.message) + (d.filename ? ' — ' + escHtml(d.filename) : ''), 'event-start');
      } catch { appendLog('● 转写开始', 'event-start'); }
    }
    else if (evt.event === 'chunk') {
      try {
        const d = JSON.parse(evt.data);
        const ts = '[' + fmtTime(d.start) + ' → ' + fmtTime(d.end) + ']';
        appendLog('<span class="timestamp">' + ts + '</span><span class="text">' + escHtml(d.text) + '</span>', 'event-chunk');
        allTexts.push(d.text);

        if (d.srt) {
          appendLog('<span class="timestamp">SRT:</span><span class="text" style="color:var(--text-muted);white-space:pre-wrap;">' + escHtml(d.srt) + '</span>', 'event-chunk');
        }
      } catch { appendLog(escHtml(evt.data), 'event-chunk'); }
    }
    else if (evt.event === 'done') {
      try {
        const d = JSON.parse(evt.data);
        appendLog('✓ 转写完成', 'event-done');
        summaryBar.innerHTML = '';
        const items = [
          '时长: ' + d.duration + 's',
          '分片: ' + d.chunks_total,
          '空片: ' + d.chunks_empty,
          '心跳: ' + heartbeats
        ];
        items.forEach(text => {
          const span = document.createElement('span');
          span.textContent = text;
          summaryBar.appendChild(span);
        });
        summaryBar.classList.add('show');
      } catch { appendLog('✓ 转写完成', 'event-done'); }
    }
    else if (evt.event === 'error') {
      try {
        const d = JSON.parse(evt.data);
        appendLog('✗ 错误: ' + escHtml(d.message), 'event-error');
      } catch { appendLog('✗ ' + escHtml(evt.data), 'event-error'); }
    }
  }

  // ── Transcribe ─────────────────────────────
  btnStart.addEventListener('click', startTranscribe);

  async function startTranscribe() {
    if (!selectedFile || isTranscribing) return;
    isTranscribing = true;
    btnStart.disabled = true;
    btnStart.textContent = '转写中…';
    clearLog();
    resultArea.classList.add('show');
    liveIndicator.style.display = 'flex';
    heartbeatBadge.textContent = '';
    let heartbeats = 0;

    const formData = new FormData();
    formData.append('file', selectedFile);
    const lang = $('cfgLanguage').value;
    if (lang) formData.append('language', lang);
    formData.append('temperature', cfgTemp.value);
    formData.append('enable_srt', $('cfgSrt').checked);
    formData.append('enable_aligner', $('cfgAligner').checked);
    formData.append('disable_vad', $('cfgDisableVad').checked);
    const ctx = $('cfgContext').value.trim();
    if (ctx) formData.append('context', ctx);

    try {
      const response = await fetch(API_PREFIX + '/transcribe/stream', {
        method: 'POST',
        headers: { 'Authorization': AUTH_TOKEN },
        body: formData,
      });

      if (!response.ok) {
        appendLog('⚠ 请求失败: HTTP ' + response.status, 'event-error');
        finishTranscribe();
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          if (buffer.trim()) {
            const lastEvents = buffer.split('\n\n');
            for (const raw of lastEvents) {
              const trimmed = raw.trim();
              if (!trimmed || trimmed.startsWith(':')) continue;
              const evt = parseSSEEvent(trimmed);
              processSSEEvent(evt);
            }
          }
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop();

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed) continue;

          // Pure comment line (heartbeat)
          if (trimmed.startsWith(':')) {
            heartbeats++;
            heartbeatBadge.textContent = '💓' + heartbeats;
            continue;
          }

          const evt = parseSSEEvent(trimmed);
          processSSEEvent(evt);
        }
      }
    } catch (err) {
      appendLog('✗ 网络错误: ' + escHtml(err.message), 'event-error');
    }

    finishTranscribe();
  }

  function finishTranscribe() {
    isTranscribing = false;
    liveIndicator.style.display = 'none';
    btnStart.disabled = !selectedFile;
    btnStart.textContent = '开始转写';
    if (allTexts.length > 0) {
      btnCopy.style.display = 'inline-block';
    }
    btnClear.style.display = 'inline-block';
  }

  // ── Copy / Clear ───────────────────────────
  btnCopy.addEventListener('click', () => {
    const text = allTexts.join('\n');
    navigator.clipboard.writeText(text).then(() => {
      btnCopy.textContent = '已复制 ✓';
      setTimeout(() => { btnCopy.textContent = '复制全部文本'; }, 1500);
    });
  });
  btnClear.addEventListener('click', () => {
    clearLog();
    resultArea.classList.remove('show');
  });

  // ── Utils ──────────────────────────────────
  function fmtTime(sec) {
    if (sec == null) return '--';
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    return m > 0 ? m + ':' + s.padStart(4, '0') : s + 's';
  }
  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ═══════════════════════════════════════════════
  // ── Recording (WebSocket real-time transcribe) ─
  // ═══════════════════════════════════════════════
  const WS_SAMPLE_RATE = 16000;

  const btnRecord      = $('btnRecord');
  const btnStopRecord  = $('btnStopRecord');
  const recWsStatus    = $('recWsStatus');
  const recTimer       = $('recTimer');
  const recResultLog   = $('recResultLog');
  const btnRecCopy     = $('btnRecCopy');
  const btnRecClear    = $('btnRecClear');
  const recChunkSeconds= $('recChunkSeconds');
  const recChunkSecondsVal = $('recChunkSecondsVal');

  let recState = 'idle'; // idle | connecting | ready | recording | stopping | done
  let recWs = null;
  let recAudioCtx = null;
  let recStream = null;
  let recProcessor = null;
  let recSource = null;
  let recTimerInterval = null;
  let recStartTime = 0;
  let recTexts = [];

  recChunkSeconds.addEventListener('input', () => {
    recChunkSecondsVal.textContent = recChunkSeconds.value;
  });

  // ── Downsample utility ────────────────────────
  function downsample(float32, fromRate, toRate) {
    if (fromRate === toRate) return float32;
    const ratio = fromRate / toRate;
    const newLen = Math.round(float32.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const idx = i * ratio;
      const lo = Math.floor(idx);
      const hi = Math.min(lo + 1, float32.length - 1);
      const frac = idx - lo;
      result[i] = float32[lo] * (1 - frac) + float32[hi] * frac;
    }
    return result;
  }

  function float32ToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
  }

  // ── State management ──────────────────────────
  function setRecState(state) {
    recState = state;
    btnRecord.disabled = (state !== 'idle' && state !== 'done');
    btnStopRecord.disabled = (state !== 'recording' && state !== 'ready');

    if (state === 'recording') {
      btnRecord.classList.add('recording');
    } else {
      btnRecord.classList.remove('recording');
    }

    // WS status badge
    recWsStatus.className = 'rec-ws-status';
    switch (state) {
      case 'idle':
      case 'done':
        recWsStatus.textContent = '\u672a\u8fde\u63a5';
        break;
      case 'connecting':
        recWsStatus.textContent = '\u8fde\u63a5\u4e2d\u2026';
        recWsStatus.classList.add('connecting');
        break;
      case 'ready':
      case 'recording':
        recWsStatus.textContent = '\u5df2\u8fde\u63a5';
        recWsStatus.classList.add('connected');
        break;
      case 'stopping':
        recWsStatus.textContent = '\u7b49\u5f85\u7ed3\u679f\u2026';
        recWsStatus.classList.add('connecting');
        break;
    }
  }

  // ── Timer ─────────────────────────────────────
  function startTimer() {
    recStartTime = Date.now();
    recTimer.classList.add('active');
    updateTimer();
    recTimerInterval = setInterval(updateTimer, 1000);
  }
  function stopTimer() {
    clearInterval(recTimerInterval);
    recTimerInterval = null;
    recTimer.classList.remove('active');
  }
  function updateTimer() {
    const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
    const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const ss = String(elapsed % 60).padStart(2, '0');
    recTimer.textContent = mm + ':' + ss;
  }

  // ── Log helpers (recording) ───────────────────
  function recAppendLog(html, cls) {
    const div = document.createElement('div');
    div.className = 'log-entry ' + (cls || '');
    div.innerHTML = html;
    recResultLog.appendChild(div);
    recResultLog.scrollTop = recResultLog.scrollHeight;
  }
  function recClearLog() {
    recResultLog.innerHTML = '';
    recTexts = [];
    btnRecCopy.style.display = 'none';
    btnRecClear.style.display = 'none';
  }

  // ── WebSocket connect ─────────────────────────
  function connectWS() {
    const language = $('recLanguage').value;
    const temperature = cfgTemp.value;
    const chunkSeconds = recChunkSeconds.value;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let wsUrl = `${protocol}//${location.host}${API_PREFIX}/transcribe/ws?sample_rate=${WS_SAMPLE_RATE}&chunk_seconds=${chunkSeconds}&is_wav=false`;
    if (language) wsUrl += `&language=${encodeURIComponent(language)}`;
    if (temperature) wsUrl += `&temperature=${temperature}`;

    setRecState('connecting');
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      // wait for ready event
    };

    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }

      switch (msg.event) {
        case 'ready':
          setRecState('ready');
          recAppendLog('\u25cf \u5df2\u8fde\u63a5\uff0c\u5f00\u59cb\u91c7\u96c6\u97f3\u9891\u2026', 'event-start');
          beginCapture();
          break;
        case 'chunk':
          if (msg.data) {
            const d = msg.data;
            const ts = '[' + fmtTime(d.start) + ' \u2192 ' + fmtTime(d.end) + ']';
            recAppendLog('<span class="timestamp">' + ts + '</span><span class="text">' + escHtml(d.text) + '</span>', 'event-chunk');
            recTexts.push(d.text);
          }
          break;
        case 'done':
          recAppendLog('\u2713 \u8f6c\u5199\u5b8c\u6210', 'event-done');
          cleanupRecording();
          setRecState('done');
          if (recTexts.length > 0) btnRecCopy.style.display = 'inline-block';
          btnRecClear.style.display = 'inline-block';
          break;
        case 'error':
          recAppendLog('\u2717 \u9519\u8bef: ' + escHtml(msg.data && msg.data.message || JSON.stringify(msg.data)), 'event-error');
          cleanupRecording();
          setRecState('done');
          btnRecClear.style.display = 'inline-block';
          break;
      }
    };

    ws.onerror = () => {
      recWsStatus.className = 'rec-ws-status error';
      recWsStatus.textContent = '\u8fde\u63a5\u9519\u8bef';
      recAppendLog('\u2717 WebSocket \u8fde\u63a5\u9519\u8bef', 'event-error');
      cleanupRecording();
      setRecState('done');
      btnRecClear.style.display = 'inline-block';
    };

    ws.onclose = () => {
      if (recState === 'recording' || recState === 'stopping') {
        cleanupRecording();
        setRecState('done');
        if (recTexts.length > 0) btnRecCopy.style.display = 'inline-block';
        btnRecClear.style.display = 'inline-block';
      }
    };

    recWs = ws;
  }

  // ── Audio capture ─────────────────────────────
  async function beginCapture() {
    try {
      recStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: WS_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
    } catch (err) {
      recAppendLog('\u2717 \u9ea6\u514b\u98ce\u6743\u9650\u88ab\u62d2\u7edd\u6216\u4e0d\u53ef\u7528: ' + escHtml(err.message), 'event-error');
      if (recWs) { recWs.close(); recWs = null; }
      setRecState('idle');
      return;
    }

    recAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: WS_SAMPLE_RATE });
    recSource = recAudioCtx.createMediaStreamSource(recStream);

    const actualRate = recAudioCtx.sampleRate;
    recProcessor = recAudioCtx.createScriptProcessor(4096, 1, 1);

    recProcessor.onaudioprocess = (e) => {
      if (recState !== 'recording') return;
      let float32 = e.inputBuffer.getChannelData(0);
      if (actualRate !== WS_SAMPLE_RATE) {
        float32 = downsample(float32, actualRate, WS_SAMPLE_RATE);
      }
      const int16 = float32ToInt16(float32);
      if (recWs && recWs.readyState === WebSocket.OPEN) {
        recWs.send(int16.buffer);
      }
    };

    recSource.connect(recProcessor);
    recProcessor.connect(recAudioCtx.destination);

    setRecState('recording');
    startTimer();
  }

  // ── Cleanup ───────────────────────────────────
  function cleanupRecording() {
    stopTimer();
    if (recProcessor) { recProcessor.disconnect(); recProcessor = null; }
    if (recSource) { recSource.disconnect(); recSource = null; }
    if (recAudioCtx) { recAudioCtx.close(); recAudioCtx = null; }
    if (recStream) {
      recStream.getTracks().forEach(t => t.stop());
      recStream = null;
    }
  }

  // ── Button handlers ───────────────────────────
  btnRecord.addEventListener('click', () => {
    if (recState !== 'idle' && recState !== 'done') return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('\u60a8\u7684\u6d4f\u89c8\u5668\u4e0d\u652f\u6301\u9ea6\u514b\u98ce\u5f55\u97f3');
      return;
    }
    recClearLog();
    connectWS();
  });

  btnStopRecord.addEventListener('click', () => {
    if (recState !== 'recording' && recState !== 'ready') return;
    setRecState('stopping');
    stopTimer();
    if (recProcessor) { recProcessor.disconnect(); recProcessor = null; }
    if (recSource) { recSource.disconnect(); recSource = null; }
    if (recAudioCtx) { recAudioCtx.close(); recAudioCtx = null; }
    if (recStream) {
      recStream.getTracks().forEach(t => t.stop());
      recStream = null;
    }
    if (recWs && recWs.readyState === WebSocket.OPEN) {
      recWs.send('stop');
    }
  });

  btnRecCopy.addEventListener('click', () => {
    const text = recTexts.join('\n');
    navigator.clipboard.writeText(text).then(() => {
      btnRecCopy.textContent = '\u5df2\u590d\u5236 \u2713';
      setTimeout(() => { btnRecCopy.textContent = '\u590d\u5236\u6587\u672c'; }, 1500);
    });
  });

  btnRecClear.addEventListener('click', () => {
    recClearLog();
  });

})();
