// Qwen3-ASR 语音转写服务 - Client Logic
(function () {
	"use strict";

	const AUTH_TOKEN = "Bearer qwen3-asr-token";
	const API_PREFIX = "/qwen3-asr/api/v1";

	// ── Elements ───────────────────────────────
	const $ = (id) => document.getElementById(id);
	const docsLink = $("docsLink");
	const healthStatus = $("healthStatus");
	const dropzone = $("dropzone");
	const fileInput = $("fileInput");
	const fileInfo = $("fileInfo");
	const fileName = $("fileName");
	const fileSize = $("fileSize");
	const removeFile = $("removeFile");
	const configToggle = $("configToggle");
	const configPanel = $("configPanel");
	const cfgTemp = $("cfgTemp");
	const cfgTempVal = $("cfgTempVal");
	const btnStart = $("btnStart");
	const resultArea = $("resultArea");
	const resultLog = $("resultLog");
	const liveIndicator = $("liveIndicator");
	const heartbeatBadge = $("heartbeatBadge");
	const summaryBar = $("summaryBar");
	const btnCopy = $("btnCopy");
	const btnClear = $("btnClear");
	const vllmHealthBox = $("vllmHealthBox");
	const vllmHealthText = $("vllmHealthText");
	const btnVllmHealth = $("btnVllmHealth");
	const btnVllmStart = $("btnVllmStart");

	let selectedFile = null;
	let isTranscribing = false;
	let isVllmTranscribing = false;
	let allTexts = [];

	// ── Docs link ──────────────────────────────
	const baseUrl = location.origin;
	docsLink.href = baseUrl + "/docs";
	docsLink.textContent = baseUrl + "/docs";

	// ── Health check ───────────────────────────
	async function checkHealth() {
		try {
			const res = await fetch(API_PREFIX + "/transcribe/health", {
				headers: { Authorization: AUTH_TOKEN },
			});
			const json = await res.json();
			const dot = healthStatus.querySelector(".status-dot");
			if (json.code !== 200 || !json.data) {
				dot.className = "status-dot fail";
				healthStatus.innerHTML = "";
				healthStatus.appendChild(dot);
				healthStatus.append(" 服务异常");
				return;
			}
			const d = json.data;
			if (d.status === "ok") {
				dot.className = "status-dot ok";
				healthStatus.innerHTML = "";
				healthStatus.appendChild(dot);
				healthStatus.append(" 服务正常");
				if (d.gpu_enabled) healthStatus.append(" · GPU");
			} else {
				dot.className = "status-dot fail";
				healthStatus.innerHTML = "";
				healthStatus.appendChild(dot);
				healthStatus.append(" 服务不可用");
			}
		} catch {
			const dot = healthStatus.querySelector(".status-dot");
			dot.className = "status-dot fail";
			healthStatus.innerHTML = "";
			healthStatus.appendChild(dot);
			healthStatus.append(" 连接失败");
		}
	}
	checkHealth();
	setInterval(checkHealth, 30000);

	// ── vLLM health check ───────────────────────
	function setVllmHealth(cls, text) {
		const dot = vllmHealthBox.querySelector(".status-dot");
		dot.className = "status-dot" + (cls ? " " + cls : "");
		vllmHealthText.textContent = text;
	}

	async function checkVllmHealth() {
		btnVllmHealth.disabled = true;
		btnVllmHealth.textContent = "检查中…";
		setVllmHealth("", "正在检查 vLLM 状态…");
		try {
			const res = await fetch(API_PREFIX + "/transcribe-vllm/health", {
				headers: { Authorization: AUTH_TOKEN },
			});
			if (!res.ok) {
				setVllmHealth("fail", `vLLM 服务不可用 (HTTP ${res.status})`);
				return;
			}
			const json = await res.json();
			if (json.code !== 200 || !json.data) {
				setVllmHealth("fail", json.msg || "vLLM 状态检查失败");
				return;
			}

			const d = json.data;
			const parts = [
				d.backend_enabled ? "后端已启用" : "后端未启用",
				d.initialized ? "模型已加载" : "模型未加载",
				"模型: " + d.model,
			];
			if (d.forced_aligner_enabled) parts.push("Aligner: " + d.forced_aligner);
			setVllmHealth(d.backend_enabled ? "ok" : "fail", parts.join(" · "));
		} catch (err) {
			setVllmHealth("fail", "vLLM 连接失败: " + err.message);
		} finally {
			btnVllmHealth.disabled = false;
			btnVllmHealth.textContent = "检查 vLLM 状态";
		}
	}

	btnVllmHealth.addEventListener("click", checkVllmHealth);
	checkVllmHealth();

	// ── File helpers ───────────────────────────
	function formatSize(bytes) {
		if (bytes < 1024) return bytes + " B";
		if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
		return (bytes / 1048576).toFixed(2) + " MB";
	}

	function setFile(file) {
		if (!file) return;
		selectedFile = file;
		fileName.textContent = file.name;
		fileSize.textContent = formatSize(file.size);
		fileInfo.classList.add("show");
		btnStart.disabled = false;
		btnVllmStart.disabled = false;
	}

	function clearFile() {
		selectedFile = null;
		fileInput.value = "";
		fileInfo.classList.remove("show");
		btnStart.disabled = true;
		btnVllmStart.disabled = true;
	}

	// ── Dropzone ───────────────────────────────
	dropzone.addEventListener("click", () => {
		if (!isTranscribing) fileInput.click();
	});
	fileInput.addEventListener("change", () => {
		if (fileInput.files[0]) setFile(fileInput.files[0]);
	});
	removeFile.addEventListener("click", (e) => {
		e.stopPropagation();
		clearFile();
	});

	dropzone.addEventListener("dragover", (e) => {
		e.preventDefault();
		dropzone.classList.add("drag-over");
	});
	dropzone.addEventListener("dragleave", () => {
		dropzone.classList.remove("drag-over");
	});
	dropzone.addEventListener("drop", (e) => {
		e.preventDefault();
		dropzone.classList.remove("drag-over");
		if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
	});

	// ── Config toggle ──────────────────────────
	configToggle.addEventListener("click", () => {
		configToggle.classList.toggle("open");
		configPanel.classList.toggle("open");
	});
	cfgTemp.addEventListener("input", () => {
		cfgTempVal.textContent = parseFloat(cfgTemp.value).toFixed(2);
	});

	// ── Log helpers ────────────────────────────
	function appendLog(html, cls) {
		const div = document.createElement("div");
		div.className = "log-entry " + (cls || "");
		div.innerHTML = html;
		resultLog.appendChild(div);
		resultLog.scrollTop = resultLog.scrollHeight;
	}

	function clearLog() {
		resultLog.innerHTML = "";
		allTexts = [];
		summaryBar.classList.remove("show");
		btnCopy.style.display = "none";
		btnClear.style.display = "none";
	}

	// ── SSE Parser ─────────────────────────────
	function parseSSEEvent(raw) {
		const lines = raw.split("\n");
		let event = "message",
			data = "",
			id = "";
		for (const line of lines) {
			if (line.startsWith(":")) continue; // comment / heartbeat
			if (line.startsWith("event:")) event = line.slice(6).trim();
			else if (line.startsWith("data:")) data = line.slice(5).trim();
			else if (line.startsWith("id:")) id = line.slice(3).trim();
		}
		return { event, data, id };
	}

	// ── SSE Event Dispatcher ──────────────────
	function processSSEEvent(evt, stats) {
		stats = stats || {};
		if (evt.event === "start") {
			try {
				const d = JSON.parse(evt.data);
				appendLog(
					"● " +
						escHtml(d.message) +
						(d.filename ? " — " + escHtml(d.filename) : ""),
					"event-start",
				);
			} catch {
				appendLog("● 转写开始", "event-start");
			}
		} else if (evt.event === "chunk") {
			try {
				const d = JSON.parse(evt.data);
				const ts = "[" + fmtTime(d.start) + " → " + fmtTime(d.end) + "]";
				appendLog(
					'<span class="timestamp">' +
						ts +
						'</span><span class="text">' +
						escHtml(d.text) +
						"</span>",
					"event-chunk",
				);
				allTexts.push(d.text);

				if (d.srt) {
					appendLog(
						'<span class="timestamp">SRT:</span><span class="text" style="color:var(--text-muted);white-space:pre-wrap;">' +
							escHtml(d.srt) +
							"</span>",
						"event-chunk",
					);
				}
			} catch {
				appendLog(escHtml(evt.data), "event-chunk");
			}
		} else if (evt.event === "done") {
			try {
				const d = JSON.parse(evt.data);
				const elapsedSeconds = stats.startedAt
					? (Date.now() - stats.startedAt) / 1000
					: null;
				const doneParts = ["✓ 转写完成"];
				if (elapsedSeconds != null) doneParts.push("耗时 " + fmtDuration(elapsedSeconds));
				if (d.duration != null) doneParts.push("音频 " + fmtDuration(d.duration));
				appendLog(doneParts.join(" · "), "event-done");

				summaryBar.innerHTML = "";
				const items = [];
				if (elapsedSeconds != null) items.push("耗时: " + fmtDuration(elapsedSeconds));
				if (d.duration != null) items.push("音频: " + fmtDuration(d.duration));
				if (d.chunks_total != null) items.push("分片: " + d.chunks_total);
				if (d.chunks_empty != null) items.push("空片: " + d.chunks_empty);
				items.push("心跳: " + (stats.heartbeats || 0));
				items.forEach((text) => {
					const span = document.createElement("span");
					span.textContent = text;
					summaryBar.appendChild(span);
				});
				summaryBar.classList.add("show");
			} catch {
				appendLog("✓ 转写完成", "event-done");
			}
		} else if (evt.event === "error") {
			try {
				const d = JSON.parse(evt.data);
				appendLog("✗ 错误: " + escHtml(d.message), "event-error");
			} catch {
				appendLog("✗ " + escHtml(evt.data), "event-error");
			}
		}
	}

	// ── Transcribe ─────────────────────────────
	btnStart.addEventListener("click", startTranscribe);

	async function startTranscribe() {
		if (!selectedFile || isTranscribing) return;
		isTranscribing = true;
		btnStart.disabled = true;
		btnVllmStart.disabled = true;
		btnStart.textContent = "转写中…";
		clearLog();
		resultArea.classList.add("show");
		liveIndicator.style.display = "flex";
		heartbeatBadge.textContent = "";
		const streamStats = {
			startedAt: Date.now(),
			heartbeats: 0,
		};

		const formData = new FormData();
		formData.append("file", selectedFile);
		const lang = $("cfgLanguage").value;
		if (lang) formData.append("language", lang);
		formData.append("temperature", cfgTemp.value);
		formData.append("enable_srt", $("cfgSrt").checked);
		formData.append("enable_aligner", $("cfgAligner").checked);
		formData.append("disable_vad", $("cfgDisableVad").checked);
		const ctx = $("cfgContext").value.trim();
		if (ctx) formData.append("context", ctx);

		try {
			const response = await fetch(API_PREFIX + "/transcribe/stream", {
				method: "POST",
				headers: { Authorization: AUTH_TOKEN },
				body: formData,
			});

			if (!response.ok) {
				appendLog("⚠ 请求失败: HTTP " + response.status, "event-error");
				finishTranscribe();
				return;
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";

			while (true) {
				const { done, value } = await reader.read();
				if (done) {
					buffer += decoder.decode();
					if (buffer.trim()) {
						const lastEvents = buffer.split("\n\n");
						for (const raw of lastEvents) {
							const trimmed = raw.trim();
							if (!trimmed || trimmed.startsWith(":")) continue;
							const evt = parseSSEEvent(trimmed);
							processSSEEvent(evt, streamStats);
						}
					}
					break;
				}
				buffer += decoder.decode(value, { stream: true });

				const parts = buffer.split("\n\n");
				buffer = parts.pop();

				for (const part of parts) {
					const trimmed = part.trim();
					if (!trimmed) continue;

					// Pure comment line (heartbeat)
					if (trimmed.startsWith(":")) {
						streamStats.heartbeats++;
						heartbeatBadge.textContent = "💓" + streamStats.heartbeats;
						continue;
					}

					const evt = parseSSEEvent(trimmed);
					processSSEEvent(evt, streamStats);
				}
			}
		} catch (err) {
			appendLog("✗ 网络错误: " + escHtml(err.message), "event-error");
		}

		finishTranscribe();
	}

	// ── vLLM file test ──────────────────────────
	btnVllmStart.addEventListener("click", startVllmTranscribe);

	async function startVllmTranscribe() {
		if (!selectedFile || isVllmTranscribing) return;
		isVllmTranscribing = true;
		btnVllmStart.disabled = true;
		btnStart.disabled = true;
		btnVllmStart.textContent = "vLLM 转写中…";
		clearLog();
		resultArea.classList.add("show");
		liveIndicator.style.display = "flex";
		heartbeatBadge.textContent = "vLLM";
		appendLog("● vLLM 转写开始 — " + escHtml(selectedFile.name), "event-start");

		const formData = new FormData();
		formData.append("file", selectedFile);
		const lang = $("vllmLanguage").value;
		if (lang) formData.append("language", lang);
		const ctx = $("vllmContext").value.trim();
		if (ctx) formData.append("context", ctx);
		formData.append("return_timestamps", $("vllmReturnTimestamps").checked);

		try {
			const response = await fetch(API_PREFIX + "/transcribe-vllm/file", {
				method: "POST",
				headers: { Authorization: AUTH_TOKEN },
				body: formData,
			});
			if (!response.ok) {
				appendLog("✗ vLLM 错误: HTTP " + response.status, "event-error");
				return;
			}
			const json = await response.json();
			if (json.code !== 200 || !json.data) {
				appendLog(
					"✗ vLLM 错误: " + escHtml(json.msg || "响应数据异常"),
					"event-error",
				);
				return;
			}

			const d = json.data;
			if (d.text) {
				appendLog('<span class="text">' + escHtml(d.text) + "</span>", "event-chunk");
				allTexts.push(d.text);
			}
			if (d.timestamps && d.timestamps.length) {
				d.timestamps.forEach((item) => {
					const ts = "[" + fmtTime(item.start) + " → " + fmtTime(item.end) + "]";
					appendLog(
						'<span class="timestamp">' +
							ts +
							'</span><span class="text">' +
							escHtml(item.text) +
							"</span>",
						"event-chunk",
					);
				});
			}
			appendLog("✓ vLLM 转写完成", "event-done");
			summaryBar.innerHTML = "";
			[
				"后端: vLLM",
				"语言: " + (d.language || "--"),
				"耗时: " + d.elapsed + "s",
			].forEach((text) => {
				const span = document.createElement("span");
				span.textContent = text;
				summaryBar.appendChild(span);
			});
			summaryBar.classList.add("show");
			setVllmHealth("ok", "后端已启用 · 模型已加载 · 模型: " + d.model);
		} catch (err) {
			appendLog("✗ vLLM 网络错误: " + escHtml(err.message), "event-error");
		} finally {
			isVllmTranscribing = false;
			liveIndicator.style.display = "none";
			heartbeatBadge.textContent = "";
			btnVllmStart.disabled = !selectedFile || isTranscribing;
			btnStart.disabled = !selectedFile || isTranscribing;
			btnVllmStart.textContent = "使用当前文件测试 vLLM";
			if (allTexts.length > 0) btnCopy.style.display = "inline-block";
			btnClear.style.display = "inline-block";
		}
	}

	function finishTranscribe() {
		isTranscribing = false;
		liveIndicator.style.display = "none";
		btnStart.disabled = !selectedFile;
		btnVllmStart.disabled = !selectedFile || isVllmTranscribing;
		btnStart.textContent = "开始转写";
		if (allTexts.length > 0) {
			btnCopy.style.display = "inline-block";
		}
		btnClear.style.display = "inline-block";
	}

	// ── Copy / Clear ───────────────────────────
	btnCopy.addEventListener("click", () => {
		const text = allTexts.join("\n");
		navigator.clipboard.writeText(text).then(() => {
			btnCopy.textContent = "已复制 ✓";
			setTimeout(() => {
				btnCopy.textContent = "复制全部文本";
			}, 1500);
		});
	});
	btnClear.addEventListener("click", () => {
		clearLog();
		resultArea.classList.remove("show");
	});

	// ── Utils ──────────────────────────────────
	function fmtTime(sec) {
		if (sec == null) return "--";
		const m = Math.floor(sec / 60);
		const s = (sec % 60).toFixed(1);
		return m > 0 ? m + ":" + s.padStart(4, "0") : s + "s";
	}
	function fmtDuration(sec) {
		if (sec == null || Number.isNaN(Number(sec))) return "--";
		return Number(sec).toFixed(2) + "s";
	}
	function escHtml(s) {
		const d = document.createElement("div");
		d.textContent = s;
		return d.innerHTML;
	}

	// ═══════════════════════════════════════════════
	// ── Recording (WebSocket real-time transcribe) ─
	// ═══════════════════════════════════════════════
	const WS_SAMPLE_RATE = 16000;

	const btnConnect = $("btnConnect");
	const btnRecord = $("btnRecord");
	const btnStopRecord = $("btnStopRecord");
	const recWsStatus = $("recWsStatus");
	const recTimer = $("recTimer");
	const recResultLog = $("recResultLog");
	const btnRecCopy = $("btnRecCopy");
	const btnRecClear = $("btnRecClear");
	const recChunkSeconds = $("recChunkSeconds");
	const recChunkSecondsVal = $("recChunkSecondsVal");

	let recState = "idle"; // idle | connecting | ready | recording | stopping | done
	let recWs = null;
	let recAudioCtx = null;
	let recStream = null;
	let recProcessor = null;
	let recSource = null;
	let recTimerInterval = null;
	let recStartTime = 0;
	let recTexts = [];

	// ── 音频能量检测 (VAD) 参数 ───────────
	const SILENCE_FLUSH_DELAY = 400; // 静音多久后触发 flush (ms)
	const FORCE_FLUSH_INTERVAL = 2000; // 最大强制 flush 间隔 (ms)，无论 VAD 结果
	const NOISE_CALIBRATION_FRAMES = 8; // 前 N 帧用于校准噪底
	const SPEECH_THRESHOLD_MULT = 3.0; // 语音判定 = 噪底 × 倍数
	const MIN_SPEECH_RMS = 0.005; // 最小绝对语音阈值（防止极安静环境误触发）
	let hasSpeechInBuffer = false; // 当前缓冲期间是否检测到语音
	let silenceStartTime = 0; // 静音开始时间
	let flushTimer = null; // flush 延迟计时器
	let lastFlushTime = 0; // 上次 flush 时间戳
	let noiseFloor = 0; // 自适应噪底 RMS
	let calibrationFrames = 0; // 已校准帧数
	let calibrationSum = 0; // 校准 RMS 累计

	recChunkSeconds.addEventListener("input", () => {
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
			int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
		}
		return int16;
	}

	// ── State management ──────────────────────────
	function setRecState(state) {
		recState = state;

		// Button states
		btnConnect.disabled = state !== "idle" && state !== "done";
		btnRecord.disabled = state !== "ready";
		btnStopRecord.disabled = state !== "recording";

		if (state === "recording") {
			btnRecord.classList.add("recording");
		} else {
			btnRecord.classList.remove("recording");
		}

		// WS status badge
		recWsStatus.className = "rec-ws-status";
		switch (state) {
			case "idle":
			case "done":
				recWsStatus.textContent = "\u672a\u8fde\u63a5";
				break;
			case "connecting":
				recWsStatus.textContent = "\u8fde\u63a5\u4e2d\u2026";
				recWsStatus.classList.add("connecting");
				break;
			case "ready":
				recWsStatus.textContent =
					"\u5df2\u8fde\u63a5\uff0c\u53ef\u5f00\u59cb\u5f55\u97f3";
				recWsStatus.classList.add("connected");
				break;
			case "recording":
				recWsStatus.textContent = "\u5f55\u97f3\u4e2d";
				recWsStatus.classList.add("connected");
				break;
			case "stopping":
				recWsStatus.textContent = "\u7b49\u5f85\u7ed3\u675f\u2026";
				recWsStatus.classList.add("connecting");
				break;
		}
	}

	// ── Timer ─────────────────────────────────────
	function startTimer() {
		recStartTime = Date.now();
		recTimer.classList.add("active");
		updateTimer();
		recTimerInterval = setInterval(updateTimer, 1000);
	}
	function stopTimer() {
		clearInterval(recTimerInterval);
		recTimerInterval = null;
		recTimer.classList.remove("active");
	}
	function updateTimer() {
		const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
		const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
		const ss = String(elapsed % 60).padStart(2, "0");
		recTimer.textContent = mm + ":" + ss;
	}

	// ── Log helpers (recording) ───────────────────
	function recAppendLog(html, cls) {
		const div = document.createElement("div");
		div.className = "log-entry " + (cls || "");
		div.innerHTML = html;
		recResultLog.appendChild(div);
		recResultLog.scrollTop = recResultLog.scrollHeight;
	}
	function recClearLog() {
		recResultLog.innerHTML = "";
		recTexts = [];
		btnRecCopy.style.display = "none";
		btnRecClear.style.display = "none";
	}

	// ── WebSocket connect ─────────────────────────
	function connectWS() {
		const language = $("recLanguage").value;
		const temperature = cfgTemp.value;
		const chunkSeconds = recChunkSeconds.value;

		const protocol = location.protocol === "https:" ? "wss:" : "ws:";
		let wsUrl = `${protocol}//${location.host}${API_PREFIX}/transcribe/ws?sample_rate=${WS_SAMPLE_RATE}&chunk_seconds=${chunkSeconds}&is_wav=false`;
		if (language) wsUrl += `&language=${encodeURIComponent(language)}`;
		if (temperature) wsUrl += `&temperature=${temperature}`;

		setRecState("connecting");
		recAppendLog(
			"\u25cb \u6b63\u5728\u8fde\u63a5 WebSocket\u2026",
			"event-start",
		);
		const ws = new WebSocket(wsUrl);

		ws.onopen = () => {
			// wait for ready event from server
		};

		ws.onmessage = (event) => {
			let msg;
			try {
				msg = JSON.parse(event.data);
			} catch {
				return;
			}

			switch (msg.event) {
				case "ready":
					setRecState("ready");
					recAppendLog(
						"\u25cf WebSocket \u5df2\u8fde\u63a5\uff0c\u8bf7\u70b9\u51fb\u201c\u5f00\u59cb\u5f55\u97f3\u201d",
						"event-start",
					);
					break;
				case "chunk":
					if (msg.data) {
						const d = msg.data;
						const ts =
							"[" + fmtTime(d.start) + " \u2192 " + fmtTime(d.end) + "]";
						recAppendLog(
							'<span class="timestamp">' +
								ts +
								'</span><span class="text">' +
								escHtml(d.text) +
								"</span>",
							"event-chunk",
						);
						recTexts.push(d.text);
					}
					break;
				case "done":
					recAppendLog("\u2713 \u8f6c\u5199\u5b8c\u6210", "event-done");
					cleanupRecording();
					setRecState("done");
					if (recTexts.length > 0) btnRecCopy.style.display = "inline-block";
					btnRecClear.style.display = "inline-block";
					break;
				case "error":
					recAppendLog(
						"\u2717 \u9519\u8bef: " +
							escHtml(
								(msg.data && msg.data.message) || JSON.stringify(msg.data),
							),
						"event-error",
					);
					cleanupRecording();
					setRecState("done");
					btnRecClear.style.display = "inline-block";
					break;
			}
		};

		ws.onerror = () => {
			recWsStatus.className = "rec-ws-status error";
			recWsStatus.textContent = "\u8fde\u63a5\u9519\u8bef";
			recAppendLog("\u2717 WebSocket \u8fde\u63a5\u9519\u8bef", "event-error");
			cleanupRecording();
			setRecState("done");
			btnRecClear.style.display = "inline-block";
		};

		ws.onclose = () => {
			if (recState === "recording" || recState === "stopping") {
				cleanupRecording();
				setRecState("done");
				if (recTexts.length > 0) btnRecCopy.style.display = "inline-block";
				btnRecClear.style.display = "inline-block";
			} else if (recState === "ready") {
				setRecState("done");
				btnRecClear.style.display = "inline-block";
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
					noiseSuppression: true,
				},
			});
		} catch (err) {
			recAppendLog(
				"\u2717 \u9ea6\u514b\u98ce\u6743\u9650\u88ab\u62d2\u7edd\u6216\u4e0d\u53ef\u7528: " +
					escHtml(err.message),
				"event-error",
			);
			if (recWs) {
				recWs.close();
				recWs = null;
			}
			setRecState("idle");
			return;
		}

		recAudioCtx = new (window.AudioContext || window.webkitAudioContext)({
			sampleRate: WS_SAMPLE_RATE,
		});
		recSource = recAudioCtx.createMediaStreamSource(recStream);

		const actualRate = recAudioCtx.sampleRate;
		recProcessor = recAudioCtx.createScriptProcessor(4096, 1, 1);

		recProcessor.onaudioprocess = (e) => {
			if (recState !== "recording") return;
			const rawFloat32 = e.inputBuffer.getChannelData(0);

			// 计算 RMS 能量
			let sum = 0;
			for (let i = 0; i < rawFloat32.length; i++) {
				sum += rawFloat32[i] * rawFloat32[i];
			}
			const rms = Math.sqrt(sum / rawFloat32.length);

			// 自适应噪底校准（前几帧）
			if (calibrationFrames < NOISE_CALIBRATION_FRAMES) {
				calibrationSum += rms;
				calibrationFrames++;
				if (calibrationFrames === NOISE_CALIBRATION_FRAMES) {
					noiseFloor = calibrationSum / NOISE_CALIBRATION_FRAMES;
					console.log("[VAD] 噪底校准完成:", noiseFloor.toFixed(4));
				}
			}

			// 动态语音阈值 = max(噪底×倍数, 最小绝对值)
			const speechThreshold = Math.max(
				noiseFloor * SPEECH_THRESHOLD_MULT,
				MIN_SPEECH_RMS,
			);
			const isSpeech = rms > speechThreshold;
			const now = Date.now();

			// 语音活动检测
			if (isSpeech) {
				hasSpeechInBuffer = true;
				silenceStartTime = 0;
				if (flushTimer) {
					clearTimeout(flushTimer);
					flushTimer = null;
				}
			} else if (hasSpeechInBuffer && !flushTimer) {
				// 有语音后进入静音，开始计时
				if (!silenceStartTime) {
					silenceStartTime = now;
				}
				if (now - silenceStartTime >= SILENCE_FLUSH_DELAY) {
					// 静音时间足够，立即发送 flush
					if (recWs && recWs.readyState === WebSocket.OPEN) {
						recWs.send("flush");
					}
					hasSpeechInBuffer = false;
					silenceStartTime = 0;
					lastFlushTime = now;
				}
			}

			// 强制 flush：无论 VAD 结果，超过最大间隔就强制触发
			if (
				hasSpeechInBuffer &&
				lastFlushTime &&
				now - lastFlushTime >= FORCE_FLUSH_INTERVAL
			) {
				if (recWs && recWs.readyState === WebSocket.OPEN) {
					recWs.send("flush");
				}
				hasSpeechInBuffer = false;
				silenceStartTime = 0;
				lastFlushTime = now;
			}

			// 重采样并发送
			let float32 = rawFloat32;
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

		setRecState("recording");
		lastFlushTime = Date.now(); // 录音开始时初始化
		startTimer();
	}

	// ── Cleanup ───────────────────────────────────
	function cleanupRecording() {
		stopTimer();
		if (flushTimer) {
			clearTimeout(flushTimer);
			flushTimer = null;
		}
		hasSpeechInBuffer = false;
		silenceStartTime = 0;
		lastFlushTime = 0;
		noiseFloor = 0;
		calibrationFrames = 0;
		calibrationSum = 0;
		if (recProcessor) {
			recProcessor.disconnect();
			recProcessor = null;
		}
		if (recSource) {
			recSource.disconnect();
			recSource = null;
		}
		if (recAudioCtx) {
			recAudioCtx.close();
			recAudioCtx = null;
		}
		if (recStream) {
			recStream.getTracks().forEach((t) => t.stop());
			recStream = null;
		}
	}

	// ── Button handlers ───────────────────────────
	btnConnect.addEventListener("click", () => {
		if (recState !== "idle" && recState !== "done") return;
		// Check microphone support before connecting
		if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
			const isInsecure =
				location.protocol === "http:" &&
				location.hostname !== "localhost" &&
				location.hostname !== "127.0.0.1";
			if (isInsecure) {
				alert(
					"\u9ea6\u514b\u98ce\u5f55\u97f3\u9700\u8981 HTTPS \u5b89\u5168\u8fde\u63a5\u3002\n\u5f53\u524d\u662f HTTP \u8bbf\u95ee\uff0c\u8bf7\u4f7f\u7528 https:// \u6216 localhost \u8bbf\u95ee\u672c\u9875\u9762\u3002",
				);
			} else {
				alert(
					"\u60a8\u7684\u6d4f\u89c8\u5668\u4e0d\u652f\u6301\u9ea6\u514b\u98ce\u5f55\u97f3\uff08navigator.mediaDevices \u4e0d\u53ef\u7528\uff09\u3002\n\u8bf7\u4f7f\u7528 Chrome/Firefox/Edge \u7b49\u73b0\u4ee3\u6d4f\u89c8\u5668\uff0c\u5e76\u786e\u4fdd\u901a\u8fc7 HTTPS \u6216 localhost \u8bbf\u95ee\u3002",
				);
			}
			return;
		}
		recClearLog();
		connectWS();
	});

	btnRecord.addEventListener("click", () => {
		if (recState !== "ready") return;
		recAppendLog(
			"\u25cf \u5f00\u59cb\u91c7\u96c6\u97f3\u9891\u2026",
			"event-start",
		);
		beginCapture();
	});

	btnStopRecord.addEventListener("click", () => {
		if (recState !== "recording" && recState !== "ready") return;
		setRecState("stopping");
		stopTimer();
		if (flushTimer) {
			clearTimeout(flushTimer);
			flushTimer = null;
		}
		hasSpeechInBuffer = false;
		silenceStartTime = 0;
		lastFlushTime = 0;
		noiseFloor = 0;
		calibrationFrames = 0;
		calibrationSum = 0;
		if (recProcessor) {
			recProcessor.disconnect();
			recProcessor = null;
		}
		if (recSource) {
			recSource.disconnect();
			recSource = null;
		}
		if (recAudioCtx) {
			recAudioCtx.close();
			recAudioCtx = null;
		}
		if (recStream) {
			recStream.getTracks().forEach((t) => t.stop());
			recStream = null;
		}
		if (recWs && recWs.readyState === WebSocket.OPEN) {
			recWs.send("stop");
		}
	});

	btnRecCopy.addEventListener("click", () => {
		const text = recTexts.join("\n");
		navigator.clipboard.writeText(text).then(() => {
			btnRecCopy.textContent = "\u5df2\u590d\u5236 \u2713";
			setTimeout(() => {
				btnRecCopy.textContent = "\u590d\u5236\u6587\u672c";
			}, 1500);
		});
	});

	btnRecClear.addEventListener("click", () => {
		recClearLog();
	});
})();
