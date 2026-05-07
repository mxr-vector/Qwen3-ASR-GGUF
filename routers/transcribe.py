# coding=utf-8
"""
Transcribe 路由 — 音频转写 API 接口

提供：
  流式实时转写（Server-Sent Events）:
    POST /asr/transcribe/stream     单文件流式转写，逐分片实时推送

  WebSocket 实时麦克风录音转写:
    WS   /asr/transcribe/ws        WebSocket 实时麦克风录音转写
                                    客户端推送 PCM/WAV 音频块，服务端返回转写 JSON

  管理:
    GET  /asr/health                健康检查

流式 SSE 事件格式:
    id: <seq>
    event: start
    data: {"message":"转写开始","filename":"test.mp3"}

    id: <seq>
    event: chunk
    data: {"segment":0,"text":"你好","start":0.0,"end":30.0}

    id: <seq>
    event: done
    data: {"duration":60.0,"chunks_total":2,"chunks_empty":1}

WebSocket 消息格式（服务端 → 客户端）:
    {"id":"1","event":"ready","data":{}}
    {"id":"2","event":"chunk","data":{"segment":0,"text":"你好","start":0.0,"end":3.0,"text_itn":"..."}}
    {"id":"3","event":"done","data":{"duration":30.0,"chunks_total":3,"chunks_empty":0}}
    {"id":"4","event":"error","data":{"message":"..."}}

WebSocket 消息格式（客户端 → 服务端）:
    binary: 原始 PCM int16 或 WAV 格式音频块
    text:   "stop"  — 通知服务端结束本次转写
"""

import asyncio
import json
import os
from collections.abc import AsyncIterable
from typing import Optional

import numpy as np
from fastapi import (
    APIRouter,
    File,
    Form,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from core.config import settings
from core.logger import logger
from core.response import R
from qwen_asr_gguf.inference import exporters, itn
from services.asr_service import get_asr_service
from utils.audio import pcm16_to_float32, read_audio_bytes, save_audio_tmp
from utils.file import generate_unique_filename, check_file_size

router = APIRouter(prefix="/transcribe", tags=["ASR 语音识别"])

# 录音转写默认采样率（PCM 裸流时使用）
_DEFAULT_SAMPLE_RATE = 16000


# ─── 响应模型 ────────────────────────────────────────────────────────────────


class HealthData(BaseModel):
    """健康检查数据"""

    status: str
    engine_ready: bool
    gpu_enabled: bool


# ─── 流式实时转写（SSE） ──────────────────────────────────────────────────────
@router.post(
    "/stream",
    summary="单文件流式实时转写 (SSE)",
    description=(
        "上传音频文件后，以 **Server-Sent Events (SSE)** 格式实时推送转写结果。"
        "每处理完一个音频分片（默认 30 秒），立即推送一条事件，无需等待整段音频处理完毕。"
        "**SSE 字段映射**:"
        "- `id`: 客户端传入的 audio_id，用于关联同一次转写的所有事件"
        "- `event`: 事件类型 — chunk (分片转写) / done (转写结束) / error (异常)"
        "- `data`: JSON 负载"
        "**适合场景**: 长音频转写、需要实时展示进度的 Web/App 应用。"
        "**服务端配置**: VAD 动态分片由音频时长自动触发（超过阈值时自动启用），"
        "VAD 阈值、分片长度、记忆片段数、"
        "默认语言，均在服务启动时由全局配置决定。"
        "> 注意: 响应 Content-Type 为 `text/event-stream`，"
        "请勿通过 Swagger UI 直接测试（建议使用 curl 或前端 EventSource）。"
    ),
    response_class=EventSourceResponse,
)
async def transcribe_stream(
    file: UploadFile = File(..., description="音频文件"),
    context: Optional[str] = Form(None, description="上下文提示词"),
    language: Optional[str] = Form(None, description="语言 (Chinese/English 等)"),
    temperature: float = Form(0.0, description="解码温度"),
    enable_srt: bool = Form(False, description="是否在每个 chunk 事件中附带 SRT 字幕"),
    enable_aligner: bool = Form(
        False,
        description="是否启用对齐模型进行词级对齐 (在 chunk 事件中附带对齐时间戳)",
    ),
    disable_vad: bool = Form(
        False, description="是否禁用 VAD 语音活动检测（音乐/歌曲场景建议启用）"
    ),
) -> AsyncIterable[ServerSentEvent]:
    content = await file.read()
    check_file_size(content, file.filename or "")

    service = get_asr_service()

    HEARTBEAT_INTERVAL = 8
    _STREAM_END = object()
    chunk_count = 0
    event_id = 0
    empty_count = 0
    heartbeat_count = 0
    audio_duration = 0.0

    async def _safe_anext(aiter):
        """安全获取下一个元素，用哨兵替代 StopAsyncIteration"""
        try:
            return await aiter.__anext__()
        except StopAsyncIteration:
            return _STREAM_END

    stream = None
    try:
        logger.info(f"[流式] SSE 连接已建立: {file.filename}")
        stream = service.stream_transcribe_bytes(
            audio_bytes=content,
            filename=file.filename or generate_unique_filename(suffix=".wav"),
            context=context,
            language=language,
            temperature=temperature,
            enable_aligner=enable_aligner,
            disable_vad=disable_vad,
        )

        # 立即发送 start 事件，让客户端确认服务已接受请求
        event_id += 1
        yield ServerSentEvent(
            raw_data=json.dumps(
                {"message": "转写开始", "filename": file.filename}, ensure_ascii=False
            ),
            event="start",
            id=str(event_id),
        )

        while True:
            next_task = asyncio.ensure_future(_safe_anext(stream))

            while not next_task.done():
                done, _ = await asyncio.wait({next_task}, timeout=HEARTBEAT_INTERVAL)
                if not done:
                    heartbeat_count += 1
                    logger.debug(
                        f"[流式] 心跳 #{heartbeat_count} | "
                        f"已完成 {chunk_count} 个分片"
                    )
                    yield ServerSentEvent(comment="keepalive")

            result = next_task.result()
            if result is _STREAM_END:
                break

            chunk = result
            chunk_count += 1

            if chunk.is_last:
                stats = getattr(chunk, "_stats", {})
                audio_duration = stats.get("audio_duration", 0.0)

            # 无语音内容片段直接跳过
            if not chunk.text or not chunk.text.strip():
                empty_count += 1
                continue

            chunk_data = {
                "segment": chunk.segment_idx,
                "text": chunk.text,
                "start": round(chunk.start_sec, 3),
                "end": round(chunk.end_sec, 3),
            }

            # 按标记在 chunk 中输出 SRT / 对齐数据
            if enable_srt or enable_aligner:
                align_items = getattr(chunk, "_align_items", None)
                if enable_srt and align_items:
                    chunk_data["srt"] = exporters.alignment_to_srt(align_items)
                if enable_aligner and align_items:
                    chunk_data["alignment"] = [
                        {
                            "text": it.text,
                            "start": round(it.start_time, 3),
                            "end": round(it.end_time, 3),
                        }
                        for it in align_items
                    ]

            event_id += 1
            yield ServerSentEvent(
                raw_data=json.dumps(chunk_data, ensure_ascii=False),
                event="chunk",
                id=str(event_id),
            )

        # ── done 事件：完成信号、音频时长与分片统计 ─────────────────
        event_id += 1
        yield ServerSentEvent(
            raw_data=json.dumps(
                {
                    "duration": round(audio_duration, 2),
                    "chunks_total": chunk_count,
                    "chunks_empty": empty_count,
                },
                ensure_ascii=False,
            ),
            event="done",
            id=str(event_id),
        )

    except asyncio.CancelledError:
        logger.warning(
            f"[流式] 转写任务被取消 | "
            f"已完成 {chunk_count} 个分片, {heartbeat_count} 次心跳"
        )
        event_id += 1
        yield ServerSentEvent(
            raw_data=json.dumps({"message": "转写任务被取消"}, ensure_ascii=False),
            event="error",
            id=str(event_id),
        )

    except Exception as exc:
        logger.error(
            f"[流式] 转写异常: {exc} | "
            f"已完成 {chunk_count} 个分片, {heartbeat_count} 次心跳",
            exc_info=True,
        )
        event_id += 1
        yield ServerSentEvent(
            raw_data=json.dumps({"message": str(exc)}, ensure_ascii=False),
            event="error",
            id=str(event_id),
        )

    finally:
        if stream is not None:
            try:
                await stream.aclose()
            except Exception:
                pass
        logger.info(
            f"[流式] SSE 连接关闭: {file.filename} | "
            f"分片={chunk_count}, 空={empty_count}, 心跳={heartbeat_count}"
        )


# ─── WebSocket 实时麦克风录音转写 ────────────────────────────────────────────


@router.websocket("/ws")
async def transcribe_realtime_ws(
    websocket: WebSocket,
    context: Optional[str] = Query(None, description="上下文提示词"),
    language: Optional[str] = Query(None, description="语言 (Chinese/English 等)"),
    temperature: float = Query(0.0, description="解码温度"),
    sample_rate: int = Query(
        _DEFAULT_SAMPLE_RATE, description="PCM 裸流采样率 (Hz)，WAV 格式时忽略此参数"
    ),
    chunk_seconds: float = Query(5.0, description="积累多少秒音频后触发一次转写"),
    is_wav: bool = Query(
        False, description="True=客户端发送 WAV 格式块; False=客户端发送原始 PCM int16"
    ),
):
    """
    WebSocket 实时麦克风录音转写接口（并行架构）。

    音频接收与 ASR 推理并行执行，录音期间无阻塞延迟。

    **音频输入协议**:
    - `is_wav=false` (默认): 客户端发送原始 PCM int16 单声道字节，采样率由 `sample_rate` 指定。
    - `is_wav=true`:  客户端发送完整 WAV 格式块，soundfile 自动解析采样率与声道数。

    **控制消息** (文本帧):
    - `"stop"` — 转写剩余缓冲区并关闭连接。
    - `"flush"` — 立即转写当前缓冲区（客户端 VAD 静音检测触发）。

    **服务端响应** (JSON 文本帧):
    - `{"id":"1","event":"ready","data":{}}` — 连接就绪确认
    - `{"id":"2","event":"chunk","data":{"segment":N,"text":"...","start":0.0,"end":5.0,"text_itn":"..."}}` — 转写分片
    - `{"id":"3","event":"done","data":{"duration":30.0,"chunks_total":3,"chunks_empty":0}}` — 转写结束
    - `{"id":"4","event":"error","data":{"message":"..."}}` — 错误
    """
    await websocket.accept()
    service = get_asr_service()

    # ── 共享状态 ──────────────────────────────────────────────────────
    buf_samplerate: int = sample_rate
    chunk_samples = int(chunk_seconds * sample_rate)
    segment_idx = 0
    chunk_count = 0
    empty_count = 0
    audio_seconds_total = 0.0
    event_id = 0

    # 并行通信
    transcribe_queue: asyncio.Queue = asyncio.Queue()
    ws_send_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    async def _ws_send(data: dict):
        """线程安全的 WebSocket 发送"""
        async with ws_send_lock:
            await websocket.send_text(json.dumps(data, ensure_ascii=False))

    async def _do_transcribe(
        buf: np.ndarray, sr: int, seg_idx: int, end_seconds: float
    ) -> Optional[dict]:
        """直接对 numpy 缓冲执行转写（零文件 I/O）"""
        if buf.size == 0:
            return None
        text = await service.transcribe_buffer(
            audio=buf,
            context=context,
            language=language,
            temperature=temperature,
            lock_timeout=8.0,
        )
        if not text:
            return None
        dur = len(buf) / sr
        return {
            "segment": seg_idx,
            "text": text,
            "text_itn": itn(text) if text else "",
            "start": round(end_seconds - dur, 3),
            "end": round(end_seconds, 3),
        }

    # ── 转写工作协程（消费者） ───────────────────────────────────────
    async def _transcription_worker():
        """从队列中取出音频片段并转写，结果立即推送客户端。

        核心优化：当推理速度慢于实时音频时，队列会堆积多个片段。
        Worker 每次取任务时会合并所有待处理片段为一个大段，一次性转写，
        避免“每段 3s 音频却要 5s 推理”导致的延迟雪球。
        """
        nonlocal chunk_count, empty_count, event_id

        while True:
            item = await transcribe_queue.get()
            if item is None:
                # 收到终止信号
                break

            # ── 合并队列中所有堆积的片段 ──────────────────────
            buf, sr, seg_idx, end_sec = item
            segments_merged = 1

            while not transcribe_queue.empty():
                try:
                    next_item = transcribe_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_item is None:
                    # 终止哨兵放回去
                    await transcribe_queue.put(None)
                    break
                next_buf, next_sr, next_seg_idx, next_end_sec = next_item
                buf = np.concatenate([buf, next_buf])
                seg_idx = next_seg_idx
                end_sec = next_end_sec
                segments_merged += 1

            if segments_merged > 1:
                logger.info(
                    f"[WS录音] 合并 {segments_merged} 个片段 "
                    f"({buf.size / sr:.1f}s) 一次性转写"
                )

            try:
                import time as _time

                _t0 = _time.time()
                res = await _do_transcribe(buf, sr, seg_idx, end_sec)
                _elapsed_ms = (_time.time() - _t0) * 1000
                if res:
                    chunk_count += 1
                    event_id += 1
                    await _ws_send({"id": str(event_id), "event": "chunk", "data": res})
                    logger.info(
                        "[WS录音] 片段转写完成 | seg=%d | %.2fs音频 | 耗时=%.0fms | text=%s",
                        seg_idx,
                        buf.size / sr,
                        _elapsed_ms,
                        res["text"][:30],
                    )
                else:
                    empty_count += 1
            except Exception as exc:
                logger.error(f"[WS录音] 转写worker异常: {exc}", exc_info=True)
                empty_count += 1

    # ── 音频接收协程（生产者） ───────────────────────────────────────
    async def _audio_receiver():
        """持续接收 WebSocket 消息，积累音频并投递到转写队列"""
        nonlocal buf_samplerate, chunk_samples, segment_idx, audio_seconds_total, event_id

        buffer: np.ndarray = np.empty(0, dtype=np.float32)
        min_flush_samples = int(0.3 * buf_samplerate)

        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                # 客户端断开，处理剩余
                if buffer.size >= min_flush_samples:
                    audio_seconds_total += len(buffer) / buf_samplerate
                    await transcribe_queue.put(
                        (
                            buffer.copy(),
                            buf_samplerate,
                            segment_idx,
                            audio_seconds_total,
                        )
                    )
                    segment_idx += 1
                break

            # ── 文本控制帧 ────────────────────────────────────────────
            if message["type"] == "websocket.receive" and message.get("text"):
                cmd = message["text"].strip().lower()

                if cmd == "stop":
                    logger.info("[WS录音] 收到 stop 指令")
                    if buffer.size >= min_flush_samples:
                        audio_seconds_total += len(buffer) / buf_samplerate
                        await transcribe_queue.put(
                            (
                                buffer.copy(),
                                buf_samplerate,
                                segment_idx,
                                audio_seconds_total,
                            )
                        )
                        segment_idx += 1
                        buffer = np.empty(0, dtype=np.float32)
                    stop_event.set()
                    break

                elif cmd == "flush":
                    if buffer.size >= min_flush_samples:
                        logger.debug(
                            f"[WS录音] flush → 队列 | {buffer.size / buf_samplerate:.2f}s"
                        )
                        audio_seconds_total += len(buffer) / buf_samplerate
                        await transcribe_queue.put(
                            (
                                buffer.copy(),
                                buf_samplerate,
                                segment_idx,
                                audio_seconds_total,
                            )
                        )
                        segment_idx += 1
                        buffer = np.empty(0, dtype=np.float32)
                continue

            # ── 二进制音频帧 ──────────────────────────────────────────
            if message["type"] == "websocket.receive" and message.get("bytes"):
                raw = message["bytes"]
                if not raw:
                    continue

                if is_wav:
                    try:
                        new_samples, detected_sr = read_audio_bytes(raw)
                        buf_samplerate = detected_sr
                        chunk_samples = int(chunk_seconds * buf_samplerate)
                        min_flush_samples = int(0.3 * buf_samplerate)
                    except Exception as exc:
                        logger.warning(f"[WS录音] soundfile 解析失败: {exc}")
                        event_id += 1
                        await _ws_send(
                            {
                                "id": str(event_id),
                                "event": "error",
                                "data": {"message": f"音频解析失败: {exc}"},
                            }
                        )
                        continue
                else:
                    new_samples = pcm16_to_float32(raw)

                buffer = np.concatenate([buffer, new_samples])

                # 按 chunk_samples 分片投递（兜底机制）
                while buffer.size >= chunk_samples:
                    seg_buf = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]
                    audio_seconds_total += chunk_samples / buf_samplerate
                    await transcribe_queue.put(
                        (seg_buf, buf_samplerate, segment_idx, audio_seconds_total)
                    )
                    segment_idx += 1

    # ── 主逻辑：并行启动接收与转写 ────────────────────────────────────
    try:
        event_id += 1
        await websocket.send_text(
            json.dumps(
                {"id": str(event_id), "event": "ready", "data": {}}, ensure_ascii=False
            )
        )
        logger.info(
            f"[WS录音] 连接就绪 | sample_rate={sample_rate} "
            f"chunk_seconds={chunk_seconds} is_wav={is_wav}"
        )

        # 启动转写 worker
        worker_task = asyncio.create_task(_transcription_worker())

        # 运行音频接收（阻塞直到 stop/disconnect）
        await _audio_receiver()

        # 接收结束，发送终止信号给 worker 并等待其处理完剩余队列
        await transcribe_queue.put(None)
        await worker_task

        # 发送 done 事件
        if stop_event.is_set():
            event_id += 1
            await _ws_send(
                {
                    "id": str(event_id),
                    "event": "done",
                    "data": {
                        "duration": round(audio_seconds_total, 2),
                        "chunks_total": chunk_count,
                        "chunks_empty": empty_count,
                    },
                }
            )

    except WebSocketDisconnect:
        logger.info("[WS录音] 客户端断开连接")
    except Exception as exc:
        logger.error(f"[WS录音] 异常: {exc}", exc_info=True)
        try:
            event_id += 1
            await _ws_send(
                {
                    "id": str(event_id),
                    "event": "error",
                    "data": {"message": str(exc)},
                }
            )
        except Exception:
            pass
    finally:
        logger.info(
            f"[WS录音] 会话结束 | 分片={chunk_count}, 空={empty_count}, "
            f"总时长={audio_seconds_total:.2f}s"
        )
        try:
            await websocket.close()
        except Exception:
            pass


# ─── 健康检查 ─────────────────────────────────────────────────────────────────
@router.get("/health", summary="健康检查")
async def health_check():
    """返回 ASR 引擎运行状态"""
    service = get_asr_service()
    data = HealthData(
        status="ok" if service.is_ready else "unavailable",
        engine_ready=service.is_ready,
        gpu_enabled=settings.USE_GPU,
    )
    return R.success(data=data)
