# coding=utf-8
"""
Transcribe 路由 — 音频转写 API 接口

提供：
  流式实时转写（Server-Sent Events）:
    POST /asr/transcribe/stream     单文件流式转写，逐分片实时推送

  录音输入实时转写:
    POST /asr/transcribe/record     上传录音文件，soundfile 解析后流式转写 (SSE)
    WS   /asr/transcribe/ws        WebSocket 实时麦克风录音转写
                                    客户端推送 PCM/WAV 音频块，服务端返回转写 JSON

  管理:
    GET  /asr/health                健康检查

流式 SSE 事件格式:
    id: <seq>
    event: chunk
    data: {"segment":0,"text":"你好","start":0.0,"end":30.0}

    id: <seq>
    event: done
    data: {"duration":60.0,"chunks_total":2,"chunks_empty":0}

WebSocket 消息格式（服务端 → 客户端）:
    {"type":"chunk","segment":0,"text":"你好","start":0.0,"end":3.0}
    {"type":"done","duration":30.0,"chunks_total":3}
    {"type":"error","message":"..."}

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
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from core.config import args, settings
from core.logger import logger
from core.response import R
from qwen_asr_gguf.inference import exporters, itn
from services.asr_service import get_asr_service
from utils.audio import pcm16_to_float32, read_audio_bytes, save_audio_tmp
from utils.file import generate_unique_filename,check_file_size

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

            # 非空校验：text 为空（含 VAD 跳过）则跳过该分片输出
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


# ─── 录音输入实时转写 (soundfile) ─────────────────────────────────────────────
@router.post(
    "/record",
    summary="录音文件流式转写 (SSE)",
    description=(
        "上传麦克风录音文件（WAV / FLAC / OGG 等），"
        "服务端使用 **soundfile** 解析并校验音频格式与采样率，"
        "然后以 **Server-Sent Events** 格式实时推送转写结果。\n\n"
        "与 `/stream` 接口的区别: 本接口专为录音场景优化，"
        "会自动处理多声道→单声道转换与采样率信息校验，"
        "并在响应头中返回 `X-Audio-Duration` 与 `X-Audio-Samplerate`。"
    ),
    response_class=EventSourceResponse,
)
async def transcribe_record(
    file: UploadFile = File(..., description="录音文件 (WAV/FLAC/OGG 等)"),
    context: Optional[str] = Form(None, description="上下文提示词"),
    language: Optional[str] = Form(None, description="语言 (Chinese/English 等)"),
    temperature: float = Form(0.0, description="解码温度"),
    enable_srt: bool = Form(False, description="是否附带 SRT 字幕"),
    enable_aligner: bool = Form(False, description="是否启用词级对齐"),
) -> AsyncIterable[ServerSentEvent]:
    content = await file.read()
    check_file_size(content, file.filename or "")

    # ── soundfile 解析与校验 ──────────────────────────────────────────
    try:
        samples, samplerate = read_audio_bytes(content)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"音频格式解析失败，请上传 soundfile 支持的格式 (WAV/FLAC/OGG 等): {exc}",
        )

    audio_info_dur = round(len(samples) / samplerate, 3)
    logger.info(
        f"[录音转写] 文件={file.filename} | "
        f"采样率={samplerate}Hz | 时长={audio_info_dur:.2f}s | "
        f"声道=单声道(已合并)"
    )

    # 将解码后的音频重新写成标准 WAV 供 ASR 引擎使用
    tmp_path = save_audio_tmp(samples, samplerate)

    service = get_asr_service()
    HEARTBEAT_INTERVAL = 8
    _STREAM_END = object()
    chunk_count = 0
    event_id = 0
    empty_count = 0
    heartbeat_count = 0
    audio_duration = 0.0

    async def _safe_anext(aiter):
        try:
            return await aiter.__anext__()
        except StopAsyncIteration:
            return _STREAM_END

    stream = None
    try:
        logger.info(f"[录音转写] SSE 连接已建立: {file.filename}")
        stream = service.stream_transcribe(
            audio_path=tmp_path,
            context=context,
            language=language,
            temperature=temperature,
            enable_aligner=enable_aligner,
        )

        while True:
            next_task = asyncio.ensure_future(_safe_anext(stream))
            while not next_task.done():
                done, _ = await asyncio.wait({next_task}, timeout=HEARTBEAT_INTERVAL)
                if not done:
                    heartbeat_count += 1
                    yield ServerSentEvent(comment="keepalive")

            result = next_task.result()
            if result is _STREAM_END:
                break

            chunk = result
            chunk_count += 1
            if chunk.is_last:
                stats = getattr(chunk, "_stats", {})
                audio_duration = stats.get("audio_duration", audio_info_dur)

            if not chunk.text or not chunk.text.strip():
                empty_count += 1
                continue

            chunk_data: dict = {
                "segment": chunk.segment_idx,
                "text": chunk.text,
                "start": round(chunk.start_sec, 3),
                "end": round(chunk.end_sec, 3),
            }
            if enable_srt or enable_aligner:
                align_items = getattr(chunk, "_align_items", None)
                if enable_srt and align_items:
                    chunk_data["srt"] = exporters.alignment_to_srt(align_items)
                if enable_aligner and align_items:
                    chunk_data["alignment"] = [
                        {"text": it.text, "start": round(it.start_time, 3), "end": round(it.end_time, 3)}
                        for it in align_items
                    ]

            event_id += 1
            yield ServerSentEvent(
                raw_data=json.dumps(chunk_data, ensure_ascii=False),
                event="chunk",
                id=str(event_id),
            )

        event_id += 1
        yield ServerSentEvent(
            raw_data=json.dumps(
                {"duration": round(audio_duration or audio_info_dur, 2),
                 "samplerate": samplerate,
                 "chunks_total": chunk_count,
                 "chunks_empty": empty_count},
                ensure_ascii=False,
            ),
            event="done",
            id=str(event_id),
        )

    except asyncio.CancelledError:
        event_id += 1
        yield ServerSentEvent(
            raw_data=json.dumps({"message": "转写任务被取消"}, ensure_ascii=False),
            event="error",
            id=str(event_id),
        )
    except Exception as exc:
        logger.error(f"[录音转写] 异常: {exc}", exc_info=True)
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
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        logger.info(
            f"[录音转写] 连接关闭: {file.filename} | "
            f"分片={chunk_count}, 空={empty_count}, 心跳={heartbeat_count}"
        )


# ─── WebSocket 实时麦克风录音转写 ────────────────────────────────────────────

@router.websocket("/ws")
async def transcribe_realtime_ws(
    websocket: WebSocket,
    context: Optional[str] = Query(None, description="上下文提示词"),
    language: Optional[str] = Query(None, description="语言 (Chinese/English 等)"),
    temperature: float = Query(0.0, description="解码温度"),
    sample_rate: int = Query(_DEFAULT_SAMPLE_RATE, description="PCM 裸流采样率 (Hz)，WAV 格式时忽略此参数"),
    chunk_seconds: float = Query(5.0, description="积累多少秒音频后触发一次转写"),
    is_wav: bool = Query(False, description="True=客户端发送 WAV 格式块; False=客户端发送原始 PCM int16"),
):
    """
    WebSocket 实时麦克风录音转写接口。

    **音频输入协议**:
    - `is_wav=false` (默认): 客户端发送原始 PCM int16 单声道字节，采样率由 `sample_rate` 指定。
    - `is_wav=true`:  客户端发送完整 WAV 格式块，soundfile 自动解析采样率与声道数。

    **控制消息** (文本帧):
    - `"stop"` — 立即转写剩余缓冲区并关闭连接。

    **服务端响应** (JSON 文本帧):
    - `{"type":"chunk","segment":N,"text":"...","start":0.0,"end":5.0}` — 每个转写分片
    - `{"type":"done","duration":30.0,"chunks_total":3,"chunks_empty":0}` — 转写结束
    - `{"type":"error","message":"..."}` — 错误
    - `{"type":"ready"}` — 连接就绪确认
    """
    await websocket.accept()
    service = get_asr_service()

    # 音频缓冲区（float32 numpy 数组）
    buffer: np.ndarray = np.empty(0, dtype=np.float32)
    buf_samplerate: int = sample_rate  # 由第一个 WAV 块或参数决定
    chunk_samples = int(chunk_seconds * sample_rate)
    segment_idx = 0
    chunk_count = 0
    empty_count = 0
    audio_seconds_total = 0.0

    async def _transcribe_buffer(buf: np.ndarray, sr: int, seg_idx: int) -> Optional[dict]:
        """将缓冲区写成 WAV 临时文件，流式转写并聚合文本，返回结果 dict 或 None"""
        if buf.size == 0:
            return None
        tmp_path = save_audio_tmp(buf, sr)
        try:
            texts = []
            async for chunk in service.stream_transcribe(
                audio_path=tmp_path,
                context=context,
                language=language,
                temperature=temperature,
                enable_aligner=False,
            ):
                if chunk.text and chunk.text.strip():
                    texts.append(chunk.text.strip())
            text = "".join(texts)
            if not text:
                return None
            dur = len(buf) / sr
            return {
                "type": "chunk",
                "segment": seg_idx,
                "text": text,
                "text_itn": itn(text) if text else "",
                "start": round(audio_seconds_total - dur, 3),
                "end": round(audio_seconds_total, 3),
            }
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    try:
        await websocket.send_text(json.dumps({"type": "ready"}, ensure_ascii=False))
        logger.info(f"[WS录音] 连接就绪 | sample_rate={sample_rate} chunk_seconds={chunk_seconds} is_wav={is_wav}")

        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break

            # ── 文本控制帧 ──────────────────────────────────────────────
            if message["type"] == "websocket.receive" and message.get("text"):
                cmd = message["text"].strip().lower()
                if cmd == "stop":
                    logger.info("[WS录音] 收到 stop 指令，处理剩余缓冲区")
                    if buffer.size > 0:
                        audio_seconds_total += len(buffer) / buf_samplerate
                        res = await _transcribe_buffer(buffer, buf_samplerate, segment_idx)
                        if res:
                            chunk_count += 1
                            segment_idx += 1
                            await websocket.send_text(json.dumps(res, ensure_ascii=False))
                        else:
                            empty_count += 1
                        buffer = np.empty(0, dtype=np.float32)
                    await websocket.send_text(
                        json.dumps({
                            "type": "done",
                            "duration": round(audio_seconds_total, 2),
                            "chunks_total": chunk_count,
                            "chunks_empty": empty_count,
                        }, ensure_ascii=False)
                    )
                    break
                continue

            # ── 二进制音频帧 ────────────────────────────────────────────
            if message["type"] == "websocket.receive" and message.get("bytes"):
                raw = message["bytes"]
                if not raw:
                    continue

                if is_wav:
                    # soundfile 解析 WAV/FLAC/OGG 块
                    try:
                        new_samples, detected_sr = read_audio_bytes(raw)
                        buf_samplerate = detected_sr
                        chunk_samples = int(chunk_seconds * buf_samplerate)
                    except Exception as exc:
                        logger.warning(f"[WS录音] soundfile 解析失败: {exc}")
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": f"音频解析失败: {exc}"}, ensure_ascii=False)
                        )
                        continue
                else:
                    # 原始 PCM int16 → float32
                    new_samples = pcm16_to_float32(raw)

                buffer = np.concatenate([buffer, new_samples])

                # 积累够 chunk_samples 就触发转写
                while buffer.size >= chunk_samples:
                    seg_buf = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]
                    dur = chunk_samples / buf_samplerate
                    audio_seconds_total += dur

                    res = await _transcribe_buffer(seg_buf, buf_samplerate, segment_idx)  # noqa: E501
                    segment_idx += 1
                    if res:
                        chunk_count += 1
                        await websocket.send_text(json.dumps(res, ensure_ascii=False))
                    else:
                        empty_count += 1

    except WebSocketDisconnect:
        logger.info("[WS录音] 客户端断开连接")
    except Exception as exc:
        logger.error(f"[WS录音] 异常: {exc}", exc_info=True)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
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
        gpu_enabled=args.use_gpu,
    )
    return R.success(data=data)
