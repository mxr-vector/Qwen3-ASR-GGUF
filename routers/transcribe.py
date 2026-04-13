# coding=utf-8
"""
Transcribe 路由 — 音频转写 API 接口

提供：
  离线转写（完整结果一次性返回）:
    POST /asr/transcribe            单文件离线转写
    POST /asr/transcribe/batch      批量文件离线转写

  流式实时转写（Server-Sent Events）:
    POST /asr/transcribe/stream     单文件流式转写，逐分片实时推送

  管理:
    GET  /asr/health                健康检查

流式接口 SSE 事件格式:
    data: {"type":"chunk","segment":0,"text":"你好","start":0.0,"end":30.0}
    data: {"type":"chunk","segment":1,"text":"世界","start":30.0,"end":60.0,"srt":"..."}
    data: {"type":"done","duration":60.0}
    data: [DONE]
    注: text 为空的分片不输出; srt/alignment 仅在对应参数启用且有对齐数据时出现。
"""

import asyncio
import json
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import args, settings
from core.logger import logger
from core.response import R
from qwen_asr_gguf.inference import exporters, itn
from services.asr_service import get_asr_service
from utils.file import generate_unique_filename

router = APIRouter(prefix="/transcribe", tags=["ASR 语音识别"])


# ─── 响应模型 ────────────────────────────────────────────────────────────────


class AlignmentItem(BaseModel):
    """单个词/字的对齐结果"""

    text: str
    start: float = Field(..., description="开始时间 (秒)")
    end: float = Field(..., description="结束时间 (秒)")


class TranscribeData(BaseModel):
    """离线转写完整结果"""

    text: str = Field(..., description="转写文本 (原始)")
    text_itn: str = Field("", description="转写文本 (ITN 数字归一化后)")
    srt: str = Field("", description="SRT 字幕内容")
    alignment: List[AlignmentItem] = Field(
        default_factory=list, description="逐词对齐时间戳"
    )
    duration: float = Field(0, description="音频时长 (秒)")


class HealthData(BaseModel):
    """健康检查数据"""

    status: str
    engine_ready: bool
    gpu_enabled: bool


# ─── 内部工具 ─────────────────────────────────────────────────────────────────


def _check_file_size(content: bytes, filename: str = "") -> None:
    """校验上传文件大小，超出限制则抛出 HTTP 413"""
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"文件过大: {size_mb:.1f} MB，"
                f"上限 {settings.MAX_FILE_SIZE_MB} MB"
                + (f" ({filename})" if filename else "")
            ),
        )


def _build_transcribe_data(
    result,
    audio_duration: float,
    enable_srt: bool = False,
) -> TranscribeData:
    """将 TranscribeResult 转换为 API 响应模型"""
    return TranscribeData(
        text=result.text,
        text_itn=itn(result.text) if result.text else "",
        srt=(
            exporters.alignment_to_srt(result.alignment.items)
            if (enable_srt and result.alignment)
            else ""
        ),
        alignment=[
            AlignmentItem(
                text=it.text,
                start=round(it.start_time, 3),
                end=round(it.end_time, 3),
            )
            for it in (result.alignment.items if result.alignment else [])
        ],
        duration=round(audio_duration, 2),
    )


# ─── 离线转写 ──────────────────────────────────────────────────────────────────


@router.post(
    "/offline",
    summary="单文件离线转写",
    description=(
        "上传单个音频文件，等待转写完成后一次性返回完整结果"
        "支持 wav、mp3、flac、m4a、ogg 等常见格式。"
        "**适合场景**: 短音频、对延迟不敏感的批量任务。"
        "长音频（> 1 分钟）建议使用 `/transcribe/stream` 流式接口。"
        "**说明**: 当前 Web 服务的分片长度、上下文记忆数等行为，"
        "来自服务端全局配置。音频超过 `dynamic_chunk_threshold` 时，"
        "VAD 动态分片会自动启用。若需调整 `speech_threshold`、"
        "`chunk_size` 或 `memory_num`，请修改应用配置并重启服务。"
    ),
)
async def transcribe(
    file: UploadFile = File(..., description="音频文件"),
    context: Optional[str] = Form(None, description="上下文提示词（如场景描述）"),
    language: Optional[str] = Form(None, description="语言 (Chinese/English 等)"),
    temperature: float = Form(0.0, description="解码温度，0 为贪婪解码"),
    enable_srt: bool = Form(False, description="是否在响应中附带 SRT 字幕"),
    enable_aligner: bool = Form(False, description="是否启用对齐模型进行词级对齐"),
):
    content = await file.read()
    _check_file_size(content, file.filename or "")

    service = get_asr_service()
    result = await service.transcribe_bytes(
        audio_bytes=content,
        filename=file.filename or "upload.wav",
        context=context,
        language=language,
        temperature=temperature,
        enable_aligner=enable_aligner,
    )

    audio_dur = (
        result.performance.get("audio_duration", 0.0) if result.performance else 0.0
    )
    data = _build_transcribe_data(result, audio_dur, enable_srt)
    return R.success(data=data)


@router.post(
    "/offline/batch",
    summary="批量文件离线转写",
    description=(
        "上传多个音频文件，依次转写并返回结果列表。"
        "文件过大将被跳过并记录警告日志。"
        "**说明**: 批量接口复用服务启动时初始化的全局 ASR 引擎配置，"
        "包括默认语言、默认上下文与 VAD 策略。"
    ),
)
async def transcribe_batch(
    files: List[UploadFile] = File(..., description="音频文件列表"),
    context: Optional[str] = Form(None, description="上下文提示词"),
    language: Optional[str] = Form(None, description="语言 (Chinese/English 等)"),
    temperature: float = Form(0.0, description="解码温度"),
    enable_srt: bool = Form(False, description="是否附带 SRT 字幕"),
    enable_aligner: bool = Form(False, description="是否启用对齐模型进行词级对齐"),
):
    service = get_asr_service()
    results: List[TranscribeData] = []

    for file in files:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            logger.warning(f"跳过过大文件: {file.filename} ({size_mb:.1f} MB)")
            results.append(
                TranscribeData(
                    text=f"[SKIPPED] 文件过大: {file.filename} ({size_mb:.1f} MB)"
                )
            )
            continue

        try:
            result = await service.transcribe_bytes(
                audio_bytes=content,
                filename=file.filename or generate_unique_filename(suffix=".wav"),
                context=context,
                language=language,
                temperature=temperature,
                enable_aligner=enable_aligner,
            )
            # 非空校验：text 为空则跳过该条结果
            if not result.text or not result.text.strip():
                logger.info(f"跳过空结果: {file.filename}")
                continue
            audio_dur = (
                result.performance.get("audio_duration", 0.0)
                if result.performance
                else 0.0
            )
            results.append(_build_transcribe_data(result, audio_dur, enable_srt))

        except Exception as exc:
            logger.error(f"转写失败: {file.filename} | {exc}", exc_info=True)
            results.append(TranscribeData(text=f"[ERROR] {file.filename}: {exc}"))

    return R.success(data=results)


# ─── 流式实时转写（SSE） ──────────────────────────────────────────────────────


@router.post(
    "/stream",
    summary="单文件流式实时转写 (SSE)",
    description=(
        "上传音频文件后，以 **Server-Sent Events (SSE)** 格式实时推送转写结果。"
        "每处理完一个音频分片（默认 30 秒），立即推送一条事件，无需等待整段音频处理完毕。"
        "**事件类型**:"
        "- `chunk`: 单个分片的转写文本（含分片时间轴、是否被 VAD 判定为静音"
        "- `done`: 转写结束，包含完整文本、SRT、对齐时间戳及耗时统计"
        "- `[DONE]`: 流结束标志（兼容 OpenAI 风格客户端"
        "**适合场景**: 长音频转写、需要实时展示进度的 Web/App 应用。"
        "**服务端配置**: VAD 动态分片由音频时长自动触发（超过阈值时自动启用），"
        "VAD 阈值、分片长度、记忆片段数、"
        "默认语言，均在服务启动时由全局配置决定。"
        "> 注意: 响应 Content-Type 为 `text/event-stream`，"
        "请勿通过 Swagger UI 直接测试（建议使用 curl 或前端 EventSource）。"
    ),
    response_class=StreamingResponse,
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
):
    content = await file.read()
    _check_file_size(content, file.filename or "")

    service = get_asr_service()

    # 将二进制写入临时文件（service 内部管理生命周期）
    tmp_path = service._save_tmp(
        content, file.filename or generate_unique_filename(suffix=".wav")
    )  # 使用 UUID 避免文件名冲突

    async def _event_generator() -> AsyncGenerator[str, None]:
        """
        生成 SSE 事件流。
        - text 为空的分片不输出
        - 每 8 秒发送 SSE 注释心跳 `: keepalive`，防止连接被代理/客户端超时断开

        重要: 使用 asyncio.wait (非 wait_for) 实现心跳，
        确保超时时 **不取消** 底层迭代协程，避免工作线程仍在推理时
        锁被意外释放导致并发崩溃。

        心跳间隔设为 8 秒，远低于常见反向代理默认超时 (Nginx proxy_read_timeout=60s)，
        确保即使在多层代理环境下也能保持连接存活。
        """
        HEARTBEAT_INTERVAL = 8  # 心跳间隔 (秒)，需低于代理最小超时
        _STREAM_END = object()  # 流结束哨兵

        async def _safe_anext(aiter):
            """安全获取下一个元素，用哨兵替代 StopAsyncIteration"""
            try:
                return await aiter.__anext__()
            except StopAsyncIteration:
                return _STREAM_END

        stream_iter = None
        stream_aiter = None
        try:
            audio_duration = 0.0
            logger.info(f"[流式] SSE 连接已建立: {file.filename}")
            stream_iter = service.stream_transcribe(
                audio_path=tmp_path,
                context=context,
                language=language,
                temperature=temperature,
                enable_aligner=enable_aligner,
            )
            stream_aiter = stream_iter.__aiter__()

            chunk_count = 0
            empty_count = 0
            heartbeat_count = 0

            while True:
                # 为下一次迭代创建 Task（不会被心跳超时取消）
                next_task = asyncio.ensure_future(_safe_anext(stream_aiter))

                # 等待推理完成，期间每隔 HEARTBEAT_INTERVAL 发送心跳
                while not next_task.done():
                    done, _ = await asyncio.wait(
                        {next_task}, timeout=HEARTBEAT_INTERVAL
                    )
                    if not done:
                        heartbeat_count += 1
                        logger.debug(
                            f"[流式] 心跳 #{heartbeat_count} | "
                            f"已完成 {chunk_count} 个分片"
                        )
                        yield ": keepalive\n\n"

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

                chunk_event = {
                    "type": "chunk",
                    "segment": chunk.segment_idx,
                    "text": chunk.text,
                    "start": round(chunk.start_sec, 3),
                    "end": round(chunk.end_sec, 3),
                }

                # 按标记在 chunk 中输出 SRT / 对齐数据
                if enable_srt or enable_aligner:
                    align_items = getattr(chunk, "_align_items", None)
                    if enable_srt and align_items:
                        chunk_event["srt"] = exporters.alignment_to_srt(align_items)
                    if enable_aligner and align_items:
                        chunk_event["alignment"] = [
                            {
                                "text": it.text,
                                "start": round(it.start_time, 3),
                                "end": round(it.end_time, 3),
                            }
                            for it in align_items
                        ]

                yield f"data: {json.dumps(chunk_event, ensure_ascii=False)}\n\n"

            # ── done 事件：完成信号、音频时长与分片统计 ─────────────────
            done_event = {
                "type": "done",
                "duration": round(audio_duration, 2),
                "chunks_total": chunk_count,
                "chunks_empty": empty_count,
            }
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            logger.warning(
                f"[流式] 转写任务被取消 | "
                f"已完成 {chunk_count} 个分片, {heartbeat_count} 次心跳"
            )
            yield f"data: {json.dumps({'type': 'error', 'message': '转写任务被取消'}, ensure_ascii=False)}\n\n"

        except Exception as exc:
            logger.error(
                f"[流式] 转写异常: {exc} | "
                f"已完成 {chunk_count} 个分片, {heartbeat_count} 次心跳",
                exc_info=True,
            )
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

        finally:
            # 显式关闭底层异步生成器，确保 asyncio.Lock 被释放
            if stream_iter is not None:
                try:
                    await stream_iter.aclose()
                except Exception:
                    pass
            service._remove_tmp(tmp_path)
            logger.info(
                f"[流式] SSE 连接关闭: {file.filename} | "
                f"分片={chunk_count}, 空={empty_count}, 心跳={heartbeat_count}"
            )
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，保证实时推送
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",  # 显式声明分块传输
            "X-Content-Type-Options": "nosniff",  # 防止浏览器嗅探中断 SSE
        },
    )


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
