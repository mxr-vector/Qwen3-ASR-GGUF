# coding=utf-8
"""
ASR 服务层 — 线程安全的 QwenASREngine 封装

设计要点:
- 使用 asyncio.Lock 保证同一时刻只有一个推理任务运行（引擎不支持并发）
- 使用 asyncio.to_thread() 将阻塞推理放入线程池，不阻塞 FastAPI 事件循环
- 流式接口通过 asyncio.Queue + threading.Thread 桥接同步生成器与异步消费者
- 全局单例模式，由 lifespan 管理生命周期
"""
import asyncio
import os
import threading
import time
import uuid
from typing import AsyncGenerator, Optional

from core.logger import logger
from core.config import settings, args
from qwen_asr_gguf.inference import (
    QwenASREngine,
    ASREngineConfig,
    AlignerConfig,
    VADConfig,
    TranscribeResult,
    StreamChunkResult,
)


# ─── 哨兵对象，用于标识流式队列结束 ──────────────────────────
_STREAM_SENTINEL = object()


class ASRService:
    """线程安全的 ASR 服务封装（支持离线转写与流式转写）"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._engine: Optional[QwenASREngine] = None

    # ──────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────

    def initialize(self):
        """
        初始化 ASR 引擎（同步方法，在 lifespan 启动阶段调用）。
        引擎加载模型需要数秒，只在服务启动时执行一次。
        """
        logger.info("正在初始化 Qwen3-ASR 引擎...")
        t0 = time.time()

        config = self._build_engine_config()
        self._engine = QwenASREngine(config=config)

        elapsed = time.time() - t0
        logger.info(
            "Qwen3-ASR 引擎初始化完成，耗时 %.2fs",
            elapsed,
        )
        logger.debug(
            "引擎配置 - GPU: %s | chunk_size: %.1f | memory_num: %s | "
            "dynamic_chunk_threshold: %.1f | vad_threshold: %.2f | aligner: %s",
            config.use_gpu,
            config.chunk_size,
            config.memory_num,
            config.dynamic_chunk_threshold,
            config.vad_config.speech_threshold if config.vad_config else 0.0,
            config.enable_aligner,
        )

    def _build_engine_config(self) -> ASREngineConfig:
        """根据全局 settings 构建 ASR 引擎配置。"""
        align_config = AlignerConfig(
            use_gpu=settings.ALIGNER_USE_GPU,
            model_dir=settings.MODEL_DIR,
        )

        # 始终创建 VAD 配置（即使未显式启用 VAD，动态分片也可能需要延迟加载）
        vad_config = VADConfig(
            model_dir=settings.VAD_MODEL_DIR,
            use_gpu=settings.VAD_USE_GPU,
            smooth_window_size=settings.VAD_SMOOTH_WINDOW_SIZE,
            speech_threshold=settings.VAD_SPEECH_THRESHOLD,
            min_speech_frame=settings.VAD_MIN_SPEECH_FRAME,
            max_speech_frame=settings.VAD_MAX_SPEECH_FRAME,
            min_silence_frame=settings.VAD_MIN_SILENCE_FRAME,
            merge_silence_frame=settings.VAD_MERGE_SILENCE_FRAME,
            extend_speech_frame=settings.VAD_EXTEND_SPEECH_FRAME,
            chunk_max_frame=settings.VAD_CHUNK_MAX_FRAME,
            vad_min_duration=settings.VAD_MIN_DURATION,
        )

        return ASREngineConfig(
            model_dir=settings.MODEL_DIR,
            use_gpu=args.use_gpu,
            chunk_size=settings.ASR_CHUNK_SIZE,
            memory_num=settings.ASR_MEMORY_NUM,
            align_config=align_config,
            vad_config=vad_config,
            dynamic_chunk_threshold=settings.ASR_DYNAMIC_CHUNK_THRESHOLD,
            n_threads=settings.INFERENCE_CPU_THREADS,
            n_threads_batch=settings.INFERENCE_CPU_THREADS_BATCH,
        )

    def shutdown(self):
        """优雅关闭引擎"""
        if self._engine:
            logger.info("正在关闭 Qwen3-ASR 引擎...")
            self._engine.shutdown()
            self._engine = None
            logger.info("Qwen3-ASR 引擎已关闭")

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    # ──────────────────────────────────────────────────────────────────
    # 离线转写（完整结果一次性返回）
    # ──────────────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_path: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.4,
        enable_aligner: bool = False,
    ) -> TranscribeResult:
        """
        对单个音频文件执行离线转写。

        使用 asyncio.Lock 保证串行访问引擎，
        使用 asyncio.to_thread 避免阻塞事件循环。
        """
        if not self._engine:
            raise RuntimeError("ASR 引擎未初始化")

        async with self._lock:
            logger.debug(f"[离线] 开始转写: {os.path.basename(audio_path)}")
            t0 = time.time()

            result = await asyncio.to_thread(
                self._engine.transcribe,
                audio_file=audio_path,
                context=context or settings.DEFAULT_CONTEXT,
                language=language or settings.DEFAULT_LANGUAGE,
                temperature=temperature,
                enable_aligner=enable_aligner,
            )

            elapsed = time.time() - t0
            text_preview = (
                result.text[:80] + "..." if len(result.text) > 80 else result.text
            )
            logger.debug(f"[离线] 转写完成: {elapsed:.2f}s | {text_preview}")

            return result

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.4,
        enable_aligner: bool = False,
    ) -> TranscribeResult:
        """
        接收二进制音频数据，写入临时文件后执行离线转写。
        """
        tmp_path = self._save_tmp(audio_bytes, filename)
        try:
            return await self.transcribe(
                audio_path=tmp_path,
                context=context,
                language=language,
                temperature=temperature,
                enable_aligner=enable_aligner,
            )
        finally:
            self._remove_tmp(tmp_path)

    # ──────────────────────────────────────────────────────────────────
    # 流式转写（逐分片实时 yield StreamChunkResult）
    # ──────────────────────────────────────────────────────────────────

    async def stream_transcribe(
        self,
        audio_path: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.4,
        enable_aligner: bool = False,
    ) -> AsyncGenerator[StreamChunkResult, None]:
        """
        对单个音频文件执行流式转写，逐分片 yield StreamChunkResult。

        实现原理:
          1. 在独立线程中调用同步生成器 engine.transcribe_stream()
          2. 每产出一个分片结果，通过 asyncio.Queue 投递到事件循环
          3. 异步消费端 await queue.get() 后立即 yield 给 SSE 等调用方
          4. 线程结束后放入哨兵 _STREAM_SENTINEL 通知消费端退出

        锁策略:
          流式转写全程持有 _lock，与离线转写互斥，保证引擎串行访问。
        """
        if not self._engine:
            raise RuntimeError("ASR 引擎未初始化")

        async with self._lock:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue(
                maxsize=16
            )  # 适当背压，长音频需要更大缓冲
            logger.debug(f"[流式] 开始转写: {os.path.basename(audio_path)}")
            t0 = time.time()
            engine = self._engine

            def _worker():
                """在子线程中运行同步生成器，将结果投入 asyncio.Queue"""
                try:
                    for chunk in engine.transcribe_stream(
                        audio_file=audio_path,
                        context=context or settings.DEFAULT_CONTEXT,
                        language=language or settings.DEFAULT_LANGUAGE,
                        temperature=temperature,
                        enable_aligner=enable_aligner,
                    ):
                        # 线程安全地将结果放入队列（超时保护防止事件循环阻塞时永久挂起）
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result(
                            timeout=60
                        )
                except Exception as exc:
                    # 将异常传递到异步侧
                    try:
                        asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result(
                            timeout=10
                        )
                    except Exception:
                        logger.error(f"[流式] 无法将异常投递到队列: {exc}")
                finally:
                    # 哨兵使用 fire-and-forget 方式投递，避免超时导致哨兵丢失
                    # 使消费端永久阻塞在 queue.get()。只要事件循环最终恢复，
                    # put_nowait 就能成功将哨兵放入队列。
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, _STREAM_SENTINEL)
                    except Exception:
                        logger.error("[流式] 无法发送流结束信号到队列")

            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()

            # 异步消费队列
            while True:
                item = await queue.get()

                if item is _STREAM_SENTINEL:
                    break

                if isinstance(item, Exception):
                    worker_thread.join(timeout=60)
                    raise item

                yield item

            worker_thread.join(timeout=300)  # 长音频可能需要较长时间完成

            elapsed = time.time() - t0
            logger.debug(
                f"[流式] 转写完成: {elapsed:.2f}s | {os.path.basename(audio_path)}"
            )

    async def stream_transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.4,
        enable_aligner: bool = False,
    ) -> AsyncGenerator[StreamChunkResult, None]:
        """
        接收二进制音频数据，写入临时文件后执行流式转写。
        临时文件在生成器完全消费后才会被删除。
        """
        tmp_path = self._save_tmp(audio_bytes, filename)
        try:
            async for chunk in self.stream_transcribe(
                audio_path=tmp_path,
                context=context,
                language=language,
                temperature=temperature,
                enable_aligner=enable_aligner,
            ):
                yield chunk
        finally:
            self._remove_tmp(tmp_path)

    # ──────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────

    def _save_tmp(self, audio_bytes: bytes, filename: str) -> str:
        """将二进制音频保存到 uploads 目录，返回临时文件路径"""
        upload_dir = settings.upload_dir_path
        ext = os.path.splitext(filename)[1] or ".wav"
        safe_name = f"{uuid.uuid4().hex}{ext}"
        tmp_path = os.path.join(upload_dir, safe_name)
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        return tmp_path

    @staticmethod
    def _remove_tmp(path: str):
        """安全删除临时文件"""
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"清理临时文件失败: {path} | {e}")


# ─── 全局单例 ───────────────────────────────────────────────────────
asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """获取 ASR 服务单例，用于 FastAPI 依赖注入"""
    if asr_service is None:
        raise RuntimeError("ASR 服务未初始化，请检查应用 lifespan")
    return asr_service
