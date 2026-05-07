# coding=utf-8
"""
ASR 服务层 — 线程安全的 QwenASREngine 封装

设计要点:
- 使用 asyncio.Lock 保证同一时刻只有一个推理任务运行（引擎不支持并发）
- 流式接口双路径: 短音频 to_thread 快速路径 / 长音频 Thread+Queue 实时管道
- 全局单例模式，由 lifespan 管理生命周期
"""

import asyncio
import os
import threading
import time
import uuid
from typing import AsyncGenerator, Optional

import numpy as np

from core.logger import logger
from core.config import settings
from qwen_asr_gguf.inference import (
    QwenASREngine,
    ASREngineConfig,
    AlignerConfig,
    VADConfig,
    StreamChunkResult,
)

# ─── 哨兵对象，用于标识流式队列结束 ──────────────────────────
_STREAM_SENTINEL = object()


class ASRService:
    """线程安全的 ASR 服务封装（流式转写）"""

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

        # 计算最终生效的 n_gpu_layers
        if config.n_gpu_layers >= 0:
            effective_gpu_layers = config.n_gpu_layers
        else:
            effective_gpu_layers = 99 if config.use_gpu else 0

        import platform

        platform_info = f"{platform.system()} {platform.machine()}"

        logger.info(
            "Qwen3-ASR 引擎初始化完成，耗时 %.2fs",
            elapsed,
        )
        logger.info(
            "GPU 配置 - use_gpu: %s | n_gpu_layers: %d (配置值: %d) | 平台: %s",
            config.use_gpu,
            effective_gpu_layers,
            config.n_gpu_layers,
            platform_info,
        )
        logger.debug(
            "引擎配置 - chunk_size: %.1f | memory_num: %s | "
            "dynamic_chunk_threshold: %.1f | vad_threshold: %.2f | aligner: %s",
            config.chunk_size,
            config.memory_num,
            config.dynamic_chunk_threshold,
            config.vad_config.speech_threshold if config.vad_config else 0.0,
            config.enable_aligner,
        )

    def _build_engine_config(self) -> ASREngineConfig:
        """根据全局 settings 构建 ASR 引擎配置。"""
        # Aligner (0.6B) 使用 CPU 推理即可，无需占用 GPU 显存
        align_config = AlignerConfig(
            use_gpu=False,
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
            max_safe_skip_sec=settings.VAD_MAX_SAFE_SKIP_SEC,
            min_speech_coverage=settings.VAD_MIN_SPEECH_COVERAGE,
        )

        return ASREngineConfig(
            model_dir=settings.MODEL_DIR,
            use_gpu=settings.USE_GPU,
            n_gpu_layers=settings.N_GPU_LAYERS,
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
    # 流式转写（逐分片实时 yield StreamChunkResult）
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """快速获取音频时长（秒），支持所有格式（含 MP3）"""
        import os
        from pathlib import Path

        ext = Path(audio_path).suffix.lower()
        SF_FORMATS = {".wav", ".flac", ".ogg"}

        # soundfile 支持的格式优先用 soundfile（最快，只读文件头）
        if ext in SF_FORMATS:
            try:
                import soundfile as sf

                return sf.info(audio_path).duration
            except Exception:
                pass

        # 其他格式（MP3/M4A/OPUS 等）使用 ffprobe 获取时长
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass

        return float("inf")  # 无法获取时走长音频路径

    async def stream_transcribe(
        self,
        audio_path: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.4,
        enable_aligner: bool = False,
        disable_vad: bool = False,
    ) -> AsyncGenerator[StreamChunkResult, None]:
        """
        对单个音频文件执行流式转写，逐分片 yield StreamChunkResult。

        双路径策略:
          快速路径 (短音频 ≤ dynamic_chunk_threshold):
            使用 asyncio.to_thread 一次性执行同步生成器并收集结果，
            消除 Thread 创建/销毁、跨线程同步、Queue 锁竞争等开销，
            短音频性能接近离线接口。

          实时路径 (长音频 > dynamic_chunk_threshold):
            在独立线程中运行同步生成器，通过 asyncio.Queue 实时投递，
            每个分片处理完即推送，保证长音频实时流式体验。

        锁策略:
          流式转写全程持有 _lock，与离线转写互斥，保证引擎串行访问。
        """
        if not self._engine:
            raise RuntimeError("ASR 引擎未初始化")

        cancel_event = threading.Event()
        audio_duration = self._get_audio_duration(audio_path)
        threshold = settings.ASR_DYNAMIC_CHUNK_THRESHOLD
        use_fast_path = audio_duration <= threshold

        if use_fast_path:
            # ── 快速路径：短音频使用显式线程 + 完成事件 ─────────────
            # 注意：不能直接用 asyncio.to_thread，因为线程池中的线程
            # 无法被外部取消。客户端断开时 CancelledError 会让 await
            # 提前返回，但底层线程仍在继续推理，若此时释放 _lock，
            # 下一个请求会与未结束的推理并发访问 QwenASREngine。
            async with self._lock:
                logger.debug(
                    f"[流式-快速] 开始转写: {os.path.basename(audio_path)} "
                    f"({audio_duration:.1f}s ≤ {threshold}s)"
                )
                t0 = time.time()
                engine = self._engine

                result_holder: dict = {}
                done_event = threading.Event()

                def _fast_run():
                    try:
                        result_holder["chunks"] = list(
                            engine.transcribe_stream(
                                audio_file=audio_path,
                                context=context or settings.DEFAULT_CONTEXT,
                                language=language or settings.DEFAULT_LANGUAGE,
                                temperature=temperature,
                                enable_aligner=enable_aligner,
                                disable_vad=disable_vad,
                                cancel_event=cancel_event,
                            )
                        )
                    except Exception as e:
                        result_holder["exc"] = e
                    finally:
                        done_event.set()

                worker_thread = threading.Thread(target=_fast_run, daemon=True)
                worker_thread.start()

                try:
                    # 轮询等待推理线程完成：
                    # asyncio.sleep 是可取消点，客户端断开可立即进入 finally
                    while not done_event.is_set():
                        await asyncio.sleep(0.05)

                    if "exc" in result_holder:
                        raise result_holder["exc"]

                    chunks = result_holder.get("chunks", [])
                    elapsed = time.time() - t0
                    logger.debug(
                        f"[流式-快速] 转写完成: {elapsed:.2f}s | "
                        f"{os.path.basename(audio_path)} | {len(chunks)} 个分片"
                    )

                    for chunk in chunks:
                        yield chunk
                finally:
                    # 客户端断开/异常/正常结束统一处理：
                    # 1) 通知底层推理尽快终止 (_asr_core/_decode 观察此信号)
                    # 2) 等待工作线程真正回收再释放 _lock，防止并发进引擎
                    cancel_event.set()
                    if worker_thread.is_alive():
                        await asyncio.to_thread(worker_thread.join, 30)
                        if worker_thread.is_alive():
                            logger.warning(
                                "[流式-快速] 工作线程在 30s 内未能终止，"
                                "可能仍在占用引擎资源"
                            )
        else:
            # ── 实时路径：长音频 Thread+Queue 管道 ─────────────────────
            async with self._lock:
                loop = asyncio.get_event_loop()
                queue: asyncio.Queue = asyncio.Queue(maxsize=16)
                logger.debug(
                    f"[流式-实时] 开始转写: {os.path.basename(audio_path)} "
                    f"({audio_duration:.1f}s > {threshold}s)"
                )
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
                            disable_vad=disable_vad,
                            cancel_event=cancel_event,
                        ):
                            asyncio.run_coroutine_threadsafe(
                                queue.put(chunk), loop
                            ).result(timeout=60)
                    except Exception as exc:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(exc), loop
                            ).result(timeout=10)
                        except Exception:
                            logger.error(f"[流式] 无法将异常投递到队列: {exc}")
                    finally:
                        try:
                            loop.call_soon_threadsafe(
                                queue.put_nowait, _STREAM_SENTINEL
                            )
                        except Exception:
                            logger.error("[流式] 无法发送流结束信号到队列")

                worker_thread = threading.Thread(target=_worker, daemon=True)
                worker_thread.start()

                try:
                    while True:
                        item = await queue.get()
                        if item is _STREAM_SENTINEL:
                            break
                        if isinstance(item, Exception):
                            worker_thread.join(timeout=60)
                            raise item
                        yield item
                finally:
                    # 客户端断开时：设置取消信号，等待工作线程结束
                    cancel_event.set()
                    # 使用 to_thread 避免阻塞事件循环
                    await asyncio.to_thread(worker_thread.join, 30)

                elapsed = time.time() - t0
                logger.debug(
                    f"[流式-实时] 转写完成: {elapsed:.2f}s | "
                    f"{os.path.basename(audio_path)}"
                )

    async def stream_transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.4,
        enable_aligner: bool = False,
        disable_vad: bool = False,
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
                disable_vad=disable_vad,
            ):
                yield chunk
        finally:
            self._remove_tmp(tmp_path)

    async def transcribe_buffer(
        self,
        audio: "np.ndarray",
        context: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.4,
        lock_timeout: float = 8.0,
    ) -> Optional[str]:
        """
        直接对 numpy 音频缓冲执行转写（零文件 I/O），专为 WS 实时场景设计。

        与 stream_transcribe 的区别:
          - 输入为 numpy float32 数组（16kHz 单声道），不写入/读取临时文件
          - 带超时的锁获取，避免被长 SSE 任务永久阻塞
          - 内部禁用 VAD（客户端已做 VAD 分片），跳过不必要的分析
          - 锁超时时返回 None，调用方静默跳过

        Args:
            audio: 16kHz 单声道 float32 音频数据
            context: 上下文提示词
            language: 语言 (Chinese/English 等)
            temperature: 解码温度
            lock_timeout: 等待引擎锁的最大秒数，超时返回 None
        """
        if not self._engine:
            raise RuntimeError("ASR 引擎未初始化")

        audio_duration = len(audio) / 16000
        t_start = time.time()

        # 带超时的锁获取，避免被 SSE 长任务永久阻塞
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=lock_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[WS转写] 等待引擎锁超时 (%.1fs)，跳过本段音频 (%.2fs)",
                lock_timeout,
                audio_duration,
            )
            return None

        t_lock_acquired = time.time()
        lock_wait_ms = (t_lock_acquired - t_start) * 1000

        cancel_event = threading.Event()
        result_holder: dict = {}
        done_event = threading.Event()
        engine = self._engine

        def _run():
            try:
                texts = []
                for chunk in engine.asr_stream(
                    audio=audio,
                    context=context or settings.DEFAULT_CONTEXT,
                    language=language or settings.DEFAULT_LANGUAGE,
                    chunk_size_sec=1.0,
                    memory_chunks=1,
                    temperature=temperature,
                    disable_vad=True,
                    cancel_event=cancel_event,
                ):
                    if chunk.text and chunk.text.strip():
                        texts.append(chunk.text.strip())
                result_holder["text"] = "".join(texts)
            except Exception as e:
                result_holder["exc"] = e
            finally:
                done_event.set()

        worker_thread = threading.Thread(target=_run, daemon=True)
        worker_thread.start()

        try:
            while not done_event.is_set():
                await asyncio.sleep(0.05)

            t_infer_done = time.time()
            infer_ms = (t_infer_done - t_lock_acquired) * 1000

            if "exc" in result_holder:
                raise result_holder["exc"]

            text = result_holder.get("text", "").strip()

            total_ms = (time.time() - t_start) * 1000
            logger.info(
                "[WS转写] 完成 | 音频=%.2fs | 锁等待=%.0fms | 推理=%.0fms | 总耗时=%.0fms",
                audio_duration,
                lock_wait_ms,
                infer_ms,
                total_ms,
            )

            return text if text else None
        finally:
            cancel_event.set()
            if worker_thread.is_alive():
                await asyncio.to_thread(worker_thread.join, 10)
                if worker_thread.is_alive():
                    logger.warning(
                        "[WS转写] 工作线程在 10s 内未终止，可能仍在占用引擎资源"
                    )
            self._lock.release()

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
