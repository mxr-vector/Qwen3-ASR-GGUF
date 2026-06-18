# coding=utf-8
"""Lazy service wrapper for Qwen3-ASR vLLM transcription."""

import asyncio
import threading
import time
from pathlib import Path
from typing import Optional

from core.config import settings
from core.logger import logger

from .config import VLLMSettings, vllm_settings
from .schema import VLLMHealthData, VLLMTranscribeData
from .utils import serialize_timestamps


class VLLMBackendDisabledError(RuntimeError):
    """Raised when the vLLM endpoint is called while backend selection is not vLLM."""


class VLLMDependencyError(RuntimeError):
    """Raised when optional vLLM dependencies are missing."""


class VLLMConfigurationError(RuntimeError):
    """Raised when vLLM configuration cannot satisfy the request."""


class VLLMASRService:
    """Independent, lazily initialized Qwen3-ASR vLLM service."""

    def __init__(self, config: VLLMSettings = vllm_settings):
        self.config = config
        self._model = None
        self._init_lock = asyncio.Lock()
        self._transcribe_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_TRANSCRIBES)

    @property
    def initialized(self) -> bool:
        return self._model is not None

    def health(self) -> VLLMHealthData:
        return VLLMHealthData(
            backend=settings.BACKEND,
            backend_enabled=self.config.backend_enabled,
            initialized=self.initialized,
            model=self.config.MODEL,
            gpu_memory_utilization=self.config.GPU_MEMORY_UTILIZATION,
            max_model_len=self.config.MAX_MODEL_LEN,
            forced_aligner_enabled=self.config.ENABLE_FORCED_ALIGNER,
            forced_aligner=(
                self.config.FORCED_ALIGNER if self.config.ENABLE_FORCED_ALIGNER else None
            ),
        )

    async def _ensure_initialized(self):
        if self._model is not None:
            return self._model

        async with self._init_lock:
            if self._model is not None:
                return self._model

            if not self.config.backend_enabled:
                raise VLLMBackendDisabledError(
                    "vLLM 转写接口需要使用 --backend=vllm 启动服务。"
                )

            self._validate_model_path()
            logger.info(f"正在初始化 Qwen3-ASR vLLM 后端: {self.config.MODEL}")
            t0 = time.time()
            try:
                self._model = await asyncio.wait_for(
                    asyncio.to_thread(self._create_model),
                    timeout=self.config.INIT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as e:
                raise VLLMConfigurationError(
                    f"vLLM 模型初始化超时（{self.config.INIT_TIMEOUT_SECONDS:g}秒），请检查模型路径和环境配置。"
                ) from e
            logger.info(f"Qwen3-ASR vLLM 后端初始化完成，耗时 {time.time() - t0:.2f}s")
            return self._model

    def _validate_model_path(self) -> None:
        model = self.config.MODEL.strip()
        lower = model.lower()
        if lower.endswith((".gguf", ".onnx")):
            raise VLLMConfigurationError(
                "vLLM 后端需要 HF 格式 Qwen3-ASR 模型，不能直接使用 GGUF 或 ONNX 模型文件。"
            )

        model_path = Path(model).expanduser()
        if model_path.exists():
            if not model_path.is_dir():
                raise VLLMConfigurationError(
                    f"vLLM 模型路径必须是 HF 模型目录，而不是文件: {self.config.MODEL}"
                )
            return

        if self._looks_like_local_model_path(model):
            raise VLLMConfigurationError(
                f"模型目录不存在: {self.config.MODEL}，请检查 ASR_VLLM_MODEL 配置。"
            )

    @staticmethod
    def _looks_like_local_model_path(model: str) -> bool:
        normalized = model.replace("\\", "/")
        first_part = normalized.split("/", 1)[0]
        return (
            Path(model).is_absolute()
            or normalized.startswith(("./", "../", "~/"))
            or first_part in {"models", "model", "checkpoints", "checkpoint"}
        )

    def _create_model(self):
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except Exception as e:
            raise VLLMDependencyError(
                "vLLM 后端依赖不可用。请安装可选依赖，例如：uv sync --extra vllm 或 pip install '.[vllm]'。"
            ) from e

        kwargs = {
            "model": self.config.MODEL,
            "gpu_memory_utilization": self.config.GPU_MEMORY_UTILIZATION,
            "max_model_len": self.config.MAX_MODEL_LEN,
            "max_inference_batch_size": self.config.MAX_INFERENCE_BATCH_SIZE,
            "max_new_tokens": self.config.MAX_NEW_TOKENS,
        }

        if self.config.ENABLE_FORCED_ALIGNER:
            dtype = getattr(torch, self.config.FORCE_ALIGNER_DTYPE, torch.bfloat16)
            kwargs["forced_aligner"] = self.config.FORCED_ALIGNER
            kwargs["forced_aligner_kwargs"] = {
                "dtype": dtype,
                "device_map": self.config.FORCE_ALIGNER_DEVICE,
            }

        try:
            return Qwen3ASRModel.LLM(**kwargs)
        except ImportError as e:
            raise VLLMDependencyError(
                "vLLM 后端依赖不可用。请安装可选依赖，例如：uv sync --extra vllm 或 pip install '.[vllm]'。"
            ) from e

    async def transcribe_file(
        self,
        audio_path: str,
        filename: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> VLLMTranscribeData:
        if not self.config.backend_enabled:
            raise VLLMBackendDisabledError(
                "vLLM 转写接口需要使用 --backend=vllm 启动服务。"
            )
        if return_timestamps and not self.config.ENABLE_FORCED_ALIGNER:
            raise VLLMConfigurationError(
                "return_timestamps=true 需要启用 ASR_VLLM_ENABLE_FORCED_ALIGNER=1 并配置 forced aligner。"
            )

        model = await self._ensure_initialized()

        t0 = time.time()
        async with self._transcribe_semaphore:
            results = await asyncio.to_thread(
                model.transcribe,
                audio=audio_path,
                context=context or "",
                language=language,
                return_time_stamps=return_timestamps,
            )
        elapsed = time.time() - t0

        if not results:
            raise VLLMConfigurationError("vLLM 转写返回空结果，请检查音频文件是否有效。")

        result = results[0]
        return VLLMTranscribeData(
            model=self.config.MODEL,
            filename=filename,
            language=str(getattr(result, "language", "")),
            text=str(getattr(result, "text", "")),
            timestamps=serialize_timestamps(getattr(result, "time_stamps", None)),
            elapsed=round(elapsed, 3),
        )

    def shutdown(self) -> None:
        if self._model is None:
            return

        model = self._model
        self._model = None
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"释放 vLLM CUDA 缓存失败或 CUDA 不可用: {e}")


_vllm_service: Optional[VLLMASRService] = None
_service_lock = threading.Lock()


def get_vllm_service() -> VLLMASRService:
    global _vllm_service
    if _vllm_service is None:
        with _service_lock:
            if _vllm_service is None:
                _vllm_service = VLLMASRService()
    return _vllm_service
