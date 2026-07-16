# coding=utf-8
"""Configuration for the independent vLLM transcription backend."""

try:
    from pydantic import field_validator
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    from pydantic import validator as field_validator
from pydantic_settings import BaseSettings

from core.config import settings


class VLLMSettings(BaseSettings):
    """vLLM-specific settings.

    These values intentionally live outside the GGUF/ONNX engine config so the
    default backend can start without importing or initializing vLLM.
    """

    MODEL: str = settings.VLLM_MODEL
    GPU_MEMORY_UTILIZATION: float = 0.9
    MAX_MODEL_LEN: int = 32768
    MAX_NEW_TOKENS: int = 4096
    MAX_INFERENCE_BATCH_SIZE: int = 32
    MAX_CONCURRENT_TRANSCRIBES: int = 1
    INIT_TIMEOUT_SECONDS: float = 300.0
    ENABLE_FORCED_ALIGNER: bool = False
    FORCED_ALIGNER: str = settings.VLLM_FORCED_ALIGNER
    FORCE_ALIGNER_DEVICE: str = "cuda:0"
    FORCE_ALIGNER_DTYPE: str = "bfloat16"

    @field_validator("MODEL", "FORCED_ALIGNER", "FORCE_ALIGNER_DEVICE", "FORCE_ALIGNER_DTYPE")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("配置值不能为空")
        return v

    @field_validator("GPU_MEMORY_UTILIZATION")
    @classmethod
    def validate_gpu_memory_utilization(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("GPU_MEMORY_UTILIZATION 必须在 0.0 到 1.0 之间")
        return v

    @field_validator(
        "MAX_MODEL_LEN",
        "MAX_NEW_TOKENS",
        "MAX_INFERENCE_BATCH_SIZE",
        "MAX_CONCURRENT_TRANSCRIBES",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("配置值必须为正整数")
        return v

    @field_validator("INIT_TIMEOUT_SECONDS")
    @classmethod
    def validate_init_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("INIT_TIMEOUT_SECONDS 必须为正数")
        return v

    @property
    def backend_enabled(self) -> bool:
        return settings.BACKEND == "vllm"

    class Config:
        env_prefix = "ASR_VLLM_"


vllm_settings = VLLMSettings()
