# coding=utf-8
"""Pydantic schemas for independent vLLM transcription responses."""

from typing import Optional

from pydantic import BaseModel


class VLLMHealthData(BaseModel):
    """vLLM health and configuration data."""

    backend: str
    backend_enabled: bool
    initialized: bool
    model: str
    gpu_memory_utilization: float
    max_model_len: int
    forced_aligner_enabled: bool
    forced_aligner: Optional[str] = None


class VLLMTimestampItem(BaseModel):
    """Serialized forced-aligner timestamp item."""

    text: str
    start: float
    end: float


class VLLMTranscribeData(BaseModel):
    """Offline vLLM transcription response payload."""

    backend: str = "vllm"
    model: str
    filename: str
    language: str
    text: str
    timestamps: Optional[list[VLLMTimestampItem]] = None
    elapsed: float
