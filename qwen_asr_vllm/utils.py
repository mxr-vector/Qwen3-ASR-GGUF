# coding=utf-8
"""Serialization helpers for vLLM ASR results."""

from typing import Any, Optional

from .schema import VLLMTimestampItem


def serialize_timestamps(time_stamps: Any) -> Optional[list[VLLMTimestampItem]]:
    """Convert a Qwen forced-aligner result into API timestamp items."""
    if time_stamps is None:
        return None

    items = getattr(time_stamps, "items", None)
    if not items:
        return None

    return [
        VLLMTimestampItem(
            text=str(getattr(item, "text", "")),
            start=round(float(getattr(item, "start_time", 0.0)), 3),
            end=round(float(getattr(item, "end_time", 0.0)), 3),
        )
        for item in items
    ]
