# coding=utf-8
"""
音频 I/O 工具 — 基于 soundfile / numpy 的音频读写辅助函数

提供:
  read_audio_bytes(audio_bytes)   — soundfile 将任意格式音频字节解码为 float32 单声道数组
  pcm16_to_float32(raw_pcm)      — 原始 PCM int16 字节 → float32 归一化数组
  save_audio_tmp(samples, sr)    — soundfile 将 float32 数组写入临时 WAV 文件
"""

import io
import os

import numpy as np
import soundfile as sf

from core.config import settings
from utils.file import generate_unique_filename


def read_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    使用 soundfile 将音频字节解码为 float32 单声道 numpy 数组。

    支持 WAV / FLAC / OGG / AIFF 等所有 soundfile 支持的容器格式。
    多声道音频自动取均值合并为单声道。

    Args:
        audio_bytes: 任意格式的音频文件二进制内容。

    Returns:
        (samples, samplerate): float32 单声道样本数组 及 采样率 (Hz)。

    Raises:
        soundfile.SoundFileError: 无法识别的格式或损坏文件。
    """
    buf = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(buf, dtype="float32", always_2d=True)
    # 多声道 → 单声道（取均值）
    mono = data.mean(axis=1)
    return mono, samplerate


def pcm16_to_float32(raw_pcm: bytes) -> np.ndarray:
    """
    将原始 PCM int16 字节流转换为 float32 归一化数组。

    适用于浏览器 MediaRecorder 或麦克风直接输出的裸 PCM 数据（无文件头）。

    Args:
        raw_pcm: Little-endian int16 PCM 字节流（单声道）。

    Returns:
        归一化到 [-1.0, 1.0] 的 float32 numpy 数组。
    """
    arr = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
    arr /= 32768.0
    return arr


def save_audio_tmp(
    samples: np.ndarray,
    samplerate: int,
    suffix: str = ".wav",
) -> str:
    """
    将 float32 单声道 numpy 数组用 soundfile 写入临时 WAV 文件。

    文件保存在 settings.UPLOAD_DIR 目录，文件名由 generate_unique_filename 生成。
    **调用方负责在使用完毕后删除临时文件**。

    Args:
        samples:    float32 单声道音频样本数组。
        samplerate: 采样率 (Hz)。
        suffix:     文件扩展名，默认 ".wav"。

    Returns:
        临时文件的绝对路径。
    """
    upload_dir = settings.upload_dir_path
    tmp_name = generate_unique_filename(suffix=suffix)
    tmp_path = os.path.join(upload_dir, tmp_name)
    sf.write(tmp_path, samples, samplerate, subtype="PCM_16")
    return tmp_path
