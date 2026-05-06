import time
import uuid

# coding=utf-8
"""
上传校验工具 — HTTP 文件上传相关的参数校验辅助函数

提供:
  check_file_size(content, filename) — 校验上传文件字节大小，超限抛出 HTTP 413
"""

from fastapi import HTTPException

from core.config import settings


def check_file_size(content: bytes, filename: str = "") -> None:
    """
    校验上传文件的字节大小是否超出服务端限制。

    超出 settings.MAX_FILE_SIZE_MB 时抛出 HTTP 413 异常，
    由 FastAPI 的全局异常处理器统一返回给客户端。

    Args:
        content:  文件的完整二进制内容。
        filename: 文件名（仅用于错误提示，可选）。

    Raises:
        HTTPException(413): 文件大小超过 MAX_FILE_SIZE_MB 限制。
    """
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


def generate_unique_filename(suffix: str = "") -> str:
    """生成唯一文件名，格式：{timestamp}_{uuid}{suffix}"""
    return f"{int(time.time()*1000)}_{uuid.uuid4().hex}{suffix}"
