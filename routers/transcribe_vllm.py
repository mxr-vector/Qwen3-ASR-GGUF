# coding=utf-8
"""
独立 vLLM 转写路由。

该模块不复用现有 GGUF/ONNX ASRService，也不在模块导入时加载 vLLM。
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from core.config import settings
from core.logger import logger
from core.response import R
from qwen_asr_vllm.service import (
    VLLMBackendDisabledError,
    VLLMConfigurationError,
    VLLMDependencyError,
    get_vllm_service,
)
from utils.file import check_file_size, generate_unique_filename

router = APIRouter(prefix="/transcribe-vllm", tags=["ASR vLLM 语音识别"])


@router.get("/health", summary="vLLM 转写健康检查")
async def health():
    """返回 vLLM 配置和初始化状态；不会加载 vLLM 模型。"""
    service = get_vllm_service()
    return R.success(service.health())


@router.post("/file", summary="vLLM 单文件离线转写")
async def transcribe_file(
    file: UploadFile = File(..., description="音频文件"),
    context: Optional[str] = Form(None, description="上下文提示词"),
    language: Optional[str] = Form(None, description="语言 (Chinese/English 等)"),
    return_timestamps: bool = Form(False, description="是否返回 forced aligner 时间戳"),
):
    content = await file.read()
    check_file_size(content, file.filename or "")

    filename = file.filename or generate_unique_filename(suffix=".wav")
    suffix = Path(filename).suffix or ".wav"
    tmp_path = os.path.join(settings.upload_dir_path, generate_unique_filename(suffix=suffix))

    try:
        with open(tmp_path, "wb") as f:
            f.write(content)

        service = get_vllm_service()
        data = await service.transcribe_file(
            audio_path=tmp_path,
            filename=filename,
            context=context,
            language=language,
            return_timestamps=return_timestamps,
        )
        return R.success(data)
    except VLLMBackendDisabledError as e:
        logger.warning(f"vLLM 后端未启用: {e}")
        return R.fail(str(e), code=400)
    except (VLLMConfigurationError, VLLMDependencyError) as e:
        logger.warning(f"vLLM 转写配置错误: {e}")
        return R.fail(str(e), code=400)
    except Exception as e:
        logger.exception("vLLM 转写失败")
        return R.fail(f"vLLM 转写失败: {e}", code=500)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.debug("删除 vLLM 临时音频失败: %s", tmp_path)
