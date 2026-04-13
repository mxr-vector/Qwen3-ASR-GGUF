# coding=utf-8
"""
Qwen3-ASR Web 服务入口

使用方式:
    uv run python infer.py
    uv run python infer.py --port 8002 --use_gpu True
"""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.config import args, settings
from core.logger import logger
from core.gobal_exception import register_exception
from core.auto_import import load_routers
from core.middleware_request_id import RequestIDMiddleware
from core.middleware_access_log import AccessLogMiddleware
from core.middleware_auth import TokenAuthMiddleware

import services.asr_service as asr_module

import os
import sys


# 自动处理缺失的 NVIDIA 动态链接库路径 (适用于 pip 安装了 nvidia-cudnn-cu12, nvidia-cublas-cu12 等包的情况)
def _setup_nvidia_paths():
    import site

    try:
        site_packages = site.getsitepackages()[0]
        nvidia_path = os.path.join(site_packages, "nvidia")
        if os.path.exists(nvidia_path):
            lib_paths = [
                os.path.join(nvidia_path, lib, "lib")
                for lib in os.listdir(nvidia_path)
                if os.path.isdir(os.path.join(nvidia_path, lib, "lib"))
            ]
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            nvidia_ld = ":".join(lib_paths)
            if nvidia_ld and nvidia_ld not in current_ld:
                os.environ["LD_LIBRARY_PATH"] = nvidia_ld + (
                    ":" + current_ld if current_ld else ""
                )
                # 必须重启进程以生效新的 LD_LIBRARY_PATH
                os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        pass


_setup_nvidia_paths()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理:
    - 启动时: 初始化 ASR 引擎
    - 关闭时: 优雅释放引擎资源
    """
    # ─── Startup ───
    logger.info("=" * 50)
    logger.info("Qwen3-ASR Web 服务启动中...")
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  Base URL: {args.base_url}")
    logger.info(f"  GPU: {args.use_gpu}")
    logger.info("=" * 50)

    # 初始化 ASR 服务单例
    service = asr_module.ASRService()
    service.initialize()
    asr_module.asr_service = service

    yield

    # ─── Shutdown ───
    logger.info("正在关闭 Qwen3-ASR Web 服务...")
    service.shutdown()
    asr_module.asr_service = None
    logger.info("服务已停止")


# ─── 创建 FastAPI 应用 ─────────────────────────────────────
app = FastAPI(
    title="Qwen3-ASR API",
    description="Qwen3-ASR 语音识别 Web 服务 — 基于 GGUF 推理后端",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── 注册中间件 (执行顺序: 从下到上) ──────────────────────
app.add_middleware(AccessLogMiddleware)
app.add_middleware(TokenAuthMiddleware)
app.add_middleware(RequestIDMiddleware)

# ─── 注册全局异常处理 ─────────────────────────────────────
register_exception(app)

# ─── 自动加载路由 ─────────────────────────────────────────
load_routers(app)


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "service": "Qwen3-ASR API",
        "version": "1.0.0",
        "docs": f"http://{args.host}:{args.port}/docs",
    }


# ─── 启动入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "infer:app",
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,  # 使用自定义 AccessLogMiddleware
        timeout_keep_alive=600,  # 长音频流式转写需要较长的 keep-alive 超时 (秒)
    )
