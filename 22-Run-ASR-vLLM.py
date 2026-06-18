# coding=utf-8
import asyncio
import os
import sys
import time
from pathlib import Path


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

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))


# 本脚本是 vLLM 专用测试；未显式指定 --backend 时自动切到 vllm。
# 如需验证后端关闭时的报错，可手动传入 --backend=onnx。
def _ensure_vllm_backend_arg():
    has_backend_arg = any(
        arg == "--backend" or arg.startswith("--backend=") for arg in sys.argv[1:]
    )
    if not has_backend_arg:
        sys.argv.append("--backend=vllm")


_ensure_vllm_backend_arg()

from qwen_asr_vllm.service import (  # noqa: E402
    VLLMBackendDisabledError,
    VLLMConfigurationError,
    VLLMDependencyError,
    get_vllm_service,
)

# ─── 配置区 ───────────────────────────────────────────────────────────────────

AUDIO_PATH = os.getenv("DEMO_AUDIO_PATH", "uploads/1.wav")
CONTEXT = os.getenv("DEMO_CONTEXT", "")
LANGUAGE = os.getenv("DEMO_LANGUAGE", "Chinese") or None

# 是否返回 forced aligner 时间戳；与 routers/transcribe_vllm.py 的 return_timestamps 参数一致。
# 开启前需设置 ASR_VLLM_ENABLE_FORCED_ALIGNER=1 并配置 ASR_VLLM_FORCED_ALIGNER。
DEMO_RETURN_TIMESTAMPS = os.getenv("DEMO_RETURN_TIMESTAMPS", "0").strip().lower() in {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
}


# ─── vLLM 离线转写演示 ────────────────────────────────────────────────────────


async def demo_offline():
    """离线转写：复用 vLLM 路由同一套懒加载 Service 和返回 schema。"""
    audio_path = Path(AUDIO_PATH)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    service = get_vllm_service()

    health = service.health()
    print("vLLM 配置:")
    print(f"  backend: {health.backend}")
    print(f"  backend_enabled: {health.backend_enabled}")
    print(f"  initialized: {health.initialized}")
    print(f"  model: {health.model}")
    print(f"  gpu_memory_utilization: {health.gpu_memory_utilization}")
    print(f"  max_model_len: {health.max_model_len}")
    print(f"  forced_aligner_enabled: {health.forced_aligner_enabled}")
    if health.forced_aligner:
        print(f"  forced_aligner: {health.forced_aligner}")

    t0 = time.time()
    data = await service.transcribe_file(
        audio_path=str(audio_path),
        filename=audio_path.name,
        context=CONTEXT,
        language=LANGUAGE,
        return_timestamps=DEMO_RETURN_TIMESTAMPS,
    )
    init_and_decode_elapsed = time.time() - t0

    print(f"\n完成: 接口耗时 {data.elapsed:.3f}s | 总耗时 {init_and_decode_elapsed:.1f}s")
    print(f"模型: {data.model}")
    print(f"语言: {data.language}")

    print("\n转写结果:")
    print(data.text)

    if data.timestamps:
        print("\n时间戳结果 (前10):")
        for it in data.timestamps[:10]:
            print(f"  {it.text:<10} | {it.start:7.3f}s → {it.end:7.3f}s")


# ─── 主入口 ───────────────────────────────────────────────────────────────────


async def main():
    service = None
    try:
        service = get_vllm_service()
        await demo_offline()
    except VLLMBackendDisabledError as e:
        print(f"\nvLLM 后端未启用: {e}")
        print("\n运行建议:")
        print("  uv run python 22-Run-ASR-vLLM.py --backend=vllm")
    except (VLLMConfigurationError, VLLMDependencyError) as e:
        print(f"\nvLLM 转写配置错误: {e}")
        print("\n排查建议:")
        print("  1. 检查依赖: uv sync && uv pip install vllm --torch-backend=auto fireredvad transformers==4.57.6")
        print("  2. 检查模型: ASR_VLLM_MODEL 是否指向 HF 格式 Qwen3-ASR 模型目录")
        print("  3. 如需时间戳: 设置 ASR_VLLM_ENABLE_FORCED_ALIGNER=1 并配置 ASR_VLLM_FORCED_ALIGNER")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"\nvLLM 转写失败: {e}")
        print("\n排查建议:")
        print("  1. 检查 GPU 显存是否充足: nvidia-smi")
        print("  2. WSL + RTX 50 系列如遇 FlashInfer 初始化失败，可设置 VLLM_USE_FLASHINFER_SAMPLER=0")
        print("  3. 16GB 显存可设置 ASR_VLLM_GPU_MEMORY_UTILIZATION=0.9 和 ASR_VLLM_MAX_MODEL_LEN=32768")
        import traceback

        traceback.print_exc()
    finally:
        if service is not None:
            service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
