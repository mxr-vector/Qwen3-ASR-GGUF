# coding=utf-8
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

import time
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import (
    QwenASREngine,
    itn,
    load_audio,
    ASREngineConfig,
    AlignerConfig,
    VADConfig,
)
from qwen_asr_gguf.inference import exporters


# ─── 配置区 ───────────────────────────────────────────────────────────────────

AUDIO_PATH = "datasets/yn6.wav"
CONTEXT = ""

# 演示模式：
#   "offline"  - 离线转写（等待全部处理完毕后一次性输出）
#   "stream"   - 流式转写（逐分片实时打印，模拟 SSE 场景）
DEMO_MODE = "offline"

# VAD 模型路径（FireRedVAD 非流式版本）
# VAD 由 dynamic_chunk_threshold 自动控制：音频 > 阈值时延迟加载
VAD_MODEL_DIR = "models/FireRedVAD/VAD"


# ─── 引擎配置 ─────────────────────────────────────────────────────────────────


def build_config() -> ASREngineConfig:
    """构造 ASR 引擎配置

    VAD 动态分片工作流（音频 > dynamic_chunk_threshold 时自动启用）：
      1. 在转写开始前对全段音频执行一次自适应阈值 VAD 检测
      2. 按语音边界动态划分分片（不在句中截断、不在静音中切割）
      3. 每分片的 token 预算随实际语音时长等比缩放，从根本上抑制幻觉
      4. LLM 上下文仅保留前片段文本（不重放音频），避免非连续音频拼接干扰
    """
    vad_cfg = VADConfig(
        model_dir=VAD_MODEL_DIR,
        use_gpu=False,  # VAD 模型较小，CPU 即可满足实时需求
        speech_threshold=0.35,  # 初始语音帧判定阈值（自适应算法会动态调整）
    )

    return ASREngineConfig(
        model_dir="models",
        use_gpu=True,
        chunk_size=30.0,  # 动态分片模式下为单分片的最大时间跨度上限
        memory_num=1,  # 保留前 N 片文本作为上下文
        enable_aligner=False,  # 关闭对齐以降低 RTF，需要字级时间戳时再开启
        align_config=AlignerConfig(
            use_gpu=True,
            model_dir="models",
        ),
        vad_config=vad_cfg,
        dynamic_chunk_threshold=10.0,  # 音频 > 10s 时自动启用 VAD 动态分片
    )


# ─── 离线转写演示 ──────────────────────────────────────────────────────────────


def demo_offline(engine: QwenASREngine):
    """离线转写：等待完整结果后一次性输出"""
    res = engine.transcribe(
        audio_file=AUDIO_PATH,
        context=CONTEXT,
        language="Chinese",
        start_second=0,
        duration=None,  # None = 整段音频
    )

    print("\n转写结果:")
    print(res.text)
    print("\nITN:")
    print(itn(res.text))

    # 导出文件示例（按需取消注释）
    # txt_path = str(Path(AUDIO_PATH).with_suffix('.txt'))
    # exporters.export_to_txt(txt_path, res)

    # srt_path = str(Path(AUDIO_PATH).with_suffix('.srt'))
    # exporters.export_to_srt(srt_path, res)

    # json_path = str(Path(AUDIO_PATH).with_suffix('.json'))
    # exporters.export_to_json(json_path, res)

    if res.alignment:
        print("\n对齐结果 (前10):")
        for it in res.alignment.items[:10]:
            print(f"  {it.text:<10} | {it.start_time:7.3f}s → {it.end_time:7.3f}s")


# ─── 流式转写演示 ──────────────────────────────────────────────────────────────


def demo_stream(engine: QwenASREngine):
    """流式转写：逐分片实时打印，模拟 SSE 场景。"""
    t0 = time.time()
    full_text = ""
    chunk_count = 0
    skipped_count = 0

    for chunk in engine.transcribe_stream(
        audio_file=AUDIO_PATH,
        context=CONTEXT,
        language="Chinese",
        start_second=0,
        duration=None,
    ):
        chunk_count += 1

        if chunk.skipped_by_vad:
            skipped_count += 1
            print(
                f"[{chunk.segment_idx:02d}] {chunk.start_sec:.0f}s~{chunk.end_sec:.0f}s  静音跳过"
            )
        else:
            full_text += chunk.text
            print(
                f"[{chunk.segment_idx:02d}] {chunk.start_sec:.0f}s~{chunk.end_sec:.0f}s"
                f"  enc {chunk.encode_time:.1f}s dec {chunk.decode_time:.1f}s"
            )
            print(f"     {chunk.text.strip()}")

    elapsed = time.time() - t0
    skip_info = f"，VAD 跳过 {skipped_count} 片" if skipped_count else ""
    print(f"\n完成: {chunk_count} 片{skip_info} | 总耗时 {elapsed:.1f}s")

    print("\n转写结果:")
    print(full_text)
    print("\nITN:")
    print(itn(full_text))


# ─── 主入口 ───────────────────────────────────────────────────────────────────


def main():
    # 初始化引擎
    config = build_config()

    t0 = time.time()
    engine = QwenASREngine(config=config)
    print(f"引擎初始化: {time.time() - t0:.1f}s")

    try:
        if DEMO_MODE == "stream":
            demo_stream(engine)
        else:
            demo_offline(engine)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
