#!/usr/bin/env bash
set -Eeuo pipefail

# 一键串行导出/优化/量化 Qwen3-ASR 与 Aligner 模型。
# 用法：
#   ./run_export_pipeline.sh        # 依次执行全部步骤
#   bash run_export_pipeline.sh     # 未授予执行权限时使用

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STEPS=(
  "01-Export-ASR-Encoder-Frontend.py|导出 ASR Encoder 前段 (CNN)"
  "02-Export_ASR-Encoder-Backend.py|导出 ASR Encoder 后段 (Transformer)"
  "03-Optimize-ASR-Encoder.py|优化 ASR ONNX 模型"
  "04-Quantize-ASR-Encoder.py|ASR 编码器量化 (FP16/INT8/INT4)"
  "05-Export-ASR-Decoder-HF.py|提取 ASR Decoder 权重"
  "06-Convert-ASR-Decoder-GGUF.py|ASR Decoder 转为 GGUF 格式 (FP16)"
  "07-Quantize-ASR-Decoder-GGUF.py|ASR Decoder GGUF 二次量化 (Q4_K)"
  "11-Export-Aligner-Encoder-Frontend.py|导出 Aligner Encoder 前段 (CNN)"
  "12-Export-Aligner-Encoder-Backend.py|导出 Aligner Encoder 后段 (Transformer)"
  "13-Optimize-Aligner-Encoder.py|优化 Aligner ONNX 模型"
  "14-Quantize-Aligner-Encoder.py|Aligner 编码器量化"
  "15-Export-Aligner-Decoder-HF.py|提取 Aligner Decoder 权重"
  "16-Convert-Aligner-Decoder-GGUF.py|Aligner Decoder 转为 GGUF 格式"
  "17-Quantize-Aligner-Decoder-GGUF.py|Aligner Decoder GGUF 二次量化"
)

TOTAL=${#STEPS[@]}
START_TIME=$(date +%s)

finish() {
  local exit_code=$?
  local end_time elapsed
  end_time=$(date +%s)
  elapsed=$((end_time - START_TIME))

  if [[ $exit_code -eq 0 ]]; then
    echo
    echo "✅ 全部 ${TOTAL} 个步骤执行完成，用时 ${elapsed}s。"
  else
    echo
    echo "❌ 流程中断，退出码：${exit_code}，已用时 ${elapsed}s。" >&2
  fi

  exit "$exit_code"
}
trap finish EXIT

for index in "${!STEPS[@]}"; do
  IFS='|' read -r script description <<< "${STEPS[$index]}"
  step_no=$((index + 1))

  if [[ ! -f "$script" ]]; then
    echo "❌ 找不到脚本：$script" >&2
    exit 1
  fi

  echo
  echo "================================================================"
  printf '[%02d/%02d] %s\n' "$step_no" "$TOTAL" "$description"
  echo "命令：uv run $script"
  echo "================================================================"

  uv run "$script"
done
