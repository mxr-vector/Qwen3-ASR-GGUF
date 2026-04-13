# Qwen3-ASR GGUF

将 [Qwen3-ASR](https://www.modelscope.cn/collections/Qwen/Qwen3-ASR) 模型转换为可本地高效运行的混合格式，实现**快速、准确的离线语音识别**。

主要依赖 [llama.cpp](https://github.com/ggml-org/llama.cpp) 加速 LLM Decoder。

Qwen3-ASR 0.6B 与 Qwen3-ASR 1.7B 以及 Qwen3-ForceAligner 0.6B 均可用，


### 核心特性

- ✅ **纯本地运行** - 无需网络，数据不外传
- ✅ **速度快** - 混合推理架构 (ONNX Encoder + GGUF Decoder)
- ✅ **GPU 加速** - 支持 Vulkan / CUDA / ROCm / DirectML
- ✅ **流式输出** - 无限时长的音频文件，流式转录
- ✅ **字幕输出** - ForceAligner 对齐字级时间戳，输出 SRT/JSON 格式
- ✅ **上下文增强** - 可提供上下文信息，提升准确率

## 性能表现

1.7B 在 RTX 5050 笔记本上的实测数据（50秒中文音频）：

```
(fun) PS D:\qwen3-asr> python .\transcribe.py .\test.mp3 -y
╭─────── Qwen3-ASR 配置选项 ───────╮
│  模型目录    D:\qwen3-asr\model  │
│  编码精度    int4                │
│  加速设备    GPU:ON | Vulkan:ON  │
│  时间戳对齐  启用                │
│  语言设定    自动识别            │
╰──────────────────────────────────╯
--- [QwenASR] 初始化引擎 (GPU: True) ---
--- [QwenASR] 辅助进程已就绪 ---
--- [QwenASR] 引擎初始化耗时: 3.61 秒 ---

开始处理: test.mp3

...
你怎么看待两个月之前的判断？
当初的判断不变，
美国对于委内瑞拉的突袭性质依然是政治投机，
不能算是地面战争。
入侵的美国军队总数是一两百，
站在委内瑞拉领土上的时间不超过一个小时，
算是。


📊 性能统计:
  🔹 RTF (实时率) : 0.052 (越小越快)
  🔹 音频时长    : 50.20 秒
  🔹 总处理耗时  : 2.59 秒
  🔹 编码等待    : 0.21 秒
  🔹 对齐总时    : 0.83 秒 (分段异步对齐)
  🔹 LLM 预填充  : 0.420 秒 (1742 tokens, 4149.1 tokens/s)
  🔹 LLM 生成    : 1.670 秒 (191 tokens, 114.4 tokens/s)
✅ 已保存文本文件: test.txt
✅ 已生成字幕文件: test.srt
✅ 已导出时间戳: test.json
```

CPU 的速度：

```
> python .\transcribe.py --no-gpu --no-vulkan .\test.mp3 -y
╭──────── Qwen3-ASR 配置选项 ────────╮
│  模型目录    D:\qwen3-asr\model    │
│  编码精度    int4                  │
│  加速设备    GPU:OFF | Vulkan:OFF  │
│  时间戳对齐  启用                  │
│  语言设定    自动识别              │
╰────────────────────────────────────╯
--- [QwenASR] 初始化引擎 (GPU: False) ---
--- [QwenASR] 辅助进程已就绪 ---
--- [QwenASR] 引擎初始化耗时: 2.75 秒 ---

开始处理: test.mp3

...
你怎么看待两个月之前的判断？
当初的判断不变，
美国对于委内瑞拉的突袭性质依然是政治投机，
不能算是地面战争。
入侵的美国军队总数是一两百，
站在委内瑞拉领土上的时间不超过一个小时，
算是。


📊 性能统计:
  🔹 RTF (实时率) : 0.390 (越小越快)
  🔹 音频时长    : 50.20 秒
  🔹 总处理耗时  : 19.60 秒
  🔹 编码等待    : 0.80 秒
  🔹 对齐总时    : 7.90 秒 (分段异步对齐)
  🔹 LLM 预填充  : 10.742 秒 (1741 tokens, 162.1 tokens/s)
  🔹 LLM 生成    : 7.009 秒 (190 tokens, 27.1 tokens/s)   
✅ 已保存文本文件: test.txt
✅ 已生成字幕文件: test.srt
✅ 已导出时间戳: test.json
```

## 显存占用

以 1.7B ASR 和 0.6B Aligner 载入为例，Encoder int4 量化，Decoder q4_k 量化。

开启 ONNX GPU Provider（CUDA/ROCm/DirectML）时：

- ASR Encoder     占用显存 473MB
- Aligner Encoder 占用显存 420MB

开启 Vulkan 时：

- ASR Decoder     模型占用显存 1064MB，上下文占用显存 228 MB，推理占用 304MB，总共 1.6GB
- Aligner Decoder 模型占用显存  372MB，上下文占用显存 228 MB，推理占用 299MB，总共 0.9GB

所以开启 ONNX GPU Provider 需备足 900M 显存，开启 Vulkan 需备足 2.5G 显存。


## 快速开始

### 1. 安装依赖

```bash
uv sync --extra cu128
uv pip install transformers==4.57.6 modelscope accelerate fireredvad

# 生产环境（默认）
bash run.sh start

# 开发环境
ENVIRONMENT=development uv run infer.py
```
> 
> 依赖可能写得不是那么全，缺啥就装啥呗，没有需要自己编译的

从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载预编译二进制，将动态库文件放入 `qwen_asr_gguf/bin/`：

| 平台 | 下载文件 |
|------|----------|
| **Windows** | `llama-bXXXX-bin-win-vulkan-x64.zip` |
| **Linux**| `llama-b8157-bin-ubuntu-vulkan-x64.tar.gz` |

### 2. 下载模型


#### 2.1 下载模型

到 [Models Release](https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models) 下载已经转换好的模型打包文件，下载后解压到 `models` 文件夹。

ASR 模型有 0.6B 和 1.7B 的，后者精度更高，但慢些。

Aligner 模型是 0.6B 的。

为节约显存，打包的模型：

- Encoder 全部 int4 量化，与 fp16 输出的数值余弦相似度 96%
- Decoder 全部 q4_k 量化，比 fp16 输出的困惑度仅增加 8.7%

对于语音识别，量化带来的精度差异小到可以忽略。

如果执意要用其它精度（fp32、fp16、int8）可以自行手动导出。


#### 2.2 手动导出

下载原始模型：

```bash
modelscope download --model Qwen/Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B
modelscope download --model xukaituo/FireRedVAD
```

配置 `export_config.py`，定义官方模型路径、导出路径：

```python
from pathlib import Path
model_home = Path('~/.cache/modelscope/hub/models/Qwen').expanduser()

# [源模型路径] 官方下载好的 SafeTensors 模型文件夹
ASR_MODEL_DIR =  model_home / 'Qwen3-ASR-0.6B'
ALIGNER_MODEL_DIR =  model_home / 'Qwen3-ForcedAligner-0.6B'

# [导出目标路径] 转换后的 ONNX, GGUF 和权重汇总目录
EXPORT_DIR = r'./models'

```

导出模型：

```bash
# === 1. ASR 模型导出流程 ===
uv run 01-Export-ASR-Encoder-Frontend.py     # 导出 Encoder 前段 (CNN)
uv run 02-Export_ASR-Encoder-Backend.py      # 导出 Encoder 后段 (Transformer)
uv run 03-Optimize-ASR-Encoder.py            # 优化 ONNX 模型
uv run 04-Quantize-ASR-Encoder.py            # 编码器量化 (FP16/INT8/INT4)
uv run 05-Export-ASR-Decoder-HF.py           # 提取 Decoder 权重
uv run 06-Convert-ASR-Decoder-GGUF.py        # 转为 GGUF 格式 (FP16)
uv run 07-Quantize-ASR-Decoder-GGUF.py       # GGUF 二次量化 (Q4_K)

# === 2. Aligner 模型导出流程 ===
uv run 11-Export-Aligner-Encoder-Frontend.py
uv run 12-Export-Aligner-Encoder-Backend.py
uv run 13-Optimize-Aligner-Encoder.py
uv run 14-Quantize-Aligner-Encoder.py
uv run 15-Export-Aligner-Decoder-HF.py
uv run 16-Convert-Aligner-Decoder-GGUF.py
uv run 17-Quantize-Aligner-Decoder-GGUF.py
```

### 3. 转录测试

推荐使用 `transcribe.py` 命令行工具进行转录，支持丰富的参数配置：

```bash
# 基本用法
uv run transcribe.py test.mp3

# 添加参数，如禁用 GPU
uv run transcribe.py test.mp3 --prec int4 --no-gpu --no-vulkan --n-ctx 4096
```

也可以参考 `21-Run-ASR.py` 在 Python 代码中调用：

```bash
python 21-Run-ASR.py
```

### 4. Web 服务全局配置

Web API 的 `transcribe` / `stream` 接口默认复用单例 ASR 引擎。
这意味着以下能力由**服务启动时的全局配置**决定，而不是每个请求单独传参：

- 是否启用 VAD：`ASR_ENABLE_VAD`
- VAD 阈值：`ASR_VAD_SPEECH_THRESHOLD`
- VAD 最小时长：`ASR_VAD_MIN_DURATION`
- 分片长度：`ASR_ASR_CHUNK_SIZE`
- 上下文记忆片段数：`ASR_ASR_MEMORY_NUM`
- 是否启用对齐：`ASR_ENABLE_ALIGNER`
- 默认语言：`ASR_DEFAULT_LANGUAGE`
- 默认上下文：`ASR_DEFAULT_CONTEXT`

推荐长音频场景使用如下配置：

```bash
export ASR_MODEL_DIR=./models
export ASR_ENABLE_VAD=true
export ASR_VAD_MODEL_DIR=./models/FireRedVAD/VAD
export ASR_VAD_SPEECH_THRESHOLD=0.3
export ASR_VAD_MIN_DURATION=10
export ASR_ASR_CHUNK_SIZE=30
export ASR_ASR_MEMORY_NUM=1
export ASR_ENABLE_ALIGNER=false
export ASR_DEFAULT_LANGUAGE=Chinese
export ASR_DEFAULT_CONTEXT=""
```

说明：

- `ASR_ENABLE_VAD=true` 适合长音频，可显著减少静音段推理与幻觉。
- `ASR_ENABLE_ALIGNER=false` 更适合追求吞吐；仅在需要时间戳时开启。
- 修改上述配置后需要**重启 Web 服务**，新配置才会生效。

![alt text](assets/image.png)
部分代码解析：

```python
# 配置引擎
config = ASREngineConfig(
    model_dir="models", 
    use_gpu = True, 
    encoder_frontend_fn = "qwen3_asr_encoder_frontend.int4.onnx",
    encoder_backend_fn = "qwen3_asr_encoder_backend.int4.onnx",
    enable_aligner = True, 
    align_config = AlignerConfig(
        use_gpu=True, 
        model_dir="models", 
        encoder_frontend_fn = "qwen3_aligner_encoder_frontend.int4.onnx",
        encoder_backend_fn = "qwen3_aligner_encoder_backend.int4.onnx"
    )
)

# 初始化引擎
engine = QwenASREngine(config=config)

# 执行转录
res = engine.transcribe(
    audio_file=audio_path,  
    context=context,
    language="Chinese",   # 强制指定语言 (如 'Chinese', 'English', None)
    start_second=0,       # 从何处开始读音频
    duration=None         # 读取多长音频，None 表示全部读取
)

```

项目采用**纯同步顺序执行**架构，得益于 Encoder 在开启 ONNX GPU Provider 后的极速表现（30s 音频仅需约 0.04s），现已移除复杂的多进程异步流水线，简化为：

```mermaid
graph TD
    A[音频输入] --> B[QwenASREngine]
    B -- 音频切片 --> C[QwenAudioEncoder]
    C --> D[ONNX Encoder]
    D -- 固定形状 Padding/Masking --> D
    D --> E[音频特征 Embedding]
    E --> F[llama.cpp GGUF Decoder]
    F --> G[转录文本]
    G -- 启用对齐时 --> H[QwenForcedAligner]
    H --> I[字级时间戳]
    I --> B
```

- **同步执行**: `编码 -> LLM 推理 -> 对齐` 顺序完成，代码更简洁，RTF 依然保持领先。
- **DirectML 形状固定优化（Windows）**: 推理时将音频填充（Padding）到固定长度（如 40s），并配合 Attention Mask。这解决了 DirectML 在处理动态形状时频繁分配显存导致的性能抖动，显著提升了推理速度。Linux 下 CUDA/ROCm 默认采用动态形状路径。


## 项目结构

```bash
├── 01-Export-ASR-Encoder-Frontend.py        # 导出 ASR 编码器前段 (CNN)
├── 02-Export_ASR-Encoder-Backend.py         # 导出 ASR 编码器后段 (Transformer)
├── 03-Optimize-ASR-Encoder.py               # 优化 ASR 编码器 (融合常量、折叠算子)
├── 04-Quantize-ASR-Encoder.py               # ASR 编码器量化 (INT8/FP16/INT4)
├── 05-Export-ASR-Decoder-HF.py              # 提取 ASR 解码器权重
├── 06-Convert-ASR-Decoder-GGUF.py           # ASR 解码器转为 GGUF 格式 (FP16)
├── 07-Quantize-ASR-Decoder-GGUF.py          # ASR 解码器 GGUF 量化 (Q4_K)
├── 11-Export-Aligner-Encoder-Frontend.py    # 导出对齐编码器前段
├── 12-Export-Aligner-Encoder-Backend.py     # 导出对齐编码器后段
├── 13-Optimize-Aligner-Encoder.py           # 优化对齐编码器
├── 14-Quantize-Aligner-Encoder.py           # 对齐编码器量化 (INT8/FP16/INT4)
├── 15-Export-Aligner-Decoder-HF.py          # 提取对齐解码器权重
├── 16-Convert-Aligner-Decoder-GGUF.py       # 将对齐解码器转换为 GGUF
├── 17-Quantize-Aligner-Decoder-GGUF.py      # 对齐解码器 GGUF 量化
├── 18-Run-Aligner.py                        # Aligner 对齐 API 示例脚本
├── 21-Run-ASR.py                            # ASR 转录 API 示例脚本
├── transcribe.py                            # 命令行转录工具 (功能最全)
└── qwen_asr_gguf/
    └── inference/
        ├── asr.py                  # ASR 核心引擎逻辑
        ├── aligner.py              # 强行对齐逻辑
        ├── encoder.py              # 音频特征提取逻辑 (ONNX 封装)
        ├── llama.py                # llama.cpp Python 绑定
        ├── exporters.py            # SRT/JSON/TXT 导出工具
        └── chinese_itn.py          # 中文数字规整 (ITN)
```

## Docker部署

```bash
docker build -t qwen3-asr-gguf:Dockerfile .

docker builder prune --filter "until=24h"
# 启动临时容器拷贝文件到本地
docker run -it --name qwen3-asr-gguf qwen3-asr-gguf:Dockerfile /bin/bash

# 开新终端 拷贝数据
docker cp qwen3-asr-gguf:/workspace $PWD

# 退出容器并删除临时容器
docker rm -f qwen3-asr-gguf

# 拷贝模型到本地
mv <模型地址> ./workspace/models

# 启动
docker run -it -p 8002:8002 --shm-size=8g \
-v $PWD/workspace:/workspace \
--restart=always \
--name qwen3-asr-gguf qwen3-asr-gguf:Dockerfile /bin/bash

bash run.sh start
```

## 常见问题

**Q: 输出全是乱码或「!!!!」怎么办？**

Intel 集显的 FP16 计算可能溢出，设置环境变量禁用：

```python
os.environ["GGML_VK_DISABLE_F16"] = "1"
```

---

## 致谢

- [Qwen3-ASR](https://www.modelscope.cn/collections/Qwen/Qwen3-ASR) - 原始模型
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF 推理引擎
