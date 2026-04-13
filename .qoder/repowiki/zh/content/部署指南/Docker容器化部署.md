# Docker容器化部署

<cite>
**本文档引用的文件**
- [Dockerfile](file://Dockerfile)
- [.dockerignore](file://.dockerignore)
- [run.sh](file://run.sh)
- [pyproject.toml](file://pyproject.toml)
- [infer.py](file://infer.py)
- [main.py](file://main.py)
- [core/config.py](file://core/config.py)
- [routers/transcribe.py](file://routers/transcribe.py)
- [services/asr_service.py](file://services/asr_service.py)
- [qwen_asr_gguf/inference/asr.py](file://qwen_asr_gguf/inference/asr.py)
- [qwen_asr_gguf/inference/schema.py](file://qwen_asr_gguf/inference/schema.py)
- [ref/llama.cpp/docs/docker.md](file://ref/llama.cpp/docs/docker.md)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构概览](#架构概览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考虑](#性能考虑)
8. [故障排除指南](#故障排除指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介

本指南详细介绍Qwen3-ASR GGUF项目的Docker容器化部署方案。该项目基于FastAPI构建语音识别服务，采用GGUF格式的模型进行推理，支持CPU和GPU加速。文档涵盖了从Dockerfile构建到容器运行的完整流程，包括多阶段构建、依赖安装、环境配置、GPU支持配置、Docker Compose编排以及监控和故障诊断方法。

## 项目结构

Qwen3-ASR GGUF项目采用模块化的Python项目结构，主要包含以下关键目录：

```mermaid
graph TB
subgraph "项目根目录"
A[Dockerfile] --> B[.dockerignore]
C[run.sh] --> D[main.py]
E[pyproject.toml] --> F[infer.py]
end
subgraph "核心模块"
G[core/] --> H[config.py]
I[routers/] --> J[transcribe.py]
K[services/] --> L[asr_service.py]
end
subgraph "ASR实现"
M[qwen_asr/] --> N[cli/]
O[qwen_asr_gguf/] --> P[inference/]
Q[qwen_asr_gguf/] --> R[export/]
end
subgraph "参考实现"
S[ref/llama.cpp/] --> T[docs/]
U[ref/llama.cpp/] --> V[src/]
end
```

**图表来源**
- [Dockerfile:1-66](file://Dockerfile#L1-L66)
- [pyproject.toml:1-102](file://pyproject.toml#L1-L102)

**章节来源**
- [Dockerfile:1-66](file://Dockerfile#L1-L66)
- [pyproject.toml:1-102](file://pyproject.toml#L1-L102)

## 核心组件

### Dockerfile多阶段构建

项目使用多阶段构建策略优化镜像体积和构建效率：

```mermaid
flowchart TD
A[基础镜像 python:3.11-slim] --> B[配置Debian源]
B --> C[安装系统依赖]
C --> D[配置国内PyPI源]
D --> E[安装uv包管理器]
E --> F[设置工作目录]
F --> G[复制依赖文件]
G --> H[同步uv依赖]
H --> I[安装额外依赖]
I --> J[复制项目代码]
J --> K[设置执行权限]
K --> L[配置环境变量]
L --> M[暴露端口]
M --> N[设置启动命令]
```

**图表来源**
- [Dockerfile:1-66](file://Dockerfile#L1-L66)

### 依赖管理系统

项目采用uv作为包管理器，支持多种Python发行版：

- **CPU版本**: `torch==2.10.0+cpu`
- **CUDA版本**: `torch==2.10.0+cu128`
- **Windows版本**: `torch==2.10.0+cpu`

**章节来源**
- [Dockerfile:33-51](file://Dockerfile#L33-L51)
- [pyproject.toml:28-48](file://pyproject.toml#L28-L48)

## 架构概览

Qwen3-ASR GGUF的容器化架构采用微服务设计理念：

```mermaid
graph TB
subgraph "容器层"
A[Web服务容器] --> B[ASR推理容器]
C[模型存储容器] --> D[日志容器]
end
subgraph "应用层"
E[FastAPI应用] --> F[ASR服务]
F --> G[推理引擎]
G --> H[GGUF模型]
end
subgraph "基础设施"
I[GPU设备] --> J[CUDA驱动]
K[存储卷] --> L[模型文件]
M[网络] --> N[端口映射]
end
A -.-> E
B -.-> F
C -.-> L
```

**图表来源**
- [infer.py:84-123](file://infer.py#L84-L123)
- [services/asr_service.py:34-115](file://services/asr_service.py#L34-L115)

## 详细组件分析

### Web服务入口点

应用使用FastAPI框架提供RESTful API服务：

```mermaid
sequenceDiagram
participant Client as 客户端
participant API as FastAPI应用
participant Service as ASR服务
participant Engine as 推理引擎
participant GPU as GPU设备
Client->>API : POST /asr/transcribe
API->>Service : transcribe_bytes()
Service->>Engine : transcribe()
Engine->>GPU : 执行推理
GPU-->>Engine : 返回结果
Engine-->>Service : 处理结果
Service-->>API : 标准化响应
API-->>Client : JSON结果
```

**图表来源**
- [infer.py:114-123](file://infer.py#L114-L123)
- [routers/transcribe.py:134-161](file://routers/transcribe.py#L134-L161)

### ASR服务架构

ASR服务采用线程安全的设计模式：

```mermaid
classDiagram
class ASRService {
-asyncio.Lock _lock
-QwenASREngine _engine
+initialize() void
+shutdown() void
+transcribe() TranscribeResult
+transcribe_bytes() TranscribeResult
+stream_transcribe() AsyncGenerator
-_build_engine_config() ASREngineConfig
-_save_tmp() str
}
class QwenASREngine {
+transcribe() TranscribeResult
+transcribe_stream() Generator
+shutdown() void
-_asr_core() Generator
-_decode() DecodeResult
}
class ASREngineConfig {
+model_dir : str
+use_gpu : bool
+chunk_size : float
+memory_num : int
+vad_config : VADConfig
}
ASRService --> QwenASREngine : 使用
QwenASREngine --> ASREngineConfig : 配置
```

**图表来源**
- [services/asr_service.py:34-322](file://services/asr_service.py#L34-L322)
- [qwen_asr_gguf/inference/asr.py:40-142](file://qwen_asr_gguf/inference/asr.py#L40-L142)
- [qwen_asr_gguf/inference/schema.py:162-210](file://qwen_asr_gguf/inference/schema.py#L162-L210)

**章节来源**
- [services/asr_service.py:34-322](file://services/asr_service.py#L34-L322)
- [qwen_asr_gguf/inference/asr.py:40-800](file://qwen_asr_gguf/inference/asr.py#L40-L800)

### 配置管理系统

应用支持灵活的配置管理：

```mermaid
flowchart TD
A[命令行参数] --> B[配置解析器]
B --> C[Settings类]
C --> D[环境变量]
C --> E[配置文件]
F[默认值] --> B
G[GPU检测] --> B
H[模型路径] --> C
I[上传目录] --> C
B --> J[运行时配置]
C --> J
D --> J
E --> J
```

**图表来源**
- [core/config.py:19-109](file://core/config.py#L19-L109)

**章节来源**
- [core/config.py:19-109](file://core/config.py#L19-L109)

## 依赖关系分析

### Python依赖层次

项目采用分层依赖管理策略：

```mermaid
graph TB
subgraph "核心依赖"
A[fastapi] --> B[uvicorn]
C[pydantic] --> D[pydantic-settings]
E[loguru] --> F[numpy]
end
subgraph "ASR特定依赖"
G[librosa] --> H[soundfile]
I[sentencepiece] --> J[onnxruntime]
K[gguf] --> L[transformers]
end
subgraph "可选依赖"
M[torch+cpu] --> N[torchvision+cpu]
O[torch+cu128] --> P[torchaudio+cu128]
Q[onnxruntime-gpu] --> R[onnxruntime]
end
subgraph "开发工具"
S[typer] --> T[pytest]
U[mypy] --> V[black]
end
```

**图表来源**
- [pyproject.toml:7-23](file://pyproject.toml#L7-L23)
- [pyproject.toml:28-48](file://pyproject.toml#L28-L48)

**章节来源**
- [pyproject.toml:1-102](file://pyproject.toml#L1-L102)

### Docker构建依赖

```mermaid
flowchart LR
A[python:3.11-slim] --> B[Debian源配置]
B --> C[系统工具]
C --> D[Python包管理]
D --> E[uv包管理器]
E --> F[应用依赖]
F --> G[模型文件]
G --> H[运行时环境]
```

**图表来源**
- [Dockerfile:8-51](file://Dockerfile#L8-L51)

**章节来源**
- [Dockerfile:8-51](file://Dockerfile#L8-L51)

## 性能考虑

### GPU加速配置

项目支持多种GPU后端：

| GPU后端 | Python发行版 | CUDA版本 | 适用场景 |
|---------|-------------|----------|----------|
| cu128 | torch==2.10.0+cu128 | CUDA 12.8 | NVIDIA GPU推理 |
| cpu | torch==2.10.0+cpu | 无 | CPU推理 |
| win | torch==2.10.0+cpu | DirectML | Windows GPU |

### 推理性能优化

```mermaid
flowchart TD
A[音频输入] --> B[VAD检测]
B --> C[动态分片]
C --> D[音频编码]
D --> E[LLM解码]
E --> F[后处理]
F --> G[结果输出]
H[性能优化] --> B
H --> C
H --> D
H --> E
B -.-> I[跳过静音片段]
C -.-> J[自适应分片长度]
D -.-> K[GPU内存优化]
E -.-> L[采样策略优化]
```

**图表来源**
- [qwen_asr_gguf/inference/asr.py:602-800](file://qwen_asr_gguf/inference/asr.py#L602-L800)

**章节来源**
- [qwen_asr_gguf/inference/asr.py:40-800](file://qwen_asr_gguf/inference/asr.py#L40-L800)

## 故障排除指南

### 常见问题诊断

```mermaid
flowchart TD
A[容器启动失败] --> B{错误类型}
B --> |端口占用| C[检查端口映射]
B --> |依赖缺失| D[重新构建镜像]
B --> |GPU不可用| E[检查CUDA驱动]
F[服务无响应] --> G{响应状态}
G --> |500错误| H[查看日志文件]
G --> |超时错误| I[增加超时设置]
G --> |内存不足| J[调整资源限制]
K[推理性能差] --> L{性能指标}
L --> |RTF过高| M[优化分片参数]
L --> |GPU利用率低| N[检查GPU配置]
L --> |内存泄漏| O[重启容器]
```

### 日志收集策略

应用提供多层次的日志记录机制：

- **应用日志**: `logs/app.log` - Uvicorn服务器日志
- **调试日志**: `logs/{date}/debug.log` - 详细调试信息
- **访问日志**: 自定义中间件记录请求信息
- **错误日志**: 标准错误输出

**章节来源**
- [run.sh:24-28](file://run.sh#L24-L28)
- [core/logger.py:54-72](file://core/logger.py#L54-L72)

## 结论

Qwen3-ASR GGUF的Docker容器化部署提供了完整的语音识别服务解决方案。通过多阶段构建、灵活的依赖管理和GPU加速支持，该容器化方案能够高效地部署和运行ASR服务。项目的设计充分考虑了生产环境的需求，包括性能优化、故障诊断和监控支持。

## 附录

### 容器构建命令

```bash
# 构建CPU版本镜像
docker build -t qwen3-asr-gguf:cpu .

# 构建GPU版本镜像
docker build -t qwen3-asr-gguf:gpu --build-arg GPU_TYPE=cu128 .

# 使用uv同步依赖
uv sync --extra cu128
```

### 容器运行示例

```bash
# 基础运行
docker run -d \
  --name qwen3-asr \
  -p 8001:8001 \
  -v ./models:/workspace/models \
  -v ./logs:/workspace/logs \
  qwen3-asr-gguf:cpu

# GPU运行
docker run -d \
  --name qwen3-asr-gpu \
  --gpus all \
  -p 8002:8002 \
  -v ./models:/workspace/models \
  -v ./logs:/workspace/logs \
  qwen3-asr-gguf:gpu
```

### 环境变量配置

| 环境变量 | 默认值 | 描述 |
|----------|--------|------|
| ASR_HOST | 0.0.0.0 | 服务绑定地址 |
| ASR_PORT | 8002 | 服务端口号 |
| ASR_MODEL_DIR | ./models | 模型文件目录 |
| ASR_UPLOAD_DIR | ./uploads | 上传文件目录 |
| ASR_WEB_SECRET_KEY | qwen3-asr-token | API访问密钥 |