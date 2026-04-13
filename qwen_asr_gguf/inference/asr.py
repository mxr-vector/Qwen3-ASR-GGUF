# coding=utf-8
import os
import time
import re
import codecs
import dataclasses
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, List, Generator
from core.logger import logger
from .schema import (
    MsgType,
    StreamingMessage,
    DecodeResult,
    ASREngineConfig,
    TranscribeResult,
    ForcedAlignItem,
    ForcedAlignResult,
    StreamChunkResult,
)
from .utils import (
    normalize_language_name,
    validate_language,
    detect_and_fix_repetitions,
)
from .encoder import QwenAudioEncoder


@dataclasses.dataclass
class ASRS_Segment:
    """管理分片记忆及其物理时间坐标"""

    idx: int
    audio_start: float
    audio_end: float
    text: str = ""
    items: List[ForcedAlignItem] = None


class QwenASREngine:
    """Qwen3-ASR 转录引擎 (GGUF 后端)

    核心改进:
      - 集成 VAD 前置过滤，对静音片段直接跳过 ASR，大幅降低 RTF
      - 统一流水线 _asr_core (生成器)，同时支持一次性 asr() 和流式 asr_stream()
      - transcribe_stream() 供 SSE/WebSocket 场景实时推送逐片结果
    """

    def __init__(self, config: ASREngineConfig):
        self.config = config
        self.verbose = config.verbose
        if self.verbose:
            logger.debug(f"--- [QwenASR] 初始化引擎 (GPU: {config.use_gpu}) ---")

        from qwen_asr_gguf.inference import llama

        self.llama_mod = llama

        # 路径解析
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)
        frontend_path = os.path.join(config.model_dir, config.encoder_frontend_fn)
        backend_path = os.path.join(config.model_dir, config.encoder_backend_fn)

        # 1. 初始化 Encoder
        # 动态分片模式下，分片长度由 VAD 动态决定（通常 3~10s），
        # 使用动态形状模式以节省冗余计算；仅当明确禁用动态分片时才使用固定形状。
        use_dynamic = (
            config.dynamic_chunk_threshold is not None
            and config.dynamic_chunk_threshold < float("inf")
        )
        encoder_pad_to = None if use_dynamic else config.pad_to
        self.encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            use_gpu=config.use_gpu,
            pad_to=encoder_pad_to,
            verbose=self.verbose,
        )

        # 2. 初始化 Aligner (可选)
        self.aligner = None
        if config.enable_aligner and config.align_config:
            from .aligner import QwenForcedAligner

            self.aligner = QwenForcedAligner(config.align_config)

        # 3. VAD 延迟初始化：由 _ensure_vad() 在首次遇到长音频时按需加载
        self.vad = None

        # 4. 加载识别 LLM
        self.model = llama.LlamaModel(
            llm_gguf, n_gpu_layers=99 if config.use_gpu else 0
        )
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(
            self.model, n_ctx=config.n_ctx, n_batch=4096, embeddings=False
        )

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    # ──────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────

    def _ensure_vad(self) -> bool:
        """延迟初始化 VAD 引擎（用于长音频动态分片自动启用场景）。

        当配置中未显式启用 VAD 但音频超过动态分片阈值时，按需加载 VAD 模型。

        Returns:
            True  → VAD 就绪
            False → 初始化失败（缺少模型或依赖），调用方应降级为固定分片
        """
        if self.vad is not None:
            return True

        try:
            from .vad import QwenVADEngine

            vad_config = self.config.vad_config
            if vad_config is None:
                from .schema import VADConfig

                vad_config = VADConfig()

            self.vad = QwenVADEngine(vad_config)
            if self.verbose:
                logger.debug("--- [QwenASR] VAD 延迟加载完成（长音频动态分片） ---")
            return True
        except Exception as exc:
            logger.warning(f"[QwenASR] VAD 延迟加载失败，将使用固定分片: {exc}")
            return False

    def shutdown(self):
        if self.vad:
            self.vad.shutdown()
        if self.verbose:
            logger.debug("--- [QwenASR] 引擎已关闭 ---")

    # ──────────────────────────────────────────────────────────────────
    # Prompt 构建
    # ──────────────────────────────────────────────────────────────────

    def _build_prompt_embd(
        self,
        audio_embd: np.ndarray,
        prefix_text: str,
        context: Optional[str],
        language: Optional[str],
    ) -> np.ndarray:
        """构造用于 LLM 输入的 Embedding 序列 (区块化打包模式)

        严格遵循官方 Qwen Chat Template 格式：
        <|im_start|>system\\n{ctx}<|im_end|>\\n
        <|im_start|>user\\n<|audio_start|>...<|audio_end|><|im_end|>\\n
        <|im_start|>assistant\\nlanguage X<asr_text>{prefix}
        """

        def tk(t):
            return self.model.tokenize(t)

        # 区块 A: 音频之前 (System + User Header)
        # 关键：每个 <|im_end|> 后面必须跟 \n，匹配官方 chat template
        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = (
            [self.ID_IM_START]
            + tk(prefix_str)
            + [self.ID_IM_END]
            + tk("\n")
            + [self.ID_IM_START]
            + tk("user\n")
            + [self.ID_AUDIO_START]
        )

        # 区块 B: 音频之后 (Instruction + Assistant Header + History)
        suffix_head = "assistant\n"
        if language:
            suffix_head += f"language {language}"

        suffix_tokens = (
            [self.ID_AUDIO_END]
            + [self.ID_IM_END]
            + tk("\n")
            + [self.ID_IM_START]
            + tk(suffix_head)
            + [self.ID_ASR_TEXT]
            + tk(prefix_text)
        )

        # 拼接
        n_pre, n_aud, n_suf = (
            len(prefix_tokens),
            audio_embd.shape[0],
            len(suffix_tokens),
        )
        total_embd = np.zeros(
            (n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32
        )
        total_embd[:n_pre] = self.embedding_table[prefix_tokens]
        total_embd[n_pre : n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud :] = self.embedding_table[suffix_tokens]

        return total_embd

    # ──────────────────────────────────────────────────────────────────
    # 解码内核
    # ──────────────────────────────────────────────────────────────────

    def _decode(
        self,
        full_embd: np.ndarray,
        prefix_text: str,
        rollback_num: int,
        is_last_chunk: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> DecodeResult:
        """底层方法：执行单次 LLM 生成循环（物理推理）"""
        result = DecodeResult()

        total_len = full_embd.shape[0]

        # ── n_ctx 越界保护 ──────────────────────────────────────────
        # 如果序列长度超过上下文窗口，llama.cpp 会触发 GGML_ASSERT
        # 并调用 abort() 终止整个进程。这里提前拦截，返回空结果而非崩溃。
        n_ctx = self.config.n_ctx
        if total_len > n_ctx:
            logger.warning(
                f"[Decode] 序列长度 {total_len} 超过 n_ctx={n_ctx}，"
                f"跳过本次推理以避免进程崩溃"
            )
            result.text = ""
            result.is_aborted = True
            return result

        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate(
            [pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)]
        )
        batch = self.llama_mod.LlamaBatch(
            max(total_len * 4, 8192), self.model.n_embd, 1
        )
        batch.set_embd(full_embd, pos=pos_arr)

        # 1. Prefill
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start

        # 2. Generation Loop
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        text_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        seed = 0 if temperature == 0.0 else int(np.random.randint(0, 2**31 - 1))
        sampler = self.llama_mod.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)

        for _ in range(max_new_tokens):
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break

            if self.ctx.decode_token(last_sampled_token) != 0:
                break

            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    stable_text_acc += piece

            # 保守逃逸阀：仅在极端死循环时熔断（≥100 token 只有 ≤2 种 unique）
            # 正常重复交由解码完成后的 detect_and_fix_repetitions() 后处理
            if len(stable_tokens) >= 100:
                if len(set(stable_tokens[-100:])) <= 2:
                    result.is_aborted = True
                    break

            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1

        gen_time = time.time() - t_gen_start
        del sampler
        del batch

        # ── 后处理：刷出 display_queue 残留 token ─────────────────────────
        # 正常结束时完整刷出全部 rollback token，确保文本完整；
        # 熔断时 display_queue 中的 token 属于重复/幻觉内容，直接丢弃。
        if not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    stable_text_acc += piece
        # else: 熔断时 display_queue 中的 token 直接丢弃，不输出

        final_p = text_decoder.decode(b"", final=True)
        if final_p and not result.is_aborted:
            stable_text_acc += final_p

        result.text = stable_text_acc
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        return result

    def _safe_decode(
        self,
        full_embd: np.ndarray,
        prefix_text: str,
        rollback_num: int,
        is_last_chunk: bool,
        temperature: float,
        max_new_tokens: int = 512,
    ) -> DecodeResult:
        """带熔断加温重试的高层推理封装"""
        for i in range(3):
            res = self._decode(
                full_embd,
                prefix_text,
                rollback_num,
                is_last_chunk,
                temperature,
                max_new_tokens,
            )
            if not res.is_aborted:
                break
            temperature = min(0.6, temperature + 0.2)
            logger.warning(f"\n\n[!] 触发重试 (Temp -> {temperature:.1f})\n")

        # 后处理：使用官方去重算法修复残余重复
        res.text = detect_and_fix_repetitions(res.text)
        return res

    # ──────────────────────────────────────────────────────────────────
    # 统计打印
    # ──────────────────────────────────────────────────────────────────

    def _print_stats(self, stats: dict, audio_duration: float, t_total: float):
        """打印转录过程的性能统计指标"""
        rtf = t_total / audio_duration if audio_duration > 0 else 0
        pre_speed = (
            stats["prefill_tokens"] / stats["prefill_time"]
            if stats["prefill_time"] > 0
            else 0
        )
        gen_speed = (
            stats["decode_tokens"] / stats["decode_time"]
            if stats["decode_time"] > 0
            else 0
        )

        logger.debug(f"\n\n📊 性能统计:")
        logger.debug(f"  🔹 RTF (实时率)   : {rtf:.3f} (越小越快)")
        logger.debug(f"  🔹 音频时长       : {audio_duration:.2f} 秒")
        logger.debug(f"  🔹 总处理耗时     : {t_total:.2f} 秒")

        vad_time = stats.get("vad_time", 0.0)
        vad_skipped = stats.get("vad_skipped_chunks", 0)
        if vad_time > 0:
            logger.debug(
                f"  🔹 VAD 过滤耗时   : {vad_time:.3f} 秒 "
                f"(跳过 {vad_skipped} 个静音分片)"
            )
        if stats.get("align_time"):
            logger.debug(f"  🔹 对齐耗时       : {stats['align_time']:.3f} 秒")
        logger.debug(f"  🔹 编码耗时       : {stats['encode_time']:.3f} 秒")
        logger.debug(
            f"  🔹 LLM 预填充     : {stats['prefill_time']:.3f} 秒 "
            f"({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)"
        )
        logger.debug(
            f"  🔹 LLM 生成       : {stats['decode_time']:.3f} 秒 "
            f"({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)"
        )

    # ──────────────────────────────────────────────────────────────────
    # VAD 辅助：对单个 chunk 执行静音判断
    # ──────────────────────────────────────────────────────────────────

    def _vad_check(
        self,
        chunk_raw: np.ndarray,
        chunk_dur: float,
        chunk_idx: int,
        sr: int = 16000,
    ) -> bool:
        """
        对音频分片执行 VAD 判断。

        Returns:
            True  → 含有语音，应送入 ASR
            False → 纯静音，可安全跳过
        """
        if self.vad is None:
            return True

        if not self.vad.should_run_vad(chunk_dur):
            return True  # 分片过短，不值得运行 VAD，直接放行

        vad_result = self.vad.detect(chunk_raw, sr=sr)

        if self.verbose:
            status = (
                f"✅ 有语音 (比例 {vad_result.speech_ratio:.0%})"
                if vad_result.has_speech
                else "⏭️  静音，跳过"
            )
            logger.info(
                f"  [VAD] 分片 #{chunk_idx:02d} | {status} "
                f"| 耗时 {vad_result.detect_time:.3f}s"
            )

        return vad_result.has_speech

    # ──────────────────────────────────────────────────────────────────
    # 公共入口：transcribe (离线) & transcribe_stream (流式)
    # ──────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        start_second: float = 0.0,
        duration: Optional[float] = None,
        temperature: float = 0.0,
        rollback_num: int = 5,
        enable_aligner: bool = False,
    ) -> TranscribeResult:
        """
        离线转录入口：加载整段音频，处理完毕后返回完整 TranscribeResult。

        适合批量处理、API 离线接口等不需要实时推流的场景。
        """
        if start_second < 0:
            start_second = 0.0
        if duration is not None and duration <= 0:
            duration = None

        from .audio import load_audio

        audio = load_audio(audio_file, start_second=start_second, duration=duration)

        return self.asr(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num,
            enable_aligner=enable_aligner,
        )

    def transcribe_stream(
        self,
        audio_file: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        start_second: float = 0.0,
        duration: Optional[float] = None,
        temperature: float = 0.0,
        rollback_num: int = 5,
        enable_aligner: bool = False,
    ) -> Generator[StreamChunkResult, None, None]:
        """
        流式转录入口：逐分片 yield StreamChunkResult，调用方可实时获取结果。

        适合 SSE / WebSocket 等需要实时推送部分结果的场景。
        每次 yield 代表一个 30s 分片处理完毕：
          - skipped_by_vad=True  → VAD 判定为静音，text 为空
          - is_last=True         → 最后一个分片，full_text 含完整转写文本

        Usage::

            for chunk in engine.transcribe_stream("long.wav"):
                if not chunk.skipped_by_vad:
                    print(chunk.text, end="", flush=True)
                if chunk.is_last:
                    print("\\n完整文本:", chunk.full_text)
        """
        if start_second < 0:
            start_second = 0.0
        if duration is not None and duration <= 0:
            duration = None

        from .audio import load_audio

        audio = load_audio(audio_file, start_second=start_second, duration=duration)

        yield from self.asr_stream(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num,
            enable_aligner=enable_aligner,
        )

    # ──────────────────────────────────────────────────────────────────
    # 核心 API：asr (一次性) & asr_stream (生成器)
    # ──────────────────────────────────────────────────────────────────

    def asr(
        self,
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 30.0,
        memory_chunks: int = 1,
        temperature: float = 0.0,
        rollback_num: int = 5,
        enable_aligner: bool = False,
    ) -> TranscribeResult:
        """
        完整转录流水线（一次性版本）。

        在内部调用 _asr_core 生成器，收集所有分片结果后一次性返回。
        VAD 集成：若引擎启用了 VAD，静音片段将被跳过，不送入 ASR 模型。
        """
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []
        final_stats = None

        for chunk_res in self._asr_core(
            audio=audio,
            context=context,
            language=language,
            chunk_size_sec=chunk_size_sec,
            memory_chunks=memory_chunks,
            temperature=temperature,
            rollback_num=rollback_num,
            enable_aligner=enable_aligner,
        ):
            if not chunk_res.skipped_by_vad:
                total_full_text += chunk_res.text
            # 对齐数据通过私有属性传递
            align_items = getattr(chunk_res, "_align_items", None)
            if align_items:
                all_aligned_items.extend(align_items)
            if chunk_res.is_last:
                final_stats = getattr(chunk_res, "_stats", None)

        all_aligned_items.sort(key=lambda x: x.start_time)
        return TranscribeResult(
            text=total_full_text,
            alignment=(
                ForcedAlignResult(items=all_aligned_items)
                if all_aligned_items
                else None
            ),
            performance=final_stats,
        )

    def asr_stream(
        self,
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 30.0,
        memory_chunks: int = 1,
        temperature: float = 0.0,
        rollback_num: int = 5,
        enable_aligner: bool = False,
    ) -> Generator[StreamChunkResult, None, None]:
        """
        核心流式转录生成器（numpy 输入版本）。

        直接透传 _asr_core 的 yield 流，供 transcribe_stream() 和其他
        需要按片段实时处理的场景使用。
        """
        yield from self._asr_core(
            audio=audio,
            context=context,
            language=language,
            chunk_size_sec=chunk_size_sec,
            memory_chunks=memory_chunks,
            temperature=temperature,
            rollback_num=rollback_num,
            enable_aligner=enable_aligner,
        )

    # ──────────────────────────────────────────────────────────────────
    # 内部统一流水线核心
    # ──────────────────────────────────────────────────────────────────

    def _asr_core(
        self,
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float,
        memory_chunks: int,
        temperature: float,
        rollback_num: int,
        enable_aligner: bool = False,
    ) -> Generator[StreamChunkResult, None, None]:
        """
        统一流水线核心（生成器）。asr() 和 asr_stream() 均调用此方法。

        ┌─ VAD 动态分片模式（长音频 > dynamic_chunk_threshold）───────────┐
        │ 1. 对全段音频执行一次自适应阈值 VAD，获取语音时间戳              │
        │ 2. 按语音边界动态组合分片（不在静音中间截断，不在句中切断）       │
        │ 3. 每分片仅送入实际语音帧（trimmed + padded），消除尾部静音幻觉   │
        │ 4. LLM 记忆仅保留前 N 片的文本（不重放音频），                   │
        │    避免非连续音频拼接导致的模型混乱                              │
        │ 5. max_new_tokens 按实际语音时长等比缩放，从根本上限制幻觉空间   │
        └──────────────────────────────────────────────────────────────────┘
        ┌─ 固定分片模式（短音频 ≤ 阈值 或 VAD 不可用时降级）───────────────┐
        │ • 短音频：单一分片直接处理                                        │
        │ • 降级：保持原有 30s 等长切割 + 音频/文本双重记忆上下文           │
        └──────────────────────────────────────────────────────────────────┘
        共同抗幻觉措施（_decode 内）:
          • token 级重复熔断（15-token 窗口，≤3 种 token）
          • n-gram 短语级重复熔断（5/8-char 短语出现 ≥4 次）
          • max_new_tokens 上限（speech_sec × 12，最大 512）
        """
        # ── 语言归一化 ──────────────────────────────────────────────
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        total_duration = total_len / sr

        asr_memory = deque(maxlen=memory_chunks)  # (audio_embd_or_None, text)
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []

        stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "encode_time": 0.0,
            "align_time": 0.0,
            "vad_time": 0.0,
            "vad_skipped_chunks": 0,
        }
        t_main_start = time.time()

        # ── 选择分片策略 ──────────────────────────────────────────────
        #
        # 三级策略:
        #   1. 短音频 (≤ dynamic_chunk_threshold): 不分片，作为单一 chunk 直接处理
        #   2. 长音频 + VAD 可用: 自适应阈值 VAD 动态分片
        #   3. 长音频 + VAD 不可用 (降级): 固定等长分片
        #
        dynamic_threshold = self.config.dynamic_chunk_threshold

        if total_duration <= dynamic_threshold:
            # ── 短音频：不分片 ─────────────────────────────────────
            vad_mode = False
            all_chunks = [
                ASRS_Segment(
                    idx=0,
                    audio_start=0.0,
                    audio_end=total_duration,
                )
            ]
            if self.verbose:
                logger.debug(
                    f"[QwenASR] 短音频 ({total_duration:.1f}s "
                    f"≤ {dynamic_threshold}s)，单一分片直接处理"
                )

        elif self.vad is not None or self._ensure_vad():
            # ── 长音频：VAD 自适应动态分片 ─────────────────────────
            vad_mode = True
            from .schema import VADChunk

            t_vad = time.time()
            vad_result = self.vad.adaptive_detect(audio, sr)
            stats["vad_time"] += time.time() - t_vad

            all_chunks = self.vad.build_chunks(
                timestamps=vad_result.timestamps,
                total_dur=total_duration,
                max_span_sec=chunk_size_sec,
            )
            if self.verbose:
                n_speech = sum(1 for c in all_chunks if c.has_speech)
                n_silence = len(all_chunks) - n_speech
                logger.debug(
                    f"[VAD] 动态分片完成 | 音频 {total_duration:.1f}s "
                    f"| 耗时 {stats['vad_time']:.2f}s "
                    f"| 语音分片 {n_speech}，静音分片 {n_silence}"
                )

        else:
            # ── 降级：VAD 不可用，固定等长分片 ────────────────────
            vad_mode = False
            num_fixed = int(np.ceil(total_len / samples_per_chunk))
            all_chunks = [
                ASRS_Segment(
                    idx=i,
                    audio_start=i * chunk_size_sec,
                    audio_end=min((i + 1) * chunk_size_sec, total_duration),
                )
                for i in range(num_fixed)
            ]
            if self.verbose:
                logger.debug(f"[QwenASR] VAD 不可用，使用固定分片 ({num_fixed} 片)")

        num_chunks = len(all_chunks)
        logger.info(
            f"[ASR] 开始转写 | 音频时长={total_duration:.1f}s | 分片数={num_chunks} "
            f"| 模式={'VAD动态' if vad_mode else '固定'}"
        )

        # ── 主循环 ────────────────────────────────────────────────────
        for i, chunk_def in enumerate(all_chunks):
            is_last = i == num_chunks - 1

            # 从 chunk_def 提取时间坐标和语音标记
            if vad_mode:
                start_sec = chunk_def.start_sec
                end_sec = chunk_def.end_sec
                has_speech = chunk_def.has_speech
                speech_sec = chunk_def.speech_sec
            else:
                start_sec = chunk_def.audio_start
                end_sec = chunk_def.audio_end
                has_speech = True  # 固定模式下统一送 ASR
                speech_sec = end_sec - start_sec

            try:
                s_smpl = int(start_sec * sr)
                e_smpl = min(int(end_sec * sr), total_len)
                chunk_raw = audio[s_smpl:e_smpl]

                # ── 边界音频缓冲（仅固定分片模式）────────────────────────
                # 非末尾分片：在 chunk 尾部额外附加 1 秒音频，让编码器
                # "多听一秒"，使 LLM 能在边界处解码出完整的词句而非截断。
                # 此缓冲仅影响编码输入，不影响报告的 start_sec/end_sec。
                BOUNDARY_PAD_SEC = 1.0
                if not vad_mode and not is_last:
                    padded_end = min(int((end_sec + BOUNDARY_PAD_SEC) * sr), total_len)
                    if padded_end > e_smpl:
                        chunk_raw = audio[s_smpl:padded_end]

                # ── Step 1: 静音跳过 ──────────────────────────────────────
                if not has_speech:
                    stats["vad_skipped_chunks"] += 1
                    chunk_result = StreamChunkResult(
                        segment_idx=i,
                        text="",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        is_last=is_last,
                        skipped_by_vad=True,
                        full_text=total_full_text if is_last else "",
                    )
                    if is_last:
                        t_total = time.time() - t_main_start
                        if self.verbose:
                            self._print_stats(stats, total_duration, t_total)
                        stats["audio_duration"] = total_duration
                        setattr(chunk_result, "_stats", stats)
                        setattr(chunk_result, "_align_items", [])
                    yield chunk_result
                    continue

                if vad_mode:
                    # VAD 模式：直接按实际语音长度编码，无需补零
                    # 效果：5s 语音仅处理 5s 数据，而非 pad 到 30s 再处理
                    audio_feature, enc_time = self.encoder.encode(chunk_raw)
                else:
                    # 固定分片模式：补零至标准分片长度，保持 Encoder 固定输入尺寸
                    chunk_padded = chunk_raw
                    if len(chunk_padded) < samples_per_chunk:
                        chunk_padded = np.pad(
                            chunk_padded, (0, samples_per_chunk - len(chunk_padded))
                        )
                    audio_feature, enc_time = self.encoder.encode(chunk_padded)
                stats["encode_time"] += enc_time

                # ── Step 3: LLM 解码 ─────────────────────────────────────
                if vad_mode:
                    # VAD 模式：仅用文本上下文，不重放前片段音频。
                    # 原因：VAD 分片之间可能有大段静音（时间不连续），将非连续
                    # 音频拼接后送 LLM 会导致模型混乱，产生重复或幻觉。
                    # 限制前缀长度（末尾 40 字符）：保持语境连贯，同时防止
                    # 前缀过长时 LLM 倾向于复读历史文本。
                    prefix_text = "".join(m[1] for m in asr_memory)
                    if len(prefix_text) > 100:
                        prefix_text = prefix_text[-100:]
                    full_embd = self._build_prompt_embd(
                        audio_feature, prefix_text, context, language
                    )
                else:
                    # 固定分片模式：保留音频 + 文本双重记忆（原有行为）
                    prefix_text = "".join(m[1] for m in asr_memory)
                    combined_audio = np.concatenate(
                        [m[0] for m in asr_memory] + [audio_feature], axis=0
                    )
                    full_embd = self._build_prompt_embd(
                        combined_audio, prefix_text, context, language
                    )
                    # n_ctx 安全估算：合并记忆后超过上下文窗口时，回退为仅当前分片
                    if full_embd.shape[0] > self.config.n_ctx:
                        logger.warning(
                            f"[分片 {i}] 合并记忆后序列长度 {full_embd.shape[0]} "
                            f"超过 n_ctx={self.config.n_ctx}，回退为仅当前分片"
                        )
                        full_embd = self._build_prompt_embd(
                            audio_feature, prefix_text, context, language
                        )

                # token 预算：按实际语音时长等比缩放（12 tokens/s 上限）
                # 例：5s 语音 → 最多 60 tokens，防止在短/稀疏音频上过度生成
                max_new_tokens = min(512, max(64, int(speech_sec * 16)))

                res = self._safe_decode(
                    full_embd,
                    prefix_text,
                    rollback_num,
                    is_last,
                    temperature,
                    max_new_tokens,
                )

                # 更新记忆
                if vad_mode:
                    asr_memory.append((None, res.text))  # VAD 模式不缓存音频特征
                else:
                    asr_memory.append((audio_feature, res.text))

                total_full_text += res.text
                stats["prefill_tokens"] += res.n_prefill
                stats["prefill_time"] += res.t_prefill
                stats["decode_tokens"] += res.n_generate
                stats["decode_time"] += res.t_generate

                # ── Step 4: 对齐（可选，同步）────────────────────────────
                chunk_aligned_items: List[ForcedAlignItem] = []
                if enable_aligner and res.text.strip():
                    if self.aligner is None and self.config.align_config:
                        from .aligner import QwenForcedAligner

                        self.aligner = QwenForcedAligner(self.config.align_config)

                    if self.aligner:
                        t_align_start = time.time()
                        s_al = int(start_sec * sr)
                        e_al = int(end_sec * sr)
                        audio_slice = audio[s_al:e_al]

                        align_res = self.aligner.align(
                            audio_slice,
                            res.text,
                            language=language,
                            offset_sec=float(start_sec),
                        )
                        chunk_aligned_items = align_res.items
                        all_aligned_items.extend(align_res.items)
                        stats["align_time"] += time.time() - t_align_start

                # ── Step 5: Yield 分片结果 ───────────────────────────────
                chunk_result = StreamChunkResult(
                    segment_idx=i,
                    text=res.text,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    is_last=is_last,
                    skipped_by_vad=False,
                    full_text=total_full_text if is_last else "",
                    encode_time=enc_time,
                    decode_time=res.t_generate,
                    prefill_time=res.t_prefill,
                )
                setattr(chunk_result, "_align_items", chunk_aligned_items)
                if is_last:
                    t_total = time.time() - t_main_start
                    if self.verbose:
                        self._print_stats(stats, total_duration, t_total)
                    stats["audio_duration"] = total_duration
                    setattr(chunk_result, "_stats", stats)

                yield chunk_result

            except Exception as exc:
                logger.error(
                    f"[分片 {i}/{num_chunks}] 处理异常，跳过本分片: {exc}",
                    exc_info=True,
                )
                chunk_result = StreamChunkResult(
                    segment_idx=i,
                    text="",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    is_last=is_last,
                    skipped_by_vad=False,
                    full_text=total_full_text if is_last else "",
                )
                setattr(chunk_result, "_align_items", [])
                if is_last:
                    t_total = time.time() - t_main_start
                    stats["audio_duration"] = total_duration
                    setattr(chunk_result, "_stats", stats)
                yield chunk_result
