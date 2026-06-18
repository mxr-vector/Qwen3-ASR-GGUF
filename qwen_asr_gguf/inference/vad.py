# coding=utf-8
"""
vad.py — 语音活动检测 (VAD) 引擎封装

基于 FireRedVAD Non-Streaming 模式，为 ASR 流水线提供分片级静音前置过滤：
- 对每个音频分片执行 VAD 检测，判断是否含有效语音
- 跳过纯静音片段，避免 ASR 模型在无语音输入时产生幻觉并浪费算力
- 接口设计与 asr.py / aligner.py 保持一致，便于统一管理
- 使用 (int16_array, sample_rate) 元组直接传递音频，无临时文件 I/O

使用示例:
    config = VADConfig(model_dir="models/FireRedVAD/VAD")
    vad = QwenVADEngine(config)

    audio = np.zeros(16000 * 30, dtype=np.float32)   # 30s 静音 (float32 PCM)
    result = vad.detect(audio)
    print(result.has_speech)   # False
"""

import os
import time
import numpy as np
from typing import Optional, List, Tuple

from .schema import VADConfig, VADResult
from core.logger import logger


class QwenVADEngine:
    """
    FireRedVAD 非流式语音活动检测引擎封装。

    职责:
      - 加载 FireRedVAD 模型
      - 将 numpy 音频片段写入临时 WAV 文件供 FireRedVAD 使用
      - 解析检测结果，返回标准化的 VADResult 数据类

    线程安全: 每次 detect() 调用使用独立临时文件，可安全用于多线程场景。
    """

    def __init__(self, config: VADConfig):
        self.config = config
        self._vad = None
        self._last_probs = None  # 缓存最近一次 detect() 的帧级概率，供自适应阈值使用
        self._load_model()

    # ──────────────────────────────────────────────────────────────────
    # 初始化
    # ──────────────────────────────────────────────────────────────────

    def _load_model(self):
        """加载 FireRedVAD 模型（延迟导入，避免无 fireredvad 时整包不可用）"""
        try:
            from fireredvad import FireRedVad, FireRedVadConfig
        except ImportError as exc:
            raise ImportError(
                "fireredvad 未安装，请执行: pip install fireredvad\n"
                "或参考 https://github.com/FireRedTeam/FireRedVAD 手动安装"
            ) from exc

        cfg = self.config
        vad_cfg = FireRedVadConfig(
            use_gpu=cfg.use_gpu,
            smooth_window_size=cfg.smooth_window_size,
            speech_threshold=cfg.speech_threshold,
            min_speech_frame=cfg.min_speech_frame,
            max_speech_frame=cfg.max_speech_frame,
            min_silence_frame=cfg.min_silence_frame,
            merge_silence_frame=cfg.merge_silence_frame,
            extend_speech_frame=cfg.extend_speech_frame,
            chunk_max_frame=cfg.chunk_max_frame,
        )

        logger.info(f"[VAD] 正在加载 FireRedVAD 模型: {cfg.model_dir}")
        t0 = time.time()
        self._vad = FireRedVad.from_pretrained(cfg.model_dir, vad_cfg)
        logger.info(
            f"[VAD] 模型加载完成，耗时 {time.time() - t0:.2f}s "
            f"(GPU: {cfg.use_gpu}, threshold: {cfg.speech_threshold})"
        )

    # ──────────────────────────────────────────────────────────────────
    # 核心检测接口
    # ──────────────────────────────────────────────────────────────────

    def detect(self, audio: np.ndarray, sr: int = 16000) -> VADResult:
        """
        对 numpy 音频数组执行语音活动检测。

        Args:
            audio: 单声道 float32 PCM 音频，采样率需与 sr 参数一致。
            sr:    采样率 (Hz)，默认 16000。

        Returns:
            VADResult: 包含 has_speech / timestamps / duration / detect_time。

        注意:
            FireRedVAD 原生接受文件路径，此方法会将 numpy 数组写入系统临时目录
            的 WAV 文件，检测完成后立即删除，不留残留文件。
        """
        if self._vad is None:
            raise RuntimeError("[VAD] 引擎未初始化，请先调用 _load_model()")

        audio_dur = len(audio) / sr
        t0 = time.time()

        # FireRedVAD 的 detect() 支持 (wav_np, sample_rate) 元组接口，
        # 完全避免临时文件 I/O，与文件路径接口结果完全一致。
        # 内部 AudioFeat 以 int16 范围处理波形，故需从 float32 PCM (-1..1) 转换。
        audio_in = np.asarray(audio, dtype=np.float32)
        if audio_in.ndim > 1:
            audio_in = audio_in.mean(axis=1)
        audio_int16 = (audio_in * 32767.0).clip(-32768, 32767).astype(np.int16)

        raw_result, _probs = self._vad.detect((audio_int16, sr))
        self._last_probs = _probs  # 缓存帧级概率，供 adaptive_detect() 使用

        detect_time = time.time() - t0

        # 解析 FireRedVAD 返回格式:
        # {'dur': float, 'timestamps': [(start_sec, end_sec), ...], 'wav_path': str}
        timestamps: List[Tuple[float, float]] = raw_result.get("timestamps", [])
        reported_dur: float = raw_result.get("dur", audio_dur)

        result = VADResult(
            has_speech=len(timestamps) > 0,
            timestamps=timestamps,
            duration=reported_dur,
            detect_time=detect_time,
            probs=np.asarray(_probs, dtype=np.float32).flatten() if _probs is not None else None,
        )

        if result.has_speech:
            segs = ", ".join(f"[{s:.2f}s~{e:.2f}s]" for s, e in timestamps)
            logger.debug(
                f"[VAD] 检测到语音 | 时长={reported_dur:.2f}s "
                f"| 语音比={result.speech_ratio:.1%} | 片段: {segs} "
                f"| 耗时={detect_time:.3f}s"
            )
        else:
            logger.debug(
                f"[VAD] 未检测到语音 | 时长={reported_dur:.2f}s "
                f"| 耗时={detect_time:.3f}s"
            )

        return result

    def has_speech(self, audio: np.ndarray, sr: int = 16000) -> bool:
        """
        快速判断音频分片是否含有语音（仅返回布尔值，忽略时间戳细节）。

        等价于 ``self.detect(audio, sr).has_speech``，适合在 ASR 热路径中
        进行简单的通过 / 跳过判断。
        """
        return self.detect(audio, sr).has_speech

    # ──────────────────────────────────────────────────────────────────
    # 自适应阈值检测
    # ──────────────────────────────────────────────────────────────────

    def adaptive_detect(self, audio: np.ndarray, sr: int = 16000) -> VADResult:
        """
        自适应阈值 VAD 检测（两遍法）。

        算法流程：
          1. 以配置阈值执行首次检测，获取帧级语音概率分布
          2. 分析概率分布，取高于噪声底的帧概率的 30% 分位数作为自适应阈值
          3. 若自适应阈值与初始值差异显著，用新阈值对帧概率重新分割语音段
          4. 否则直接返回首次检测结果

        适用场景：长音频离线转写，可更精确地适配不同录音环境的信噪比。
        """
        # 第一遍：标准检测
        result = self.detect(audio, sr)
        probs = self._last_probs

        # 无概率数据时直接返回
        if probs is None:
            return result

        try:
            probs_arr = np.asarray(probs, dtype=np.float32).flatten()
        except (ValueError, TypeError):
            return result

        if len(probs_arr) == 0:
            return result

        # 仅考虑高于噪声底 (>0.1) 的帧概率，避免静音帧拉低分位数
        speech_probs = probs_arr[probs_arr > 0.1]
        if len(speech_probs) == 0:
            return result

        initial_threshold = self.config.speech_threshold
        # 取 30% 分位数，限制在 [0.20, 0.65] 安全区间。
        # recall/balanced 模式下只允许降低阈值，避免自适应逻辑牺牲召回。
        adaptive_threshold = float(np.clip(np.percentile(speech_probs, 30), 0.20, 0.65))
        if self.config.vad_mode in {"recall", "balanced"}:
            adaptive_threshold = min(adaptive_threshold, initial_threshold)

        # recall/balanced 模式下，即使小幅降低阈值也重新分割，优先保护召回。
        # speed 模式保留旧的 0.05 抖动过滤，避免为微小差异多做处理。
        if adaptive_threshold >= initial_threshold or (
            self.config.vad_mode == "speed"
            and abs(adaptive_threshold - initial_threshold) < 0.05
        ):
            logger.debug(
                f"[VAD] 自适应阈值未带来有效降低 "
                f"({adaptive_threshold:.3f} vs {initial_threshold:.3f})，保持原结果"
            )
            return result

        logger.debug(
            f"[VAD] 自适应阈值调整: {initial_threshold:.3f} -> {adaptive_threshold:.3f}"
        )

        # 第二遍：用自适应阈值对帧概率重新分割
        audio_dur = len(audio) / sr
        timestamps = self._probs_to_timestamps(probs_arr, adaptive_threshold, audio_dur)

        if not timestamps:
            # 自适应分割无结果 → 保留首次结果
            return result

        return VADResult(
            has_speech=True,
            timestamps=timestamps,
            duration=audio_dur,
            detect_time=result.detect_time,
            probs=probs_arr,
        )

    def _probs_to_timestamps(
        self,
        probs: np.ndarray,
        threshold: float,
        audio_duration: float,
    ) -> List[Tuple[float, float]]:
        """
        基于帧级概率和阈值重新生成语音时间戳。

        实现逻辑：
          1. 滑动窗口平滑帧级概率，消除毛刺
          2. 应用阈值判定每帧是否为语音
          3. 连续语音帧合并为片段
          4. 过滤过短片段、合并近邻片段、扩展语音边界
        """
        if len(probs) == 0:
            return []

        frame_dur = audio_duration / len(probs)

        # 平滑概率
        window = max(1, self.config.smooth_window_size)
        if window > 1 and len(probs) > window:
            kernel = np.ones(window, dtype=np.float32) / window
            smoothed = np.convolve(probs, kernel, mode="same")
        else:
            smoothed = probs

        # 应用阈值 → 二值语音掩码
        is_speech = smoothed >= threshold

        # 连续语音帧 → 时间段
        segments: List[Tuple[float, float]] = []
        in_speech = False
        start_idx = 0

        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                start_idx = i
                in_speech = True
            elif not is_speech[i] and in_speech:
                segments.append((start_idx * frame_dur, i * frame_dur))
                in_speech = False

        if in_speech:
            segments.append((start_idx * frame_dur, len(probs) * frame_dur))

        # 过滤过短片段 (min_speech_frame × frame_dur)
        min_speech_sec = self.config.min_speech_frame * frame_dur
        segments = [(s, e) for s, e in segments if e - s >= min_speech_sec]

        # 合并近邻片段 (min_silence_frame × frame_dur)
        min_silence_sec = self.config.min_silence_frame * frame_dur
        merged: List[Tuple[float, float]] = []
        for s, e in segments:
            if merged and s - merged[-1][1] < min_silence_sec:
                merged.append((merged[-1][0], max(merged[-1][1], e)))
                merged.pop(-2)
            else:
                merged.append((s, e))

        # 扩展语音边界 (extend_speech_frame × frame_dur)
        extend_sec = self.config.extend_speech_frame * frame_dur
        if extend_sec > 0:
            merged = [
                (max(0.0, s - extend_sec), min(audio_duration, e + extend_sec))
                for s, e in merged
            ]

        return merged

    # ──────────────────────────────────────────────────────────────────
    # 高级工具方法
    # ──────────────────────────────────────────────────────────────────

    def build_chunks(
        self,
        timestamps: List[Tuple[float, float]],
        total_dur: float,
        max_span_sec: float = 30.0,
        merge_gap_sec: float = 1.0,
        context_pre_sec: Optional[float] = None,
        context_post_sec: Optional[float] = None,
        protect_long_gaps: bool = True,
        probs: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        sr: int = 16000,
    ) -> list:
        """
        根据 VAD 时间戳构建对齐语音边界的音频分片列表（VADChunk）。

        算法：
          1. 合并间隔 < merge_gap_sec 的相邻语音段
          2. 贪心打包：在不超过 max_span_sec 的前提下，将连续语音段组合为一个
             分片；每个分片在首段前补 context_pre_sec、末段后补 context_post_sec
          3. 在语音分片之间插入静音分片，使输出完整覆盖 0 ~ total_dur 全域
          4. 超过当前 VAD 策略安全跳过阈值的静音分片会被拆成 fallback 分片送 ASR 复核
          5. 若提供帧级概率/音频，则对 VAD 阴性区间做二次分级：高置信静音继续跳过，
             近阈值可疑区间提升为 fallback，避免只按时长在“漏话/幻觉”之间二选一。
        """
        from .schema import VADChunk

        probs_arr: Optional[np.ndarray] = None
        if probs is not None:
            try:
                probs_arr = np.asarray(probs, dtype=np.float32).flatten()
                if len(probs_arr) == 0:
                    probs_arr = None
            except (ValueError, TypeError):
                probs_arr = None

        audio_arr: Optional[np.ndarray] = None
        if audio is not None:
            try:
                audio_arr = np.asarray(audio, dtype=np.float32).flatten()
                if len(audio_arr) == 0:
                    audio_arr = None
            except (ValueError, TypeError):
                audio_arr = None

        if context_pre_sec is None:
            context_pre_sec = self.config.context_pre_sec
        if context_post_sec is None:
            context_post_sec = self.config.context_post_sec

        def _renumber(chunks: list) -> list:
            for idx, chunk in enumerate(chunks):
                chunk.idx = idx
            return chunks

        def _gap_stats(start: float, end: float) -> Tuple[float, float, float]:
            """返回 VAD 阴性区间的 (max_prob, mean_prob, rms)。"""
            max_prob = 0.0
            mean_prob = 0.0
            if probs_arr is not None and total_dur > 0:
                p_start = max(0, int(start / total_dur * len(probs_arr)))
                p_end = min(len(probs_arr), int(np.ceil(end / total_dur * len(probs_arr))))
                if p_end > p_start:
                    gap_probs = probs_arr[p_start:p_end]
                    max_prob = float(np.max(gap_probs))
                    mean_prob = float(np.mean(gap_probs))

            rms = 0.0
            if audio_arr is not None:
                s_smpl = max(0, int(start * sr))
                e_smpl = min(len(audio_arr), int(np.ceil(end * sr)))
                if e_smpl > s_smpl:
                    gap_audio = audio_arr[s_smpl:e_smpl]
                    rms = float(np.sqrt(np.mean(np.square(gap_audio))))

            return max_prob, mean_prob, rms

        def _make_silence_chunk(
            start: float,
            end: float,
            reason: str,
            prob_max: float,
            prob_mean: float,
            rms: float,
        ):
            return VADChunk(
                idx=0,
                start_sec=start,
                end_sec=end,
                has_speech=False,
                source="silence",
                skip_reason=reason,
                vad_prob_max=prob_max,
                vad_prob_mean=prob_mean,
                rms=rms,
            )

        def _make_fallback_chunk(
            start: float,
            end: float,
            prob_max: float,
            prob_mean: float,
            rms: float,
        ):
            return VADChunk(
                idx=0,
                start_sec=start,
                end_sec=end,
                has_speech=True,
                source="fallback",
                vad_prob_max=prob_max,
                vad_prob_mean=prob_mean,
                rms=rms,
            )

        def _build_silence_chunks(start: float, end: float) -> list:
            if end <= start:
                return []
            span = end - start
            vad_mode = self.config.vad_mode
            if vad_mode == "speed":
                max_safe_skip = self.config.speed_max_safe_skip_sec
            elif vad_mode == "balanced":
                max_safe_skip = self.config.balanced_max_safe_skip_sec
            else:
                max_safe_skip = self.config.max_safe_skip_sec

            prob_max, prob_mean, rms = _gap_stats(start, end)
            has_prob_or_energy = probs_arr is not None or audio_arr is not None
            high_conf_silence = (
                has_prob_or_energy
                and prob_max <= self.config.silence_prob_threshold
                and rms <= self.config.silence_rms_threshold
            )
            suspicious_gap = (
                probs_arr is not None
                and prob_max >= self.config.suspicious_prob_threshold
                and vad_mode in {"recall", "balanced"}
            )

            if high_conf_silence:
                return [
                    _make_silence_chunk(
                        start,
                        end,
                        "vad_high_conf_silence",
                        prob_max,
                        prob_mean,
                        rms,
                    )
                ]

            if not protect_long_gaps or (span <= max_safe_skip and not suspicious_gap):
                return [
                    _make_silence_chunk(
                        start,
                        end,
                        "vad_silence",
                        prob_max,
                        prob_mean,
                        rms,
                    )
                ]

            chunks = []
            cursor = start
            while cursor < end:
                chunk_end = min(cursor + max_span_sec, end)
                c_prob_max, c_prob_mean, c_rms = _gap_stats(cursor, chunk_end)
                chunks.append(
                    _make_fallback_chunk(
                        cursor,
                        chunk_end,
                        c_prob_max,
                        c_prob_mean,
                        c_rms,
                    )
                )
                cursor = chunk_end
            return chunks

        if not timestamps:
            return _renumber(_build_silence_chunks(0.0, total_dur))

        # Step 1: 合并近邻语音段
        merged: List[List[float]] = []
        for s, e in sorted(timestamps):
            s = max(0.0, min(float(s), total_dur))
            e = max(s, min(float(e), total_dur))
            if e <= s:
                continue
            if merged and s - merged[-1][1] < merge_gap_sec:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        if not merged:
            return _renumber(_build_silence_chunks(0.0, total_dur))

        # Step 2: 贪心打包 → speech_chunks = [(chunk_start, chunk_end, segs), ...]
        speech_chunks: list = []
        chunk_segs: List[Tuple[float, float]] = []
        chunk_start_sec: Optional[float] = None

        for raw_s, raw_e in merged:
            c_start = max(0.0, raw_s - context_pre_sec)
            c_end = min(total_dur, raw_e + context_post_sec)

            if chunk_start_sec is None:
                chunk_start_sec = c_start
                chunk_segs = [(raw_s, raw_e)]
            else:
                if c_end - chunk_start_sec > max_span_sec:
                    last_end = min(total_dur, chunk_segs[-1][1] + context_post_sec)
                    speech_chunks.append((chunk_start_sec, last_end, list(chunk_segs)))
                    chunk_start_sec = c_start
                    chunk_segs = [(raw_s, raw_e)]
                else:
                    chunk_segs.append((raw_s, raw_e))

        if chunk_segs:
            last_end = min(total_dur, chunk_segs[-1][1] + context_post_sec)
            speech_chunks.append((chunk_start_sec, last_end, list(chunk_segs)))

        # Step 3: 插入静音/fallback 分片，完整覆盖 [0, total_dur]
        result: list = []
        cursor = 0.0

        for cs, ce, segs in speech_chunks:
            if cs > cursor + 0.5:
                result.extend(_build_silence_chunks(cursor, cs))
            result.append(
                VADChunk(
                    idx=0,
                    start_sec=cs,
                    end_sec=ce,
                    has_speech=True,
                    speech_segments=segs,
                    source="vad",
                )
            )
            cursor = ce

        if cursor < total_dur - 0.5:
            result.extend(_build_silence_chunks(cursor, total_dur))

        return _renumber(result)

    # ──────────────────────────────────────────────────────────────────

    def get_speech_segments(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        提取音频中所有语音片段的 numpy 子数组。

        Returns:
            List of (segment_audio, start_sec, end_sec) 三元组，
            按时间顺序排列。若无语音则返回空列表。
        """
        result = self.detect(audio, sr)
        if not result.has_speech:
            return []

        segments = []
        for start_sec, end_sec in result.timestamps:
            s = int(start_sec * sr)
            e = int(end_sec * sr)
            s = max(0, s)
            e = min(len(audio), e)
            if e > s:
                segments.append((audio[s:e], start_sec, end_sec))

        return segments

    def should_run_vad(self, chunk_duration: float) -> bool:
        """
        判断当前音频分片时长是否达到启用 VAD 过滤的阈值。

        Args:
            chunk_duration: 分片时长 (秒)。

        Returns:
            True 表示应当执行 VAD；False 表示分片过短，直接送入 ASR 即可。
        """
        return chunk_duration >= self.config.vad_min_duration

    # ──────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────

    def shutdown(self):
        """释放引擎资源（当前 FireRedVAD 无显式释放接口，预留扩展用）"""
        self._vad = None
        logger.info("[VAD] 引擎已关闭")

    def __repr__(self) -> str:
        status = "ready" if self._vad is not None else "unloaded"
        return (
            f"QwenVADEngine("
            f"model='{self.config.model_dir}', "
            f"threshold={self.config.speech_threshold}, "
            f"gpu={self.config.use_gpu}, "
            f"status={status})"
        )
