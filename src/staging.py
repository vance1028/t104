from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


STAGE_WAKE = "wake"
STAGE_LIGHT = "light"
STAGE_DEEP = "deep"
STAGE_REM = "rem"

STAGE_ORDER = [STAGE_WAKE, STAGE_LIGHT, STAGE_DEEP, STAGE_REM]

STAGE_RANK: Dict[str, int] = {
    STAGE_WAKE: 0,
    STAGE_LIGHT: 1,
    STAGE_DEEP: 2,
    STAGE_REM: 3,
}


@dataclass
class StagingConfig:
    hr_resting: float = 65.0
    hr_wake_threshold: float = 72.0
    movement_wake_threshold: float = 0.35
    movement_deep_threshold: float = 0.08
    hr_deep_bonus: float = 55.0
    hr_rem_spike_window: int = 5
    hr_rem_spike_min: float = 8.0
    hr_rem_spike_count: int = 2
    smooth_window: int = 5
    min_wake_duration: int = 4
    min_light_duration: int = 6
    min_deep_duration: int = 8
    min_rem_duration: int = 6


@dataclass
class StageSegment:
    stage: str
    start_index: int
    end_index: int
    duration_minutes: int


@dataclass
class StagingResult:
    stages: List[str] = field(default_factory=list)
    segments: List[StageSegment] = field(default_factory=list)
    sleep_onset_index: Optional[int] = None
    final_wake_index: Optional[int] = None
    sleep_onset_timestamp: Optional[pd.Timestamp] = None
    final_wake_timestamp: Optional[pd.Timestamp] = None


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) == 0:
        return arr.copy() if isinstance(arr, np.ndarray) else np.array(arr)
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr.astype(float), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _count_spikes(hr: np.ndarray, window: int, min_delta: float) -> np.ndarray:
    n = len(hr)
    counts = np.zeros(n, dtype=int)
    if n < 3:
        return counts
    diffs = np.diff(hr)
    for i in range(1, n - 1):
        if diffs[i - 1] > 0 and diffs[i] < 0:
            peak = hr[i]
            left_base = hr[max(0, i - window) : i]
            right_base = hr[i + 1 : min(n, i + 1 + window)]
            base_left = np.min(left_base) if len(left_base) else peak
            base_right = np.min(right_base) if len(right_base) else peak
            base = min(base_left, base_right)
            if peak - base >= min_delta:
                start = max(0, i - window)
                end = min(n, i + 1 + window)
                counts[start:end] += 1
    return counts


def _initial_classify(
    hr: np.ndarray,
    movement: np.ndarray,
    spike_counts: np.ndarray,
    cfg: StagingConfig,
) -> List[str]:
    n = len(hr)
    result: List[str] = []
    for i in range(n):
        h = hr[i]
        m = movement[i]
        sc = spike_counts[i]
        if h >= cfg.hr_wake_threshold or m >= cfg.movement_wake_threshold:
            result.append(STAGE_WAKE)
        elif h <= cfg.hr_deep_bonus and m <= cfg.movement_deep_threshold:
            result.append(STAGE_DEEP)
        elif sc >= cfg.hr_rem_spike_count and m < cfg.movement_wake_threshold:
            result.append(STAGE_REM)
        else:
            result.append(STAGE_LIGHT)
    return result


def _remove_short_phase(
    stages: List[str],
    stage: str,
    min_duration: int,
    replacement: str,
) -> List[str]:
    if len(stages) == 0:
        return stages
    arr = list(stages)
    i = 0
    n = len(arr)
    while i < n:
        if arr[i] == stage:
            j = i
            while j < n and arr[j] == stage:
                j += 1
            length = j - i
            if length < min_duration:
                for k in range(i, j):
                    arr[k] = replacement
            i = j
        else:
            i += 1
    return arr


def _find_dominant_among(stages: List[str], candidates: List[str]) -> str:
    counts: Dict[str, int] = {s: 0 for s in candidates}
    for s in stages:
        if s in counts:
            counts[s] += 1
    best = candidates[0]
    best_count = -1
    for s in candidates:
        if counts[s] > best_count:
            best_count = counts[s]
            best = s
    return best


def _smooth_short_transitions(stages: List[str], cfg: StagingConfig) -> List[str]:
    if len(stages) == 0:
        return stages
    arr = list(stages)
    for _ in range(2):
        arr = _remove_short_phase(arr, STAGE_WAKE, cfg.min_wake_duration, STAGE_LIGHT)
        arr = _remove_short_phase(arr, STAGE_DEEP, cfg.min_deep_duration, STAGE_LIGHT)
        arr = _remove_short_phase(arr, STAGE_REM, cfg.min_rem_duration, STAGE_LIGHT)
        arr = _remove_short_phase(arr, STAGE_LIGHT, cfg.min_light_duration, STAGE_DEEP)
    return arr


def _head_tail_unified(stages: List[str], window: int = 8) -> List[str]:
    if len(stages) == 0:
        return stages
    arr = list(stages)
    head_len = min(window, len(arr) // 6)
    if head_len > 0 and arr[0] == STAGE_WAKE:
        for i in range(head_len):
            if arr[i] in (STAGE_LIGHT, STAGE_DEEP, STAGE_REM):
                break
            arr[i] = STAGE_WAKE
    tail_len = min(window, len(arr) // 6)
    if tail_len > 0 and arr[-1] == STAGE_WAKE:
        for i in range(1, tail_len + 1):
            if arr[-i] in (STAGE_LIGHT, STAGE_DEEP, STAGE_REM):
                break
            arr[-i] = STAGE_WAKE
    return arr


def _to_segments(stages: List[str]) -> List[StageSegment]:
    segments: List[StageSegment] = []
    if len(stages) == 0:
        return segments
    current = stages[0]
    start = 0
    for i, s in enumerate(stages):
        if s != current:
            segments.append(StageSegment(current, start, i, i - start))
            current = s
            start = i
    segments.append(StageSegment(current, start, len(stages), len(stages) - start))
    return segments


def _find_sleep_onset(stages: List[str], min_run: int = 6) -> Optional[int]:
    n = len(stages)
    for i in range(n):
        if stages[i] != STAGE_WAKE:
            run = 1
            j = i + 1
            while j < n and stages[j] != STAGE_WAKE:
                run += 1
                j += 1
            if run >= min_run:
                return i
    return None


def _find_final_wake(stages: List[str], min_run: int = 8) -> Optional[int]:
    n = len(stages)
    for i in range(n - 1, -1, -1):
        if stages[i] == STAGE_WAKE:
            run = 1
            j = i - 1
            while j >= 0 and stages[j] == STAGE_WAKE:
                run += 1
                j -= 1
            if run >= min_run:
                return i
    return None


def _find_final_wake_after_sleep(
    stages: List[str], onset_index: int, min_run: int = 8
) -> Optional[int]:
    n = len(stages)
    wake_run_start: Optional[int] = None
    for i in range(onset_index, n):
        if stages[i] == STAGE_WAKE:
            if wake_run_start is None:
                wake_run_start = i
            run_length = i - wake_run_start + 1
            if run_length >= min_run:
                return wake_run_start
        else:
            wake_run_start = None
    if wake_run_start is not None:
        run_length = n - wake_run_start
        if run_length >= max(1, min_run // 2):
            return wake_run_start
    return None


def run_staging(
    night_df: pd.DataFrame,
    cfg: Optional[StagingConfig] = None,
) -> StagingResult:
    cfg = cfg or StagingConfig()
    if night_df.empty:
        return StagingResult()

    hr = night_df["heart_rate"].to_numpy(dtype=float)
    movement = night_df["movement"].to_numpy(dtype=float)

    hr_smooth = _moving_average(hr, cfg.smooth_window)
    mov_smooth = _moving_average(movement, max(1, cfg.smooth_window // 2 + 1))
    spikes = _count_spikes(hr_smooth, cfg.hr_rem_spike_window, cfg.hr_rem_spike_min)

    raw_stages = _initial_classify(hr_smooth, mov_smooth, spikes, cfg)
    smoothed = _smooth_short_transitions(raw_stages, cfg)
    smoothed = _head_tail_unified(smoothed)

    segments = _to_segments(smoothed)
    onset_idx = _find_sleep_onset(smoothed, min_run=cfg.min_light_duration)

    final_wake_idx: Optional[int] = None
    if onset_idx is not None:
        final_wake_idx = _find_final_wake_after_sleep(
            smoothed, onset_idx, min_run=cfg.min_wake_duration + 2
        )
    if final_wake_idx is None:
        final_wake_idx = _find_final_wake(smoothed, min_run=max(2, cfg.min_wake_duration))

    ts = night_df["timestamp"].to_numpy()
    result = StagingResult(
        stages=smoothed,
        segments=segments,
        sleep_onset_index=onset_idx,
        final_wake_index=final_wake_idx,
        sleep_onset_timestamp=pd.Timestamp(ts[onset_idx]) if onset_idx is not None else None,
        final_wake_timestamp=pd.Timestamp(ts[final_wake_idx]) if final_wake_idx is not None else None,
    )
    return result


def stages_to_dataframe(
    night_df: pd.DataFrame, staging_result: StagingResult
) -> pd.DataFrame:
    df = night_df.copy().reset_index(drop=True)
    df["stage"] = staging_result.stages
    return df
