from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .staging import (
    STAGE_DEEP,
    STAGE_LIGHT,
    STAGE_REM,
    STAGE_WAKE,
    StageSegment,
    StagingResult,
)


@dataclass
class SleepCycle:
    start_index: int
    end_index: int
    duration_minutes: int
    rem_index: Optional[int] = None
    has_deep: bool = False
    has_rem: bool = False


@dataclass
class SleepMetrics:
    total_bed_minutes: int = 0
    total_sleep_minutes: int = 0
    sleep_efficiency: float = 0.0
    sleep_latency_minutes: int = 0
    awakenings_count: int = 0
    wake_after_onset_minutes: int = 0
    light_minutes: int = 0
    deep_minutes: int = 0
    rem_minutes: int = 0
    wake_minutes_total: int = 0
    light_ratio: float = 0.0
    deep_ratio: float = 0.0
    rem_ratio: float = 0.0
    sleep_cycles: List[SleepCycle] = field(default_factory=list)
    sleep_cycle_count: int = 0
    average_cycle_minutes: float = 0.0


def _count_stage_minutes(stages: List[str], stage: str) -> int:
    return sum(1 for s in stages if s == stage)


def _count_awakenings(stages: List[str], onset_idx: int, final_wake_idx: Optional[int]) -> int:
    if onset_idx is None:
        return 0
    end = final_wake_idx if final_wake_idx is not None else len(stages)
    window = stages[onset_idx:end]
    count = 0
    in_sleep = False
    for s in window:
        if s != STAGE_WAKE:
            in_sleep = True
        elif in_sleep:
            count += 1
            in_sleep = False
    return count


def _wake_after_onset(
    stages: List[str], onset_idx: Optional[int], final_wake_idx: Optional[int]
) -> int:
    if onset_idx is None:
        return 0
    end = final_wake_idx if final_wake_idx is not None else len(stages)
    window = stages[onset_idx:end]
    return sum(1 for s in window if s == STAGE_WAKE)


def _extract_cycles(
    stages: List[str], onset_idx: Optional[int], final_wake_idx: Optional[int]
) -> List[SleepCycle]:
    cycles: List[SleepCycle] = []
    if onset_idx is None:
        return cycles
    end = final_wake_idx if final_wake_idx is not None else len(stages)
    sleep_stages = stages[onset_idx:end]
    n = len(sleep_stages)
    if n == 0:
        return cycles

    rem_positions: List[int] = []
    i = 0
    while i < n:
        if sleep_stages[i] == STAGE_REM:
            j = i
            while j < n and sleep_stages[j] == STAGE_REM:
                j += 1
            rem_positions.append(i + (j - i) // 2)
            i = j
        else:
            i += 1

    if len(rem_positions) == 0:
        cycle = SleepCycle(
            start_index=onset_idx,
            end_index=end,
            duration_minutes=n,
            rem_index=None,
            has_deep=STAGE_DEEP in sleep_stages,
            has_rem=False,
        )
        return [cycle]

    prev_end = 0
    for rem_mid in rem_positions:
        c_start_global = onset_idx + prev_end
        c_end_global = onset_idx + rem_mid + 1
        c_slice = sleep_stages[prev_end : rem_mid + 1]
        cycles.append(
            SleepCycle(
                start_index=c_start_global,
                end_index=c_end_global,
                duration_minutes=len(c_slice),
                rem_index=onset_idx + rem_mid,
                has_deep=STAGE_DEEP in c_slice,
                has_rem=True,
            )
        )
        prev_end = rem_mid + 1

    if prev_end < n:
        c_start_global = onset_idx + prev_end
        c_end_global = end
        c_slice = sleep_stages[prev_end:n]
        cycles.append(
            SleepCycle(
                start_index=c_start_global,
                end_index=c_end_global,
                duration_minutes=len(c_slice),
                rem_index=None,
                has_deep=STAGE_DEEP in c_slice,
                has_rem=False,
            )
        )
    return cycles


def compute_sleep_metrics(
    night_df: pd.DataFrame, staging: StagingResult
) -> SleepMetrics:
    stages = staging.stages
    n = len(stages)
    if n == 0:
        return SleepMetrics()

    onset_idx = staging.sleep_onset_index
    final_wake_idx = staging.final_wake_index

    total_bed = n
    wake_total = _count_stage_minutes(stages, STAGE_WAKE)
    light = _count_stage_minutes(stages, STAGE_LIGHT)
    deep = _count_stage_minutes(stages, STAGE_DEEP)
    rem = _count_stage_minutes(stages, STAGE_REM)
    total_sleep = light + deep + rem

    efficiency = (total_sleep / total_bed * 100.0) if total_bed > 0 else 0.0

    latency = 0
    if onset_idx is not None:
        latency = onset_idx

    awakenings = _count_awakenings(stages, onset_idx, final_wake_idx)
    waso = _wake_after_onset(stages, onset_idx, final_wake_idx)

    sleep_ratio_denom = total_sleep if total_sleep > 0 else 1
    light_ratio = light / sleep_ratio_denom
    deep_ratio = deep / sleep_ratio_denom
    rem_ratio = rem / sleep_ratio_denom

    cycles = _extract_cycles(stages, onset_idx, final_wake_idx)
    cycle_count = len(cycles)
    avg_cycle = (
        sum(c.duration_minutes for c in cycles) / cycle_count if cycle_count > 0 else 0.0
    )

    return SleepMetrics(
        total_bed_minutes=total_bed,
        total_sleep_minutes=total_sleep,
        sleep_efficiency=round(efficiency, 1),
        sleep_latency_minutes=latency,
        awakenings_count=awakenings,
        wake_after_onset_minutes=waso,
        light_minutes=light,
        deep_minutes=deep,
        rem_minutes=rem,
        wake_minutes_total=wake_total,
        light_ratio=round(light_ratio, 3),
        deep_ratio=round(deep_ratio, 3),
        rem_ratio=round(rem_ratio, 3),
        sleep_cycles=cycles,
        sleep_cycle_count=cycle_count,
        average_cycle_minutes=round(avg_cycle, 1),
    )


def metrics_to_dict(metrics: SleepMetrics) -> Dict:
    d = {
        "total_bed_minutes": metrics.total_bed_minutes,
        "total_sleep_minutes": metrics.total_sleep_minutes,
        "sleep_efficiency_pct": metrics.sleep_efficiency,
        "sleep_latency_minutes": metrics.sleep_latency_minutes,
        "awakenings_count": metrics.awakenings_count,
        "wake_after_onset_minutes": metrics.wake_after_onset_minutes,
        "light_minutes": metrics.light_minutes,
        "deep_minutes": metrics.deep_minutes,
        "rem_minutes": metrics.rem_minutes,
        "wake_minutes_total": metrics.wake_minutes_total,
        "light_ratio": metrics.light_ratio,
        "deep_ratio": metrics.deep_ratio,
        "rem_ratio": metrics.rem_ratio,
        "sleep_cycle_count": metrics.sleep_cycle_count,
        "average_cycle_minutes": metrics.average_cycle_minutes,
        "cycles": [
            {
                "start_index": c.start_index,
                "end_index": c.end_index,
                "duration_minutes": c.duration_minutes,
                "has_deep": c.has_deep,
                "has_rem": c.has_rem,
            }
            for c in metrics.sleep_cycles
        ],
    }
    return d
