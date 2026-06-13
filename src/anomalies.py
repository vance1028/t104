from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class AnomalyEvent:
    event_type: str
    start_index: int
    end_index: int
    duration_minutes: int
    severity: str
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    start_timestamp: Optional[pd.Timestamp] = None
    end_timestamp: Optional[pd.Timestamp] = None


@dataclass
class AnomalyReport:
    events: List[AnomalyEvent] = field(default_factory=list)


def _running_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=float)
    if n == 0 or window <= 1:
        return arr.copy()
    w = min(window, n)
    for i in range(n):
        lo = max(0, i - w + 1)
        out[i] = np.min(arr[lo : i + 1])
    return out


def _find_drop_runs(
    values: np.ndarray, baseline: float, min_drop: float, min_duration: int
) -> List[tuple]:
    threshold = baseline - min_drop
    mask = values <= threshold
    n = len(mask)
    runs: List[tuple] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            length = j - i
            if length >= min_duration:
                runs.append((i, j, float(np.min(values[i:j]))))
            i = j
        else:
            i += 1
    return runs


def _find_movement_runs(
    movement: np.ndarray, threshold: float, min_duration: int
) -> List[tuple]:
    mask = movement >= threshold
    n = len(mask)
    runs: List[tuple] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            length = j - i
            if length >= min_duration:
                runs.append((i, j, float(np.max(movement[i:j]))))
            i = j
        else:
            i += 1
    return runs


def detect_anomalies(night_df: pd.DataFrame) -> AnomalyReport:
    events: List[AnomalyEvent] = []
    if night_df.empty:
        return AnomalyReport(events=events)

    df_sorted = night_df.sort_values("minute_index").reset_index(drop=True)
    hr = df_sorted["heart_rate"].to_numpy(dtype=float)
    spo2 = df_sorted["spo2"].to_numpy(dtype=float)
    movement = df_sorted["movement"].to_numpy(dtype=float)
    timestamps = df_sorted["timestamp"].to_numpy()

    hr_trim = hr[max(10, len(hr) // 8) :]
    hr_baseline = float(np.median(hr_trim)) if len(hr_trim) > 0 else 65.0
    spo2_baseline = float(np.median(spo2)) if len(spo2) > 0 else 96.0

    spo2_drops = _find_drop_runs(spo2, spo2_baseline, min_drop=4.0, min_duration=3)
    for a, b, min_v in spo2_drops:
        severity = "high" if (spo2_baseline - min_v) >= 7.0 else "medium"
        events.append(
            AnomalyEvent(
                event_type="spo2_drop",
                start_index=int(a),
                end_index=int(b),
                duration_minutes=int(b - a),
                severity=severity,
                description=f"血氧从基线{spo2_baseline:.1f}下降，最低{min_v:.1f}",
                min_value=round(min_v, 1),
                start_timestamp=pd.Timestamp(timestamps[a]),
                end_timestamp=pd.Timestamp(timestamps[b - 1]),
            )
        )

    hr_drops = _find_drop_runs(hr, hr_baseline, min_drop=10.0, min_duration=4)
    for a, b, min_v in hr_drops:
        severity = "high" if (hr_baseline - min_v) >= 16.0 else "medium"
        events.append(
            AnomalyEvent(
                event_type="hr_drop",
                start_index=int(a),
                end_index=int(b),
                duration_minutes=int(b - a),
                severity=severity,
                description=f"心率从基线{hr_baseline:.1f}下降，最低{min_v:.1f}",
                min_value=round(min_v, 1),
                start_timestamp=pd.Timestamp(timestamps[a]),
                end_timestamp=pd.Timestamp(timestamps[b - 1]),
            )
        )

    mov_runs = _find_movement_runs(movement, threshold=0.45, min_duration=4)
    for a, b, max_v in mov_runs:
        severity = "high" if (b - a) >= 8 else "medium"
        events.append(
            AnomalyEvent(
                event_type="movement_burst",
                start_index=int(a),
                end_index=int(b),
                duration_minutes=int(b - a),
                severity=severity,
                description=f"持续大幅体动，最大强度{max_v:.2f}",
                max_value=round(max_v, 2),
                start_timestamp=pd.Timestamp(timestamps[a]),
                end_timestamp=pd.Timestamp(timestamps[b - 1]),
            )
        )

    events.sort(key=lambda e: e.start_index)
    return AnomalyReport(events=events)
