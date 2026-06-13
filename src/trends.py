from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CIRCULAR_DAY_MINUTES = 24 * 60


@dataclass
class NightSummary:
    night_id: str
    record_date: object
    bed_timestamp: Optional[pd.Timestamp] = None
    sleep_onset_timestamp: Optional[pd.Timestamp] = None
    final_wake_timestamp: Optional[pd.Timestamp] = None
    total_sleep_minutes: int = 0
    sleep_efficiency: float = 0.0
    sleep_latency_minutes: int = 0
    bed_minute_of_day: Optional[int] = None
    onset_minute_of_day: Optional[int] = None


@dataclass
class CircularStats:
    mean_minute_of_day: Optional[int]
    mean_clock_str: str
    resultant_length: float
    circular_std_minutes: float
    regularity_score_0_100: float


@dataclass
class TrendAnalysis:
    nights: List[NightSummary] = field(default_factory=list)
    bed_time_circular: Optional[CircularStats] = None
    onset_time_circular: Optional[CircularStats] = None
    sleep_debt_minutes: int = 0
    cumulative_sleep_minutes: int = 0
    cumulative_target_minutes: int = 0
    average_sleep_minutes: float = 0.0
    target_sleep_per_night_minutes: int = 480


def _minute_of_day_from_ts(ts: Optional[pd.Timestamp]) -> Optional[int]:
    if ts is None:
        return None
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    return ts.hour * 60 + ts.minute


def _minutes_to_circular_xy(minutes_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(minutes_list, dtype=float)
    angles = 2 * math.pi * arr / CIRCULAR_DAY_MINUTES
    x = np.cos(angles)
    y = np.sin(angles)
    return x, y


def circular_mean_and_dispersion(
    minutes_list: List[int],
) -> Tuple[Optional[int], float, float]:
    if not minutes_list:
        return None, 0.0, 0.0
    x, y = _minutes_to_circular_xy(minutes_list)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    R = math.sqrt(mean_x * mean_x + mean_y * mean_y)
    if R < 1e-9:
        return None, R, float("inf")
    mean_angle = math.atan2(mean_y, mean_x)
    if mean_angle < 0:
        mean_angle += 2 * math.pi
    mean_min = int(round(mean_angle * CIRCULAR_DAY_MINUTES / (2 * math.pi))) % CIRCULAR_DAY_MINUTES
    if R > 1.0:
        R = 1.0
    if R > 0.999999:
        circ_std = 0.0
    else:
        circ_std = math.sqrt(-2.0 * math.log(R))
        circ_std = circ_std * CIRCULAR_DAY_MINUTES / (2 * math.pi)
    return mean_min, R, circ_std


def _regularity_score_from_r(R: float, std_minutes: float) -> float:
    if math.isinf(std_minutes):
        return 0.0
    r_component = max(0.0, min(1.0, R))
    std_bounded = min(std_minutes, 240.0)
    std_component = 1.0 - (std_bounded / 240.0)
    score = 60.0 * r_component + 40.0 * std_component
    return round(max(0.0, min(100.0, score)), 1)


def compute_circular_stats(minutes_list: List[int]) -> CircularStats:
    mean_min, R, std_min = circular_mean_and_dispersion(minutes_list)
    if mean_min is None:
        clock_str = "--:--"
    else:
        h = mean_min // 60
        m = mean_min % 60
        clock_str = f"{h:02d}:{m:02d}"
    score = _regularity_score_from_r(R, std_min)
    return CircularStats(
        mean_minute_of_day=mean_min,
        mean_clock_str=clock_str,
        resultant_length=round(R, 4),
        circular_std_minutes=round(std_min, 1),
        regularity_score_0_100=score,
    )


def _shift_to_reference_window(
    minute_list: List[int], reference_hour: int = 12
) -> List[int]:
    threshold = reference_hour * 60
    shifted: List[int] = []
    for m in minute_list:
        if m < threshold:
            shifted.append(m + CIRCULAR_DAY_MINUTES)
        else:
            shifted.append(m)
    return shifted


def _circular_paired_diffs_minutes(minutes_list: List[int]) -> List[float]:
    if len(minutes_list) < 2:
        return []
    diffs: List[float] = []
    for i in range(1, len(minutes_list)):
        a = minutes_list[i - 1]
        b = minutes_list[i]
        raw = b - a
        while raw <= -CIRCULAR_DAY_MINUTES / 2:
            raw += CIRCULAR_DAY_MINUTES
        while raw > CIRCULAR_DAY_MINUTES / 2:
            raw -= CIRCULAR_DAY_MINUTES
        diffs.append(abs(raw))
    return diffs


def build_night_summaries(
    all_data: pd.DataFrame,
    per_night_staging: Dict[str, object],
    per_night_metrics: Dict[str, object],
) -> List[NightSummary]:
    summaries: List[NightSummary] = []
    for night_id, grp in all_data.groupby("night_id", sort=False):
        grp_sorted = grp.sort_values("minute_index").reset_index(drop=True)
        bed_ts = pd.Timestamp(grp_sorted["timestamp"].iloc[0])
        record_date = grp_sorted["record_date"].iloc[0]
        staging = per_night_staging.get(night_id)
        metrics = per_night_metrics.get(night_id)
        onset_ts = getattr(staging, "sleep_onset_timestamp", None)
        wake_ts = getattr(staging, "final_wake_timestamp", None)
        sleep_min = getattr(metrics, "total_sleep_minutes", 0)
        efficiency = getattr(metrics, "sleep_efficiency", 0.0)
        latency = getattr(metrics, "sleep_latency_minutes", 0)
        summaries.append(
            NightSummary(
                night_id=night_id,
                record_date=record_date,
                bed_timestamp=bed_ts,
                sleep_onset_timestamp=onset_ts,
                final_wake_timestamp=wake_ts,
                total_sleep_minutes=int(sleep_min),
                sleep_efficiency=float(efficiency),
                sleep_latency_minutes=int(latency),
                bed_minute_of_day=_minute_of_day_from_ts(bed_ts),
                onset_minute_of_day=_minute_of_day_from_ts(onset_ts),
            )
        )
    summaries.sort(key=lambda s: (s.record_date, s.night_id))
    return summaries


def compute_sleep_debt(
    summaries: List[NightSummary], target_minutes_per_night: int = 480
) -> Tuple[int, int, int]:
    if not summaries:
        return 0, 0, 0
    cum_sleep = 0
    cum_target = 0
    for s in summaries:
        cum_sleep += s.total_sleep_minutes
        cum_target += target_minutes_per_night
    debt = cum_target - cum_sleep
    return debt, cum_sleep, cum_target


def analyze_trends(
    summaries: List[NightSummary],
    target_sleep_minutes: int = 480,
) -> TrendAnalysis:
    bed_minutes = [
        s.bed_minute_of_day for s in summaries if s.bed_minute_of_day is not None
    ]
    onset_minutes = [
        s.onset_minute_of_day for s in summaries if s.onset_minute_of_day is not None
    ]

    bed_stats = compute_circular_stats(bed_minutes) if bed_minutes else None
    onset_stats = compute_circular_stats(onset_minutes) if onset_minutes else None

    debt, cum_sleep, cum_target = compute_sleep_debt(summaries, target_sleep_minutes)
    avg_sleep = (cum_sleep / len(summaries)) if summaries else 0.0

    return TrendAnalysis(
        nights=summaries,
        bed_time_circular=bed_stats,
        onset_time_circular=onset_stats,
        sleep_debt_minutes=debt,
        cumulative_sleep_minutes=cum_sleep,
        cumulative_target_minutes=cum_target,
        average_sleep_minutes=round(avg_sleep, 1),
        target_sleep_per_night_minutes=target_sleep_minutes,
    )


def format_minutes_clock(total_minutes: int) -> str:
    sign = "-" if total_minutes < 0 else ""
    abs_m = abs(int(total_minutes))
    h = abs_m // 60
    m = abs_m % 60
    return f"{sign}{h}h {m:02d}m"
