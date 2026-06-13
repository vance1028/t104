from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class NightType(str, Enum):
    GOOD = "good"
    ONSET_DIFFICULTY = "onset_difficulty"
    FREQUENT_AWAKENINGS = "frequent_awakenings"
    ARRHYTHMIA = "arrhythmia"


@dataclass
class NightConfig:
    night_type: NightType
    bed_hour: int
    bed_minute: int
    total_minutes: int
    seed: int


NIGHT_PRESETS: List[NightConfig] = [
    NightConfig(NightType.GOOD, 23, 0, 480, seed=1001),
    NightConfig(NightType.GOOD, 22, 45, 495, seed=1002),
    NightConfig(NightType.ONSET_DIFFICULTY, 23, 30, 510, seed=1003),
    NightConfig(NightType.FREQUENT_AWAKENINGS, 0, 15, 470, seed=1004),
    NightConfig(NightType.ARRHYTHMIA, 1, 0, 460, seed=1005),
    NightConfig(NightType.GOOD, 23, 15, 475, seed=1006),
    NightConfig(NightType.FREQUENT_AWAKENINGS, 23, 50, 490, seed=1007),
    NightConfig(NightType.ONSET_DIFFICULTY, 2, 30, 450, seed=1008),
]


def _gen_timestamps(start_dt: datetime, n_minutes: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_dt, periods=n_minutes, freq="min")


def _base_hr_curve(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n)
    base = 72 - 18 * np.sin(np.pi * t)
    noise = rng.normal(0, 2.2, n)
    return np.clip(base + noise, 42, 110)


def _base_movement(n: int, rng: np.random.Generator) -> np.ndarray:
    mov = np.zeros(n)
    n_turns = max(4, int(n / 60))
    turn_positions = rng.choice(np.arange(5, n - 5), size=n_turns, replace=False)
    for pos in turn_positions:
        length = rng.integers(2, 6)
        mov[pos : pos + length] = rng.uniform(0.4, 1.0, length)
    mov += rng.exponential(0.04, n)
    return np.clip(mov, 0.0, 1.0)


def _base_spo2(n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full(n, 96.8, dtype=float)
    base += rng.normal(0, 0.6, n)
    dippers = rng.choice(np.arange(20, n - 20), size=int(n / 90), replace=False)
    for d in dippers:
        drop_len = rng.integers(3, 9)
        base[d : d + drop_len] -= rng.uniform(1.2, 3.5, drop_len)
    return np.clip(base, 82.0, 100.0)


def _base_resp(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n)
    base = 16.5 - 2.3 * np.sin(np.pi * t)
    noise = rng.normal(0, 0.9, n)
    return np.clip(base + noise, 9.0, 26.0)


def _apply_stage_shape(values: np.ndarray, stage: str, rng: np.random.Generator) -> None:
    n = len(values)
    if n == 0:
        return
    if stage == "wake":
        values[:] += rng.normal(6, 1.8, n)
    elif stage == "rem":
        values[:] += rng.normal(-1, 2.2, n)
        spikes = rng.choice(n, size=max(1, n // 25), replace=False)
        values[spikes] += rng.uniform(5, 14, len(spikes))
    elif stage == "deep":
        values[:] += rng.normal(-9, 1.5, n)
    else:
        values[:] += rng.normal(-4, 1.6, n)


def _apply_stage_movement(mov: np.ndarray, stage: str, rng: np.random.Generator) -> None:
    n = len(mov)
    if n == 0:
        return
    if stage == "wake":
        mov[:] = np.maximum(mov, rng.uniform(0.25, 0.9, n))
    elif stage == "rem":
        pass
    elif stage == "deep":
        mov[:] = np.minimum(mov, rng.exponential(0.03, n))
    else:
        pass


def _build_good_night(n: int, rng: np.random.Generator) -> List[str]:
    stages: List[str] = []
    stages += ["wake"] * 14
    stages += ["light"] * 18
    stages += ["deep"] * 52
    stages += ["light"] * 30
    stages += ["rem"] * 18
    remaining = max(0, n - len(stages))
    cycles = 4
    per_cycle = remaining // cycles
    for _ in range(cycles):
        stages += ["light"] * max(22, int(per_cycle * 0.35))
        stages += ["deep"] * max(10, int(per_cycle * 0.20))
        stages += ["light"] * max(18, int(per_cycle * 0.25))
        stages += ["rem"] * max(14, int(per_cycle * 0.20))
    stages = stages[:n]
    if len(stages) < n:
        stages += ["wake"] * (n - len(stages))
    tail = min(12, n)
    stages[-tail:] = ["wake"] * tail
    return stages


def _build_onset_night(n: int, rng: np.random.Generator) -> List[str]:
    stages: List[str] = []
    onset_len = min(72, n // 5)
    stages += ["wake"] * onset_len
    rest_n = n - onset_len
    good_stages = _build_good_night(max(rest_n, 60), rng)
    cut = min(len(good_stages), rest_n)
    stages += good_stages[:cut]
    return stages[:n]


def _build_awakenings_night(n: int, rng: np.random.Generator) -> List[str]:
    stages = _build_good_night(n, rng)
    arr = np.array(stages)
    n_awakenings = 7
    candidates = np.arange(90, n - 45)
    if len(candidates) >= n_awakenings:
        positions = rng.choice(candidates, size=n_awakenings, replace=False)
        for pos in positions:
            length = rng.integers(5, 16)
            end = min(n, pos + length)
            arr[pos:end] = "wake"
    return arr.tolist()


def _build_arrhythmia_night(n: int, rng: np.random.Generator) -> List[str]:
    stages = _build_good_night(n, rng)
    arr = np.array(stages)
    drops = 4
    positions = rng.choice(np.arange(60, n - 60), size=drops, replace=False)
    for pos in positions:
        length = rng.integers(10, 22)
        end = min(n, pos + length)
        arr[pos:end] = "light"
    return arr.tolist()


def _arrhythmia_perturbations(
    hr: np.ndarray, spo2: np.ndarray, rng: np.random.Generator
) -> None:
    n = len(hr)
    events = 5
    positions = rng.choice(np.arange(30, n - 30), size=events, replace=False)
    for pos in positions:
        drop_len = rng.integers(6, 16)
        end = min(n, pos + drop_len)
        hr[pos:end] -= rng.uniform(6, 16, drop_len)
        spo2[pos:end] -= rng.uniform(3, 8, drop_len)


def generate_single_night(cfg: NightConfig, base_date: Optional[datetime] = None) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    if base_date is None:
        base_date = datetime(2026, 1, cfg.seed % 28 + 1, cfg.bed_hour, cfg.bed_minute)
    start_dt = datetime(
        base_date.year, base_date.month, base_date.day, cfg.bed_hour, cfg.bed_minute
    )
    if start_dt.hour < 12:
        start_dt = start_dt - timedelta(days=1)
    timestamps = _gen_timestamps(start_dt, cfg.total_minutes)

    if cfg.night_type == NightType.GOOD:
        stage_seq = _build_good_night(cfg.total_minutes, rng)
    elif cfg.night_type == NightType.ONSET_DIFFICULTY:
        stage_seq = _build_onset_night(cfg.total_minutes, rng)
    elif cfg.night_type == NightType.FREQUENT_AWAKENINGS:
        stage_seq = _build_awakenings_night(cfg.total_minutes, rng)
    elif cfg.night_type == NightType.ARRHYTHMIA:
        stage_seq = _build_arrhythmia_night(cfg.total_minutes, rng)
    else:
        stage_seq = _build_good_night(cfg.total_minutes, rng)

    hr = _base_hr_curve(cfg.total_minutes, rng)
    mov = _base_movement(cfg.total_minutes, rng)
    spo2 = _base_spo2(cfg.total_minutes, rng)
    resp = _base_resp(cfg.total_minutes, rng)

    segments: List[tuple] = []
    current = stage_seq[0]
    start_i = 0
    for i, s in enumerate(stage_seq):
        if s != current:
            segments.append((current, start_i, i))
            current = s
            start_i = i
    segments.append((current, start_i, len(stage_seq)))

    for seg_stage, a, b in segments:
        _apply_stage_shape(hr[a:b], seg_stage, rng)
        _apply_stage_movement(mov[a:b], seg_stage, rng)

    if cfg.night_type == NightType.ARRHYTHMIA:
        _arrhythmia_perturbations(hr, spo2, rng)

    hr = np.clip(hr, 38, 120)
    spo2 = np.clip(spo2, 80.0, 100.0)

    night_id = f"N{cfg.seed}_{cfg.night_type.value}"
    record_date = (start_dt + timedelta(hours=12)).date()

    df = pd.DataFrame(
        {
            "night_id": night_id,
            "record_date": record_date,
            "timestamp": timestamps,
            "minute_index": np.arange(cfg.total_minutes, dtype=int),
            "heart_rate": np.round(hr, 1),
            "movement": np.round(mov, 3),
            "spo2": np.round(spo2, 1),
            "respiratory_rate": np.round(resp, 1),
            "night_type": cfg.night_type.value,
        }
    )
    return df


def generate_all_nights(presets: Optional[List[NightConfig]] = None) -> pd.DataFrame:
    presets = presets or NIGHT_PRESETS
    frames = []
    anchor = datetime(2026, 5, 10, 0, 0)
    for i, cfg in enumerate(presets):
        base = anchor + timedelta(days=i)
        frames.append(generate_single_night(cfg, base_date=base))
    return pd.concat(frames, ignore_index=True)


def save_dataset(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


def load_or_generate(in_path: Optional[str] = None) -> pd.DataFrame:
    if in_path is not None:
        try:
            df = pd.read_csv(in_path, parse_dates=["timestamp"])
            df["record_date"] = pd.to_datetime(df["record_date"]).dt.date
            return df
        except FileNotFoundError:
            pass
    df = generate_all_nights()
    return df
