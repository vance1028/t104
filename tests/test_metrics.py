from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as _dt
import pandas as pd
import numpy as np
import pytest

from src.staging import (
    STAGE_WAKE,
    STAGE_LIGHT,
    STAGE_DEEP,
    STAGE_REM,
    StagingResult,
    StageSegment,
)
from src.metrics import (
    SleepCycle,
    SleepMetrics,
    _count_stage_minutes,
    _count_awakenings,
    _wake_after_onset,
    _extract_cycles,
    compute_sleep_metrics,
    metrics_to_dict,
)


def _build_staging(stages, onset_idx=None, final_wake_idx=None, start_hour=23):
    base = _dt.datetime(2026, 5, 1, start_hour, 0, 0)
    if start_hour < 12:
        base = base - _dt.timedelta(days=1)
    ts = pd.date_range(base, periods=len(stages), freq="min")
    if onset_idx is None:
        for i, s in enumerate(stages):
            if s != STAGE_WAKE:
                onset_idx = i
                break
    if final_wake_idx is None:
        for i in range(len(stages) - 1, -1, -1):
            if stages[i] == STAGE_WAKE:
                final_wake_idx = i
                break
    segs = []
    i = 0
    while i < len(stages):
        j = i
        while j < len(stages) and stages[j] == stages[i]:
            j += 1
        segs.append(StageSegment(stages[i], i, j, j - i))
        i = j
    return StagingResult(
        stages=list(stages),
        segments=segs,
        sleep_onset_index=onset_idx,
        final_wake_index=final_wake_idx,
        sleep_onset_timestamp=pd.Timestamp(ts[onset_idx]) if onset_idx is not None else None,
        final_wake_timestamp=pd.Timestamp(ts[final_wake_idx]) if final_wake_idx is not None else None,
    )


def _make_night_df(n, start_hour=23):
    base = _dt.datetime(2026, 5, 1, start_hour, 0, 0)
    if start_hour < 12:
        base = base - _dt.timedelta(days=1)
    ts = pd.date_range(base, periods=n, freq="min")
    return pd.DataFrame(
        {
            "night_id": ["T"] * n,
            "record_date": [pd.Timestamp(base + _dt.timedelta(hours=12)).date()] * n,
            "timestamp": list(ts),
            "minute_index": list(range(n)),
            "heart_rate": [60.0] * n,
            "movement": [0.05] * n,
            "spo2": [96.0] * n,
            "respiratory_rate": [16.0] * n,
            "night_type": ["t"] * n,
        }
    )


class TestCountStageMinutes:
    def test_counts_wake(self):
        stages = [STAGE_WAKE, STAGE_LIGHT, STAGE_WAKE, STAGE_WAKE]
        assert _count_stage_minutes(stages, STAGE_WAKE) == 3

    def test_empty(self):
        assert _count_stage_minutes([], STAGE_WAKE) == 0


class TestAwakeningsAndWASO:
    """睡眠效率、觉醒次数、入睡后清醒时长可手算。"""

    def test_two_awakenings(self):
        # 0-10 清醒(上床)
        # 10-30 浅睡
        # 30-35 醒 (1st)
        # 35-55 深睡
        # 55-60 醒 (2nd)
        # 60-90 REM
        # 90-100 清醒(早上)
        stages = (
            [STAGE_WAKE] * 10
            + [STAGE_LIGHT] * 20
            + [STAGE_WAKE] * 5
            + [STAGE_DEEP] * 20
            + [STAGE_WAKE] * 5
            + [STAGE_REM] * 30
            + [STAGE_WAKE] * 10
        )
        staging = _build_staging(stages, onset_idx=10, final_wake_idx=90)
        count = _count_awakenings(stages, 10, 90)
        assert count == 2

    def test_waso_excludes_initial_wake(self):
        # onset=10，final_wake=90
        # 窗口内(10..90)的wake: 30-35(5分) + 55-60(5分) = 10分
        stages = (
            [STAGE_WAKE] * 10
            + [STAGE_LIGHT] * 20
            + [STAGE_WAKE] * 5
            + [STAGE_DEEP] * 20
            + [STAGE_WAKE] * 5
            + [STAGE_REM] * 30
            + [STAGE_WAKE] * 10
        )
        waso = _wake_after_onset(stages, 10, 90)
        assert waso == 10

    def test_no_awakenings(self):
        stages = [STAGE_WAKE] * 5 + [STAGE_LIGHT] * 50 + [STAGE_WAKE] * 5
        count = _count_awakenings(stages, 5, 55)
        assert count == 0
        assert _wake_after_onset(stages, 5, 55) == 0


class TestSleepEfficiency:
    def test_perfect_efficiency(self):
        stages = [STAGE_LIGHT] * 60
        staging = _build_staging(stages, onset_idx=0, final_wake_idx=59)
        df = _make_night_df(60)
        m = compute_sleep_metrics(df, staging)
        assert m.total_sleep_minutes == 60
        assert m.total_bed_minutes == 60
        assert abs(m.sleep_efficiency - 100.0) < 0.5

    def test_half_efficiency(self):
        # 60 bed minutes, 30 sleep + 30 wake
        stages = [STAGE_LIGHT] * 30 + [STAGE_WAKE] * 30
        staging = _build_staging(stages, onset_idx=0, final_wake_idx=30)
        df = _make_night_df(60)
        m = compute_sleep_metrics(df, staging)
        assert m.total_sleep_minutes == 30
        assert abs(m.sleep_efficiency - 50.0) < 0.5


class TestSleepLatency:
    def test_15_min_latency(self):
        stages = [STAGE_WAKE] * 15 + [STAGE_LIGHT] * 60 + [STAGE_WAKE] * 10
        staging = _build_staging(stages, onset_idx=15, final_wake_idx=75)
        df = _make_night_df(len(stages))
        m = compute_sleep_metrics(df, staging)
        assert m.sleep_latency_minutes == 15


class TestStageRatios:
    def test_ratios_sum_to_one(self):
        stages = (
            [STAGE_WAKE] * 10
            + [STAGE_LIGHT] * 30
            + [STAGE_DEEP] * 20
            + [STAGE_REM] * 10
            + [STAGE_WAKE] * 10
        )
        staging = _build_staging(stages)
        df = _make_night_df(len(stages))
        m = compute_sleep_metrics(df, staging)
        total = m.light_ratio + m.deep_ratio + m.rem_ratio
        assert abs(total - 1.0) < 0.01
        assert m.light_minutes == 30
        assert m.deep_minutes == 20
        assert m.rem_minutes == 10


class TestCycleExtraction:
    def test_single_cycle_no_rem(self):
        stages = [STAGE_LIGHT] * 10 + [STAGE_DEEP] * 15 + [STAGE_LIGHT] * 10
        onset = 0
        end = len(stages)
        cycles = _extract_cycles(stages, onset, end)
        assert len(cycles) == 1
        assert cycles[0].duration_minutes == 35
        assert cycles[0].has_deep is True
        assert cycles[0].has_rem is False

    def test_two_cycles_split_by_rem(self):
        # REM块中点位于 22.5 (len=5 中点为index 22+2=24)
        # 0-25: cycle 1 (包含 L-D-L-REM中点)
        # 25-35: cycle 2 余波(L)
        stages = (
            [STAGE_LIGHT] * 10
            + [STAGE_DEEP] * 10
            + [STAGE_LIGHT] * 5
            + [STAGE_REM] * 5
            + [STAGE_LIGHT] * 10
        )
        onset = 0
        end = len(stages)
        cycles = _extract_cycles(stages, onset, end)
        assert len(cycles) >= 2
        total_dur = sum(c.duration_minutes for c in cycles)
        assert total_dur == end - onset

    def test_cycles_cover_full_sleep(self):
        stages = (
            [STAGE_WAKE] * 10
            + [STAGE_LIGHT] * 20
            + [STAGE_DEEP] * 30
            + [STAGE_LIGHT] * 10
            + [STAGE_REM] * 15
            + [STAGE_LIGHT] * 15
            + [STAGE_DEEP] * 10
            + [STAGE_LIGHT] * 10
            + [STAGE_REM] * 20
            + [STAGE_WAKE] * 10
        )
        onset = 10
        final_wake = len(stages) - 10
        cycles = _extract_cycles(stages, onset, final_wake)
        assert len(cycles) >= 2
        total = sum(c.duration_minutes for c in cycles)
        assert total == final_wake - onset


class TestFullMetricsIntegration:
    def test_handcalculated_scenario(self):
        # 手算场景：
        # 卧床 120 分钟
        # 0-10: wake (latency=10)
        # 10-35: light (25)
        # 35-60: deep (25)
        # 60-70: wake → 觉醒1次, WASO +10
        # 70-95: rem (25)
        # 95-115: light (20)
        # 115-120: wake (final, 不算WASO, 用final_wake_idx=115)
        stages = (
            [STAGE_WAKE] * 10
            + [STAGE_LIGHT] * 25
            + [STAGE_DEEP] * 25
            + [STAGE_WAKE] * 10
            + [STAGE_REM] * 25
            + [STAGE_LIGHT] * 20
            + [STAGE_WAKE] * 5
        )
        staging = _build_staging(stages, onset_idx=10, final_wake_idx=115)
        df = _make_night_df(len(stages))
        m = compute_sleep_metrics(df, staging)
        assert m.total_bed_minutes == 120
        assert m.sleep_latency_minutes == 10
        assert m.light_minutes == 25 + 20
        assert m.deep_minutes == 25
        assert m.rem_minutes == 25
        assert m.total_sleep_minutes == 25 + 25 + 25 + 20  # =95
        expected_eff = 95 / 120 * 100.0
        assert abs(m.sleep_efficiency - round(expected_eff, 1)) < 0.5
        assert m.awakenings_count == 1
        # WASO = 60-70 (10分), final wake段不计
        assert m.wake_after_onset_minutes == 10

    def test_empty(self):
        df = pd.DataFrame()
        staging = StagingResult()
        m = compute_sleep_metrics(df, staging)
        assert m.total_bed_minutes == 0
        assert m.total_sleep_minutes == 0

    def test_metrics_to_dict_serializable(self):
        stages = [STAGE_LIGHT] * 30 + [STAGE_WAKE] * 10
        staging = _build_staging(stages, onset_idx=0, final_wake_idx=30)
        df = _make_night_df(len(stages))
        m = compute_sleep_metrics(df, staging)
        d = metrics_to_dict(m)
        assert isinstance(d, dict)
        assert d["total_sleep_minutes"] == 30
        assert "cycles" in d
