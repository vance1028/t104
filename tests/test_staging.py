from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from src.staging import (
    STAGE_WAKE,
    STAGE_LIGHT,
    STAGE_DEEP,
    STAGE_REM,
    StagingConfig,
    StagingResult,
    StageSegment,
    _moving_average,
    _count_spikes,
    _initial_classify,
    _remove_short_phase,
    _smooth_short_transitions,
    _find_sleep_onset,
    _find_final_wake,
    _find_final_wake_after_sleep,
    run_staging,
    stages_to_dataframe,
)


def _make_df(n: int, hr_values, mov_values, start_hour=23):
    import datetime as _dt
    base = _dt.datetime(2026, 5, 1, start_hour, 0, 0)
    if start_hour < 12:
        base = base - _dt.timedelta(days=1)
    ts = pd.date_range(base, periods=n, freq="min")
    hr = np.array(hr_values, dtype=float)
    mv = np.array(mov_values, dtype=float)
    if len(hr) == 1:
        hr = np.full(n, float(hr_values[0]))
    if len(mv) == 1:
        mv = np.full(n, float(mov_values[0]))
    return pd.DataFrame(
        {
            "night_id": ["TST"] * n,
            "record_date": [pd.Timestamp(base + _dt.timedelta(hours=12)).date()] * n,
            "timestamp": list(ts),
            "minute_index": list(range(n)),
            "heart_rate": hr[:n],
            "movement": mv[:n],
            "spo2": [96.0] * n,
            "respiratory_rate": [16.0] * n,
            "night_type": ["test"] * n,
        }
    )


class TestMovingAverage:
    def test_flat(self):
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        out = _moving_average(arr, 3)
        assert np.allclose(out, 5.0)

    def test_window_1(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = _moving_average(arr, 1)
        assert np.allclose(out, arr)

    def test_smoothes_spike(self):
        arr = np.array([10.0, 10.0, 100.0, 10.0, 10.0])
        out = _moving_average(arr, 3)
        assert 10.0 < out[2] < 100.0


class TestSpikeDetection:
    def test_no_spikes(self):
        hr = np.full(20, 60.0)
        out = _count_spikes(hr, 5, 8.0)
        assert np.all(out == 0)

    def test_single_spike(self):
        hr = np.full(20, 60.0)
        hr[10] = 80.0
        out = _count_spikes(hr, 5, 8.0)
        assert np.max(out) >= 1


class TestShortPhaseRemoval:
    def test_short_wake_merged_to_light(self):
        stages = [STAGE_LIGHT] * 5 + [STAGE_WAKE] * 2 + [STAGE_LIGHT] * 5
        result = _remove_short_phase(stages, STAGE_WAKE, 4, STAGE_LIGHT)
        assert STAGE_WAKE not in result

    def test_long_wake_preserved(self):
        stages = [STAGE_LIGHT] * 3 + [STAGE_WAKE] * 6 + [STAGE_LIGHT] * 3
        result = _remove_short_phase(stages, STAGE_WAKE, 4, STAGE_LIGHT)
        assert result[3:9] == [STAGE_WAKE] * 6

    def test_short_deep_merged(self):
        stages = [STAGE_LIGHT] * 10 + [STAGE_DEEP] * 3 + [STAGE_LIGHT] * 10
        cfg = StagingConfig()
        result = _remove_short_phase(stages, STAGE_DEEP, cfg.min_deep_duration, STAGE_LIGHT)
        assert STAGE_DEEP not in result


class TestInitialClassify:
    def test_high_hr_is_wake(self):
        cfg = StagingConfig()
        n = 10
        hr = np.full(n, 80.0)
        mv = np.full(n, 0.1)
        sc = np.zeros(n)
        out = _initial_classify(hr, mv, sc, cfg)
        assert all(s == STAGE_WAKE for s in out)

    def test_high_mov_is_wake(self):
        cfg = StagingConfig()
        n = 10
        hr = np.full(n, 60.0)
        mv = np.full(n, 0.8)
        sc = np.zeros(n)
        out = _initial_classify(hr, mv, sc, cfg)
        assert all(s == STAGE_WAKE for s in out)

    def test_low_hr_low_mov_is_deep(self):
        cfg = StagingConfig()
        n = 10
        hr = np.full(n, 50.0)
        mv = np.full(n, 0.02)
        sc = np.zeros(n)
        out = _initial_classify(hr, mv, sc, cfg)
        assert all(s == STAGE_DEEP for s in out)


class TestSleepOnsetAndWake:
    def test_find_onset_simple(self):
        stages = (
            [STAGE_WAKE] * 10
            + [STAGE_LIGHT] * 10
            + [STAGE_DEEP] * 20
        )
        idx = _find_sleep_onset(stages, min_run=6)
        assert idx == 10

    def test_find_onset_with_prelight_too_short(self):
        stages = (
            [STAGE_WAKE] * 5
            + [STAGE_LIGHT] * 2
            + [STAGE_WAKE] * 3
            + [STAGE_LIGHT] * 15
        )
        idx = _find_sleep_onset(stages, min_run=6)
        assert idx == 10

    def test_no_onset(self):
        stages = [STAGE_WAKE] * 50
        assert _find_sleep_onset(stages, min_run=6) is None

    def test_find_final_wake_simple(self):
        stages = (
            [STAGE_LIGHT] * 20
            + [STAGE_WAKE] * 10
        )
        idx = _find_final_wake(stages, min_run=8)
        assert idx is not None
        assert stages[idx] == STAGE_WAKE

    def test_find_final_wake_after_sleep(self):
        stages = (
            [STAGE_WAKE] * 8
            + [STAGE_LIGHT] * 10
            + [STAGE_DEEP] * 10
            + [STAGE_LIGHT] * 5
            + [STAGE_WAKE] * 12
        )
        onset = 8
        idx = _find_final_wake_after_sleep(stages, onset, min_run=8)
        assert idx == 33


class TestTransientMovementNotAwake:
    """核心测试：半夜2分钟翻身体动不应判为正式清醒。"""

    def test_two_minute_turn_not_wake_after_smoothing(self):
        cfg = StagingConfig()
        cfg.smooth_window = 1
        n = 60
        hr = np.full(n, 58.0)
        mv = np.full(n, 0.03)
        mv[30:32] = 0.7
        df = _make_df(n, hr.tolist(), mv.tolist())
        result = run_staging(df, cfg=cfg)
        window = result.stages[25:40]
        wake_count = sum(1 for s in window if s == STAGE_WAKE)
        assert wake_count <= 3, f"短暂体动误判为清醒段太长: {window}"

    def test_five_minute_mov_preserved_as_wake(self):
        cfg = StagingConfig()
        cfg.smooth_window = 1
        cfg.min_wake_duration = 4
        n = 80
        hr = np.full(n, 58.0)
        mv = np.full(n, 0.03)
        mv[40:45] = 0.7
        hr[40:45] = 78.0
        df = _make_df(n, hr.tolist(), mv.tolist())
        result = run_staging(df, cfg=cfg)
        middle_window = result.stages[38:48]
        assert STAGE_WAKE in middle_window


class TestMinimumDurationSmoothing:
    def test_3min_deep_merged_into_light(self):
        cfg = StagingConfig()
        stages = (
            [STAGE_LIGHT] * 10
            + [STAGE_DEEP] * 3
            + [STAGE_LIGHT] * 10
        )
        smoothed = _smooth_short_transitions(stages, cfg)
        assert STAGE_DEEP not in smoothed

    def test_10min_deep_survives(self):
        cfg = StagingConfig()
        stages = (
            [STAGE_LIGHT] * 5
            + [STAGE_DEEP] * 10
            + [STAGE_LIGHT] * 5
        )
        smoothed = _smooth_short_transitions(stages, cfg)
        assert STAGE_DEEP in smoothed


class TestFullStagingIntegration:
    def test_simple_night_has_onset_and_wake(self):
        n = 200
        hr = np.concatenate([
            np.full(20, 78.0),
            np.full(160, 58.0),
            np.full(20, 76.0),
        ])
        mv = np.concatenate([
            np.full(20, 0.6),
            np.full(160, 0.03),
            np.full(20, 0.55),
        ])
        df = _make_df(n, hr.tolist(), mv.tolist())
        result = run_staging(df)
        assert result.sleep_onset_index is not None
        assert result.final_wake_index is not None
        assert 15 <= result.sleep_onset_index <= 30
        assert 170 <= result.final_wake_index <= 200
        assert len(result.stages) == n
        assert len(result.segments) >= 2

    def test_stages_dataframe_shape(self):
        n = 100
        hr = np.full(n, 60.0)
        mv = np.full(n, 0.05)
        df = _make_df(n, hr.tolist(), mv.tolist())
        result = run_staging(df)
        sdf = stages_to_dataframe(df, result)
        assert "stage" in sdf.columns
        assert len(sdf) == n

    def test_empty_input(self):
        df = pd.DataFrame()
        result = run_staging(df)
        assert result.stages == []
        assert result.segments == []
        assert result.sleep_onset_index is None
