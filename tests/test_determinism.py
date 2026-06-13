from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as _dt
import copy
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
    run_staging,
)
from src.metrics import compute_sleep_metrics, metrics_to_dict
from src.data_generator import (
    NightConfig,
    NightType,
    generate_single_night,
    generate_all_nights,
)
from src.trends import (
    NightSummary,
    circular_mean_and_dispersion,
    compute_sleep_debt,
    compute_circular_stats,
)


def _make_controlled_df():
    """造一个完全受控、不依赖随机的数据，确保每次相同。"""
    n = 160
    base = _dt.datetime(2026, 5, 10, 23, 0, 0)
    ts = pd.date_range(base, periods=n, freq="min")
    hr = np.concatenate([
        np.full(20, 78.0),
        np.full(20, 64.0),
        np.full(40, 50.0),
        np.full(30, 60.0),
        np.full(30, 62.0),
        np.full(20, 77.0),
    ])
    mov = np.concatenate([
        np.full(20, 0.55),
        np.full(20, 0.10),
        np.full(40, 0.02),
        np.full(30, 0.12),
        np.full(30, 0.15),
        np.full(20, 0.50),
    ])
    return pd.DataFrame(
        {
            "night_id": ["DET"] * n,
            "record_date": [pd.Timestamp(base + _dt.timedelta(hours=12)).date()] * n,
            "timestamp": list(ts),
            "minute_index": list(range(n)),
            "heart_rate": hr,
            "movement": mov,
            "spo2": [96.5] * n,
            "respiratory_rate": [16.0] * n,
            "night_type": ["test"] * n,
        }
    )


class TestStagingDeterminism:
    def test_staging_same_input_10_runs(self):
        df = _make_controlled_df()
        cfg = StagingConfig()
        first = run_staging(df, cfg=cfg)
        for _ in range(9):
            r = run_staging(df, cfg=cfg)
            assert r.stages == first.stages
            assert len(r.segments) == len(first.segments)
            for a, b in zip(r.segments, first.segments):
                assert (a.stage, a.start_index, a.end_index, a.duration_minutes) == (
                    b.stage, b.start_index, b.end_index, b.duration_minutes
                )
            assert r.sleep_onset_index == first.sleep_onset_index
            assert r.final_wake_index == first.final_wake_index

    def test_staging_config_independent_copy(self):
        df = _make_controlled_df()
        cfg1 = StagingConfig()
        cfg2 = copy.deepcopy(cfg1)
        r1 = run_staging(df, cfg=cfg1)
        r2 = run_staging(df, cfg=cfg2)
        assert r1.stages == r2.stages


class TestMetricsDeterminism:
    def test_metrics_same_staging_10_runs(self):
        df = _make_controlled_df()
        staging = run_staging(df)
        m1 = compute_sleep_metrics(df, staging)
        d1 = metrics_to_dict(m1)
        for _ in range(9):
            m = compute_sleep_metrics(df, staging)
            d = metrics_to_dict(m)
            assert d == d1

    def test_cycle_extraction_repeatable(self):
        df = _make_controlled_df()
        staging = run_staging(df)
        cycles_1 = [
            (c.start_index, c.end_index, c.duration_minutes, c.has_deep, c.has_rem)
            for c in compute_sleep_metrics(df, staging).sleep_cycles
        ]
        for _ in range(5):
            cycles_k = [
                (c.start_index, c.end_index, c.duration_minutes, c.has_deep, c.has_rem)
                for c in compute_sleep_metrics(df, staging).sleep_cycles
            ]
            assert cycles_k == cycles_1


class TestTrendDeterminism:
    def test_circular_mean_repeatable(self):
        minutes = [23 * 60, 30, 0, 22 * 60 + 45, 1 * 60 + 15]
        mean_1, R_1, std_1 = circular_mean_and_dispersion(minutes)
        for _ in range(10):
            m, R, s = circular_mean_and_dispersion(minutes)
            assert m == mean_1
            assert abs(R - R_1) < 1e-9
            assert abs(s - std_1) < 1e-6

    def test_circular_stats_object_repeatable(self):
        minutes = [20 * 60, 21 * 60, 23 * 60 + 30, 0, 30]
        s1 = compute_circular_stats(minutes)
        for _ in range(5):
            s = compute_circular_stats(minutes)
            assert s.mean_minute_of_day == s1.mean_minute_of_day
            assert s.mean_clock_str == s1.mean_clock_str
            assert abs(s.resultant_length - s1.resultant_length) < 1e-9
            assert abs(s.circular_std_minutes - s1.circular_std_minutes) < 1e-6
            assert abs(s.regularity_score_0_100 - s1.regularity_score_0_100) < 1e-6

    def test_sleep_debt_repeatable(self):
        s = [
            NightSummary(night_id=str(i), record_date=_dt.date(2026, 5, i + 1), total_sleep_minutes=420 + i * 10)
            for i in range(4)
        ]
        d1, c1, t1 = compute_sleep_debt(s, 480)
        for _ in range(10):
            d, c, t = compute_sleep_debt(s, 480)
            assert d == d1
            assert c == c1
            assert t == t1


class TestDataGeneratorDeterminism:
    def test_same_seed_same_output(self):
        cfg = NightConfig(NightType.GOOD, 23, 0, 400, seed=2024)
        df1 = generate_single_night(cfg)
        for _ in range(5):
            df2 = generate_single_night(cfg)
            pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_output(self):
        cfg_a = NightConfig(NightType.GOOD, 23, 0, 300, seed=100)
        cfg_b = NightConfig(NightType.GOOD, 23, 0, 300, seed=200)
        df_a = generate_single_night(cfg_a)
        df_b = generate_single_night(cfg_b)
        assert not np.allclose(df_a["heart_rate"].to_numpy(), df_b["heart_rate"].to_numpy())

    def test_all_nights_preset_deterministic(self):
        df1 = generate_all_nights()
        for _ in range(3):
            df2 = generate_all_nights()
            pd.testing.assert_frame_equal(df1, df2)


class TestFullPipelineDeterminism:
    def test_end_to_end_on_generated_data(self):
        cfg = NightConfig(NightType.GOOD, 23, 0, 300, seed=777)
        frames_1 = []
        frames_2 = []
        for i in range(3):
            df = generate_single_night(cfg)
            staging = run_staging(df)
            metrics = compute_sleep_metrics(df, staging)
            frames_1.append((list(staging.stages), metrics_to_dict(metrics)))
        for i in range(3):
            df = generate_single_night(cfg)
            staging = run_staging(df)
            metrics = compute_sleep_metrics(df, staging)
            frames_2.append((list(staging.stages), metrics_to_dict(metrics)))
        for a, b in zip(frames_1, frames_2):
            assert a[0] == b[0]
            assert a[1] == b[1]
