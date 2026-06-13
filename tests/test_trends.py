from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import datetime as _dt
import pandas as pd
import numpy as np
import pytest

from src.trends import (
    CIRCULAR_DAY_MINUTES,
    CircularStats,
    NightSummary,
    TrendAnalysis,
    circular_mean_and_dispersion,
    compute_circular_stats,
    compute_sleep_debt,
    build_night_summaries,
    analyze_trends,
    format_minutes_clock,
    _circular_paired_diffs_minutes,
)


class TestCircularMean:
    """核心：入睡时间是环形时间，跨午夜不能直接算术平均。"""

    def test_same_time_no_variance(self):
        # 三天都是 23:00 = 23*60 = 1380
        minutes = [23 * 60, 23 * 60, 23 * 60]
        mean_min, R, std = circular_mean_and_dispersion(minutes)
        assert mean_min == 23 * 60
        assert abs(R - 1.0) < 0.01
        assert abs(std - 0.0) < 0.1

    def test_11pm_and_1am_mean_is_midnight_not_noon(self):
        """关键测试：23:00（晚上11点）和 01:00（凌晨1点）
        算术平均 = (1380 + 60) / 2 = 720 分钟 = 中午12点 ❌
        环形平均 = 00:00 = 0 分钟 ✅
        """
        minutes = [23 * 60, 1 * 60]  # 23:00 和 01:00
        mean_min, R, std = circular_mean_and_dispersion(minutes)
        # 环形平均应该靠近午夜 (0或1440都表示00:00)
        # 允许 ±30 分钟误差
        dist_from_midnight = min(abs(mean_min - 0), abs(mean_min - 1440))
        assert dist_from_midnight < 30, (
            f"平均={mean_min}分, 应靠近午夜(0/1440)，差{dist_from_midnight}分"
        )
        # 离散度应该很小（两天只差2小时）
        assert std < 120

    def test_three_days_clustered(self):
        # 三天：00:30, 23:30, 01:00
        # 都围绕午夜，平均应靠近00:00
        minutes = [30, 23 * 60 + 30, 60]
        mean_min, R, std = circular_mean_and_dispersion(minutes)
        dist = min(abs(mean_min - 0), abs(mean_min - 1440))
        assert dist < 60, f"平均={mean_min}分离午夜太远"

    def test_three_days_930pm_consistent(self):
        # 21:30 ± 15 分钟，都是同一天晚上
        minutes = [21 * 60 + 15, 21 * 60 + 30, 21 * 60 + 45]
        mean_min, R, std = circular_mean_and_dispersion(minutes)
        assert 21 * 60 + 10 <= mean_min <= 21 * 60 + 50
        assert abs(R - 1.0) < 0.02

    def test_empty_list(self):
        mean_min, R, std = circular_mean_and_dispersion([])
        assert mean_min is None
        assert R == 0.0

    def test_opposite_points_give_low_R(self):
        # 12:00 和 00:00 正好相对
        minutes = [720, 0]
        mean_min, R, std = circular_mean_and_dispersion(minutes)
        # R 应该接近 0（两个相反方向向量抵消）
        assert R < 0.1


class TestCircularStatsObject:
    def test_mean_clock_format(self):
        minutes = [23 * 60, 23 * 60, 23 * 60]
        s = compute_circular_stats(minutes)
        assert s.mean_clock_str == "23:00"
        assert s.mean_minute_of_day == 23 * 60
        assert 90.0 <= s.regularity_score_0_100 <= 100.0

    def test_very_irregular(self):
        # 非常分散：晚8点、凌晨2点、中午12点
        minutes = [20 * 60, 2 * 60, 12 * 60]
        s = compute_circular_stats(minutes)
        assert s.regularity_score_0_100 < 70.0


class TestPairedCircularDiffs:
    def test_no_wrap(self):
        # 22:00 -> 22:30 = 30分钟
        minutes = [22 * 60, 22 * 60 + 30]
        diffs = _circular_paired_diffs_minutes(minutes)
        assert len(diffs) == 1
        assert abs(diffs[0] - 30) < 1

    def test_wrap_midnight(self):
        # 23:30 -> 00:30 = 60分钟（不是 -1380）
        minutes = [23 * 60 + 30, 30]
        diffs = _circular_paired_diffs_minutes(minutes)
        assert len(diffs) == 1
        assert abs(diffs[0] - 60) < 1


class TestSleepDebt:
    """睡眠债累计 = 目标总时长 - 实际总时长"""

    def test_all_matching_target_zero_debt(self):
        s1 = NightSummary(night_id="1", record_date=_dt.date(2026, 5, 1), total_sleep_minutes=480)
        s2 = NightSummary(night_id="2", record_date=_dt.date(2026, 5, 2), total_sleep_minutes=480)
        s3 = NightSummary(night_id="3", record_date=_dt.date(2026, 5, 3), total_sleep_minutes=480)
        debt, cum_sleep, cum_target = compute_sleep_debt([s1, s2, s3], 480)
        assert debt == 0
        assert cum_sleep == 1440
        assert cum_target == 1440

    def test_1hour_debt_per_night_3nights(self):
        # 目标8h(480)，实际只睡7h(420)，3晚 → 3h 债
        s = [
            NightSummary(night_id=str(i), record_date=_dt.date(2026, 5, i + 1), total_sleep_minutes=420)
            for i in range(3)
        ]
        debt, cum_sleep, cum_target = compute_sleep_debt(s, 480)
        assert debt == 3 * 60
        assert cum_sleep == 3 * 420
        assert cum_target == 3 * 480

    def test_surplus_is_negative_debt(self):
        # 实际睡 9h，目标8h → -1h 债
        s = [NightSummary(night_id="1", record_date=_dt.date(2026, 5, 1), total_sleep_minutes=540)]
        debt, _, _ = compute_sleep_debt(s, 480)
        assert debt == -60

    def test_empty_summaries(self):
        debt, cs, ct = compute_sleep_debt([], 480)
        assert debt == 0
        assert cs == 0
        assert ct == 0


class TestFormatMinutesClock:
    def test_positive(self):
        assert format_minutes_clock(125) == "2h 05m"

    def test_negative(self):
        assert format_minutes_clock(-90) == "-1h 30m"

    def test_zero(self):
        assert format_minutes_clock(0) == "0h 00m"


class _FakeStaging:
    def __init__(self, onset_ts, wake_ts):
        self.sleep_onset_timestamp = onset_ts
        self.final_wake_timestamp = wake_ts


class _FakeMetrics:
    def __init__(self, sleep_min, eff, latency):
        self.total_sleep_minutes = sleep_min
        self.sleep_efficiency = eff
        self.sleep_latency_minutes = latency
        self.deep_minutes = 60
        self.rem_minutes = 90
        self.awakenings_count = 0


class TestBuildSummariesAndTrend:
    def test_build_summaries_count(self):
        # 直接构造测试数据
        nights = []
        base = _dt.datetime(2026, 5, 1, 23, 0, 0)
        per_staging = {}
        per_metrics = {}
        rows = []
        for i in range(3):
            nid = f"n{i}"
            bed_dt = base + _dt.timedelta(days=i)
            onset_dt = bed_dt + _dt.timedelta(minutes=15)
            wake_dt = bed_dt + _dt.timedelta(minutes=500)
            n_min = 510
            for j in range(n_min):
                rows.append({
                    "night_id": nid,
                    "record_date": (bed_dt + _dt.timedelta(hours=12)).date(),
                    "timestamp": bed_dt + _dt.timedelta(minutes=j),
                    "minute_index": j,
                    "heart_rate": 60.0,
                    "movement": 0.05,
                    "spo2": 96.0,
                    "respiratory_rate": 16.0,
                    "night_type": "test",
                })
            per_staging[nid] = _FakeStaging(pd.Timestamp(onset_dt), pd.Timestamp(wake_dt))
            per_metrics[nid] = _FakeMetrics(420, 85.0, 15)
        df = pd.DataFrame(rows)
        summaries = build_night_summaries(df, per_staging, per_metrics)
        assert len(summaries) == 3
        assert summaries[0].total_sleep_minutes == 420
        assert summaries[0].bed_minute_of_day == 23 * 60

    def test_analyze_trends_circular_and_debt(self):
        s1 = NightSummary(
            night_id="1", record_date=_dt.date(2026, 5, 1),
            bed_timestamp=pd.Timestamp("2026-05-01 23:00"),
            sleep_onset_timestamp=pd.Timestamp("2026-05-01 23:15"),
            bed_minute_of_day=23 * 60, onset_minute_of_day=23 * 60 + 15,
            total_sleep_minutes=420, sleep_efficiency=85.0, sleep_latency_minutes=15,
        )
        s2 = NightSummary(
            night_id="2", record_date=_dt.date(2026, 5, 2),
            bed_timestamp=pd.Timestamp("2026-05-03 00:30"),
            sleep_onset_timestamp=pd.Timestamp("2026-05-03 00:45"),
            bed_minute_of_day=0 * 60 + 30, onset_minute_of_day=0 * 60 + 45,
            total_sleep_minutes=420, sleep_efficiency=85.0, sleep_latency_minutes=15,
        )
        trend = analyze_trends([s1, s2], target_sleep_minutes=480)
        # 两晚目标 480*2=960，实际 420*2=840 → 睡眠债 120 分
        assert trend.sleep_debt_minutes == 120
        assert trend.cumulative_target_minutes == 960
        assert trend.cumulative_sleep_minutes == 840
        assert abs(trend.average_sleep_minutes - 420.0) < 0.1
        # 入睡时间环形平均：23:15 和 00:45 → 应该在午夜附近
        assert trend.onset_time_circular is not None
        oc = trend.onset_time_circular
        dist = min(abs(oc.mean_minute_of_day - 0), abs(oc.mean_minute_of_day - 1440))
        assert dist < 60

    def test_empty_trend(self):
        trend = analyze_trends([], target_sleep_minutes=480)
        assert trend.sleep_debt_minutes == 0
        assert trend.bed_time_circular is None
