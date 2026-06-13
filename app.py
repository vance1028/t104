from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.anomalies import AnomalyReport, detect_anomalies
from src.data_generator import load_or_generate, save_dataset, generate_all_nights
from src.metrics import SleepMetrics, compute_sleep_metrics
from src.staging import (
    STAGE_DEEP,
    STAGE_LIGHT,
    STAGE_REM,
    STAGE_WAKE,
    StagingResult,
    run_staging,
    stages_to_dataframe,
)
from src.trends import (
    NightSummary,
    TrendAnalysis,
    analyze_trends,
    build_night_summaries,
    format_minutes_clock,
)


STAGE_COLORS = {
    STAGE_WAKE: "#EF4444",
    STAGE_LIGHT: "#60A5FA",
    STAGE_DEEP: "#1E3A8A",
    STAGE_REM: "#A855F7",
}

STAGE_Y = {
    STAGE_WAKE: 0,
    STAGE_LIGHT: 1,
    STAGE_DEEP: 2,
    STAGE_REM: 3,
}


@st.cache_data(show_spinner=False)
def load_all_data(csv_path: str):
    df = load_or_generate(csv_path if csv_path else None)
    return df


@st.cache_data(show_spinner=False)
def process_all_nights(all_data: pd.DataFrame):
    staging_map: Dict[str, StagingResult] = {}
    metrics_map: Dict[str, SleepMetrics] = {}
    anomalies_map: Dict[str, AnomalyReport] = {}
    staged_dfs: Dict[str, pd.DataFrame] = {}

    for night_id, grp in all_data.groupby("night_id", sort=False):
        night_df = grp.sort_values("minute_index").reset_index(drop=True)
        staging = run_staging(night_df)
        metrics = compute_sleep_metrics(night_df, staging)
        anomalies = detect_anomalies(night_df)
        staging_map[night_id] = staging
        metrics_map[night_id] = metrics
        anomalies_map[night_id] = anomalies
        staged_dfs[night_id] = stages_to_dataframe(night_df, staging)

    summaries = build_night_summaries(all_data, staging_map, metrics_map)
    trend = analyze_trends(summaries, target_sleep_minutes=480)

    return staging_map, metrics_map, anomalies_map, staged_dfs, summaries, trend


def _stage_block_figures(staged_df: pd.DataFrame):
    shapes = []
    stage_values = staged_df["stage"].tolist()
    n = len(stage_values)
    i = 0
    while i < n:
        s = stage_values[i]
        j = i
        while j < n and stage_values[j] == s:
            j += 1
        shapes.append(
            dict(
                type="rect",
                x0=i,
                x1=j,
                y0=-0.5,
                y1=3.5,
                fillcolor=STAGE_COLORS[s],
                opacity=0.18,
                line=dict(width=0),
                layer="below",
            )
        )
        i = j
    return shapes


def plot_hypnogram_with_signals(staged_df: pd.DataFrame):
    stage_numeric = [STAGE_Y[s] for s in staged_df["stage"]]
    shapes = _stage_block_figures(staged_df)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.32, 0.26],
        vertical_spacing=0.04,
        subplot_titles=("睡眠阶段 (Hypnogram)", "心率 (bpm)", "体动强度"),
    )

    fig.add_trace(
        go.Scatter(
            x=staged_df["minute_index"],
            y=stage_numeric,
            mode="lines",
            line=dict(color="#111827", width=2, shape="hv"),
            name="阶段",
            hovertext=staged_df["stage"],
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        tickvals=list(STAGE_Y.values()),
        ticktext=list(STAGE_Y.keys()),
        range=[-0.5, 3.5],
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=staged_df["minute_index"],
            y=staged_df["heart_rate"],
            mode="lines",
            line=dict(color="#2563EB", width=1.3),
            name="心率",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=staged_df["minute_index"],
            y=staged_df["spo2"],
            mode="lines",
            line=dict(color="#DC2626", width=1.1),
            name="血氧(%)",
            yaxis="y4",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=staged_df["minute_index"],
            y=staged_df["movement"],
            marker_color="#F59E0B",
            name="体动",
            opacity=0.7,
        ),
        row=3,
        col=1,
    )

    for s in shapes:
        for row in (1, 2, 3):
            sh = dict(s)
            sh["xref"] = f"x{row}" if row > 1 else "x"
            sh["yref"] = f"y{row} domain" if row > 1 else "y domain"
            sh["y0"] = 0
            sh["y1"] = 1
            fig.add_shape(sh, row=row, col=1)

    fig.update_layout(
        height=620,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="bpm / %", row=2, col=1)
    fig.update_yaxes(title_text="强度", row=3, col=1)
    fig.update_xaxes(title_text="卧床分钟数", row=3, col=1)
    return fig


def plot_stage_pie(metrics: SleepMetrics):
    labels = ["浅睡", "深睡", "REM"]
    values = [metrics.light_minutes, metrics.deep_minutes, metrics.rem_minutes]
    colors = [STAGE_COLORS[STAGE_LIGHT], STAGE_COLORS[STAGE_DEEP], STAGE_COLORS[STAGE_REM]]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.45,
                sort=False,
            )
        ]
    )
    fig.update_layout(height=320, title="睡眠阶段构成")
    return fig


def plot_sleep_trend(summaries: List[NightSummary]):
    dates = [str(s.record_date) for s in summaries]
    sleep = [s.total_sleep_minutes / 60.0 for s in summaries]
    eff = [s.sleep_efficiency for s in summaries]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        subplot_titles=("睡眠时长 (小时)", "睡眠效率 (%)"),
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sleep,
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=8),
            name="睡眠时长(h)",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=8.0, line_dash="dash", line_color="#10B981", annotation_text="目标 8h", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=eff,
            mode="lines+markers",
            line=dict(color="#7C3AED", width=2.5),
            marker=dict(size=8),
            name="效率(%)",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=480, showlegend=False, margin=dict(l=40, r=30, t=50, b=40))
    fig.update_yaxes(title_text="小时", row=1, col=1)
    fig.update_yaxes(title_text="%", range=[50, 105], row=2, col=1)
    return fig


def plot_bed_onset_trend(summaries: List[NightSummary]):
    dates = [str(s.record_date) for s in summaries]

    def shift(m):
        return m + 24 * 60 if m < 12 * 60 else m

    bed_y = [shift(s.bed_minute_of_day) - 24 * 60 if s.bed_minute_of_day and shift(s.bed_minute_of_day) >= 24 * 60 else (shift(s.bed_minute_of_day) if s.bed_minute_of_day else None) for s in summaries]
    bed_y_plot = []
    for s in summaries:
        if s.bed_minute_of_day is None:
            bed_y_plot.append(None)
        else:
            m = s.bed_minute_of_day
            bed_y_plot.append(m / 60.0 if m >= 12 * 60 else (m + 24 * 60) / 60.0)

    onset_y_plot = []
    for s in summaries:
        if s.onset_minute_of_day is None:
            onset_y_plot.append(None)
        else:
            m = s.onset_minute_of_day
            onset_y_plot.append(m / 60.0 if m >= 12 * 60 else (m + 24 * 60) / 60.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bed_y_plot,
            mode="lines+markers",
            name="上床时间",
            line=dict(color="#F59E0B", width=2.5),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=onset_y_plot,
            mode="lines+markers",
            name="入睡时间",
            line=dict(color="#8B5CF6", width=2.5),
            marker=dict(size=8),
        )
    )
    tick_vals = [18, 20, 22, 24, 26, 28, 30]
    tick_labels = ["18:00", "20:00", "22:00", "00:00", "02:00", "04:00", "06:00"]
    fig.update_layout(
        height=360,
        title="上床 / 入睡时间趋势 (跨午夜显示为次日+小时)",
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(
        title_text="时间 (h)",
        tickvals=tick_vals,
        ticktext=tick_labels,
        range=[17.5, 30.5],
    )
    return fig


def _metric_card(title, value, sub=None, color="#1F2937"):
    sub_html = f'<div style="font-size:12px;color:#6B7280;margin-top:4px">{sub}</div>' if sub else ""
    return f"""
    <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:16px;margin-bottom:10px">
      <div style="font-size:12px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px">{title}</div>
      <div style="font-size:24px;font-weight:700;margin-top:6px;color:{color}">{value}</div>
      {sub_html}
    </div>
    """


def render_single_night_view(
    night_id: str,
    staged_df: pd.DataFrame,
    metrics: SleepMetrics,
    staging: StagingResult,
    anomalies: AnomalyReport,
):
    st.subheader(f"单晚详情 · {night_id}")
    record_date = staged_df["record_date"].iloc[0]
    night_type = staged_df["night_type"].iloc[0]
    st.caption(f"记录日期：{record_date} ｜ 夜晚类型：{night_type}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            _metric_card(
                "总睡眠时长",
                format_minutes_clock(metrics.total_sleep_minutes),
                f"卧床 {format_minutes_clock(metrics.total_bed_minutes)}",
                color="#1D4ED8",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card(
                "睡眠效率",
                f"{metrics.sleep_efficiency}%",
                f"入睡潜伏期 {metrics.sleep_latency_minutes} 分钟",
                color="#059669",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card(
                "夜醒次数",
                f"{metrics.awakenings_count} 次",
                f"醒着 {metrics.wake_after_onset_minutes} 分钟",
                color="#DC2626",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _metric_card(
                "睡眠周期",
                f"{metrics.sleep_cycle_count} 个",
                f"平均 {metrics.average_cycle_minutes:.0f} 分钟",
                color="#7C3AED",
            ),
            unsafe_allow_html=True,
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown(
            _metric_card("深睡", format_minutes_clock(metrics.deep_minutes), f"占比 {metrics.deep_ratio*100:.0f}%", color="#1E3A8A"),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            _metric_card("浅睡", format_minutes_clock(metrics.light_minutes), f"占比 {metrics.light_ratio*100:.0f}%", color="#2563EB"),
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown(
            _metric_card("REM", format_minutes_clock(metrics.rem_minutes), f"占比 {metrics.rem_ratio*100:.0f}%", color="#9333EA"),
            unsafe_allow_html=True,
        )

    onset_ts = staging.sleep_onset_timestamp
    wake_ts = staging.final_wake_timestamp
    if onset_ts or wake_ts:
        st.info(
            f"入睡时刻：{onset_ts.strftime('%H:%M') if onset_ts else '--:--'} ｜ "
            f"最终清醒：{wake_ts.strftime('%H:%M') if wake_ts else '--:--'}"
        )

    st.plotly_chart(plot_hypnogram_with_signals(staged_df), use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(plot_stage_pie(metrics), use_container_width=True)
    with right:
        st.markdown("#### 可疑异常事件")
        st.caption("仅为数据特征参考，非医学结论")
        if not anomalies.events:
            st.success("未检出明显异常")
        else:
            for e in anomalies.events:
                sev_color = "#DC2626" if e.severity == "high" else "#F59E0B"
                t_start = e.start_timestamp.strftime("%H:%M") if e.start_timestamp else "?"
                t_end = e.end_timestamp.strftime("%H:%M") if e.end_timestamp else "?"
                st.markdown(
                    f"<div style='border-left:4px solid {sev_color};padding:6px 10px;margin:6px 0;background:#FFF7ED;border-radius:4px'>"
                    f"<b>{e.event_type}</b> <span style='color:{sev_color}'>[{e.severity}]</span><br>"
                    f"<small>{t_start} ~ {t_end}，持续 {e.duration_minutes} 分钟</small><br>"
                    f"<small style='color:#374151'>{e.description}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def render_multi_night_view(
    trend: TrendAnalysis,
    summaries: List[NightSummary],
    metrics_map: Dict[str, SleepMetrics],
):
    st.subheader("多晚趋势总览")

    c1, c2, c3 = st.columns(3)
    with c1:
        debt_h = trend.sleep_debt_minutes / 60.0
        debt_sign = "+" if trend.sleep_debt_minutes > 0 else ("-" if trend.sleep_debt_minutes < 0 else "")
        st.markdown(
            _metric_card(
                "累计睡眠债",
                f"{debt_sign}{abs(debt_h):.1f} h",
                f"目标 {trend.target_sleep_per_night_minutes//60}h × {len(summaries)} 晚",
                color="#DC2626" if trend.sleep_debt_minutes > 0 else "#059669",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        reg = trend.onset_time_circular.regularity_score_0_100 if trend.onset_time_circular else 0.0
        st.markdown(
            _metric_card(
                "作息规律性评分",
                f"{reg:.0f} / 100",
                f"基于入睡时间环形方差",
                color="#7C3AED",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _metric_card(
                "平均睡眠时长",
                f"{trend.average_sleep_minutes/60.0:.1f} h",
                f"共 {len(summaries)} 晚数据",
                color="#2563EB",
            ),
            unsafe_allow_html=True,
        )

    if trend.bed_time_circular or trend.onset_time_circular:
        bc = trend.bed_time_circular
        oc = trend.onset_time_circular
        st.markdown(
            f"> **平均上床** {bc.mean_clock_str if bc else '--:--'} "
            f"(离散度 {bc.circular_std_minutes:.0f} 分) ｜ "
            f"**平均入睡** {oc.mean_clock_str if oc else '--:--'} "
            f"(离散度 {oc.circular_std_minutes:.0f} 分，R={oc.resultant_length if oc else 0:.2f})"
        )

    st.plotly_chart(plot_sleep_trend(summaries), use_container_width=True)
    st.plotly_chart(plot_bed_onset_trend(summaries), use_container_width=True)

    st.markdown("#### 逐晚数据明细")
    rows = []
    for s in summaries:
        mm = metrics_map.get(s.night_id)
        rows.append(
            {
                "日期": str(s.record_date),
                "night_id": s.night_id,
                "上床": s.bed_timestamp.strftime("%H:%M") if s.bed_timestamp else "--",
                "入睡": s.sleep_onset_timestamp.strftime("%H:%M") if s.sleep_onset_timestamp else "--",
                "睡眠(h)": round(s.total_sleep_minutes / 60.0, 2),
                "效率(%)": s.sleep_efficiency,
                "潜伏期(分)": s.sleep_latency_minutes,
                "深睡(分)": mm.deep_minutes if mm else 0,
                "REM(分)": mm.rem_minutes if mm else 0,
                "夜醒次数": mm.awakenings_count if mm else 0,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main():
    st.set_page_config(
        page_title="本地睡眠分析平台",
        page_icon="💤",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("💤 本地睡眠分析平台")
    st.caption("纯本地分析 · 逐分钟信号 · 睡眠分期 / 指标 / 趋势 / 异常参考")

    with st.sidebar:
        st.header("数据与控制")
        data_file = st.text_input("数据 CSV 路径（留空用内置模拟）", value="data/sleep_dataset.csv")
        if st.button("🔄 重新生成模拟数据"):
            df = generate_all_nights()
            os.makedirs(os.path.dirname(os.path.abspath(data_file)) or ".", exist_ok=True)
            save_dataset(df, data_file)
            st.success(f"已生成并保存到 {data_file}")
        target_sleep_h = st.slider("目标每晚睡眠 (小时)", 5.0, 10.0, 8.0, 0.5)

    with st.spinner("加载并分析所有夜晚..."):
        all_data = load_all_data(data_file)
        staging_map, metrics_map, anomalies_map, staged_dfs, summaries, trend_raw = process_all_nights(
            all_data
        )
        if int(target_sleep_h * 60) != trend_raw.target_sleep_per_night_minutes:
            trend = analyze_trends(summaries, target_sleep_minutes=int(target_sleep_h * 60))
        else:
            trend = trend_raw

    tab1, tab2 = st.tabs(["单晚详情", "多晚趋势"])

    with tab2:
        render_multi_night_view(trend, summaries, metrics_map)

    with tab1:
        night_ids = list(dict.fromkeys(all_data["night_id"].tolist()))
        default_idx = 0
        sel_nid = st.selectbox("选择夜晚", night_ids, index=default_idx)
        if sel_nid:
            render_single_night_view(
                sel_nid,
                staged_dfs[sel_nid],
                metrics_map[sel_nid],
                staging_map[sel_nid],
                anomalies_map[sel_nid],
            )


if __name__ == "__main__":
    main()
