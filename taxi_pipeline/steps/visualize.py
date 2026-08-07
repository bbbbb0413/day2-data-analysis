"""정제 데이터로 정적 차트와 인터랙티브 차트를 생성한다."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
import seaborn as sns                     # noqa: E402

from ..config import Config               # noqa: E402
# 색상·라벨·폰트는 viz.style 한곳에서 관리한다(Agg 백엔드 설정도 그쪽에 있다).
from ..viz.style import (ACCENT, DOW, PAYMENT, PAYMENT_LABEL, POSITIVE,  # noqa: E402
                         PRIMARY, SECONDARY, setup_style)
from .base import Artifact, StepResult    # noqa: E402

log = logging.getLogger(__name__)


def mpl_artifact(fig, name: str, caption: str, dpi: int) -> Artifact:
    """Matplotlib Figure를 PNG 산출물로 변환한다."""
    def save(path: Path) -> None:
        """Figure를 PNG로 저장한다."""
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return Artifact(name=name, save=save, caption=caption, kind="figure")


def _plotly(fig, name: str, caption: str) -> Artifact:
    """Plotly Figure를 인터랙티브 HTML 산출물로 변환한다.

    PNG 스냅샷은 만들지 않는다. kaleido가 정적 이미지를 뽑을 때 실제 Chrome을
    띄워 렌더링하는데(차트 3개면 브라우저도 3번), 그렇게 만든 PNG를 참조하는
    곳이 없었다. Chrome이 없는 서버·CI에서는 저장이 실패해 파이프라인 전체가
    죽기까지 했다. 그림이 필요하면 HTML을 열어 내보내면 된다.
    """
    def save(path: Path) -> None:
        """Plotly 차트를 HTML로 저장한다."""
        fig.write_html(str(path), include_plotlyjs="cdn")

    return Artifact(name=name, save=save, caption=caption, kind="plotly")


def _tip_rate(df: pd.DataFrame) -> pd.Series:
    """카드결제 건의 팁 비율을 계산한다."""
    card = df[df["payment_type"] == 1]
    r = card["tip_amount"] / card["fare_amount"].replace(0, np.nan)
    return r[r.notna() & (r < 2)]        # 계산할 수 없는 값과 극단값을 제외한다.


# ============================================================================
# 1. 수치형 분포 4종 (Seaborn · 분포)
# ============================================================================
def _chart_distributions(df: pd.DataFrame, cfg: Config):
    """주요 수치형 변수의 분포와 평균, 중앙값을 비교한다."""
    dur = (df[cfg.columns.dropoff] - df[cfg.columns.pickup]).dt.total_seconds() / 60
    panels = [
        ("trip_distance", df["trip_distance"], "이동거리 (mile)", 20),
        ("fare_amount", df["fare_amount"], "기본요금 (USD)", 80),
        ("duration_min", dur, "소요시간 (분)", 60),
        ("total_amount", df["total_amount"], "총액 (USD)", 100),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    stats = {}
    for ax, (key, s, label, xmax) in zip(axes.ravel(), panels):
        shown = s[(s >= 0) & (s <= xmax)]
        mean, median = float(s.mean()), float(s.median())
        ax.hist(shown, bins=60, color=PRIMARY, edgecolor="none")
        ax.axvline(mean, color=ACCENT, linestyle="--", linewidth=1.8,
                   label=f"평균 {mean:.2f}")
        ax.axvline(median, color=POSITIVE, linewidth=1.8, label=f"중앙값 {median:.2f}")
        ax.set_title(f"{label} — 평균/중앙값 {mean / median:.2f}배", fontsize=11)
        ax.set_xlabel(label)
        ax.set_ylabel("운행 건수")
        ax.legend(fontsize=8)
        stats[key] = {"mean": mean, "median": median,
                      "ratio": mean / median if median else None,
                      "coverage": float((s <= xmax).mean())}

    fig.suptitle("수치형 변수 분포 — 네 변수 모두 오른쪽 꼬리가 길다", fontsize=13)
    fig.tight_layout()

    worst = max(stats.items(), key=lambda kv: kv[1]["ratio"] or 0)
    caption = (f"네 변수 모두 평균이 중앙값보다 크다. 가장 심한 것은 {worst[0]}로 "
               f"{worst[1]['ratio']:.2f}배다. 소수의 극단값이 평균을 끌어올린 결과이므로 "
               f"'평균'으로 전형적인 운행을 설명하면 안 된다. "
               f"가독성을 위해 각 패널의 x축 상한을 잘랐다(표시 범위가 전체의 "
               f"{min(v['coverage'] for v in stats.values()):.1%} 이상).")
    return mpl_artifact(fig, "01_numeric_distributions.png", caption, cfg.visualize.dpi), stats


# ============================================================================
# 2. 상관계수 히트맵 (Seaborn · 상관관계)
# ============================================================================
def _chart_correlation(df: pd.DataFrame, cfg: Config):
    """Pearson과 Spearman 상관계수를 비교한다."""
    cols = [c for c in cfg.statistics.correlation_columns if c in df.columns]
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mat, title in ((axes[0], pearson, "피어슨 (선형 관계)"),
                           (axes[1], spearman, "스피어만 (순위 관계)")):
        # 두 히트맵의 색상 범위를 동일하게 맞춘다.
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                    square=True, ax=ax, cbar=False, annot_kws={"size": 9})
        ax.set_title(title, fontsize=12)

    # copy=True는 필수다. pandas copy-on-write에서 .values/.to_numpy()는 read-only
    # 배열을 주는데, np.fill_diagonal()이 제자리 수정이라 "underlying array is
    # read-only"로 터진다.
    diff = (pearson - spearman).abs().to_numpy(copy=True)
    np.fill_diagonal(diff, 0)
    i, j = np.unravel_index(np.argmax(diff), diff.shape)
    gap = float(diff[i, j])
    fig.suptitle(f"상관계수 — {cols[i]} ~ {cols[j]} 에서 두 계수가 {gap:.3f} 벌어진다",
                 fontsize=13)
    fig.tight_layout()

    caption = (f"팁이 얽힌 변수쌍에서 두 계수가 크게 어긋난다(최대 {cols[i]}~{cols[j]}, "
               f"차이 {gap:.3f}). 피어슨은 선형 크기를, 스피어만은 순위를 본다. "
               f"팁이 0인 건이 대량이라 순위 계산에서 동점이 생긴 결과다. "
               f"상관은 인과가 아니다.")
    metrics = {"max_gap": {"pair": [cols[i], cols[j]], "gap": gap}}
    return mpl_artifact(fig, "02_correlation_heatmap.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 3. 시간대별 패턴 (Seaborn · 그룹 비교)
# ============================================================================
def _chart_hourly(df: pd.DataFrame, cfg: Config):
    """시간대별 운행량과 평균 요금, 거리를 비교한다."""
    pu = cfg.columns.pickup
    g = df.groupby(df[pu].dt.hour).agg(
        trips=("total_amount", "size"), fare=("fare_amount", "mean"),
        dist=("trip_distance", "mean"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(g.index, g["trips"], color=PRIMARY)
    axes[0].set_title("시간대별 운행량", fontsize=12)
    axes[0].set_ylabel("운행 건수")

    axes[1].plot(g.index, g["fare"], marker="o", color=ACCENT, label="평균 요금 (USD)")
    axes[1].plot(g.index, g["dist"], marker="s", color=POSITIVE, label="평균 거리 (mile)")
    axes[1].set_title("시간대별 평균 요금·거리", fontsize=12)
    axes[1].set_xlabel("승차 시간대 (시)")
    axes[1].set_ylabel("평균값")
    axes[1].set_xticks(range(0, 24))
    axes[1].legend()
    fig.tight_layout()

    peak, low = int(g["trips"].idxmax()), int(g["trips"].idxmin())
    fmax, fmin = int(g["fare"].idxmax()), int(g["fare"].idxmin())
    caption = (f"운행량은 {peak}시에 최다({int(g.loc[peak, 'trips']):,}건), "
               f"{low}시에 최소({int(g.loc[low, 'trips']):,}건)로 "
               f"{g.loc[peak, 'trips'] / g.loc[low, 'trips']:.1f}배 차이난다. "
               f"평균 요금은 {fmax}시에 가장 높고({g.loc[fmax, 'fare']:.2f} USD) "
               f"{fmin}시에 가장 낮다({g.loc[fmin, 'fare']:.2f} USD). "
               f"운행량이 적은 새벽에 요금이 오히려 높은 것은 이 시간대에 "
               f"장거리 운행 비중이 크기 때문이다(평균 거리 선 참고).")
    metrics = {"peak_hour": peak, "quietest_hour": low,
               "max_fare_hour": fmax, "min_fare_hour": fmin}
    return mpl_artifact(fig, "03_hourly_pattern.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 4. 요일별 패턴 (Seaborn · 그룹 비교)
# ============================================================================
def _chart_weekday(df: pd.DataFrame, cfg: Config):
    """요일별 운행량과 평균 요금, 팁을 비교한다."""
    pu = cfg.columns.pickup
    g = df.groupby(df[pu].dt.dayofweek).agg(
        trips=("total_amount", "size"), fare=("fare_amount", "mean"),
        tip=("tip_amount", "mean")).sort_index()
    labels = [DOW[i] for i in g.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(labels, g["trips"], color=PRIMARY)
    axes[0].set_title("요일별 운행량", fontsize=12)
    axes[0].set_xlabel("요일")
    axes[0].set_ylabel("운행 건수")

    x = np.arange(len(labels))
    w = 0.38
    axes[1].bar(x - w / 2, g["fare"], w, label="평균 요금 (USD)", color=ACCENT)
    axes[1].bar(x + w / 2, g["tip"], w, label="평균 팁 (USD)", color=POSITIVE)
    axes[1].set_title("요일별 평균 요금·팁", fontsize=12)
    axes[1].set_xlabel("요일")
    axes[1].set_ylabel("금액 (USD)")
    axes[1].set_xticks(x, labels)
    axes[1].legend()
    fig.tight_layout()

    tmax, tmin = int(g["trips"].idxmax()), int(g["trips"].idxmin())
    pmax, pmin = int(g["tip"].idxmax()), int(g["tip"].idxmin())
    caption = (f"운행량은 {DOW[tmax]}요일이 최다({int(g.loc[tmax, 'trips']):,}건), "
               f"{DOW[tmin]}요일이 최소({int(g.loc[tmin, 'trips']):,}건)다. "
               f"평균 팁은 {DOW[pmax]}요일이 가장 높고({g.loc[pmax, 'tip']:.2f} USD) "
               f"{DOW[pmin]}요일이 가장 낮다({g.loc[pmin, 'tip']:.2f} USD). "
               f"운행량은 절대 건수라 요일별 일수 차이(2026-05는 금·토가 5회, "
               f"나머지는 4회)의 영향을 받는다.")
    metrics = {"busiest_dow": tmax, "quietest_dow": tmin, "best_tip_dow": pmax}
    return mpl_artifact(fig, "04_weekday_pattern.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 5. 거리-요금 관계 (Seaborn · 상관관계)
# ============================================================================
def _chart_distance_fare(df: pd.DataFrame, cfg: Config):
    """이동거리 구간별 평균 요금과 마일당 요금을 비교한다."""
    bins = [0, 1, 2, 3, 5, 10, 20, 100]
    b = df.assign(bucket=pd.cut(df["trip_distance"], bins),
                  per_mile=df["fare_amount"] / df["trip_distance"])
    g = b.groupby("bucket", observed=True).agg(
        trips=("fare_amount", "size"), fare=("fare_amount", "mean"),
        per_mile=("per_mile", "median"))
    labels = [f"{int(i.left)}~{int(i.right)}" for i in g.index]

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.bar(labels, g["fare"], color=PRIMARY, label="평균 요금 (USD)")
    ax1.set_xlabel("이동거리 구간 (mile)")
    ax1.set_ylabel("평균 요금 (USD)", color=PRIMARY)
    ax1.tick_params(axis="y", labelcolor=PRIMARY)

    # 마일당 요금은 보조축에 표시한다.
    ax2 = ax1.twinx()
    ax2.plot(labels, g["per_mile"], marker="o", color=ACCENT, linewidth=2,
             label="마일당 요금 중앙값 (USD/mile)")
    ax2.set_ylabel("마일당 요금 (USD/mile)", color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)
    ax2.grid(False)

    ax1.set_title("거리 구간별 요금 — 거리가 늘수록 마일당 단가는 떨어진다", fontsize=13)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    fig.tight_layout()

    first, last = float(g["per_mile"].iloc[0]), float(g["per_mile"].iloc[-1])
    caption = (f"평균 요금은 거리에 따라 단조 증가하지만, 마일당 단가는 "
               f"{labels[0]} mile 구간 {first:.2f} USD/mile에서 "
               f"{labels[-1]} mile 구간 {last:.2f} USD/mile로 {first / last:.1f}배 떨어진다. "
               f"기본요금이 짧은 운행에 크게 반영되기 때문이다. "
               f"마일당 요금은 극단값에 끌리지 않도록 중앙값을 썼다.")
    metrics = {"per_mile_first": first, "per_mile_last": last,
               "buckets": {l: int(n) for l, n in zip(labels, g["trips"])}}
    return mpl_artifact(fig, "05_distance_fare_relation.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 6. 결제수단별 팁 (Seaborn · 그룹 비교)
# ============================================================================
def _chart_payment_tip(df: pd.DataFrame, cfg: Config):
    """결제수단별 운행량과 팁 0 비율을 비교한다."""
    d = df.copy()
    d[PAYMENT_LABEL] = d["payment_type"].map(PAYMENT).fillna("미기재")
    g = d.groupby(PAYMENT_LABEL).agg(
        trips=("tip_amount", "size"), mean_tip=("tip_amount", "mean"),
        zero_ratio=("tip_amount", lambda s: float((s == 0).mean())))
    g = g.sort_values("trips", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(g.index, g["trips"], color=PRIMARY)
    axes[0].set_title("결제수단별 운행 건수", fontsize=12)
    axes[0].set_xlabel("결제수단")
    axes[0].set_ylabel("운행 건수")

    axes[1].bar(g.index, g["zero_ratio"] * 100, color=ACCENT)
    axes[1].set_title("결제수단별 '팁 0' 비율", fontsize=12)
    axes[1].set_xlabel("결제수단")
    axes[1].set_ylabel("팁이 0인 비율 (%)")
    axes[1].set_ylim(0, 105)
    for i, (name, row) in enumerate(g.iterrows()):
        axes[1].text(i, row["zero_ratio"] * 100 + 2, f"{row['zero_ratio']:.1%}",
                     ha="center", fontsize=9)
    fig.tight_layout()

    cash = g.loc["현금", "zero_ratio"] if "현금" in g.index else float("nan")
    card = g.loc["카드", "zero_ratio"] if "카드" in g.index else float("nan")
    caption = (f"현금 결제는 팁이 {cash:.1%} 비율로 0이고, 카드는 {card:.1%}다. "
               f"승객이 현금으로 팁을 주지 않는다는 뜻이 아니라 현금 팁이 미터기에 "
               f"입력되지 않아 시스템이 기록하지 못하는 것이다. "
               f"팁을 다루는 분석에서 결제수단을 섞으면 '결제수단 차이'를 "
               f"'팁 행동 차이'로 오독하게 된다.")
    metrics = {k: {"trips": int(v["trips"]), "mean_tip": float(v["mean_tip"]),
                   "zero_ratio": float(v["zero_ratio"])} for k, v in g.iterrows()}
    return mpl_artifact(fig, "06_payment_tip.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 7. 소스별 프로파일 (Seaborn · 그룹 비교)
# ============================================================================
def _chart_source_profile(df: pd.DataFrame, cfg: Config):
    """record_source별 운행량과 평균값을 비교한다."""
    if "record_source" not in df.columns or df["record_source"].nunique() < 2:
        return None, {}

    g = df.groupby("record_source").agg(
        trips=("total_amount", "size"), dist=("trip_distance", "mean"),
        fare=("fare_amount", "mean"), tip=("tip_amount", "mean"),
        total=("total_amount", "mean"))
    metrics_cols = [("dist", "평균 거리 (mile)"), ("fare", "평균 요금 (USD)"),
                    ("tip", "평균 팁 (USD)"), ("total", "평균 총액 (USD)")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(g.index, g["trips"], color=[PRIMARY, SECONDARY])
    axes[0].set_title("소스별 운행 건수", fontsize=12)
    axes[0].set_xlabel("record_source")
    axes[0].set_ylabel("운행 건수")
    for i, v in enumerate(g["trips"]):
        axes[0].text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

    x = np.arange(len(metrics_cols))
    w = 0.38
    for k, (src, row) in enumerate(g.iterrows()):
        axes[1].bar(x + (k - 0.5) * w, [row[c] for c, _ in metrics_cols], w, label=src)
    axes[1].set_title("소스별 평균값 비교", fontsize=12)
    axes[1].set_xlabel("지표")
    axes[1].set_ylabel("평균값")
    axes[1].set_xticks(x, [lab for _, lab in metrics_cols], fontsize=9)
    axes[1].legend()
    fig.tight_layout()

    tips = g["tip"]
    ratio = tips.max() / tips.min() if tips.min() > 0 else float("inf")
    caption = (f"두 소스의 평균 팁이 {tips.max():.2f} vs {tips.min():.2f} USD로 "
               f"{ratio:.1f}배 차이난다. partial 소스는 팁 정보가 연동되지 않아 "
               f"대부분 0으로 기록되기 때문이다. 섞어서 평균을 내면 팁이 과소 추정되고, "
               f"한쪽을 지우면 표본이 편향된다. 결측행을 삭제하지 않고 "
               f"record_source로 구분해 둔 이유가 여기 있다.")
    metrics = {str(k): {"trips": int(v["trips"]), "mean_tip": float(v["tip"]),
                        "mean_fare": float(v["fare"])} for k, v in g.iterrows()}
    return mpl_artifact(fig, "07_source_profile.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 8. 시간대 × 요일 히트맵 (Plotly · 인터랙티브)
# ============================================================================
def _chart_demand_heatmap(df: pd.DataFrame, cfg: Config):
    """요일과 시간대별 운행량을 히트맵으로 만든다."""
    import plotly.graph_objects as go

    pu = cfg.columns.pickup
    # 원본 행을 그대로 넘기지 않고 집계한 결과를 사용한다.
    pivot = (df.groupby([df[pu].dt.dayofweek, df[pu].dt.hour])
             .size().unstack(fill_value=0).sort_index())

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{h}시" for h in pivot.columns],
        y=[DOW[i] for i in pivot.index],
        # 운행량 크기를 표현하기 위해 순차형 색상을 사용한다.
        colorscale="YlOrRd", colorbar=dict(title="운행 건수"),
        hovertemplate="%{y}요일 %{x}<br>운행 %{z:,}건<extra></extra>",
    ))
    fig.update_layout(
        title="시간대 × 요일 운행량 — 셀에 마우스를 올리면 정확한 건수가 표시됩니다",
        xaxis_title="승차 시간대", yaxis_title="요일",
        # 월요일부터 읽을 수 있도록 y축 순서를 뒤집는다.
        yaxis=dict(autorange="reversed"),
        width=1000, height=460, font=dict(size=12),
    )

    flat = pivot.stack()
    (pd_, ph_), peak = flat.idxmax(), int(flat.max())
    (ld_, lh_), low = flat.idxmin(), int(flat.min())
    caption = (f"최다는 {DOW[pd_]}요일 {ph_}시({peak:,}건), 최소는 "
               f"{DOW[ld_]}요일 {lh_}시({low:,}건)로 {peak / max(low, 1):.0f}배 "
               f"차이난다. 평일 출퇴근 시간대와 주말 심야에 수요가 몰린다.")
    metrics = {"peak": {"dayofweek": int(pd_), "hour": int(ph_), "trips": peak},
               "lowest": {"dayofweek": int(ld_), "hour": int(lh_), "trips": low}}
    return _plotly(fig, "08_demand_heatmap.html", caption), metrics


# ============================================================================
# 9. 일자별 운행량 추이 (Plotly · 인터랙티브)
# ============================================================================
def _chart_daily_trend(df: pd.DataFrame, cfg: Config):
    """일자별 운행량을 인터랙티브 차트로 만든다."""
    import plotly.graph_objects as go

    pu = cfg.columns.pickup
    g = df.groupby(df[pu].dt.date).agg(
        trips=("total_amount", "size"), fare=("fare_amount", "mean"))
    dates = pd.to_datetime(g.index)
    # 주말 막대는 다른 색으로 표시한다.
    colors = [ACCENT if d.dayofweek >= 5 else PRIMARY for d in dates]

    fig = go.Figure()
    fig.add_bar(x=dates, y=g["trips"], marker_color=colors, name="운행 건수",
                hovertemplate="%{x|%m월 %d일}<br>운행 %{y:,}건<extra></extra>")
    fig.update_layout(
        title="일자별 운행량 — 붉은 막대가 주말입니다 (드래그로 구간 확대 가능)",
        xaxis_title="날짜", yaxis_title="운행 건수",
        width=1000, height=440, font=dict(size=12), showlegend=False,
    )

    peak, low = g["trips"].idxmax(), g["trips"].idxmin()
    caption = (f"운행량이 가장 많은 날은 {peak}({int(g.loc[peak, 'trips']):,}건), "
               f"가장 적은 날은 {low}({int(g.loc[low, 'trips']):,}건)로 "
               f"{g.loc[peak, 'trips'] / g.loc[low, 'trips']:.1f}배 차이난다. "
               f"주 단위 주기가 뚜렷하며, 특정 날짜의 급감은 날씨나 공휴일 같은 "
               f"외부 요인일 수 있으나 이 데이터만으로는 확인할 수 없다.")
    metrics = {"peak_date": str(peak), "peak_trips": int(g.loc[peak, "trips"]),
               "lowest_date": str(low), "lowest_trips": int(g.loc[low, "trips"]),
               "days": len(g)}
    return _plotly(fig, "09_daily_trend.html", caption), metrics


# ============================================================================
# 10. 승차 존 TOP 20 (Plotly · 인터랙티브)
# ============================================================================
def _chart_top_zones(df: pd.DataFrame, cfg: Config):
    """승차 건수가 많은 지역 20개를 차트로 만든다."""
    import plotly.graph_objects as go

    top = df["PULocationID"].value_counts().head(20)
    zone_stats = (df[df["PULocationID"].isin(top.index)]
                  .groupby("PULocationID")
                  .agg(fare=("fare_amount", "mean"), dist=("trip_distance", "mean")))
    labels = [f"존 {int(z)}" for z in top.index]
    share = top / len(df) * 100

    fig = go.Figure(go.Bar(
        x=labels, y=top.values, marker_color=PRIMARY,
        customdata=np.stack([share.values,
                             zone_stats.loc[top.index, "fare"].values,
                             zone_stats.loc[top.index, "dist"].values], axis=-1),
        hovertemplate=("%{x}<br>운행 %{y:,}건 (전체의 %{customdata[0]:.2f}%)"
                       "<br>평균 요금 %{customdata[1]:.2f} USD"
                       "<br>평균 거리 %{customdata[2]:.2f} mile<extra></extra>"),
    ))
    fig.update_layout(
        title="승차 존 TOP 20 — 막대에 마우스를 올리면 요금·거리도 표시됩니다",
        xaxis_title="승차 존 (TLC LocationID)", yaxis_title="운행 건수",
        width=1000, height=460, font=dict(size=12),
    )

    caption = (f"상위 20개 존이 전체 운행의 {share.sum():.1f}%를 차지한다. "
               f"1위는 존 {int(top.index[0])}로 {int(top.iloc[0]):,}건"
               f"({share.iloc[0]:.2f}%)이다. 승차가 소수 지역에 크게 몰려 있어, "
               f"지역을 모델 피처로 쓸 때 희귀 존을 그대로 두면 학습이 불안정해진다. "
               f"Unknown(264)·구역 외(265)는 정제 단계에서 결측 처리되어 제외됐다.")
    metrics = {"top20_share": float(share.sum()),
               "top_zone": int(top.index[0]), "top_zone_trips": int(top.iloc[0])}
    return _plotly(fig, "10_top_zones.html", caption), metrics


# ============================================================================
# 시각화 단계를 구성한다.
# ============================================================================
_BUILDERS = [
    ("numeric_distributions", _chart_distributions),
    ("correlation_heatmap", _chart_correlation),
    ("hourly_pattern", _chart_hourly),
    ("weekday_pattern", _chart_weekday),
    ("distance_fare_relation", _chart_distance_fare),
    ("payment_tip", _chart_payment_tip),
    ("source_profile", _chart_source_profile),
    ("demand_heatmap", _chart_demand_heatmap),
    ("daily_trend", _chart_daily_trend),
    ("top_zones", _chart_top_zones),
]


def visualize_step(df: pd.DataFrame, cfg: Config) -> StepResult:
    """차트를 생성하고 Artifact 목록으로 반환한다."""
    font = setup_style(cfg)
    log.info("시각화 시작 (폰트 %s, %s행)", font, f"{len(df):,}")

    artifacts, metrics, notes = [], {"font": font}, []
    for key, build in _BUILDERS:
        art, m = build(df, cfg)
        if art is None:                  # 필요한 컬럼이 없으면 해당 차트를 건너뛴다.
            log.warning("%s 생략 (데이터 없음)", key)
            continue
        artifacts.append(art)
        metrics[key] = m
        notes.append(f"[{art.name}] {art.caption}")

    metrics["figure_count"] = len(artifacts)
    log.info("차트 %d개 생성 (정적 %d · 인터랙티브 %d)", len(artifacts),
             sum(1 for a in artifacts if a.kind == "figure"),
             sum(1 for a in artifacts if a.kind == "plotly"))
    return StepResult(df=df, metrics=metrics, notes=notes, artifacts=artifacts)
