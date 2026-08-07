"""원본 데이터 시각화 — 정제 전 raw 데이터로 EDA를 수행한다.

목적이 다른 두 시각화 단계를 분리한 이유:
  - `visualize.py`(정제 후) : 정제된 데이터가 어떻게 생겼는지 "보여주는" 용도
  - `visualize_raw.py`(정제 전) : 정제 기준을 "발견"하는 용도(EDA 본연의 목적)

여기서 나온 발견(결측 구조, 극단 이상치가 상관관계를 가리는 문제, 시간대 패턴 등)이
이후 missing/duplicates/outliers/features 단계의 처리 기준을 설계하는 근거가 된다.

결측·품질이슈·IQR 이상치 차트는 `taxi_pipeline/viz/quality_charts.py`를 호출해서
만든다(노트북 Part 5·12를 함수로 뺀 모듈). `correlation_trap`·`hourly_pattern_raw`·
`payment_tip_raw`는 그 모듈에 없어서 이 파일에서 직접 그린다.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                       # noqa: E402

from ..config import Config               # noqa: E402
from ..viz.quality_charts import generate_data_quality_charts  # noqa: E402
from ..viz.style import (ACCENT, PAYMENT, PAYMENT_LABEL, POSITIVE,  # noqa: E402
                         PRIMARY, setup_style)
from .base import Artifact, StepResult    # noqa: E402
# is_rush_hour 정의를 파생변수 단계와 공유한다. 각자 하드코딩하면 정의가 갈린다.
from .features import RUSH_HOURS          # noqa: E402
from .visualize import mpl_artifact       # noqa: E402

log = logging.getLogger(__name__)


def _bytes_artifact(name: str, data: bytes, caption: str) -> Artifact:
    """이미 메모리에 있는 PNG 바이트를 그대로 저장하는 Artifact."""
    def save(dst: Path) -> None:
        dst.write_bytes(data)
    return Artifact(name=name, save=save, caption=caption, kind="figure")


def _chart_data_quality_handoff(df: pd.DataFrame, cfg: Config):
    """결측(matrix/bar/heatmap) + 품질이슈 + IQR 박스플롯을 quality_charts 모듈로 만든다.

    generate_data_quality_charts()는 output_dir에 PNG를 즉시 파일로 써버리는 함수라
    (Figure를 지연 반환하는 mpl_artifact()와 다른 계약), 스크래치 디렉터리에서 실행시키고
    바이트를 메모리로 읽은 뒤 즉시 지운다. runner.py가 Artifact.save()를 호출하는
    시점은 이 함수가 끝난 한참 뒤라, TemporaryDirectory의 with블록을 여기서 닫아버리면
    그때는 이미 파일이 사라진 뒤라서 바이트로 읽어서 들고 있어야 한다.
    """
    nullable = [c for c in cfg.columns.missing_group if c in df.columns]

    tmp = tempfile.mkdtemp(prefix="dq_charts_")
    try:
        dq = generate_data_quality_charts(df, output_dir=tmp, month_label=cfg.month,
                                          nullable_columns=nullable)
        image_bytes = {k: Path(p).read_bytes() for k, p in dq["images"].items()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return image_bytes, dq["stats"]


# ============================================================================
# 1. 극단 이상치 & 상관관계 함정
# ============================================================================
def _chart_correlation_trap(df: pd.DataFrame, cfg: Config):
    """이상치를 안 지우면 trip_distance-fare_amount 상관관계가 사라짐을 보인다."""
    corr_raw = float(df["trip_distance"].corr(df["fare_amount"]))

    sample = df[["trip_distance", "fare_amount"]].sample(
        n=min(30_000, len(df)), random_state=42)
    filt = sample[(sample["trip_distance"] < 50) & (sample["fare_amount"] < 200)
                  & (sample["trip_distance"] > 0)]
    corr_filtered = float(filt["trip_distance"].corr(filt["fare_amount"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(sample["trip_distance"], sample["fare_amount"], s=4, alpha=0.3, color=PRIMARY)
    axes[0].set_title(f"원본 그대로 (r={corr_raw:.3f})", fontsize=11)
    axes[0].set_xlabel("trip_distance (mile)")
    axes[0].set_ylabel("fare_amount (USD)")

    axes[1].scatter(filt["trip_distance"], filt["fare_amount"], s=4, alpha=0.3, color=POSITIVE)
    axes[1].set_title(f"거리<50mi·요금<$200 필터링 (r={corr_filtered:.3f})", fontsize=11)
    axes[1].set_xlabel("trip_distance (mile)")

    fig.suptitle(f"trip_distance ↔ fare_amount — 최댓값 {df['trip_distance'].max():,.0f}mile 등"
                 f" 극단 이상치가 상관관계를 가린다", fontsize=12)
    fig.tight_layout()

    caption = (f"원본 그대로 계산하면 상관계수가 r={corr_raw:.3f}로 거의 무관해 보이지만, "
               f"상식적으로 불가능한 극단치(최댓값 {df['trip_distance'].max():,.0f}mile)를 "
               f"제외하면 r={corr_filtered:.3f}로 뚜렷한 양의 관계가 드러난다. 소수의 "
               f"극단 이상치가 전체 상관관계를 완전히 무력화시킬 수 있다는 근거이며, "
               f"이상치를 먼저 제거한 뒤 상관·회귀분석을 해야 하는 이유다.")
    metrics = {"corr_raw": corr_raw, "corr_filtered": corr_filtered,
               "trip_distance_max": float(df["trip_distance"].max())}
    return mpl_artifact(fig, "raw_03_correlation_trap.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 2. 시간대별 운행 패턴 (원본)
# ============================================================================
def _chart_hourly_raw(df: pd.DataFrame, cfg: Config):
    """정제 전 원본 기준 시간대별 트립 수를 본다(러시아워 가정 검증용)."""
    pu = cfg.columns.pickup
    counts = df[pu].dt.hour.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [ACCENT if h in RUSH_HOURS else PRIMARY for h in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("시간대별 운행 수 (원본) — 빨강 = 현재 is_rush_hour 정의(7-9시·16-19시)", fontsize=11)
    ax.set_xlabel("승차 시간대 (시)")
    ax.set_ylabel("운행 건수")
    ax.set_xticks(range(24))
    fig.tight_layout()

    peak, low = int(counts.idxmax()), int(counts.idxmin())
    morning_slope = float(counts.loc[9] - counts.loc[7]) if 7 in counts.index and 9 in counts.index else None
    caption = (f"운행량은 {peak}시에 최다({int(counts[peak]):,}건), {low}시에 최저"
               f"({int(counts[low]):,}건)다. 오전 7~9시 구간은 완만히 증가할 뿐 뚜렷한 "
               f"피크로 보기 어려워, 현재 is_rush_hour 정의(7-9시·16-19시 이중 피크 가정)와 "
               f"실제 분포가 다소 차이난다 — 저녁 단일 피크에 가깝다.")
    metrics = {"peak_hour": peak, "peak_trips": int(counts[peak]),
               "low_hour": low, "low_trips": int(counts[low]),
               "morning_increase_7_to_9": morning_slope}
    return mpl_artifact(fig, "raw_04_hourly_pattern.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 3. 결제수단별 팁 분포 (원본)
# ============================================================================
def _chart_payment_tip_raw(df: pd.DataFrame, cfg: Config):
    """현금 결제의 팁이 구조적으로 0인지 원본에서 바로 확인한다."""
    d = df[df["payment_type"].isin(PAYMENT)].copy()
    d[PAYMENT_LABEL] = d["payment_type"].map(PAYMENT)
    sample = d.sample(n=min(50_000, len(d)), random_state=42)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = list(PAYMENT.values())
    data = [sample.loc[sample[PAYMENT_LABEL] == p, "tip_amount"].clip(upper=20) for p in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title("결제 방식별 팁 금액 분포 (원본, 5만행 샘플)", fontsize=12)
    ax.set_ylabel("팁 (USD)")
    fig.tight_layout()

    cash_zero_ratio = float((d.loc[d[PAYMENT_LABEL] == "현금", "tip_amount"] == 0).mean())
    card_median = float(d.loc[d[PAYMENT_LABEL] == "카드", "tip_amount"].median())
    caption = (f"현금 결제는 팁이 {cash_zero_ratio:.1%} 0으로 기록되는 반면, 카드 결제는 "
               f"중앙값 ${card_median:.2f}로 폭넓게 분포한다. 현금 팁이 실제로 없었던 게 "
               f"아니라 시스템에 기록되지 않는 구조적 특성이므로, 팁 분석은 카드결제만 "
               f"필터링해야 한다.")
    metrics = {"cash_zero_ratio": cash_zero_ratio, "card_tip_median": card_median}
    return mpl_artifact(fig, "raw_05_payment_tip.png", caption, cfg.visualize.dpi), metrics


# ============================================================================
# 단계 진입점
# ============================================================================
_BUILDERS = [
    ("correlation_trap", _chart_correlation_trap),
    ("hourly_pattern_raw", _chart_hourly_raw),
    ("payment_tip_raw", _chart_payment_tip_raw),
]

_HANDOFF_CAPTIONS = {
    "missing_matrix": "결측치 위치 패턴(5만행 샘플).",
    "missing_bar": "컬럼별 non-null 개수 — 5개 결측 컬럼이 시각적으로 드러난다.",
    "missing_heatmap": "컬럼 간 결측 동시발생 상관관계.",
    "quality_issues": "정제 전 데이터 품질 이슈 요약.",
    "iqr_boxplot": "IQR 기준 수치형 컬럼 이상치.",
}


def visualize_raw_step(df: pd.DataFrame, cfg: Config) -> StepResult:
    """정제 전 원본 데이터를 시각화해 정제 기준 설계의 근거를 남긴다. DataFrame은 바꾸지 않는다."""
    font = setup_style(cfg)
    log.info("원본 데이터 시각화 시작 (폰트: %s)", font)

    artifacts: list[Artifact] = []
    metrics: dict = {}
    notes: list[str] = []

    # ---- 결측(matrix/bar/heatmap) + 품질이슈 + IQR 박스플롯 ----------------------
    try:
        image_bytes, dq_stats = _chart_data_quality_handoff(df, cfg)
    except Exception as e:
        # 차트가 빠진 사실을 리포트에도 남긴다. 로그만 남기면 리포트만 읽는
        # 사람은 5개 차트가 원래 없었던 것으로 오해한다.
        log.exception("데이터 품질 차트 생성 실패 — 건너뛴다")
        notes.append(
            f"[한계] 데이터 품질 차트(결측 matrix/bar/heatmap · 품질이슈 · IQR 박스플롯) "
            f"생성에 실패해 이 리포트에서 빠졌다: {type(e).__name__}: {e}")
    else:
        for key, data in image_bytes.items():
            artifacts.append(_bytes_artifact(
                f"raw_dq_{key}.png", data, _HANDOFF_CAPTIONS.get(key, "")))
        metrics["data_quality_handoff"] = dq_stats
        if dq_stats.get("missing_all_pairs_correlated"):
            notes.append(
                "[원본EDA] (노은서 handoff) 5개 결측 컬럼이 정확히 같은 건수만큼 결측 "
                "— 동일 행에서 함께 비어있다.")
        if iqr := dq_stats.get("iqr_outliers"):
            top_col, top_v = max(iqr.items(), key=lambda kv: kv[1].get("pct") or 0)
            notes.append(
                f"[원본EDA] (노은서 handoff) IQR 기준 이상치 비율이 가장 큰 컬럼: "
                f"{top_col} {top_v['pct']}%.")

    # ---- 나머지 원본 시각화(상관관계 함정/시간대/결제수단) -----------------------
    for name, builder in _BUILDERS:
        try:
            art, m = builder(df, cfg)
        except Exception:
            log.exception("원본 시각화 %s 생성 실패 — 건너뛴다", name)
            continue
        artifacts.append(art)
        metrics[name] = m

    if cq := metrics.get("correlation_trap"):
        notes.append(
            f"[원본EDA] trip_distance-fare_amount 상관계수가 원본 그대로면 "
            f"r={cq['corr_raw']:.3f}이지만 극단 이상치를 제외하면 r={cq['corr_filtered']:.3f}"
            f"로 바뀐다 — 이상치를 먼저 제거해야 상관·회귀분석이 의미를 갖는다는 근거.")
    if hp := metrics.get("hourly_pattern_raw"):
        notes.append(
            f"[원본EDA] 원본 시간대별 운행량은 {hp['peak_hour']}시 최다·{hp['low_hour']}시 "
            f"최저다. 오전 7~9시는 완만한 증가 구간일 뿐이라, is_rush_hour의 아침 경계는 "
            f"재검토할 여지가 있다(engineer_features 단계 참고).")

    log.info("원본 데이터 시각화 완료 (%d개 산출물)", len(artifacts))
    return StepResult(df=df, metrics=metrics, notes=notes, artifacts=artifacts)
