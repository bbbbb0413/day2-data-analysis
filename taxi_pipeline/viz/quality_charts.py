"""
자동화 파이프라인에 붙이는 '결측치 & 이상치' 시각화 모듈.

visualization_original.ipynb의 Part 5(결측치 & 데이터 품질 이슈)와
Part 12(수치형 이상치 박스플롯)를 그대로 재현하되, 노트북이 아니라
어떤 월의 DataFrame이 들어와도 재사용 가능한 함수 형태로 뺀 것이다.

파이프라인 쪽에서 할 일은 단 하나:
    from taxi_pipeline.viz.quality_charts import generate_data_quality_charts
    result = generate_data_quality_charts(df, output_dir=..., month_label="2026-06")
그리고 result["stats"] 안의 숫자를 리포트 템플릿에 채워 넣으면 된다
(현재 `steps/visualize_raw.py`가 이 방식으로 쓴다). result["markdown"]도 있다.
"""

from __future__ import annotations

import platform
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns


def _setup_korean_font() -> None:
    """한글 라벨이 깨지지 않도록 폰트를 설정한다 (노트북 Part 1과 동일한 로직)."""
    if platform.system() == "Darwin":
        font_list = [f.name for f in fm.fontManager.ttflist]
        plt.rcParams["font.family"] = "AppleGothic" if "AppleGothic" in font_list else "NanumGothic"
    else:
        # 리눅스 CI/서버 환경: 나눔고딕이 설치돼 있어야 한다 (예: apt-get install fonts-nanum)
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


_setup_korean_font()

# 이 5개 컬럼은 원본 NYC Yellow Taxi 스키마에서 결측이 발생하는 컬럼이다.
# 월이 바뀌어도 스키마(컬럼 구성) 자체는 동일하다고 가정한다 — 스키마가
# 달라지는 경우에는 호출부에서 다른 값을 넘기면 된다.
DEFAULT_NULLABLE_COLUMNS = [
    "passenger_count", "RatecodeID", "store_and_fwd_flag",
    "congestion_surcharge", "Airport_fee",
]


def _infer_numeric_cols(df: pd.DataFrame) -> list[str]:
    id_like = {"VendorID", "PULocationID", "DOLocationID",
               "payment_type", "RatecodeID", "passenger_count"}
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in id_like]


def generate_data_quality_charts(
    df: pd.DataFrame,
    output_dir: Path | str,
    month_label: str,
    nullable_columns: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    missing_sample_n: int = 50_000,
    random_state: int = 42,
) -> dict:
    """결측치(Part 5) + IQR 이상치(Part 12) 차트를 생성하고 이미지 경로/수치를 반환한다.

    Parameters
    ----------
    df : 정제하지 않은 원본 월별 DataFrame (예: 2026-06 raw parquet을 읽은 것)
    output_dir : PNG를 저장할 디렉터리. 월별로 구분하고 싶으면 호출부에서
                 output/2026-06 처럼 미리 만들어서 넘기면 된다 (이 함수는 강제하지 않음).
    month_label : 리포트 문구에 쓸 라벨, 예: "2026-06"
    nullable_columns : 결측 대상 컬럼 목록. 생략하면 DEFAULT_NULLABLE_COLUMNS 사용.
    numeric_cols : IQR 박스플롯 대상 수치형 컬럼. 생략하면 스키마에서 자동 추론.

    Returns
    -------
    dict with keys:
      - "images": {"missing_matrix": Path, "missing_bar": Path, "missing_heatmap": Path,
                    "quality_issues": Path, "iqr_boxplot": Path}
      - "stats": {
            "total_rows": int,
            "missing": {col: {"count": int, "pct": float}, ...},
            "missing_all_pairs_correlated": bool,   # 5개 컬럼이 전부 함께 비는지
            "quality_issues": {label: {"count": int, "pct": float}, ...},
            "iqr_outliers": {col: {"count": int, "pct": float}}, ...},
        }
      - "markdown": 위 stats를 그대로 문장/표로 렌더링한 마크다운 블록 (str)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nullable_columns = nullable_columns or DEFAULT_NULLABLE_COLUMNS
    numeric_cols = numeric_cols or _infer_numeric_cols(df)
    total_rows = len(df)

    images: dict[str, Path] = {}
    stats: dict = {"total_rows": total_rows}

    # ---------- Part 5-A. missingno matrix / bar / heatmap ----------
    sample_n = min(missing_sample_n, total_rows)
    msno_sample = df.sample(sample_n, random_state=random_state)

    ax = msno.matrix(msno_sample, figsize=(12, 6), fontsize=10,
                      color=(0.27, 0.52, 0.70))
    ax.set_title(f"결측치 위치 패턴 ({month_label}, {sample_n:,}행 샘플)", fontsize=13)
    p = output_dir / "05a_missing_matrix.png"
    ax.get_figure().savefig(p, dpi=150, bbox_inches="tight")
    plt.close(ax.get_figure())
    images["missing_matrix"] = p

    ax = msno.bar(df, figsize=(10, 6), fontsize=10, color="steelblue")
    ax.set_title(f"컬럼별 non-null 개수 ({month_label}, 전체 {total_rows:,}행)", fontsize=13)
    p = output_dir / "05a_missing_bar.png"
    ax.get_figure().savefig(p, dpi=150, bbox_inches="tight")
    plt.close(ax.get_figure())
    images["missing_bar"] = p

    ax = msno.heatmap(df, figsize=(8, 6), fontsize=10, cmap="RdBu")
    ax.set_title(f"컬럼 간 결측 동시발생 상관관계 ({month_label})", fontsize=13)
    p = output_dir / "05a_missing_heatmap.png"
    ax.get_figure().savefig(p, dpi=150, bbox_inches="tight")
    plt.close(ax.get_figure())
    images["missing_heatmap"] = p

    na_counts = df[nullable_columns].isna().sum()
    stats["missing"] = {
        col: {"count": int(n), "pct": round(n / total_rows * 100, 2)}
        for col, n in na_counts.items()
    }
    # 5개 컬럼이 전부 정확히 같은 건수만큼 비어있으면 "동일 행에서 함께 결측"일 가능성이 높다.
    stats["missing_all_pairs_correlated"] = na_counts.nunique() == 1 and na_counts.min() > 0

    # ---------- Part 5-B. 정제 전 데이터 품질 이슈 요약 ----------
    dur_min = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    issues = {
        f"결측치 포함 행 ({len(nullable_columns)}개 컬럼 중 1개+)": df[nullable_columns].isna().any(axis=1).sum(),
        "total_amount 음수": (df["total_amount"] < 0).sum(),
        "fare_amount 음수": (df["fare_amount"] < 0).sum(),
        "trip_distance ≤ 0": (df["trip_distance"] <= 0).sum(),
        "소요시간(승차→하차) ≤ 0": (dur_min <= 0).sum(),
        "passenger_count = 0": (df["passenger_count"] == 0).sum(),
    }
    issue_df = pd.DataFrame({"이슈": list(issues.keys()), "건수": list(issues.values())})
    issue_df["비율(%)"] = (issue_df["건수"] / total_rows * 100).round(2)
    issue_df = issue_df.sort_values("건수")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(issue_df["이슈"], issue_df["건수"], color="indianred", alpha=0.85)
    for bar, pct in zip(bars, issue_df["비율(%)"]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=9)
    ax.set_title(f"정제 전 데이터 품질 이슈 요약 ({month_label}, 전체 {total_rows:,}행 기준)", fontsize=13)
    ax.set_xlabel("건수")
    plt.tight_layout()
    p = output_dir / "05b_data_quality_issues.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    images["quality_issues"] = p

    stats["quality_issues"] = {
        row["이슈"]: {"count": int(row["건수"]), "pct": float(row["비율(%)"])}
        for _, row in issue_df.iterrows()
    }

    # ---------- Part 12. 수치형 이상치 — 박스플롯 (IQR) ----------
    n_c = 3
    n_r = int(np.ceil(len(numeric_cols) / n_c))
    fig, axes = plt.subplots(n_r, n_c, figsize=(18, n_r * 4))
    axes = np.atleast_1d(axes).flatten()

    iqr_stats: dict = {}
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        col_data = df[col].dropna()
        if col_data.empty:
            # 그 달 데이터에서 이 컬럼이 전부 결측이면 박스플롯을 그릴 수 없다 —
            # 빈 서브플롯 대신 안내 문구만 표시하고 통계는 0으로 기록한다.
            ax.text(0.5, 0.5, "데이터 없음\n(전체 결측)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(col, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            iqr_stats[col] = {"count": 0, "pct": 0.0, "q1": None, "q3": None, "lo": None, "hi": None}
            continue

        sns.boxplot(x=col_data, ax=ax, color="lightcoral",
                    flierprops=dict(markersize=2, alpha=0.3))
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((col_data < lo) | (col_data > hi)).sum())
        ax.set_title(f"{col}\nIQR 이상치: {n_out:,}건 ({n_out/total_rows*100:.1f}%)", fontsize=10)
        ax.set_xlabel("")
        iqr_stats[col] = {
            "count": n_out,
            "pct": round(n_out / total_rows * 100, 2),
            "q1": float(q1), "q3": float(q3), "lo": float(lo), "hi": float(hi),
        }

    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"수치형 컬럼 박스플롯 — IQR 이상치 탐지 ({month_label})", fontsize=14, y=1.01)
    plt.tight_layout()
    p = output_dir / "12_numeric_boxplots.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    images["iqr_boxplot"] = p

    stats["iqr_outliers"] = iqr_stats

    markdown = _render_markdown(month_label, images, stats)
    return {"images": images, "stats": stats, "markdown": markdown}


def _render_markdown(month_label: str, images: dict[str, Path], stats: dict) -> str:
    total_rows = stats["total_rows"]
    missing_lines = "\n".join(
        f"| {col} | {v['count']:,} | {v['pct']}% |"
        for col, v in stats["missing"].items()
    )
    same_row_note = (
        "5개 컬럼 모두 정확히 같은 건수만큼 결측 → 동일 행에서 함께 비어있을 가능성이 높음 (heatmap으로 확인)"
        if stats["missing_all_pairs_correlated"]
        else "컬럼별 결측 건수가 서로 달라 개별적으로 발생했을 가능성이 있음"
    )
    issue_lines = "\n".join(
        f"| {label} | {v['count']:,} | {v['pct']}% |"
        for label, v in sorted(stats["quality_issues"].items(), key=lambda kv: -kv[1]["count"])
    )
    iqr_lines = "\n".join(
        f"| {col} | {v['count']:,} | {v['pct']}% |"
        for col, v in sorted(stats["iqr_outliers"].items(), key=lambda kv: -kv[1]["pct"])
    )

    return f"""## 결측치 & 데이터 품질 이슈 시각화 ({month_label})

전체 {total_rows:,}행 기준.

![결측치 위치 패턴]({images['missing_matrix']})
![컬럼별 non-null 개수]({images['missing_bar']})
![결측 동시발생 상관관계]({images['missing_heatmap']})

| 컬럼 | 결측 건수 | 비율 |
|---|---:|---:|
{missing_lines}

{same_row_note}

![데이터 품질 이슈 요약]({images['quality_issues']})

| 이슈 | 건수 | 비율 |
|---|---:|---:|
{issue_lines}

## 수치형 이상치 — 박스플롯 (IQR, {month_label})

![IQR 박스플롯]({images['iqr_boxplot']})

| 컬럼 | IQR 이상치 건수 | 비율 |
|---|---:|---:|
{iqr_lines}
"""


if __name__ == "__main__":
    # 간단한 동작 확인용 (파이프라인에서는 이 블록을 쓰지 않음)
    import sys
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/yellow_tripdata_2026-05.parquet")
    month = sys.argv[2] if len(sys.argv) > 2 else "2026-05"
    pdf = pd.read_parquet(raw_path)
    result = generate_data_quality_charts(pdf, output_dir="output", month_label=month)
    print(result["markdown"])
