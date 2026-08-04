"""[단계 4] 기본 EDA — 지표를 '출력'하지 않고 '반환'한다.

자동화에서 EDA의 역할이 달라진다. 사람이 한 번 보고 끝나는 게 아니라,
매 실행 같은 지표를 남겨 **이전 실행과 비교**하는 것이 목적이 된다.
평균 요금이 갑자기 20% 뛰면 그건 분석 결과가 아니라 데이터 사고 신호다.

그래서 모든 결과를 JSON으로 직렬화 가능한 dict로 만든다.
사람이 읽는 형태는 report.py가 이 dict로부터 렌더링한다.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def profile(df: pd.DataFrame, cfg: Config) -> StepResult:
    """정제된 데이터의 기본 분포를 지표로 수집한다. DataFrame은 바꾸지 않는다.

    ★ 결측 소스를 어떻게 다뤘는지 지표마다 명시한다.
      record_source='partial'은 승객수·요율 컬럼이 아예 없으므로, 해당 컬럼을
      쓰는 지표는 'full' 한정으로 계산한다(문서 §2.7 기준 ④).
      반대로 요금·거리·시각은 결측이 0건이라 전체를 쓴다(기준 ③).
    """
    c = cfg.columns
    pu = c.pickup
    has_source = "record_source" in df.columns
    full = df[df["record_source"] == "full"] if has_source else df
    notes: list[str] = []

    num_cols = [x for x in ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
                if x in df.columns]

    # ---- 수치형 기술통계 (전체 사용 — 기준 ③) -------------------------------
    desc = df[num_cols].describe(percentiles=[.25, .5, .75, .95, .99]).T
    describe = {col: {k: float(v) for k, v in row.items()} for col, row in desc.iterrows()}

    # 평균÷중앙값으로 쏠림을 본다. 표준편차는 이상치에 같이 부풀려져 기준이 안 된다.
    skew = {}
    for col in num_cols:
        med = desc.loc[col, "50%"]
        if med > 0:
            skew[col] = float(desc.loc[col, "mean"] / med)
    if heavy := {k: v for k, v in skew.items() if v > 1.5}:
        notes.append("평균이 중앙값의 1.5배를 넘는 컬럼: "
                     + ", ".join(f"{k} {v:.2f}배" for k, v in heavy.items())
                     + " — 평균만 보고 판단하면 안 된다.")

    metrics: dict = {
        "rows": len(df),
        "describe": describe,
        "mean_over_median": skew,
    }

    # ---- 소스별 프로파일 (결측 처리 판단의 사후 검증) ------------------------
    # 두 소스의 프로파일이 크게 다르면 dropna()를 안 한 판단이 옳았다는 증거가 된다.
    if has_source and df["record_source"].nunique() > 1:
        prof = df.groupby("record_source").agg(
            rows=("total_amount", "size"),
            mean_distance=("trip_distance", "mean"),
            mean_fare=("fare_amount", "mean"),
            mean_tip=("tip_amount", "mean"),
            mean_total=("total_amount", "mean"),
        )
        metrics["by_record_source"] = {
            str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in prof.iterrows()
        }
        tips = prof["mean_tip"]
        if tips.min() > 0:
            notes.append(
                f"소스별 평균 팁이 {tips.max():.2f} vs {tips.min():.2f}로 "
                f"{tips.max()/tips.min():.1f}배 차이 — 섞어서 평균 내면 왜곡되고 "
                "한쪽을 지우면 편향된다. 구분 보존이 옳았다.")

    # ---- 시간 패턴 (전체 사용) ----------------------------------------------
    by_hour = df.groupby(df[pu].dt.hour).agg(
        rows=("total_amount", "size"), mean_fare=("fare_amount", "mean"))
    by_dow = df.groupby(df[pu].dt.dayofweek).agg(
        rows=("total_amount", "size"), mean_fare=("fare_amount", "mean"),
        mean_tip=("tip_amount", "mean"))
    metrics["by_hour"] = {int(k): {kk: float(vv) for kk, vv in v.items()}
                          for k, v in by_hour.iterrows()}
    metrics["by_dayofweek"] = {int(k): {kk: float(vv) for kk, vv in v.items()}
                               for k, v in by_dow.iterrows()}
    metrics["peak_hour"] = int(by_hour["rows"].idxmax())
    metrics["quietest_hour"] = int(by_hour["rows"].idxmin())

    # ---- 승하차 존 TOP (sentinel 변환으로 Unknown/구역외는 이미 NaN) ---------
    if "PULocationID" in df.columns:
        top = df["PULocationID"].value_counts().head(10)
        metrics["top_pickup_zones"] = {int(k): int(v) for k, v in top.items()}

    # ---- 범주형 분포 (full 한정 — 기준 ④) ------------------------------------
    cat: dict[str, dict] = {}
    for col in c.categorical:
        if col not in full.columns:
            continue
        vc = full[col].value_counts(dropna=False).head(6)
        cat[col] = {("null" if pd.isna(k) else str(k)): int(v) for k, v in vc.items()}
    metrics["categorical_full_only"] = cat
    metrics["categorical_base_rows"] = len(full)
    if has_source:
        notes.append(f"범주형 분포는 record_source='full' {len(full):,}행 한정으로 계산했다 "
                     "— partial 소스는 해당 컬럼 자체가 없어 분모에 넣으면 오염된다.")

    # ---- 요금 구조 -----------------------------------------------------------
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        metrics["correlation"] = {i: {j: float(corr.loc[i, j]) for j in num_cols}
                                  for i in num_cols}

    # 마일당 요금은 중앙값을 쓴다: 0.01마일에 요금 8달러 같은 행이 섞이면
    # 비율의 평균이 800/mile로 튀어 구간 대표값 역할을 못 한다.
    bins = [0, 1, 2, 5, 10, 100]
    binned = df.assign(
        bucket=pd.cut(df["trip_distance"], bins),
        per_mile=df["fare_amount"] / df["trip_distance"],
    )
    per_bin = binned.groupby("bucket", observed=True).agg(
        rows=("fare_amount", "size"), mean_fare=("fare_amount", "mean"),
        median_per_mile=("per_mile", "median"))
    metrics["by_distance_bucket"] = {
        str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in per_bin.iterrows()
    }

    log.info("EDA 완료: %s행, 지표 %s종", f"{len(df):,}", len(metrics))
    return StepResult(df=df, metrics=metrics, notes=notes)
