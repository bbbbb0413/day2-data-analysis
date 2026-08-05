"""결측 구조를 분석하고 결측 여부를 구분한다."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def analyze_missing(df: pd.DataFrame, cfg: Config) -> StepResult:
    """결측 컬럼이 함께 비는 구조인지 확인한다."""
    c = cfg.columns
    group = [x for x in c.missing_group if x in df.columns]
    na = df[group].isna()
    all_na = na.all(axis=1)
    partial = int((na.any(axis=1) & ~all_na).sum())

    metrics: dict = {
        "rows": len(df),
        "missing_rows": int(all_na.sum()),
        "missing_ratio": float(all_na.mean()),
        "partial_missing_rows": partial,
        "by_column": {col: int(df[col].isna().sum()) for col in df.columns
                      if df[col].isna().any()},
    }
    notes: list[str] = []

    # 결측 컬럼이 같은 행에서 함께 비는지 확인한다.
    metrics["is_structural"] = bool(partial == 0 and all_na.any())
    if metrics["is_structural"]:
        notes.append(
            f"{len(group)}개 컬럼이 정확히 같은 {int(all_na.sum()):,}행에서만 결측이고 "
            f"일부만 결측인 행은 0건 → 구조적 결측(MNAR). 삭제·대체 모두 부적절.")

    if not all_na.any():
        return StepResult(df=df, metrics=metrics, notes=notes)

    # 결측 행의 데이터 소스 차이를 확인한다.
    if "payment_type" in df.columns:
        pt0 = float((df.loc[all_na, "payment_type"] == 0).mean())
        metrics["missing_payment_type_zero_ratio"] = pt0
        notes.append(f"결측행의 payment_type=0 비율 {pt0:.1%} (0은 TLC 코드북에 없는 값)")

    if c.vendor in df.columns:
        na_v = {int(v) for v in df.loc[all_na, c.vendor].dropna().unique()}
        ok_v = {int(v) for v in df.loc[~all_na, c.vendor].dropna().unique()}
        metrics["vendor_only_in_missing"] = sorted(na_v - ok_v)
        metrics["vendor_only_in_full"] = sorted(ok_v - na_v)
        if na_v - ok_v:
            notes.append(f"VendorID {sorted(na_v - ok_v)}는 결측행에만 존재 → 제출 소스가 다름")

    # dropna 적용 시 시간대와 요일의 편향을 확인한다.
    pu = c.pickup
    by_dow = all_na.groupby(df[pu].dt.dayofweek).mean()
    by_hour = all_na.groupby(df[pu].dt.hour).mean()
    metrics["dropna_impact"] = {
        "kept_rows": int((~all_na).sum()),
        "kept_ratio": float((~all_na).mean()),
        "missing_ratio_by_dayofweek": {int(k): float(v) for k, v in by_dow.items()},
        "missing_ratio_by_hour": {int(k): float(v) for k, v in by_hour.items()},
        "mean_distance_all": float(df["trip_distance"].mean()),
        "mean_distance_after_dropna": float(df.loc[~all_na, "trip_distance"].mean()),
    }
    hi, lo = int(by_hour.idxmax()), int(by_hour.idxmin())
    notes.append(
        f"dropna() 시 표본 {(~all_na).mean():.2%}만 남고, 결측 비중이 "
        f"{hi}시 {by_hour[hi]:.1%} vs {lo}시 {by_hour[lo]:.1%}로 쏠려 있어 "
        f"심야·주말이 선택적으로 삭제된다.")

    # 총액을 역산하여 fillna(0)의 왜곡을 확인한다.
    flag = c.missing_flag
    parts = [x for x in c.amount_parts if x in df.columns and x != flag]
    if flag in df.columns and "total_amount" in df.columns:
        resid = (df.loc[all_na, "total_amount"]
                 - df.loc[all_na, parts].fillna(0).sum(axis=1)).round(2)
        top = resid.value_counts().head(5)
        real_mean = float(df[flag].mean())            # 결측 제외한 실측 평균
        zero_mean = float(df[flag].fillna(0).mean())  # 0으로 채운 평균
        metrics["fillna_zero_check"] = {
            "column": flag,
            "residual_top": {float(k): int(v) for k, v in top.items()},
            "mean_observed": real_mean,
            "mean_if_filled_zero": zero_mean,
            "distortion": float(zero_mean / real_mean - 1) if real_mean else None,
        }
        if not top.empty:
            val, cnt = float(top.index[0]), int(top.iloc[0])
            notes.append(
                f"{flag} 결측행의 {cnt/len(resid):.1%}는 총액 역산 결과 실제 {val:.2f}가 "
                f"부과됨 → fillna(0) 시 평균이 {real_mean:.3f}→{zero_mean:.3f}로 왜곡.")

    return StepResult(df=df, metrics=metrics, notes=notes)


def prepare_missing(df: pd.DataFrame, cfg: Config) -> StepResult:
    """sentinel 값을 변환하고 record_source를 추가한다."""
    c = cfg.columns
    df = df.copy()
    notes: list[str] = []

    # sentinel 값을 NaN으로 변환한다.
    converted: dict[str, int] = {}
    for col, bad in cfg.sentinels.items():
        if col not in df.columns:
            continue
        hit = int(df[col].isin(bad).sum())
        if hit:
            # 기존 dtype을 유지하면서 결측값을 변환한다.
            df[col] = df[col].where(~df[col].isin(bad), np.nan)
            converted[col] = hit
    total_converted = sum(converted.values())
    if total_converted:
        notes.append(
            f"위장 결측 {total_converted:,}건을 NaN으로 변환 "
            f"({', '.join(f'{k} {v:,}' for k, v in converted.items())}) "
            "— isna()만으로는 놓쳤을 결측이다.")

    # 결측 여부에 따라 record_source를 구분한다.
    flag = c.missing_flag
    df["record_source"] = np.where(df[flag].isna(), "partial", "full")
    counts = df["record_source"].value_counts().to_dict()
    notes.append(
        f"record_source 부여 — full {counts.get('full', 0):,} / "
        f"partial {counts.get('partial', 0):,}. 행 삭제 없음.")
    # partial 행은 일부 분석에서 제외한다.
    if counts.get("partial"):
        notes.append(
            f"[한계] partial 소스 {counts['partial']:,}행은 행을 남겼지만 승객수·요율·"
            f"결제수단 컬럼 자체가 없다. 이 컬럼을 쓰는 분석(범주형 분포, 팁 모델)에서는 "
            f"자동으로 제외되므로, 그 결과는 전체가 아닌 부분집합에 대한 것이다.")

    log.info("결측 처리: sentinel %s건 변환, record_source 부여", f"{total_converted:,}")
    return StepResult(
        df=df,
        metrics={
            "sentinels_converted": converted,
            "sentinels_total": total_converted,
            "record_source_counts": {k: int(v) for k, v in counts.items()},
            "rows_dropped": 0,          # 이 단계에서는 행을 삭제하지 않는다.
        },
        notes=notes,
    )
