"""[단계 1] 결측치 처리 — 문서 §2

이 단계의 결론: **행을 지우지도, 값을 채우지도 않는다.**
근거를 지표로 남겨, 나중에 누군가 dropna()를 넣으려 할 때 반박 자료가 되게 한다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def analyze_missing(df: pd.DataFrame, cfg: Config) -> StepResult:
    """결측의 '구조'를 먼저 확인한다. 처리 방식은 이 결과로 결정된다.

    DataFrame은 건드리지 않고 근거만 수집하는 분석 전용 단계다.
    """
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

    # ---- 핵심 판정: '일부만 결측인 행'이 0건인가? (문서 §2.2) -----------------
    # 0건이면 결측이 우연히 흩어진 것(MCAR)일 수 없다. 값을 못 받은 게 아니라
    # 애초에 그 필드를 제출하지 않는 다른 소스의 레코드라는 뜻이다.
    # 이 판정 하나로 dropna/fillna를 쓸지 말지가 갈린다.
    metrics["is_structural"] = bool(partial == 0 and all_na.any())
    if metrics["is_structural"]:
        notes.append(
            f"{len(group)}개 컬럼이 정확히 같은 {int(all_na.sum()):,}행에서만 결측이고 "
            f"일부만 결측인 행은 0건 → 구조적 결측(MNAR). 삭제·대체 모두 부적절.")

    if not all_na.any():
        return StepResult(df=df, metrics=metrics, notes=notes)

    # ---- 결측행이 '다른 소스'라는 증거 (문서 §2.3) ---------------------------
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

    # ---- dropna()를 쓰면 무엇이 사라지는가 (문서 §2.4) -----------------------
    # 결측이 시간대·요일에 쏠려 있으면 dropna()는 '심야·주말을 골라 버리는' 처리가 된다.
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

    # ---- fillna(0)이 틀린 이유 (문서 §2.5) ----------------------------------
    # total_amount에서 결측 컬럼을 뺀 나머지 항목 합을 빼면
    # '총액에는 반영됐지만 컬럼에는 안 적힌 금액' = 결측된 실제 값이 나온다.
    flag = c.missing_flag
    parts = [x for x in c.amount_parts if x in df.columns and x != flag]
    if flag in df.columns and "total_amount" in df.columns:
        resid = (df.loc[all_na, "total_amount"]
                 - df.loc[all_na, parts].fillna(0).sum(axis=1)).round(2)
        top = resid.value_counts().head(5)
        real_mean = float(df[flag].mean())            # 결측 제외한 실측 평균
        zero_mean = float(df[flag].fillna(0).mean())  # 0으로 채웠을 때
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
    """문서 §2.7 기준을 적용한다: 삭제·대체 없이 '구분'만 한다."""
    c = cfg.columns
    df = df.copy()
    notes: list[str] = []

    # ---- 기준 ⑤ : 위장 결측(sentinel)을 NaN으로 명시 변환 --------------------
    # payment_type=0, RatecodeID=99 등은 isna()에 안 잡히지만 의미상 결측이다.
    # 변환하지 않으면 최빈값·평균 같은 집계가 조용히 오염된다.
    converted: dict[str, int] = {}
    for col, bad in cfg.sentinels.items():
        if col not in df.columns:
            continue
        hit = int(df[col].isin(bad).sum())
        if hit:
            # where + NaN: 수치형 dtype을 유지하면서 결측만 표시한다
            df[col] = df[col].where(~df[col].isin(bad), np.nan)
            converted[col] = hit
    total_converted = sum(converted.values())
    if total_converted:
        notes.append(
            f"위장 결측 {total_converted:,}건을 NaN으로 변환 "
            f"({', '.join(f'{k} {v:,}' for k, v in converted.items())}) "
            "— isna()만으로는 놓쳤을 결측이다.")

    # ---- 기준 ① : 행 삭제 대신 소스 구분 플래그 -----------------------------
    # 결측행은 '불량 데이터'가 아니라 '필드 구성이 다른 소스의 데이터'다.
    # 지우면 심야·주말·특정 사업자가 통째로 사라지므로(§2.4) 표시만 한다.
    # 이후 승객수·요율 분석은 record_source == "full" 부분집합에서만 수행한다.
    flag = c.missing_flag
    df["record_source"] = np.where(df[flag].isna(), "partial", "full")
    counts = df["record_source"].value_counts().to_dict()
    notes.append(
        f"record_source 부여 — full {counts.get('full', 0):,} / "
        f"partial {counts.get('partial', 0):,}. 행 삭제 없음.")

    log.info("결측 처리: sentinel %s건 변환, record_source 부여", f"{total_converted:,}")
    return StepResult(
        df=df,
        metrics={
            "sentinels_converted": converted,
            "sentinels_total": total_converted,
            "record_source_counts": {k: int(v) for k, v in counts.items()},
            "rows_dropped": 0,          # 이 단계는 절대 행을 지우지 않는다
        },
        notes=notes,
    )
