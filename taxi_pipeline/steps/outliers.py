"""기간, 소요시간, 거리와 금액 기준으로 이상치를 처리한다."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)

# TLC 원본은 거리를 mile로 기록하지만 속력 상한(cfg.outliers.speed_max_kmh)은
# km/h로 잡혀 있어 환산이 필요하다.
MILES_TO_KM = 1.60934


def filter_outliers(df: pd.DataFrame, cfg: Config) -> StepResult:
    """물리적으로 불가능하거나 명백한 오기록을 제거한다."""
    c, o = cfg.columns, cfg.outliers
    pu, do = c.pickup, c.dropoff
    month = pd.Period(cfg.month)
    dur = (df[do] - df[pu]).dt.total_seconds()
    notes: list[str] = []

    duration_ok = dur.between(o.duration_min_sec, o.duration_max_sec)

    # 속력(km/h) = 거리(mile→km 환산) / 소요시간(초→시간 환산) 으로 계산한다.
    # 입력 df를 건드리지 않기 위해 별도 Series로 두고, 필터링한 결과에만 컬럼으로 붙인다.
    speed_kmh = pd.Series(
        np.where(dur > 0, df["trip_distance"] * MILES_TO_KM / (dur / 3600), np.nan),
        index=df.index, name="speed_kmh")

    rules: list[tuple[str, pd.Series]] = [
        # 승차시각이 기준 월에 포함되는지 확인한다.
        ("period", df[pu].between(month.start_time, month.end_time)),
        # 이동거리가 설정 범위에 포함되는지 확인한다.
        ("distance", df["trip_distance"].between(o.distance_min, o.distance_max)),
        # 총액과 요금이 정상 범위인지 확인한다.
        ("amount", (df["total_amount"] > 0) & (df["fare_amount"] >= 0)),

        # 거리, 시간을 각각 봐서는 못 잡는 거리, 시간 조합형 오기록을 잡는다!
        ("speed", speed_kmh.isna() | (speed_kmh <= o.speed_max_kmh)),
    ]

    # 소요시간 정책에 따라 행을 제거하거나 플래그를 추가한다.
    if o.duration_policy == "drop":
        rules.insert(1, ("duration", duration_ok))

    keep = pd.Series(True, index=df.index)
    per_rule: dict[str, int] = {}
    prev = len(df)
    for name, cond in rules:
        keep &= cond                    # 규칙을 순서대로 누적 적용한다.
        now = int(keep.sum())
        per_rule[name] = prev - now
        prev = now

    out = df[keep].copy()
    out["speed_kmh"] = speed_kmh[keep]
    if o.duration_policy == "flag":
        # 행은 유지하고 duration_valid 컬럼으로 표시한다.
        out["duration_valid"] = duration_ok.loc[out.index]
        notes.append(
            f"duration_policy='flag' — 소요시간 이상 {int((~duration_ok).sum()):,}건을 "
            "삭제하지 않고 duration_valid=False로 표시했다.")

    metrics: dict = {
        "rows_in": len(df),
        "rows_out": len(out),
        "rows_dropped": len(df) - len(out),
        "dropped_by_rule": per_rule,
        "duration_policy": o.duration_policy,
        "retention_ratio": len(out) / len(df) if len(df) else 0.0,
        "negative_total_after": int((out["total_amount"] < 0).sum()),
    }

    # 필터 적용 전후의 사업자 구성을 비교한다.
    vendor = c.vendor
    wiped = []
    if vendor in df.columns:
        before_v = set(df[vendor].dropna().unique())
        after_v = set(out[vendor].dropna().unique())
        for v in sorted(before_v - after_v):
            sub = df[df[vendor] == v]
            sub_dur = (sub[do] - sub[pu]).dt.total_seconds()
            wiped.append({
                "vendor": int(v),
                "rows": int(len(sub)),
                "nonpositive_duration_ratio": float((sub_dur <= 0).mean()),
            })
    metrics["vendors_wiped_out"] = wiped
    if wiped:
        for w in wiped:
            notes.append(
                f"[경고] VendorID={w['vendor']}가 {w['rows']:,}건 전량 삭제됐다 "
                f"(소요시간<=0 비율 {w['nonpositive_duration_ratio']:.1%}). "
                "값이 이상한 게 아니라 하차시각을 기록하지 않는 사업자일 수 있다. "
                "운행량·요금 분석이 목적이면 duration_policy='flag'로 바꿔라.")
            log.warning("이상치 필터로 VendorID=%s 전량(%s건) 삭제됨",
                        w["vendor"], f"{w['rows']:,}")
        notes.append(
            f"[한계] 소요시간 규칙(현재 정책 '{o.duration_policy}')이 특정 사업자를 "
            f"통째로 제거했다. 값이 이상한 게 아니라 기록 방식이 다른 것이므로, "
            f"운행량·요금·존 분석이 목적이라면 duration_policy='flag'로 바꿔 "
            f"행을 보존해야 한다.")

    log.info("이상치 처리: %s행 제거 (보존율 %.2f%%)",
             f"{len(df) - len(out):,}", metrics["retention_ratio"] * 100)
    return StepResult(df=out, metrics=metrics, notes=notes)
