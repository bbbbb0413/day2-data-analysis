"""[단계 3] 이상치 처리 — 문서 §4

중복 처리를 '먼저' 끝낸 뒤 실행해야 한다. 상쇄쌍을 제거하면 음수 금액이 89% 줄어드는데,
순서를 바꾸면 이상치 필터가 취소 쌍의 한쪽만 지워 짝이 깨진 반쪽짜리 기록이 남는다.
runner의 단계 순서가 그래서 고정되어 있다.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def filter_outliers(df: pd.DataFrame, cfg: Config) -> StepResult:
    """물리적으로 불가능하거나 명백한 오기록을 제거한다."""
    c, o = cfg.columns, cfg.outliers
    pu, do = c.pickup, c.dropoff
    month = pd.Period(cfg.month)
    dur = (df[do] - df[pu]).dt.total_seconds()
    notes: list[str] = []

    duration_ok = dur.between(o.duration_min_sec, o.duration_max_sec)

    rules: list[tuple[str, pd.Series]] = [
        # 기간: 승차시각으로만 판정한다. 하차가 다음 달로 넘어간 785건은
        #       자정을 넘긴 정상 심야운행이므로 하차시각으로 자르면 안 된다.
        ("period", df[pu].between(month.start_time, month.end_time)),
        # 거리: 0마일은 미터기만 켠 기록. 상한 100마일은 최댓값 307,491마일
        #       (지구 12바퀴) 같은 오기록을 걸러낸다.
        ("distance", df["trip_distance"].between(o.distance_min, o.distance_max)),
        # 금액: 상쇄쌍 제거 후에도 남은 잔여 음수·0원 건.
        ("amount", (df["total_amount"] > 0) & (df["fare_amount"] >= 0)),
    ]

    # 소요시간: 정책에 따라 '제거'와 '표시'가 갈린다.
    # 이 규칙만 정책 스위치를 둔 이유는 아래 편향 점검에서 설명한다.
    if o.duration_policy == "drop":
        rules.insert(1, ("duration", duration_ok))

    keep = pd.Series(True, index=df.index)
    per_rule: dict[str, int] = {}
    prev = len(df)
    for name, cond in rules:
        keep &= cond                    # 누적 적용 — 규칙 간 중복 카운트를 피한다
        now = int(keep.sum())
        per_rule[name] = prev - now
        prev = now

    out = df[keep].copy()
    if o.duration_policy == "flag":
        # 행은 남기고 표시만 한다. 소요시간을 쓰는 집계에서만 걸러 쓴다.
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

    # ---- 편향 자동 점검 -----------------------------------------------------
    # 결측 처리에서 dropna()를 거부한 이유(§2.4)와 똑같은 위험이 이 필터에도 있다.
    # "물리적으로 불가능한 값"이 사실은 '특정 사업자의 기록 방식'일 수 있고,
    # 그러면 필터가 그 사업자를 통째로 지운다. 조용히 넘어가면 발견하지 못하므로
    # 필터 전후 VendorID 구성을 비교해 전멸한 사업자를 경고한다.
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
