"""중복 그룹을 유형별로 판정하여 필요한 행만 제거한다."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def _classify(df: pd.DataFrame, cfg: Config):
    """중복 그룹의 최소 금액과 최대 금액으로 유형을 판정한다."""
    key = cfg.duplicates.key
    dup_mask = df.duplicated(subset=key, keep=False)   # 중복 그룹의 모든 행을 선택한다.
    if not dup_mask.any():
        empty = pd.Index([])
        return empty, empty, np.array([], bool), np.array([], bool)

    dup = df.loc[dup_mask, key + ["total_amount"]].copy()
    # 중복 그룹에 정수 ID를 부여한다.
    dup["gid"] = dup.groupby(key, dropna=False).ngroup()
    # 금액순으로 정렬한 후 그룹별 최소 행과 최대 행을 선택한다.
    dup = dup.sort_values(["gid", "total_amount"])
    lo_idx = dup.groupby("gid").head(1).index
    hi_idx = dup.groupby("gid").tail(1).index

    lo = df.loc[lo_idx, "total_amount"].to_numpy()
    hi = df.loc[hi_idx, "total_amount"].to_numpy()

    # 상쇄쌍은 두 금액의 합이 0에 가까운 경우로 판정한다.
    is_void = np.abs(lo + hi) < cfg.duplicates.void_tolerance
    # 이중계상은 큰 금액이 작은 금액의 약 2배인 경우로 판정한다.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = hi / np.where(lo == 0, np.nan, lo)
    is_double = (~is_void
                 & (ratio > cfg.duplicates.double_ratio_min)
                 & (ratio < cfg.duplicates.double_ratio_max))
    return lo_idx, hi_idx, is_void, is_double


def deduplicate(df: pd.DataFrame, cfg: Config) -> StepResult:
    """중복 유형에 따라 제거할 행을 선택한다."""
    key = cfg.duplicates.key
    before = len(df)
    notes: list[str] = []

    # 완전중복 여부를 확인한다.
    n_exact = int(df.duplicated().sum())
    if n_exact == 0:
        notes.append("20개 컬럼 완전중복 0건 — drop_duplicates()는 아무 행도 지우지 않는다. "
                     "부분키로 봐야 중복이 보인다.")

    lo_idx, hi_idx, is_void, is_double = _classify(df, cfg)
    n_groups = len(lo_idx)

    metrics: dict = {
        "rows_in": before,
        "exact_duplicates": n_exact,
        "duplicate_groups": n_groups,
        "duplicate_rows": int(df.duplicated(subset=key, keep=False).sum()),
        "void_pairs": int(is_void.sum()),
        "double_rows": int(is_double.sum()),
        "other_groups": int((~(is_void | is_double)).sum()),
    }

    if n_groups == 0:
        metrics.update(rows_out=before, rows_dropped=0)
        return StepResult(df=df, metrics=metrics, notes=notes)

    # keep='first' 적용 시 제거되는 행을 확인한다.
    naive = df.index[df.duplicated(subset=key, keep="first")]
    metrics["naive_keep_first"] = {
        "positive_rows_dropped": int((df.loc[naive, "total_amount"] > 0).sum()),
        "negative_rows_dropped": int((df.loc[naive, "total_amount"] < 0).sum()),
        "negative_rows_left": int((df.drop(index=naive)["total_amount"] < 0).sum()),
    }
    notes.append(
        f"drop_duplicates(keep='first')를 썼다면 양수 총액 "
        f"{metrics['naive_keep_first']['positive_rows_dropped']:,}건(정상 운행)이 지워지고 "
        f"음수 {metrics['naive_keep_first']['negative_rows_left']:,}건(취소 기록)이 남는다. "
        "순서 기반 제거는 사용 금지.")

    # 상쇄쌍은 양쪽 행을 모두 제거한다.
    # 이중계상은 큰 금액 행만 제거한다.
    # 나머지 중복 그룹은 유지한다.
    drop_idx = lo_idx[is_void].union(hi_idx[is_void])
    drop_idx = drop_idx.union(hi_idx[is_double])
    out = df.drop(index=drop_idx)

    notes.append(
        f"상쇄쌍 {int(is_void.sum()):,}쌍은 양쪽 제거, 이중계상 {int(is_double.sum()):,}건은 "
        f"큰 쪽만 제거, 기타 {metrics['other_groups']:,}그룹은 유지.")
    notes.append(
        f"음수 total_amount가 {int((df['total_amount'] < 0).sum()):,} → "
        f"{int((out['total_amount'] < 0).sum()):,}건으로 감소 "
        "— 이상치 필터로 음수를 지우는 것보다 원인(취소쌍) 제거가 정확하다.")

    metrics.update(
        rows_out=len(out),
        rows_dropped=len(drop_idx),
        negative_total_before=int((df["total_amount"] < 0).sum()),
        negative_total_after=int((out["total_amount"] < 0).sum()),
    )
    log.info("중복 처리: %s행 제거 (상쇄쌍 %s쌍, 이중계상 %s건)",
             f"{len(drop_idx):,}", f"{int(is_void.sum()):,}", f"{int(is_double.sum()):,}")
    return StepResult(df=out, metrics=metrics, notes=notes)
