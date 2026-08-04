"""[단계 2] 중복 처리 — 문서 §3

이 데이터에서 drop_duplicates()를 그냥 쓰면 안 되는 이유가 두 가지다.
  1) 20개 컬럼 완전중복은 0건 → 아무 것도 안 지워지고 '중복 없음'으로 오판
  2) 부분키에 keep='first'를 쓰면 정상 운행을 지우고 취소 기록을 남긴다(§3.5)

그래서 '지우기 전에 그룹의 금액 관계로 유형을 판정'하는 구조를 취한다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def _classify(df: pd.DataFrame, cfg: Config):
    """중복 그룹을 (작은 total, 큰 total) 쌍으로 정리하고 유형을 판정한다.

    반환: (lo_idx, hi_idx, is_void, is_double)
    분류 로직만 떼어 둔 이유는 단위 테스트에서 이 함수만 검증하기 위함이다.
    """
    key = cfg.duplicates.key
    dup_mask = df.duplicated(subset=key, keep=False)   # keep=False: 구성원 전부
    if not dup_mask.any():
        empty = pd.Index([])
        return empty, empty, np.array([], bool), np.array([], bool)

    dup = df.loc[dup_mask, key + ["total_amount"]].copy()
    # ngroup()으로 그룹에 정수 ID를 부여하면 head/tail 연산이 간단해진다.
    dup["gid"] = dup.groupby(key, dropna=False).ngroup()
    # 그룹 내부를 total_amount 오름차순 정렬 → head=작은 값, tail=큰 값.
    # 상쇄쌍이면 (음수, 양수), 이중계상이면 (정상, 2배)가 각각 (lo, hi)로 잡힌다.
    dup = dup.sort_values(["gid", "total_amount"])
    lo_idx = dup.groupby("gid").head(1).index
    hi_idx = dup.groupby("gid").tail(1).index

    lo = df.loc[lo_idx, "total_amount"].to_numpy()
    hi = df.loc[hi_idx, "total_amount"].to_numpy()

    # ① 상쇄쌍: 두 행의 합이 0 → 운행이 취소·환불되어 장부상 지워진 쌍
    is_void = np.abs(lo + hi) < cfg.duplicates.void_tolerance
    # ② 이중계상: 큰 쪽이 작은 쪽의 정확히 2배 → 금액이 두 번 더해진 행
    #    0으로 나누기를 피하려고 분모의 0을 NaN으로 바꾼다(비교 시 자동 False)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = hi / np.where(lo == 0, np.nan, lo)
    is_double = (~is_void
                 & (ratio > cfg.duplicates.double_ratio_min)
                 & (ratio < cfg.duplicates.double_ratio_max))
    return lo_idx, hi_idx, is_void, is_double


def deduplicate(df: pd.DataFrame, cfg: Config) -> StepResult:
    """유형별로 다르게 제거한다 (문서 §3.6)."""
    key = cfg.duplicates.key
    before = len(df)
    notes: list[str] = []

    # ---- 완전중복 확인 (문서 §3.1) ------------------------------------------
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

    # ---- keep='first'가 왜 위험한지 실측으로 남긴다 (문서 §3.5) --------------
    # 파일에 저장된 행 순서상 상쇄쌍은 취소 행(음수)이 먼저 오는 경우가 많다.
    # 따라서 keep='first'는 '실제 운행'을 지우고 '취소 기록'을 남긴다.
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

    # ---- 유형별 선택 제거 (문서 §3.6) ---------------------------------------
    # ① 상쇄쌍   → 양쪽 모두 제거. 취소된 운행은 '발생하지 않은 운행'으로 본다.
    #              (한쪽만 남기면 매출 집계가 그 구간에서 음수가 된다)
    # ② 이중계상 → 큰 쪽만 제거, 작은 쪽 유지. 작은 쪽이 동일 포맷 단독행의
    #              total/fare 분포(중앙값 1.231)와 일치하기 때문이다.
    # ③ 기타     → 유지. 서로 다른 운행일 가능성이 높고 규모가 무시 가능하다.
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
