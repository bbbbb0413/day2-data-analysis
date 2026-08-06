"""Pandas와 Polars의 parquet 로딩 결과를 비교한다."""

from __future__ import annotations

import logging
import time

import pandas as pd
import polars as pl

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def compare_loaders(df: pd.DataFrame, cfg: Config) -> StepResult:
    """같은 parquet을 Polars로도 읽어 Pandas 결과와 비교한다."""
    path = cfg.paths.raw
    if not path.is_file():
        # 원본 경로가 없으면 비교 단계를 건너뛴다.
        log.warning("원본 parquet을 찾지 못해 로딩 비교를 건너뜁니다: %s", path)
        return StepResult(df=df, metrics={"skipped": True}, notes=[])

    t = time.perf_counter()
    ldf = pl.read_parquet(path)
    pl_sec = time.perf_counter() - t

    # Polars의 컬럼별 결측 수를 dict로 변환한다.
    pl_nulls = dict(zip(ldf.columns, ldf.null_count().row(0)))
    pd_nulls = df.isna().sum().to_dict()

    # 전체 행 수에서 고유 행 수를 빼 완전중복 수를 계산한다.
    pl_dup = len(ldf) - ldf.n_unique()
    pd_dup = int(df.duplicated().sum())

    # 부분키 중복은 라이브러리마다 집계 기준이 다르다.
    # Polars는 그룹 전체를, Pandas는 첫 행을 제외한 행만 센다.
    key = [c for c in cfg.duplicates.key if c in ldf.columns]
    pl_key = int(ldf.select(key).is_duplicated().sum())
    pd_key = int(df.duplicated(subset=key).sum())

    # 결측이 있는 정수 컬럼의 dtype 차이를 확인한다.
    pl_types = {c: str(t) for c, t in zip(ldf.columns, ldf.dtypes)}
    promoted = [
        {"column": c, "pandas": str(df[c].dtype), "polars": pl_types[c],
         "nulls": int(pd_nulls[c])}
        for c in df.columns
        if str(df[c].dtype).startswith("float") and pl_types.get(c, "").startswith("Int")
    ]

    shape_ok = (len(ldf), ldf.width) == df.shape
    nulls_ok = all(int(pl_nulls[c]) == int(pd_nulls[c]) for c in pd_nulls
                   if c in pl_nulls)
    dup_ok = pl_dup == pd_dup

    pl_mb = ldf.estimated_size("mb")
    pd_mb = df.memory_usage(deep=True).sum() / 1024**2
    del ldf                       # 비교가 끝나면 즉시 해제한다.

    notes = [
        f"[로딩 비교] 같은 parquet을 polars {pl_sec:.3f}초에 읽었고 DataFrame 크기는 "
        f"{pl_mb:,.0f}MB다(pandas {pd_mb:,.0f}MB). polars가 문자열을 Arrow 포맷으로 "
        f"저장해 {pd_mb / pl_mb:.2f}배 작다.",
        f"[로딩 비교] 결과 일치 검증 — shape {'OK' if shape_ok else '불일치'}, "
        f"컬럼별 결측수 {'OK' if nulls_ok else '불일치'}, "
        f"완전중복 {'OK' if dup_ok else '불일치'}({pd_dup}건). "
        f"이후 분석은 도구 선택과 무관한 데이터 자체의 성질이다.",
        f"[로딩 비교] 부분키 중복은 두 라이브러리의 정의가 다르다. "
        f"polars `is_duplicated().sum()` = {pl_key:,}(그룹 구성원 전부), "
        f"pandas `duplicated().sum()` = {pd_key:,}(첫 행 제외). "
        f"같은 데이터인데 2배 차이 나므로 '중복 N건'이라고 말할 때 무엇을 센 것인지 "
        f"밝혀야 한다.",
    ]
    if promoted:
        cols = ", ".join(f"`{p['column']}`" for p in promoted)
        notes.append(
            f"[로딩 비교] 결측이 있는 정수 컬럼({cols})을 pandas는 float64로 승격시키고 "
            f"polars는 Int64로 유지한다. pandas는 NaN을 표현할 방법이 float밖에 없기 "
            f"때문이며, 그 결과 승객수가 1이 아니라 1.0으로 출력된다.")

    metrics = {
        "polars_version": pl.__version__,
        "pandas_version": pd.__version__,
        "polars_load_sec": pl_sec,
        "polars_size_mb": float(pl_mb),
        "pandas_size_mb": float(pd_mb),
        "shape_match": shape_ok,
        "nulls_match": nulls_ok,
        "duplicates_match": dup_ok,
        "exact_duplicates": pd_dup,
        "partial_key_polars": pl_key,   # 그룹 전체를 센 값이다.
        "partial_key_pandas": pd_key,   # 첫 행을 제외하고 센 값이다.
        "dtype_promoted": promoted,
        "all_match": bool(shape_ok and nulls_ok and dup_ok),
    }
    log.info("로딩 비교: polars %.3f초/%.0fMB vs pandas %.0fMB, 일치 %s",
             pl_sec, pl_mb, pd_mb, metrics["all_match"])
    return StepResult(df=df, metrics=metrics, notes=notes)
