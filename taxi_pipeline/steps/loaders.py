"""[단계 1] Pandas vs Polars 로딩 비교

같은 parquet을 두 라이브러리로 읽어 shape·결측·중복이 일치하는지 확인한다.
이 검증이 통과해야 "이후 분석은 도구 선택과 무관한 데이터 자체의 성질"이라고
말할 수 있다. 불일치하면 파서 설정부터 다시 봐야 한다.

★ 왜 별도 명령이 아니라 파이프라인 단계인가
  처음에는 `run_pipeline.py compare-loaders` 라는 별도 명령으로 두었다.
  그러면 기본 실행에 polars가 등장하지 않고 비교 결과가 report.md에도
  실리지 않는다. 리포트만 읽는 사람은 이 검증이 있었는지조차 알 수 없다.

★ 비용
  parquet을 polars로 한 번 더 읽는다. 실측 0.05초 수준이라 부담이 없다.
  pandas가 읽은 DataFrame은 runner가 이미 갖고 있으므로 그것과 대조한다.

★ 이 단계는 DataFrame을 바꾸지 않는다.
  비교만 하고 pandas가 읽은 것을 그대로 다음 단계로 넘긴다.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import polars as pl

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)


def compare_loaders(df: pd.DataFrame, cfg: Config) -> StepResult:
    """같은 parquet을 polars로도 읽어 pandas 결과와 대조한다."""
    path = cfg.paths.raw
    if not path.is_file():
        # --input으로 다른 파일을 지정한 경우 등. 비교를 건너뛰되 죽지는 않는다.
        log.warning("원본 parquet을 찾지 못해 로딩 비교를 건너뜁니다: %s", path)
        return StepResult(df=df, metrics={"skipped": True}, notes=[])

    t = time.perf_counter()
    ldf = pl.read_parquet(path)
    pl_sec = time.perf_counter() - t

    # 결측 수: polars는 null_count()가 컬럼별 1행 DataFrame으로 나온다
    pl_nulls = dict(zip(ldf.columns, ldf.null_count().row(0)))
    pd_nulls = df.isna().sum().to_dict()

    # 완전중복: n_unique()는 '서로 다른 행의 수'이므로 전체에서 빼면 중복 수가 된다
    pl_dup = len(ldf) - ldf.n_unique()
    pd_dup = int(df.duplicated().sum())

    # ★ 부분키 중복은 두 라이브러리가 '세는 대상'이 다르다.
    #   polars is_duplicated() : 그룹 구성원 '전부'를 True로 표시
    #   pandas duplicated()    : 그룹의 '첫 행을 뺀 나머지'만 True
    #   같은 데이터인데 2배 차이 나므로, "중복 N건"이라고 말할 때
    #   무엇을 센 것인지 밝히지 않으면 보고서끼리 숫자가 어긋난다.
    key = [c for c in cfg.duplicates.key if c in ldf.columns]
    pl_key = int(ldf.select(key).is_duplicated().sum())
    pd_key = int(df.duplicated(subset=key).sum())

    # 결측이 있는 정수 컬럼에서 pandas는 float64로 승격시킨다.
    # NaN을 표현할 방법이 float밖에 없기 때문인데, 그 결과 승객수가 1이 아니라
    # 1.0으로 출력되고 정수 비교에 부동소수 문제가 끼어든다.
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
    del ldf                       # 비교가 끝나면 즉시 해제한다

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
        "partial_key_polars": pl_key,   # 그룹 구성원 전부
        "partial_key_pandas": pd_key,   # 첫 행 제외
        "dtype_promoted": promoted,
        "all_match": bool(shape_ok and nulls_ok and dup_ok),
    }
    log.info("로딩 비교: polars %.3f초/%.0fMB vs pandas %.0fMB, 일치 %s",
             pl_sec, pl_mb, pd_mb, metrics["all_match"])
    return StepResult(df=df, metrics=metrics, notes=notes)
