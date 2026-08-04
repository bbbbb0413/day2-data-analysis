"""[Day 2 종합실습] 3. 데이터 준비 + 기본 EDA — Stack Overflow Developer Survey 2024
Stack Overflow Survey 2024 데이터를 Pandas/Polars 양쪽으로 로딩해 결과를 비교하고,
핵심 컬럼 위주로 정제한 뒤 기본 EDA를 수행한다.

주의: 이 데이터는 114개 컬럼짜리 설문 응답이라 대부분 컬럼이 선택 질문(스킵 로직)이라서
결측치가 원래 정상적으로 많다. Adult/Taxi처럼 전체 컬럼에 dropna/fillna를 적용하면
의미가 왜곡되므로, 실제 분석에 쓰는 핵심 컬럼만 골라서 정제한다.
"""
from __future__ import annotations

import sys
import timeit
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.error import URLError

import pandas as pd
import polars as pl

DATA_URL = "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2024/results.csv"
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "stackoverflow_2024_results.csv"

# 전체 114개 컬럼 중 분석에 실제로 쓸 핵심 컬럼만 추림
KEY_COLUMNS = [
    "ResponseId", "Age", "Employment", "RemoteWork", "Country",
    "EdLevel", "YearsCodePro", "DevType", "ConvertedCompYearly", "JobSat",
]

BENCHMARK_NUMBER = 3


def print_title(title: str) -> None:
    """출력 결과의 가독성을 높이기 위해 상하로 구분선이 포함된 타이틀 출력."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def benchmark(name: str, function: Callable[[], object]) -> dict[str, float | str]:
    """timeit 모듈을 활용하여 지정된 함수 실행 평균 소요 시간 측정."""
    seconds = timeit.timeit(function, number=BENCHMARK_NUMBER)
    return {"tool": name, "seconds": seconds / BENCHMARK_NUMBER}


def ensure_raw_data() -> Path:
    """원본 CSV가 없으면 GitHub에서 다운로드하여 캐시 형태로 저장."""
    RAW_PATH.parent.mkdir(exist_ok=True)
    if not RAW_PATH.exists():
        try:
            urllib.request.urlretrieve(DATA_URL, RAW_PATH)
        except (URLError, OSError, TimeoutError) as error:
            raise RuntimeError(f"데이터 다운로드 실패: {error}") from error
    return RAW_PATH


def load_with_pandas(path: Path) -> pd.DataFrame:
    """Pandas로 CSV 전체(114컬럼) 로딩."""
    return pd.read_csv(path, low_memory=False)


def load_with_polars(path: Path) -> pl.DataFrame:
    """Polars로 CSV 전체(114컬럼) 로딩.
    컬럼이 많고 혼합 타입이 섞여 있어 infer_schema_length를 넉넉히 줘서
    타입 추론 오류(스키마 미스매치)를 방지한다.

    이 CSV는 결측치가 빈 칸이 아니라 문자열 "NA"로 기록돼 있다.
    Pandas는 read_csv 기본 na_values 목록에 "NA"가 포함돼 있어 자동으로 걸러지지만,
    Polars는 기본적으로 빈 칸만 null로 보므로 null_values를 명시하지 않으면
    "NA"를 그냥 일반 문자열로 읽어버려 결측치 개수가 0으로 나온다."""
    return pl.read_csv(path, infer_schema_length=10_000, null_values=["NA"])


def compare_pandas_polars(pdf: pd.DataFrame, pldf: pl.DataFrame, raw_path: Path) -> None:
    """Pandas와 Polars로 각각 로딩한 결과의 형태·핵심 컬럼 결측치 비교 + timeit 속도 비교."""
    print_title("1. Pandas vs Polars 로딩 결과 비교")
    print(f"Pandas shape : {pdf.shape}")
    print(f"Polars shape : {pldf.shape}")
    print(f"행 수 일치   : {pdf.shape[0] == pldf.shape[0]}")

    pandas_na = pdf[KEY_COLUMNS].isna().sum()
    polars_na = pldf.select(KEY_COLUMNS).null_count()
    print("\n[핵심 컬럼 결측치 수 비교]")
    for col in KEY_COLUMNS:
        p_count, pl_count = int(pandas_na[col]), int(polars_na[col][0])
        print(f"  {col:20s} Pandas={p_count:6,d}  Polars={pl_count:6,d}  일치={p_count == pl_count}")

    print(f"\n[로딩 속도 비교 (timeit, {BENCHMARK_NUMBER}회 반복 평균)]")
    timing_results = [
        benchmark("Pandas", lambda: load_with_pandas(raw_path)),
        benchmark("Polars", lambda: load_with_polars(raw_path)),
    ]
    for result in sorted(timing_results, key=lambda r: r["seconds"]):
        print(f"  {result['tool']:8s} 평균 {result['seconds']:.3f} s")


def clean_data(pdf: pd.DataFrame) -> pd.DataFrame:
    """설문 데이터 정제.

    전체 114컬럼에 dropna를 걸면 선택 문항 특성상 거의 모든 행이 날아가서 의미가 없다.
    그래서 핵심 컬럼만 추린 뒤, 문항 성격에 따라 다르게 처리한다.
    - Age/Employment/Country: 필수 성격 문항 -> 결측 행 제거
    - YearsCodePro: "Less than 1 year"/"More than 50 years" 텍스트를 숫자로 변환
    - ConvertedCompYearly/JobSat: 선택 문항 -> 결측을 정상으로 보고 그대로 둠
    """
    print_title("2. 결측치·중복 처리")
    dup_count = int(pdf.duplicated().sum())
    print(f"완전 중복 행: {dup_count}건 (설문 응답 데이터라 원래 거의 없음)")

    subset = pdf[KEY_COLUMNS].copy()

    # YearsCodePro는 숫자 문자열 사이에 텍스트 값이 섞여 있어 수치형으로 못 바로 못 씀
    subset["YearsCodePro"] = (
        subset["YearsCodePro"]
        .replace({"Less than 1 year": 0.5, "More than 50 years": 51})
        .astype(float)
    )

    print("\n[핵심 컬럼별 결측 비율(%)]")
    print((subset.isna().sum() / len(subset) * 100).round(1))

    before = len(subset)
    cleaned = subset.dropna(subset=["Age", "Employment", "Country"]).copy()
    print(f"\n필수 문항(Age/Employment/Country) 결측 행 제거: {before:,} -> {len(cleaned):,}")

    return cleaned


def run_basic_eda(df: pd.DataFrame) -> None:
    """핵심 컬럼 기준 기본 EDA."""
    print_title("3. 기본 EDA")
    print(f"shape: {df.shape}")

    print("\n[연령대(Age) 분포]")
    print(df["Age"].value_counts())

    print("\n[국가(Country) 상위 10개]")
    print(df["Country"].value_counts().head(10))

    print("\n[재택근무(RemoteWork) 분포, 결측 포함]")
    print(df["RemoteWork"].value_counts(dropna=False))

    print("\n[YearsCodePro 기술통계 (결측 제외)]")
    print(df["YearsCodePro"].describe())

    print("\n[연봉(ConvertedCompYearly, USD) 기술통계 (결측 제외)]")
    print(df["ConvertedCompYearly"].describe())
    # 설문 특성상 극단적으로 큰 값을 적는 응답자가 있어 평균보다 중앙값이 더 대표값에 가까움
    print(f"중앙값: {df['ConvertedCompYearly'].median():,.0f}")


def main() -> None:
    raw_path = ensure_raw_data()
    pdf = load_with_pandas(raw_path)
    pldf = load_with_polars(raw_path)

    compare_pandas_polars(pdf, pldf, raw_path)
    cleaned = clean_data(pdf)
    run_basic_eda(cleaned)

    out_path = BASE_DIR / "data" / "stackoverflow_2024_key_columns_cleaned.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"\n정제된 핵심 컬럼 데이터 저장 완료: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"[데이터 오류] {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except Exception as error:
        print(f"[실행 오류] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
