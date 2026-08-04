"""Practice 3: Pandas EDA, Polars Lazy, DuckDB SQL 비교.
[Day 2 종합실습] 1. 데이터 준비 + 기본 EDA
Adult Census Income 데이터셋을 Pandas/Polars 양쪽으로 로딩해 결과를 비교하고,
결측치·중복 행을 정리한 뒤 기본 EDA를 수행한다.
EDA(Exploratory Data Analysis, 탐색적 데이터 분석)
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

# ---------------------------------------------------------------------
# 전역 상수 및 설정 정의
# ---------------------------------------------------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
BASE_DIR = Path(__file__).resolve().parent  # 현재 스크립트의 절대 경로 기준 디렉토리 추출
RAW_PATH = BASE_DIR / "data" / "adult.data"  # 원본 데이터가 저장될 로컬 파일 경로 지정

# UCI Adult Dataset의 15개 컬럼명 정의 (헤더가 없는 원본 파일에 직접 부여)
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

# 원본 데이터셋 내에서 누락된 값이 '?' 문자로 표기되어 결측치 처리가 필요한 주요 컬럼 정의
NULLABLE_COLUMNS = ["workclass", "occupation", "native-country"]

# timeit으로 로딩 속도를 비교할 때 각 함수를 반복 실행할 횟수
BENCHMARK_NUMBER = 5


def print_title(title: str) -> None:
    """출력 결과의 가독성을 높이기 위해 상하로 구분선이 포함된 타이틀 출력."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def benchmark(name: str, function: Callable[[], object]) -> dict[str, float | str]:
    """timeit 모듈을 활용하여 지정된 함수 실행 평균 소요 시간 측정."""

    # timeit.timeit()
    # 인자로 받은 함수를 지정된 횟수(number)만큼 반복 실행하는 데 소요된 총 시간을 초 단위로 측정
    seconds = timeit.timeit(function, number=BENCHMARK_NUMBER)

    # 총 소요 시간을 반복 횟수로 나누어 1회 실행 당 평균 소요 시간 계산 후 딕셔너리 형태로 반환
    return {"tool": name, "seconds": seconds / BENCHMARK_NUMBER}


def ensure_raw_data() -> Path:
    """원본 CSV 파일이 로컬에 없으면 UCI 저장소에서 다운로드하여 캐시 형태로 저장."""
    RAW_PATH.parent.mkdir(exist_ok=True)
    if not RAW_PATH.exists():
        try:
            urllib.request.urlretrieve(DATA_URL, RAW_PATH)
        except (URLError, OSError, TimeoutError) as error:
            raise RuntimeError(f"데이터 다운로드 실패: {error}") from error
    return RAW_PATH


def load_with_pandas(path: Path) -> pd.DataFrame:
    """Pandas를 활용한 CSV 파일 로딩 함수.
    
    - na_values='?': '?' 문자를 명시적으로 판다스의 Na(결측치)로 매핑
    - skipinitialspace=True: 쉼표(,) 뒤에 존재하는 공백을 미리 제거하여 정상 매칭 유도
    """
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)


def load_with_polars(path: Path) -> pl.DataFrame:
    """Polars를 활용한 CSV 파일 로딩 함수.
    
    - null_values=' ?': 공백이 포함된 ' ?' 형태의 결측 문자열을 널값으로 처리
    - strip_chars(): 문자열 컬럼 전체의 양쪽 공백 제거 정제 수행
    - filter(~pl.all_horizontal(...)): 파일 마지막 줄의 빈 줄(전체 널 행) 필터링 제거
    """
    df = pl.read_csv(path, has_header=False, new_columns=COLUMNS, null_values=" ?")
    str_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.String]
    df = df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])
    return df.filter(~pl.all_horizontal(pl.all().is_null()))


def compare_pandas_polars(pdf: pd.DataFrame, pldf: pl.DataFrame, raw_path: Path) -> None:
    """Pandas와 Polars로 각각 로딩한 결과 데이터의 형태(Shape)와 컬럼별 결측치 개수 비교 검증.
    추가로 timeit을 이용해 두 로딩 함수의 평균 실행 시간도 비교한다."""
    print_title("1. Pandas vs Polars 로딩 결과 비교")
    print(f"Pandas shape : {pdf.shape}")
    print(f"Polars shape : {pldf.shape}")
    print(f"행 수 일치   : {pdf.shape[0] == pldf.shape[0]}")

    pandas_na = pdf[NULLABLE_COLUMNS].isna().sum()
    polars_na = pldf.select(NULLABLE_COLUMNS).null_count()

    print("\n[컬럼별 결측치 수 비교]")
    for col in NULLABLE_COLUMNS:
        p_count, pl_count = int(pandas_na[col]), int(polars_na[col][0])
        print(f"  {col:15s} Pandas={p_count:5d}  Polars={pl_count:5d}  일치={p_count == pl_count}")

    # timeit으로 두 로딩 함수의 평균 실행 시간 비교 (같은 반복 횟수 BENCHMARK_NUMBER로 공정 비교)
    print(f"\n[로딩 속도 비교 (timeit, {BENCHMARK_NUMBER}회 반복 평균)]")
    timing_results = [
        benchmark("Pandas", lambda: load_with_pandas(raw_path)),
        benchmark("Polars", lambda: load_with_polars(raw_path)),
    ]
    for result in sorted(timing_results, key=lambda r: r["seconds"]):
        print(f"  {result['tool']:8s} 평균 {result['seconds'] * 1000:.2f} ms")


def clean_data(pdf: pd.DataFrame) -> pd.DataFrame:
    """데이터 정제(Data Cleaning) 수행 함수.
    
    - 완전 중복 행(Duplicated Rows) 제거
    - 범주형 결측치(Null)가 존재하는 컬럼에 대해 해당 컬럼의 최빈값(Mode)으로 대체
    - 정제 완료 후 잔여 결측치 존재 여부를 assert 문으로 엄격하게 검증
    """
    print_title("2. 결측치·중복 처리")
    dup_count = int(pdf.duplicated().sum())
    cleaned = pdf.drop_duplicates().copy()
    print(f"중복 행 제거: {dup_count}건 ({len(pdf)} -> {len(cleaned)})")

    for col in NULLABLE_COLUMNS:
        missing = int(cleaned[col].isna().sum())
        if missing:
            # Pandas의 mode()[0]을 이용해 최빈값 추출 후 결측치 채우기
            mode_value = cleaned[col].mode()[0]
            cleaned[col] = cleaned[col].fillna(mode_value)
            print(f"{col}: 결측 {missing}건 -> 최빈값 '{mode_value}'로 대체")

    # 정제 공정 후 결측치가 0개가 되었는지 최종 무결성 검증
    assert cleaned.isna().sum().sum() == 0, "정제 후에도 결측치가 남아있습니다."
    return cleaned


def run_basic_eda(df: pd.DataFrame) -> None:
    """정제된 데이터를 바탕으로 기본 탐색적 데이터 분석(EDA) 수행."""
    print_title("3. 기본 EDA")
    print(f"shape: {df.shape}")
    print("\n[dtypes]")
    print(df.dtypes)
    print("\n[수치형 기술통계]")
    print(df.describe())
    print("\n[income 분포]")
    print(df["income"].value_counts())
    print("\n[성별 x income 교차비율(%)]")
    print(pd.crosstab(df["sex"], df["income"], normalize="index").mul(100).round(1))


def main() -> None:
    """전체 데이터 준비, 로딩 비교, 정제 및 EDA 파이프라인 제어 메인 함수."""
    raw_path = ensure_raw_data()
    
    # 1. 두 라이브러리로 각각 원본 CSV 로드
    pdf = load_with_pandas(raw_path)
    pldf = load_with_polars(raw_path)

    # 2. 로딩 결과 정합성 검증 + 로딩 속도 비교
    compare_pandas_polars(pdf, pldf, raw_path)
    
    # 3. 데이터 정제 및 EDA 실행 (현재 예제에서는 검증 완료된 Pandas 객체 기준 정제 진행)
    cleaned = clean_data(pdf)
    run_basic_eda(cleaned)

    # 4. 정제된 데이터를 새로운 CSV 파일로 로컬 저장
    out_path = BASE_DIR / "data" / "adult_cleaned.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"\n정제된 데이터 저장 완료: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"[데이터 오류] {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except Exception as error:
        print(f"[실행 오류] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error