"""[Day 2 종합실습] 2. 데이터 준비 + 기본 EDA — NYC Yellow Taxi
NYC Yellow Taxi 2026-05 trip 데이터를 Pandas/Polars 양쪽으로 로딩해 결과를 비교하고,
결측치·중복 행을 정리한 뒤 기본 EDA를 수행한다.
"""
from __future__ import annotations

import sys
import timeit
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.error import URLError

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

# macOS 기본 폰트는 한글 지원 안 해서 차트 라벨이 깨짐 -> 한글 폰트로 교체
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
# PULocationID/DOLocationID는 숫자 코드라 그 자체로는 어디인지 알 수 없음
# TLC가 공식 배포하는 조회표로 Borough/Zone 실제 지명에 매핑함
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "yellow_tripdata_2026-05.parquet"
ZONE_LOOKUP_PATH = BASE_DIR / "data" / "taxi_zone_lookup.csv"

# 전부 955,371건으로 동일 -> 특정 벤더/트립타입에서만 채워지지 않는 컬럼으로 보임
NULLABLE_COLUMNS = ["passenger_count", "RatecodeID", "store_and_fwd_flag", "congestion_surcharge", "Airport_fee"]

BENCHMARK_NUMBER = 3  # parquet라 로딩 자체가 빨라서 3회로도 충분히 안정적임


def print_title(title: str) -> None:
    """출력 결과의 가독성을 높이기 위해 상하로 구분선이 포함된 타이틀 출력."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def benchmark(name: str, function: Callable[[], object]) -> dict[str, float | str]:
    """timeit 모듈을 활용하여 지정된 함수 실행 평균 소요 시간 측정."""
    seconds = timeit.timeit(function, number=BENCHMARK_NUMBER)
    return {"tool": name, "seconds": seconds / BENCHMARK_NUMBER}


def ensure_raw_data() -> Path:
    """원본 parquet 파일이 로컬에 없으면 CloudFront에서 다운로드하여 캐시 형태로 저장."""
    RAW_PATH.parent.mkdir(exist_ok=True)
    if not RAW_PATH.exists():
        try:
            urllib.request.urlretrieve(DATA_URL, RAW_PATH)
        except (URLError, OSError, TimeoutError) as error:
            raise RuntimeError(f"데이터 다운로드 실패: {error}") from error
    return RAW_PATH


def ensure_zone_lookup() -> Path:
    """LocationID -> Borough/Zone 매핑표(TLC 공식 조회표)가 로컬에 없으면 다운로드."""
    ZONE_LOOKUP_PATH.parent.mkdir(exist_ok=True)
    if not ZONE_LOOKUP_PATH.exists():
        try:
            urllib.request.urlretrieve(ZONE_LOOKUP_URL, ZONE_LOOKUP_PATH)
        except (URLError, OSError, TimeoutError) as error:
            raise RuntimeError(f"zone lookup 다운로드 실패: {error}") from error
    return ZONE_LOOKUP_PATH


def load_with_pandas(path: Path) -> pd.DataFrame:
    """Pandas로 parquet 로딩."""
    return pd.read_parquet(path)


def load_with_polars(path: Path) -> pl.DataFrame:
    """Polars로 parquet 로딩."""
    return pl.read_parquet(path)


def compare_pandas_polars(pdf: pd.DataFrame, pldf: pl.DataFrame, raw_path: Path) -> None:
    """Pandas와 Polars로 각각 로딩한 결과의 형태·결측치 비교 + timeit 속도 비교."""
    print_title("1. Pandas vs Polars 로딩 결과 비교")
    print(f"Pandas shape : {pdf.shape}")
    print(f"Polars shape : {pldf.shape}")
    print(f"행 수 일치   : {pdf.shape[0] == pldf.shape[0]}")

    pandas_na = pdf[NULLABLE_COLUMNS].isna().sum()
    polars_na = pldf.select(NULLABLE_COLUMNS).null_count()
    print("\n[컬럼별 결측치 수 비교]")
    for col in NULLABLE_COLUMNS:
        p_count, pl_count = int(pandas_na[col]), int(polars_na[col][0])
        print(f"  {col:22s} Pandas={p_count:8,d}  Polars={pl_count:8,d}  일치={p_count == pl_count}")

    # timeit으로 두 로딩 함수의 평균 실행 시간 비교 (같은 반복 횟수로 공정 비교)
    print(f"\n[로딩 속도 비교 (timeit, {BENCHMARK_NUMBER}회 반복 평균)]")
    timing_results = [
        benchmark("Pandas", lambda: load_with_pandas(raw_path)),
        benchmark("Polars", lambda: load_with_polars(raw_path)),
    ]
    for result in sorted(timing_results, key=lambda r: r["seconds"]):
        print(f"  {result['tool']:8s} 평균 {result['seconds']:.3f} s")


def clean_data(pdf: pd.DataFrame) -> pd.DataFrame:
    """중복 제거 + 결측치 처리.

    - passenger_count: 중앙값으로 대체 (수치형)
    - RatecodeID, store_and_fwd_flag: 최빈값으로 대체 (범주형)
    - congestion_surcharge, Airport_fee: 0으로 대체 (결측 = 해당 요금 미부과로 해석)
    """
    print_title("2. 결측치·중복 처리")
    dup_count = int(pdf.duplicated().sum())
    cleaned = pdf.drop_duplicates().copy()
    print(f"중복 행 제거: {dup_count:,}건 ({len(pdf):,} -> {len(cleaned):,})")

    median_val = cleaned["passenger_count"].median()
    missing = int(cleaned["passenger_count"].isna().sum())
    cleaned["passenger_count"] = cleaned["passenger_count"].fillna(median_val)
    print(f"passenger_count: 결측 {missing:,}건 -> 중앙값 {median_val}로 대체")

    for col in ["RatecodeID", "store_and_fwd_flag"]:
        mode_val = cleaned[col].mode()[0]
        missing = int(cleaned[col].isna().sum())
        cleaned[col] = cleaned[col].fillna(mode_val)
        print(f"{col}: 결측 {missing:,}건 -> 최빈값 '{mode_val}'로 대체")

    for col in ["congestion_surcharge", "Airport_fee"]:
        missing = int(cleaned[col].isna().sum())
        cleaned[col] = cleaned[col].fillna(0)
        print(f"{col}: 결측 {missing:,}건 -> 0으로 대체 (요금 미부과로 해석)")

    return cleaned


def run_basic_eda(df: pd.DataFrame) -> None:
    """정제된 데이터의 기본 구조·분포·이상치 확인."""
    print_title("3. 기본 EDA")
    print(f"shape: {df.shape}")

    print("\n[수치형 기술통계]")
    print(df[["trip_distance", "fare_amount", "tip_amount", "total_amount"]].describe())

    print("\n[payment_type 분포] (1=신용카드, 2=현금, 3=무료, 4=분쟁, 5=미상, 6=취소)")
    print(df["payment_type"].value_counts().sort_index())

    # 승차~하차 소요시간(분) 계산 -> 음수/0분이면 데이터 오류 가능성
    duration_min = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    print(f"\n[트립 소요시간] 평균 {duration_min.mean():.1f}분, 중앙값 {duration_min.median():.1f}분")
    print(f"소요시간 0분 이하 트립: {(duration_min <= 0).sum():,}건 (데이터 오류 의심)")

    # IQR 기준 total_amount 이상치 탐지 (제거는 안 하고 리포트만 함)
    q1, q3 = df["total_amount"].quantile(0.25), df["total_amount"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ~df["total_amount"].between(lower, upper)
    print(f"\n[total_amount IQR 이상치] 정상범위 [{lower:.2f}, {upper:.2f}]")
    print(f"이상치 {outliers.sum():,}건 ({outliers.mean() * 100:.1f}%)")
    print(f"음수 total_amount(환불 등) 건수: {(df['total_amount'] < 0).sum():,}건")


def map_location_names(df: pd.DataFrame, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    """PULocationID/DOLocationID 숫자 코드를 실제 Borough/Zone 이름으로 매핑해 컬럼 추가."""
    lookup = zone_lookup.set_index("LocationID")[["Borough", "Zone"]]

    result = df.join(lookup.add_prefix("PU_"), on="PULocationID")
    result = result.join(lookup.add_prefix("DO_"), on="DOLocationID")
    return result


def show_location_map(df: pd.DataFrame) -> Path:
    """픽업 위치를 Borough/Zone 실제 이름 기준으로 요약 + 막대차트로 저장."""
    print_title("4. 위치(Zone) 매핑")

    print("[Borough별 픽업 트립 수 · 평균 총요금]")
    borough_summary = (
        df.groupby("PU_Borough", observed=True)
        .agg(trips=("total_amount", "count"), avg_total=("total_amount", "mean"))
        .sort_values("trips", ascending=False)
    )
    print(borough_summary)

    top_zones = df["PU_Zone"].value_counts().head(10)
    print("\n[픽업 Zone 상위 10]")
    print(top_zones)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_zones.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("픽업 트립 수 상위 10 Zone")
    ax.set_xlabel("트립 수")
    plt.tight_layout()

    out_path = BASE_DIR / "nyc_taxi_top_zones.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[4] 위치 매핑 차트 저장: {out_path}")
    return out_path


def main() -> None:
    raw_path = ensure_raw_data()
    pdf = load_with_pandas(raw_path)
    pldf = load_with_polars(raw_path)

    compare_pandas_polars(pdf, pldf, raw_path)
    cleaned = clean_data(pdf)
    run_basic_eda(cleaned)

    zone_lookup = pd.read_csv(ensure_zone_lookup())
    mapped = map_location_names(cleaned, zone_lookup)
    show_location_map(mapped)

    out_path = BASE_DIR / "data" / "nyc_taxi_cleaned.parquet"
    cleaned.to_parquet(out_path, index=False)
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
