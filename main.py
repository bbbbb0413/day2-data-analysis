"""
프로그램명: NYC Yellow Taxi End-to-End 데이터 분석
작성자: 윤서준

설명:
    NYC TLC Yellow Taxi 2026년 5월 Parquet 데이터를 Pandas와 Polars로
    각각 로딩하여 결과를 비교하고, 결측치·중복·비정상값 처리, EDA,
    시각화, 통계 검정, 머신러닝 Pipeline 학습, 모델 저장,
    report.md 자동 생성을 순서대로 수행합니다.

분석 주제:
    1. 평일과 주말의 카드 결제 팁 비율이 다른지 검정합니다.
    2. 운행 종료 시점에 확인 가능한 정보로 팁 비율 20% 이상 여부를 예측합니다.

주의:
    NYC TLC 데이터 사전에 따르면 현금 팁은 tip_amount에 포함되지 않습니다.
    따라서 팁 관련 분석과 모델 학습은 신용카드 결제(payment_type == 1)만 사용합니다.
"""

from __future__ import annotations

import gc
import io
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==============================================================================
# [설정] 데이터와 결과 파일 경로
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2026-05.parquet"
)
DATA_FILE = DATA_DIR / "yellow_tripdata_2026-05.parquet"
REPORT_FILE = BASE_DIR / "report.md"

LOADER_COMPARISON_FILE = OUTPUT_DIR / "loader_comparison.csv"
LOADER_CHECK_FILE = OUTPUT_DIR / "loader_consistency_check.json"
PANDAS_MISSING_FILE = OUTPUT_DIR / "pandas_missing_values.csv"
POLARS_MISSING_FILE = OUTPUT_DIR / "polars_missing_values.csv"
POLARS_STATS_FILE = OUTPUT_DIR / "polars_descriptive_statistics.csv"
POLARS_PAYMENT_FILE = OUTPUT_DIR / "polars_payment_type_summary.csv"
CLEANING_SUMMARY_FILE = OUTPUT_DIR / "cleaning_summary.csv"
MISSING_BEFORE_FILE = OUTPUT_DIR / "missing_before_cleaning.csv"
MISSING_AFTER_FILE = OUTPUT_DIR / "missing_after_eda_imputation.csv"
CLEANED_PREVIEW_FILE = OUTPUT_DIR / "cleaned_preview_10000.csv"
DESCRIPTIVE_STATS_FILE = OUTPUT_DIR / "descriptive_statistics.csv"
CORRELATION_FILE = OUTPUT_DIR / "correlation_matrix.csv"
PAYMENT_SUMMARY_FILE = OUTPUT_DIR / "payment_type_summary.csv"
TARGET_SUMMARY_FILE = OUTPUT_DIR / "high_tip_target_summary.csv"
TTEST_RESULT_FILE = OUTPUT_DIR / "ttest_result.json"
MODEL_METRICS_FILE = OUTPUT_DIR / "model_metrics.json"
MODEL_FILE = OUTPUT_DIR / "yellow_taxi_high_tip_pipeline.joblib"
SEABORN_CHART_FILE = OUTPUT_DIR / "seaborn_tip_rate_weekday_weekend.png"
PLOTLY_CHART_FILE = OUTPUT_DIR / "plotly_hourly_trip_share.html"

# 대용량 데이터를 노트북·일반 PC에서도 실행할 수 있도록 분석용 표본을 사용합니다.
# None으로 변경하면 Pandas 분석에 전체 데이터를 사용합니다.
ANALYSIS_SAMPLE_SIZE: int | None = None
ML_SAMPLE_SIZE = 150_000
PLOT_SAMPLE_SIZE = 50_000
TTEST_MAX_PER_GROUP = 100_000

RANDOM_STATE = 42
TEST_SIZE = 0.2
ALPHA = 0.05

# 프로젝트에서 정의한 분류 기준입니다.
# 결제 전 금액 대비 팁 비율이 20% 이상이면 high_tip=1로 정의합니다.
HIGH_TIP_THRESHOLD = 0.20

ANALYSIS_START = pd.Timestamp("2026-05-01 00:00:00")
ANALYSIS_END = pd.Timestamp("2026-06-01 00:00:00")

REQUIRED_COLUMNS = {
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
}

PAYMENT_TYPE_LABELS = {
    0: "Flex Fare",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}


# ==============================================================================
# [공통 함수] 출력 구간과 파일 저장 보조 함수
# ==============================================================================


def print_section(title: str) -> None:
    """실행 단계가 구분되도록 제목을 출력합니다."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def create_directories() -> None:
    """데이터와 결과를 저장할 폴더를 생성합니다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    """NumPy·Pandas 값을 JSON으로 저장할 수 있는 기본 자료형으로 바꿉니다."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"JSON으로 변환할 수 없는 자료형입니다: {type(value)}")


def save_json(data: dict[str, Any], path: Path) -> None:
    """딕셔너리를 UTF-8 JSON 파일로 저장합니다."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


def format_markdown_value(value: Any) -> str:
    """Markdown 표에서 사용할 값을 보기 좋게 변환합니다."""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(
    dataframe: pd.DataFrame,
    include_index: bool = True,
) -> str:
    """tabulate 추가 설치 없이 DataFrame을 Markdown 표로 변환합니다."""
    table_df = dataframe.copy()

    if include_index:
        index_name = table_df.index.name or "variable"
        table_df = table_df.reset_index().rename(columns={"index": index_name})

    headers = [str(column) for column in table_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in table_df.itertuples(index=False, name=None):
        values = [format_markdown_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# ==============================================================================
# [문제 1] 데이터 다운로드
# ==============================================================================


def download_dataset() -> None:
    """데이터 파일이 없을 때 NYC TLC 주소에서 자동으로 다운로드합니다."""
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
        size_mb = DATA_FILE.stat().st_size / 1024**2
        print(f"기존 데이터 파일을 사용합니다: {DATA_FILE} ({size_mb:.1f} MB)")
        return

    print("NYC Yellow Taxi 데이터를 다운로드합니다.")
    print(f"주소: {DATA_URL}")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    temporary_file = DATA_FILE.with_suffix(".parquet.part")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with temporary_file.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(
                            f"\r다운로드 진행률: {progress:6.2f}%",
                            end="",
                            flush=True,
                        )

        temporary_file.replace(DATA_FILE)
        print(f"\n다운로드 완료: {DATA_FILE}")

    except (urllib.error.URLError, TimeoutError, OSError) as error:
        temporary_file.unlink(missing_ok=True)
        raise RuntimeError(
            "데이터 다운로드에 실패했습니다. 인터넷 연결을 확인하거나 "
            f"yellow_tripdata_2026-05.parquet 파일을 '{DATA_DIR}' 폴더에 "
            "직접 넣은 뒤 다시 실행해 주세요."
        ) from error


# ==============================================================================
# [문제 2] Pandas와 Polars 로딩 및 결과 비교
# ==============================================================================


def standardize_pandas_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """버전에 따라 달라질 수 있는 일부 컬럼명의 대소문자를 통일합니다."""
    rename_map: dict[str, str] = {}

    if "Airport_fee" in dataframe.columns and "airport_fee" not in dataframe.columns:
        rename_map["Airport_fee"] = "airport_fee"

    return dataframe.rename(columns=rename_map)


def standardize_polars_columns(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Polars 데이터에서도 일부 컬럼명의 대소문자를 통일합니다."""
    rename_map: dict[str, str] = {}

    if "Airport_fee" in dataframe.columns and "airport_fee" not in dataframe.columns:
        rename_map["Airport_fee"] = "airport_fee"

    return dataframe.rename(rename_map) if rename_map else dataframe


def validate_required_columns(columns: list[str]) -> None:
    """분석에 필요한 컬럼이 빠져 있지 않은지 확인합니다."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(columns))

    if missing_columns:
        raise ValueError(
            "데이터에 필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
        )


def sample_pandas_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """고정된 난수로 재현 가능한 분석용 표본을 추출합니다."""
    if ANALYSIS_SAMPLE_SIZE is None or len(dataframe) <= ANALYSIS_SAMPLE_SIZE:
        return dataframe.copy()

    return dataframe.sample(
        n=ANALYSIS_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).copy()


def load_and_compare() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    동일한 Parquet 파일을 Polars와 Pandas로 각각 전체 로딩합니다.

    메모리 사용량을 줄이기 위해 두 전체 DataFrame을 동시에 보관하지 않고,
    Polars 확인 후 메모리를 해제한 다음 Pandas를 로딩합니다.
    """
    # --------------------------------------------------------------------------
    # 1. Polars 전체 로딩
    # --------------------------------------------------------------------------
    polars_start = time.perf_counter()
    polars_df = pl.read_parquet(DATA_FILE)
    polars_df = standardize_polars_columns(polars_df)
    polars_seconds = time.perf_counter() - polars_start

    validate_required_columns(polars_df.columns)

    polars_rows = polars_df.height
    polars_columns = polars_df.width
    polars_column_names = polars_df.columns.copy()
    polars_missing_values = [int(value) for value in polars_df.null_count().row(0)]
    polars_missing_total = int(sum(polars_missing_values))
    polars_memory_mb = float(polars_df.estimated_size("mb"))

    polars_missing_df = pd.DataFrame(
        {
            "column": polars_column_names,
            "missing_count": polars_missing_values,
        }
    )
    polars_missing_df.to_csv(
        POLARS_MISSING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("[Polars 상위 5개 행]")
    print(polars_df.head(5))

    polars_stat_columns = [
        column
        for column in [
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "total_amount",
        ]
        if column in polars_df.columns
    ]
    polars_statistics = polars_df.select(polars_stat_columns).describe()
    polars_statistics.write_csv(POLARS_STATS_FILE)

    polars_payment_summary = (
        polars_df.group_by("payment_type")
        .agg(pl.len().alias("count"))
        .sort("payment_type")
    )
    polars_payment_summary.write_csv(POLARS_PAYMENT_FILE)

    print("\n[Polars 주요 기술통계]")
    print(polars_statistics)
    print("\n[Polars 결제 유형별 건수]")
    print(polars_payment_summary)

    # Polars 전체 DataFrame을 해제해 Pandas 로딩 시 메모리를 확보합니다.
    del polars_df
    gc.collect()

    # --------------------------------------------------------------------------
    # 2. Pandas 전체 로딩
    # --------------------------------------------------------------------------
    pandas_start = time.perf_counter()
    pandas_full_df = pd.read_parquet(DATA_FILE, engine="pyarrow")
    pandas_full_df = standardize_pandas_columns(pandas_full_df)
    pandas_seconds = time.perf_counter() - pandas_start

    validate_required_columns(pandas_full_df.columns.tolist())

    pandas_rows, pandas_columns = pandas_full_df.shape
    pandas_column_names = pandas_full_df.columns.tolist()
    pandas_missing_series = pandas_full_df.isna().sum()
    pandas_missing_total = int(pandas_missing_series.sum())
    pandas_memory_mb = float(
        pandas_full_df.memory_usage(index=True, deep=True).sum() / 1024**2
    )

    pandas_missing_df = pandas_missing_series.rename("missing_count").reset_index()
    pandas_missing_df.columns = ["column", "missing_count"]
    pandas_missing_df.to_csv(
        PANDAS_MISSING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    analysis_df = sample_pandas_dataframe(pandas_full_df)

    print("\n[Pandas 상위 5개 행]")
    print(pandas_full_df.head().to_string(index=False))
    print(f"\n전체 데이터 크기: {pandas_full_df.shape}")
    print(f"분석용 표본 크기: {analysis_df.shape}")

    # 분석용 표본만 남기고 Pandas 전체 DataFrame도 해제합니다.
    del pandas_full_df
    gc.collect()

    # --------------------------------------------------------------------------
    # 3. 두 라이브러리의 로딩 결과 비교
    # --------------------------------------------------------------------------
    loader_comparison = pd.DataFrame(
        [
            {
                "library": "Pandas",
                "rows": pandas_rows,
                "columns": pandas_columns,
                "missing_values": pandas_missing_total,
                "memory_mb": pandas_memory_mb,
                "load_seconds": pandas_seconds,
            },
            {
                "library": "Polars",
                "rows": polars_rows,
                "columns": polars_columns,
                "missing_values": polars_missing_total,
                "memory_mb": polars_memory_mb,
                "load_seconds": polars_seconds,
            },
        ]
    )
    loader_comparison.to_csv(
        LOADER_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    consistency_check = {
        "same_row_count": pandas_rows == polars_rows,
        "same_column_count": pandas_columns == polars_columns,
        "same_column_names": pandas_column_names == polars_column_names,
        "same_total_missing_count": pandas_missing_total == polars_missing_total,
        "pandas_columns": pandas_column_names,
        "polars_columns": polars_column_names,
        "source_file_size_mb": DATA_FILE.stat().st_size / 1024**2,
        "analysis_sample_rows": int(len(analysis_df)),
        "analysis_sample_rule": (
            "전체 사용"
            if ANALYSIS_SAMPLE_SIZE is None or pandas_rows <= len(analysis_df)
            else f"random_state={RANDOM_STATE}, 최대 {ANALYSIS_SAMPLE_SIZE:,}행"
        ),
    }
    save_json(consistency_check, LOADER_CHECK_FILE)

    print("\n[Pandas·Polars 로딩 비교]")
    print(loader_comparison.round(4).to_string(index=False))
    print("\n[일관성 확인]")
    for key in [
        "same_row_count",
        "same_column_count",
        "same_column_names",
        "same_total_missing_count",
    ]:
        print(f"{key}: {consistency_check[key]}")

    return analysis_df, loader_comparison, consistency_check


# ==============================================================================
# [문제 3] 중복·결측치·논리 오류 처리 및 기본 EDA
# ==============================================================================


def apply_filter_rule(
    dataframe: pd.DataFrame,
    rule_name: str,
    condition: pd.Series,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """정제 규칙을 적용하고 제거된 행 수를 기록합니다."""
    before_rows = len(dataframe)
    cleaned_df = dataframe.loc[condition.fillna(False)].copy()
    after_rows = len(cleaned_df)

    records.append(
        {
            "rule": rule_name,
            "before_rows": before_rows,
            "removed_rows": before_rows - after_rows,
            "after_rows": after_rows,
        }
    )

    return cleaned_df


def impute_for_eda(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    EDA용 복사본의 결측치를 처리합니다.

    수치형은 중앙값, 문자열·범주형은 'Unknown'으로 채웁니다.
    ML 학습에서는 이 복사본을 사용하지 않고 Pipeline 내부에서 별도로 처리합니다.
    """
    eda_df = dataframe.copy()

    code_columns = [
        column
        for column in [
            "VendorID",
            "RatecodeID",
            "PULocationID",
            "DOLocationID",
            "payment_type",
        ]
        if column in eda_df.columns
    ]
    numeric_columns = [
        column
        for column in eda_df.select_dtypes(include=np.number).columns.tolist()
        if column not in code_columns
    ]
    categorical_columns = eda_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # 숫자로 저장되어 있더라도 ID·코드 컬럼은 연속형 값이 아니므로 최빈값을 사용합니다.
    for column in code_columns:
        if eda_df[column].isna().any():
            mode_values = eda_df[column].mode(dropna=True)
            if not mode_values.empty:
                eda_df[column] = eda_df[column].fillna(mode_values.iloc[0])

    for column in numeric_columns:
        if eda_df[column].isna().any():
            median_value = eda_df[column].median()
            if pd.notna(median_value):
                eda_df[column] = eda_df[column].fillna(median_value)

    for column in categorical_columns:
        if eda_df[column].isna().any():
            eda_df[column] = eda_df[column].fillna("Unknown")

    return eda_df


def clean_analysis_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """중복 제거와 분석 목적에 맞는 논리적 범위 검사를 수행합니다."""
    working_df = dataframe.copy()

    # 날짜 컬럼은 오류가 있는 값을 NaT로 변환합니다.
    working_df["tpep_pickup_datetime"] = pd.to_datetime(
        working_df["tpep_pickup_datetime"],
        errors="coerce",
    )
    working_df["tpep_dropoff_datetime"] = pd.to_datetime(
        working_df["tpep_dropoff_datetime"],
        errors="coerce",
    )

    missing_before = working_df.isna().sum().rename("missing_count").to_frame()
    missing_before["missing_rate_percent"] = (
        missing_before["missing_count"] / len(working_df) * 100
    )
    missing_before.to_csv(MISSING_BEFORE_FILE, encoding="utf-8-sig")

    cleaning_records: list[dict[str, Any]] = []

    # 완전히 같은 행만 중복으로 판단합니다.
    duplicate_count = int(working_df.duplicated().sum())
    before_duplicate_rows = len(working_df)
    working_df = working_df.drop_duplicates().copy()
    cleaning_records.append(
        {
            "rule": "완전 중복 행 제거",
            "before_rows": before_duplicate_rows,
            "removed_rows": duplicate_count,
            "after_rows": len(working_df),
        }
    )

    # 운행 시간을 분 단위 파생변수로 만듭니다.
    working_df["trip_duration_min"] = (
        working_df["tpep_dropoff_datetime"]
        - working_df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # 다음 기준은 TLC 공식 오류 기준이 아니라 본 프로젝트의 분석용 품질 규칙입니다.
    working_df = apply_filter_rule(
        working_df,
        "승차일시가 2026년 5월 범위",
        working_df["tpep_pickup_datetime"].ge(ANALYSIS_START)
        & working_df["tpep_pickup_datetime"].lt(ANALYSIS_END),
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "하차일시가 승차일시보다 늦음",
        working_df["tpep_dropoff_datetime"]
        > working_df["tpep_pickup_datetime"],
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "운행시간 1분 이상 180분 이하",
        working_df["trip_duration_min"].between(1, 180, inclusive="both"),
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "운행거리 0.1마일 이상 100마일 이하",
        working_df["trip_distance"].between(0.1, 100, inclusive="both"),
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "미터요금 0달러 초과 500달러 이하",
        working_df["fare_amount"].gt(0)
        & working_df["fare_amount"].le(500),
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "총 결제금액 0달러 초과 1000달러 이하",
        working_df["total_amount"].gt(0)
        & working_df["total_amount"].le(1000),
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "승차·하차 위치 ID가 1~265 범위",
        working_df["PULocationID"].between(1, 265, inclusive="both")
        & working_df["DOLocationID"].between(1, 265, inclusive="both"),
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "승객 수가 결측이거나 0~8명 범위",
        working_df["passenger_count"].isna()
        | working_df["passenger_count"].between(0, 8, inclusive="both"),
        cleaning_records,
    )
    working_df = apply_filter_rule(
        working_df,
        "결제 유형이 결측이거나 0~6 범위",
        working_df["payment_type"].isna()
        | working_df["payment_type"].between(0, 6, inclusive="both"),
        cleaning_records,
    )

    # 시간 관련 파생변수를 생성합니다.
    working_df["pickup_date"] = working_df["tpep_pickup_datetime"].dt.date
    working_df["pickup_hour"] = working_df["tpep_pickup_datetime"].dt.hour
    working_df["pickup_dayofweek"] = (
        working_df["tpep_pickup_datetime"].dt.dayofweek
    )
    working_df["day_type"] = np.where(
        working_df["pickup_dayofweek"] < 5,
        "Weekday",
        "Weekend",
    )

    cleaning_summary = pd.DataFrame(cleaning_records)
    cleaning_summary.to_csv(
        CLEANING_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # EDA에서는 결측치를 미리 채운 복사본을 사용합니다.
    eda_df = impute_for_eda(working_df)
    missing_after = eda_df.isna().sum().rename("missing_count").to_frame()
    missing_after["missing_rate_percent"] = (
        missing_after["missing_count"] / len(eda_df) * 100
    )
    missing_after.to_csv(MISSING_AFTER_FILE, encoding="utf-8-sig")

    # 제출물 확인용으로 정제 데이터 앞부분만 저장합니다.
    eda_df.head(10_000).to_csv(
        CLEANED_PREVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("[정제 단계별 결과]")
    print(cleaning_summary.to_string(index=False))
    print(f"\n정제 후 분석 행 수: {len(working_df):,}")
    print(f"EDA 결측치 처리 후 남은 결측치: {int(eda_df.isna().sum().sum()):,}")

    return working_df, eda_df, cleaning_summary, missing_before, missing_after


def prepare_card_tip_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """현금 팁 누락 문제를 피하기 위해 카드 결제 운행만 팁 분석용으로 준비합니다."""
    card_df = dataframe.loc[
        (dataframe["payment_type"] == 1)
        & dataframe["tip_amount"].notna()
        & dataframe["total_amount"].notna()
    ].copy()

    # total_amount에는 카드 팁이 포함되므로 팁을 뺀 금액을 결제 전 금액으로 사용합니다.
    card_df["pre_tip_amount"] = card_df["total_amount"] - card_df["tip_amount"]
    card_df = card_df.loc[card_df["pre_tip_amount"] > 0].copy()

    card_df["tip_rate"] = card_df["tip_amount"] / card_df["pre_tip_amount"]

    # 음수 팁과 100%를 초과하는 극단값은 팁 비율 분석에서 제외합니다.
    card_df = card_df.loc[
        card_df["tip_rate"].between(0, 1, inclusive="both")
    ].copy()

    card_df["tip_rate_pct"] = card_df["tip_rate"] * 100
    card_df["high_tip"] = (
        card_df["tip_rate"] >= HIGH_TIP_THRESHOLD
    ).astype(int)

    if card_df.empty:
        raise ValueError("팁 분석에 사용할 카드 결제 데이터가 없습니다.")

    return card_df


def print_basic_eda(
    eda_df: pd.DataFrame,
    card_tip_df: pd.DataFrame,
) -> pd.DataFrame:
    """데이터 구조, 결제 유형, 목표값 분포를 출력합니다."""
    print(f"EDA 데이터 크기: {eda_df.shape}")

    print("\n[상위 5개 행]")
    print(eda_df.head().to_string(index=False))

    print("\n[데이터 구조]")
    info_buffer = io.StringIO()
    eda_df.info(buf=info_buffer)
    print(info_buffer.getvalue())

    payment_summary = (
        eda_df["payment_type"]
        .value_counts(dropna=False)
        .rename_axis("payment_type")
        .reset_index(name="count")
    )
    payment_summary["payment_label"] = payment_summary["payment_type"].map(
        PAYMENT_TYPE_LABELS
    )
    payment_summary["ratio_percent"] = (
        payment_summary["count"] / payment_summary["count"].sum() * 100
    )
    payment_summary.to_csv(
        PAYMENT_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    target_summary = (
        card_tip_df["high_tip"]
        .value_counts()
        .sort_index()
        .rename_axis("high_tip")
        .reset_index(name="count")
    )
    target_summary["label"] = target_summary["high_tip"].map(
        {0: "Below 20%", 1: "20% or more"}
    )
    target_summary["ratio_percent"] = (
        target_summary["count"] / target_summary["count"].sum() * 100
    )
    target_summary.to_csv(
        TARGET_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("[결제 유형별 건수와 비율]")
    print(payment_summary.round(4).to_string(index=False))
    print("\n[카드 결제 고팁 여부 분포]")
    print(target_summary.round(4).to_string(index=False))

    return target_summary


# ==============================================================================
# [문제 4] 기술통계와 상관계수
# ==============================================================================


def calculate_statistics(
    eda_df: pd.DataFrame,
    card_tip_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """평균·표준편차·분위수와 주요 수치형 변수의 상관계수를 계산합니다."""
    statistics_columns = [
        column
        for column in [
            "passenger_count",
            "trip_distance",
            "trip_duration_min",
            "fare_amount",
            "tip_amount",
            "tolls_amount",
            "total_amount",
        ]
        if column in eda_df.columns
    ]

    descriptive_statistics = eda_df[statistics_columns].describe(
        percentiles=[0.25, 0.5, 0.75]
    ).T
    descriptive_statistics = descriptive_statistics[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    descriptive_statistics.to_csv(
        DESCRIPTIVE_STATS_FILE,
        encoding="utf-8-sig",
    )

    # 팁 관련 상관계수는 현금 팁 누락 영향을 피하기 위해 카드 결제만 사용합니다.
    correlation_columns = [
        column
        for column in [
            "passenger_count",
            "trip_distance",
            "trip_duration_min",
            "fare_amount",
            "tolls_amount",
            "pre_tip_amount",
            "tip_amount",
            "tip_rate_pct",
            "pickup_hour",
        ]
        if column in card_tip_df.columns
    ]
    correlation_matrix = card_tip_df[correlation_columns].corr()
    correlation_matrix.to_csv(
        CORRELATION_FILE,
        encoding="utf-8-sig",
    )

    print("[기술통계]")
    print(descriptive_statistics.round(4).to_string())
    print("\n[카드 결제 데이터 상관계수]")
    print(correlation_matrix.round(4).to_string())

    return descriptive_statistics, correlation_matrix


# ==============================================================================
# [문제 5] Seaborn 정적 차트와 Plotly 인터랙티브 차트
# ==============================================================================


def create_visualizations(
    eda_df: pd.DataFrame,
    card_tip_df: pd.DataFrame,
) -> None:
    """그룹 비교를 위한 정적·인터랙티브 차트를 각각 생성합니다."""
    sns.set_theme(style="whitegrid")

    # --------------------------------------------------------------------------
    # Seaborn: 평일·주말 카드 팁 비율 Boxplot
    # --------------------------------------------------------------------------
    plot_card_df = card_tip_df.loc[card_tip_df["tip_rate_pct"] <= 50].copy()

    if len(plot_card_df) > PLOT_SAMPLE_SIZE:
        plot_card_df = plot_card_df.sample(
            n=PLOT_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    plt.figure(figsize=(8, 6))
    axis = sns.boxplot(
        data=plot_card_df,
        x="day_type",
        y="tip_rate_pct",
        order=["Weekday", "Weekend"],
        showfliers=False,
    )
    axis.set_title("Credit-Card Tip Rate: Weekday vs Weekend")
    axis.set_xlabel("Day Type")
    axis.set_ylabel("Tip Rate (%)")
    plt.tight_layout()
    plt.savefig(SEABORN_CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------------------
    # Plotly: 평일·주말 시간대별 운행 비중
    # --------------------------------------------------------------------------
    hourly_summary = (
        eda_df.groupby(["day_type", "pickup_hour"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )
    hourly_summary["trip_share_percent"] = (
        hourly_summary["trip_count"]
        / hourly_summary.groupby("day_type")["trip_count"].transform("sum")
        * 100
    )

    figure = px.line(
        hourly_summary,
        x="pickup_hour",
        y="trip_share_percent",
        color="day_type",
        markers=True,
        hover_data={
            "trip_count": ":,",
            "trip_share_percent": ":.2f",
        },
        category_orders={"day_type": ["Weekday", "Weekend"]},
        title="Hourly Trip Share: Weekday vs Weekend",
        labels={
            "pickup_hour": "Pickup Hour",
            "trip_share_percent": "Trip Share (%)",
            "day_type": "Day Type",
            "trip_count": "Trip Count in Analysis Data",
        },
    )
    figure.update_xaxes(dtick=1)
    figure.write_html(
        PLOTLY_CHART_FILE,
        include_plotlyjs=True,
        full_html=True,
    )

    print(f"Seaborn 차트 저장: {SEABORN_CHART_FILE}")
    print(f"Plotly 차트 저장: {PLOTLY_CHART_FILE}")


# ==============================================================================
# [문제 6] 독립표본 t-test와 p-value 해석
# ==============================================================================


def sample_group_for_ttest(series: pd.Series) -> pd.Series:
    """t-test 계산량과 과도한 표본 크기를 줄이기 위해 그룹별 상한을 둡니다."""
    if len(series) <= TTEST_MAX_PER_GROUP:
        return series

    return series.sample(
        n=TTEST_MAX_PER_GROUP,
        random_state=RANDOM_STATE,
    )


def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """두 독립 집단 평균 차이의 표준화 효과크기 Cohen's d를 계산합니다."""
    variance1 = group1.var(ddof=1)
    variance2 = group2.var(ddof=1)
    pooled_std = np.sqrt((variance1 + variance2) / 2)

    if pooled_std == 0 or pd.isna(pooled_std):
        return 0.0

    return float((group1.mean() - group2.mean()) / pooled_std)


def perform_t_test(card_tip_df: pd.DataFrame) -> dict[str, Any]:
    """평일과 주말 카드 결제 팁 비율 평균을 Welch t-test로 비교합니다."""
    weekday_tip = card_tip_df.loc[
        card_tip_df["day_type"] == "Weekday",
        "tip_rate_pct",
    ].dropna()
    weekend_tip = card_tip_df.loc[
        card_tip_df["day_type"] == "Weekend",
        "tip_rate_pct",
    ].dropna()

    weekday_tip = sample_group_for_ttest(weekday_tip)
    weekend_tip = sample_group_for_ttest(weekend_tip)

    if len(weekday_tip) < 2 or len(weekend_tip) < 2:
        raise ValueError("t-test를 수행할 평일 또는 주말 표본이 부족합니다.")

    test_result = ttest_ind(
        weekday_tip,
        weekend_tip,
        equal_var=False,
        nan_policy="omit",
    )

    p_value = float(test_result.pvalue)
    significant = p_value < ALPHA
    mean_difference = float(weekday_tip.mean() - weekend_tip.mean())
    cohens_d = calculate_cohens_d(weekday_tip, weekend_tip)

    if significant:
        interpretation = (
            f"p-value가 유의수준 {ALPHA}보다 작으므로 평일과 주말의 "
            "평균 카드 팁 비율이 같다는 귀무가설을 기각합니다. "
            "다만 통계적 유의성과 실제 차이의 크기는 구분하여 해석해야 합니다."
        )
    else:
        interpretation = (
            f"p-value가 유의수준 {ALPHA} 이상이므로 평일과 주말의 "
            "평균 카드 팁 비율이 다르다고 판단할 통계적 근거가 부족합니다."
        )

    result = {
        "test": "Welch independent two-sample t-test",
        "variable": "tip_rate_pct",
        "group_1": "Weekday",
        "group_2": "Weekend",
        "weekday_count": int(len(weekday_tip)),
        "weekend_count": int(len(weekend_tip)),
        "weekday_mean_percent": float(weekday_tip.mean()),
        "weekend_mean_percent": float(weekend_tip.mean()),
        "mean_difference_percentage_point": mean_difference,
        "t_statistic": float(test_result.statistic),
        "p_value": p_value,
        "alpha": ALPHA,
        "is_significant": bool(significant),
        "cohens_d": cohens_d,
        "interpretation": interpretation,
    }
    save_json(result, TTEST_RESULT_FILE)

    print(f"평일 평균 팁 비율: {result['weekday_mean_percent']:.4f}%")
    print(f"주말 평균 팁 비율: {result['weekend_mean_percent']:.4f}%")
    print(f"평균 차이: {mean_difference:.4f}%p")
    print(f"t-statistic: {result['t_statistic']:.4f}")
    print(f"p-value: {p_value:.6e}")
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"해석: {interpretation}")

    return result


# ==============================================================================
# [문제 7] ML Pipeline: 전처리 + 고팁 여부 분류
# ==============================================================================


def prepare_model_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    모델 입력에 필요한 시간 파생변수를 만들고 범주형 자료형을 통일합니다.

    저장된 Pipeline은 결측치 대체·스케일링·원-핫 인코딩·모델을 포함합니다.
    날짜에서 운행시간과 시간대 변수를 만드는 단계는 이 함수에서 수행합니다.
    """
    feature_df = dataframe.copy()

    pickup_datetime = pd.to_datetime(
        feature_df["tpep_pickup_datetime"],
        errors="coerce",
    )
    dropoff_datetime = pd.to_datetime(
        feature_df["tpep_dropoff_datetime"],
        errors="coerce",
    )

    feature_df["trip_duration_min"] = (
        dropoff_datetime - pickup_datetime
    ).dt.total_seconds() / 60
    feature_df["pickup_hour"] = pickup_datetime.dt.hour
    feature_df["pickup_dayofweek"] = pickup_datetime.dt.dayofweek

    categorical_columns = [
        "VendorID",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_dayofweek",
    ]

    # OneHotEncoder에는 한 컬럼 안에서 숫자와 문자열이 섞이지 않도록
    # 모든 코드형 범주 값을 문자열로 통일합니다.
    for column in categorical_columns:
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].map(
                lambda value: np.nan if pd.isna(value) else str(value)
            )

    return feature_df


def select_stratified_model_sample(dataframe: pd.DataFrame) -> pd.DataFrame:
    """목표값 비율을 유지하면서 모델 학습용 표본을 추출합니다."""
    if len(dataframe) <= ML_SAMPLE_SIZE:
        return dataframe.copy()

    sampled_df, _ = train_test_split(
        dataframe,
        train_size=ML_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe["high_tip"],
    )
    return sampled_df.copy()


def train_ml_pipeline(card_tip_df: pd.DataFrame) -> dict[str, Any]:
    """팁 비율 20% 이상 여부를 예측하는 분류 Pipeline을 학습합니다."""
    model_df = select_stratified_model_sample(card_tip_df)

    # total_amount와 tip_amount는 목표값 계산에 사용되므로 입력 변수에서 제외합니다.
    # 이를 포함하면 정답 정보가 모델 입력에 들어가는 데이터 누수가 발생합니다.
    model_feature_df = prepare_model_features(model_df)

    numeric_feature_candidates = [
        "passenger_count",
        "trip_distance",
        "trip_duration_min",
        "fare_amount",
        "extra",
        "mta_tax",
        "tolls_amount",
        "improvement_surcharge",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
    ]
    numeric_features = [
        column for column in numeric_feature_candidates if column in model_feature_df.columns
    ]

    categorical_feature_candidates = [
        "VendorID",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_dayofweek",
    ]
    categorical_features = [
        column
        for column in categorical_feature_candidates
        if column in model_feature_df.columns
    ]

    input_features = numeric_features + categorical_features
    feature_data = model_feature_df[input_features].copy()
    target_data = model_df["high_tip"].astype(int)

    if target_data.nunique() < 2:
        raise ValueError("ML 학습에 필요한 두 개의 목표 클래스가 존재하지 않습니다.")

    x_train, x_test, y_train, y_test = train_test_split(
        feature_data,
        target_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_data,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model_pipeline.fit(x_train, y_train)
    predictions = model_pipeline.predict(x_test)
    probabilities = model_pipeline.predict_proba(x_test)[:, 1]

    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probabilities))
    baseline_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))
    confusion = confusion_matrix(y_test, predictions).tolist()
    report_text = classification_report(
        y_test,
        predictions,
        target_names=["Below 20%", "20% or more"],
        digits=4,
        zero_division=0,
    )

    # 전처리와 모델이 모두 들어 있는 Pipeline 전체를 저장합니다.
    joblib.dump(model_pipeline, MODEL_FILE, compress=3)

    model_result: dict[str, Any] = {
        "target_definition": (
            f"tip_amount / (total_amount - tip_amount) >= {HIGH_TIP_THRESHOLD:.0%}"
        ),
        "analysis_scope": "Credit-card trips only (payment_type == 1)",
        "model_rows": int(len(model_df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "input_features": input_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion,
        "classification_report": report_text,
        "model_file": str(MODEL_FILE),
    }
    save_json(model_result, MODEL_METRICS_FILE)

    print(f"모델 표본 수: {len(model_df):,}")
    print(f"학습 데이터 수: {len(x_train):,}")
    print(f"평가 데이터 수: {len(x_test):,}")
    print(f"기준 정확도: {baseline_accuracy:.4f}")
    print(f"정확도(Accuracy): {accuracy:.4f}")
    print(f"정밀도(Precision): {precision:.4f}")
    print(f"재현율(Recall): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\n[Classification Report]")
    print(report_text)
    print(f"모델 저장: {MODEL_FILE}")

    return model_result


# ==============================================================================
# [문제 8] report.md 자동 생성
# ==============================================================================


def generate_report(
    loader_comparison: pd.DataFrame,
    consistency_check: dict[str, Any],
    cleaning_summary: pd.DataFrame,
    missing_before: pd.DataFrame,
    missing_after: pd.DataFrame,
    descriptive_statistics: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    target_summary: pd.DataFrame,
    t_test_result: dict[str, Any],
    model_result: dict[str, Any],
) -> None:
    """실행 결과를 하나의 Markdown 보고서로 자동 작성합니다."""
    missing_before_visible = missing_before.loc[
        missing_before["missing_count"] > 0
    ].copy()
    missing_after_visible = missing_after.loc[
        missing_after["missing_count"] > 0
    ].copy()

    metric_table = pd.DataFrame(
        {
            "metric": [
                "Baseline Accuracy",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
            ],
            "value": [
                model_result["baseline_accuracy"],
                model_result["accuracy"],
                model_result["precision"],
                model_result["recall"],
                model_result["f1_score"],
                model_result["roc_auc"],
            ],
        }
    )

    confusion_table = pd.DataFrame(
        model_result["confusion_matrix"],
        index=["Actual Below 20%", "Actual 20% or more"],
        columns=["Predicted Below 20%", "Predicted 20% or more"],
    )

    ttest_table = pd.DataFrame(
        {
            "item": [
                "Weekday sample count",
                "Weekend sample count",
                "Weekday mean tip rate (%)",
                "Weekend mean tip rate (%)",
                "Mean difference (%p)",
                "t-statistic",
                "p-value",
                "Cohen's d",
            ],
            "value": [
                t_test_result["weekday_count"],
                t_test_result["weekend_count"],
                t_test_result["weekday_mean_percent"],
                t_test_result["weekend_mean_percent"],
                t_test_result["mean_difference_percentage_point"],
                t_test_result["t_statistic"],
                t_test_result["p_value"],
                t_test_result["cohens_d"],
            ],
        }
    )

    missing_before_text = (
        dataframe_to_markdown(missing_before_visible)
        if not missing_before_visible.empty
        else "결측치가 없습니다."
    )
    missing_after_text = (
        dataframe_to_markdown(missing_after_visible)
        if not missing_after_visible.empty
        else "EDA용 결측치 처리 후 남은 결측치가 없습니다."
    )
    analysis_scope_text = (
        "전체 데이터"
        if ANALYSIS_SAMPLE_SIZE is None
        else f"최대 {ANALYSIS_SAMPLE_SIZE:,}행의 분석 표본"
    )

    report_content = f"""# NYC Yellow Taxi End-to-End 데이터 분석 보고서

## 1. 분석 데이터와 목적

- 데이터: NYC TLC Yellow Taxi Trip Records, 2026년 5월
- 원본 형식: Parquet
- 분석 목적 1: 평일과 주말의 카드 결제 팁 비율 평균 차이 검정
- 분석 목적 2: 운행 종료 시점의 정보로 팁 비율 20% 이상 여부 분류

`tip_amount`에는 현금 팁이 기록되지 않으므로 팁 분석과 모델 학습은
`payment_type == 1`인 카드 결제 운행으로 제한하였습니다.

고팁 목표값은 본 프로젝트에서 다음과 같이 정의하였습니다.

```text
팁 비율 = tip_amount / (total_amount - tip_amount)
high_tip = 팁 비율이 {HIGH_TIP_THRESHOLD:.0%} 이상이면 1, 아니면 0
```

## 2. Pandas와 Polars 로딩 비교

{dataframe_to_markdown(loader_comparison, include_index=False)}

- 행 수 일치: **{consistency_check['same_row_count']}**
- 열 수 일치: **{consistency_check['same_column_count']}**
- 컬럼명 및 순서 일치: **{consistency_check['same_column_names']}**
- 전체 결측치 수 일치: **{consistency_check['same_total_missing_count']}**
- 원본 파일 크기: **{consistency_check['source_file_size_mb']:.2f} MB**
- Pandas 후속 분석 행 수: **{consistency_check['analysis_sample_rows']:,}개**
- 표본 추출 기준: **{consistency_check['analysis_sample_rule']}**

로딩 시간은 실행 환경, 디스크 속도, 캐시 상태에 따라 달라질 수 있으므로
이번 실행 환경에서의 참고값으로 해석하였습니다.

## 3. 데이터 정제

{dataframe_to_markdown(cleaning_summary, include_index=False)}

정제 범위는 완전 중복 제거, 2026년 5월 승차 기록 확인, 양수 운행시간,
현실적인 분석 범위의 운행시간·거리·요금, 유효한 위치 ID 등을 포함합니다.
이 범위는 TLC 공식 오류 판정 규칙이 아니라 본 프로젝트의 분석용 품질 규칙입니다.

### 결측치 처리 전

{missing_before_text}

### EDA용 결측치 처리 후

{missing_after_text}

EDA용 데이터는 수치형 중앙값과 범주형 `Unknown`으로 처리하였습니다.
머신러닝에서는 전체 데이터에 미리 대체값을 적용하지 않고 Pipeline 내부에서
학습 데이터 기준으로 결측치를 처리하여 데이터 누수를 줄였습니다.

## 4. 기술통계

{dataframe_to_markdown(descriptive_statistics)}

- 평균은 데이터의 전체적인 중심을 보여줍니다.
- 표준편차는 평균 주변에서 값이 얼마나 퍼져 있는지 보여줍니다.
- 25%, 50%, 75% 분위수는 분포와 극단값을 확인하는 기준으로 사용하였습니다.

## 5. 카드 결제 데이터 상관계수

{dataframe_to_markdown(correlation_matrix)}

상관계수는 두 수치형 변수의 선형 관계를 나타냅니다. `total_amount`,
`tip_amount`, `pre_tip_amount`처럼 계산식으로 직접 연결된 변수는 구조적으로
높은 상관이 나타날 수 있으므로 인과관계로 해석하지 않았습니다.

## 6. 시각화

### Seaborn 정적 차트

![Credit-Card Tip Rate: Weekday vs Weekend](outputs/{SEABORN_CHART_FILE.name})

평일과 주말 카드 결제 팁 비율의 분포를 비교합니다. 차트 가독성을 위해
50% 이하의 팁 비율만 표시하고 이상점 표시는 숨겼으며, t-test에는 정제 기준을
통과한 0~100% 팁 비율을 사용하였습니다.

### Plotly 인터랙티브 차트

[시간대별 평일·주말 운행 비중 차트 열기](outputs/{PLOTLY_CHART_FILE.name})

평일과 주말의 표본 수 차이 영향을 줄이기 위해 단순 운행 건수가 아니라
각 그룹 내부의 시간대별 운행 비중을 비교하였습니다.

## 7. 독립표본 t-test

- 검정 방법: Welch independent two-sample t-test
- 귀무가설: 평일과 주말의 평균 카드 팁 비율은 같다.
- 대립가설: 평일과 주말의 평균 카드 팁 비율은 다르다.

{dataframe_to_markdown(ttest_table, include_index=False)}

**p-value 해석:** {t_test_result['interpretation']}

p-value는 차이의 존재 여부를 판단하는 기준이고, 차이의 실제 크기를 의미하지는
않으므로 평균 차이와 Cohen's d를 함께 출력하였습니다.

## 8. 고팁 여부 ML Pipeline

### 목표값 분포

{dataframe_to_markdown(target_summary, include_index=False)}

### Pipeline 구성

```text
날짜 컬럼에서 운행시간·승차시간·요일 파생변수 생성
    ↓
Pipeline 내부 수치형: 중앙값 대체 + StandardScaler
Pipeline 내부 범주형: Unknown 대체 + OneHotEncoder
    ↓
Pipeline 내부 LogisticRegression(class_weight='balanced')
```

`tip_amount`, `total_amount`, `tip_rate`, `high_tip`은 목표값과 직접 연결되므로
모델 입력에서 제외하여 데이터 누수를 방지하였습니다.

### 평가 결과

{dataframe_to_markdown(metric_table, include_index=False)}

### 혼동행렬

{dataframe_to_markdown(confusion_table)}

### Classification Report

```text
{model_result['classification_report']}
```

## 9. 자동 생성 파일

- Pandas·Polars 비교: `outputs/{LOADER_COMPARISON_FILE.name}`
- 정제 단계 결과: `outputs/{CLEANING_SUMMARY_FILE.name}`
- 기술통계: `outputs/{DESCRIPTIVE_STATS_FILE.name}`
- 상관계수: `outputs/{CORRELATION_FILE.name}`
- t-test 결과: `outputs/{TTEST_RESULT_FILE.name}`
- 모델 성능: `outputs/{MODEL_METRICS_FILE.name}`
- 저장 모델: `outputs/{MODEL_FILE.name}`

## 10. 본인 의견 및 개선 사항

이번 분석에서는 대용량 Parquet 파일을 Pandas와 Polars로 각각 로딩하여
동일한 데이터가 읽혔는지 확인한 뒤, 메모리 부담이 큰 시각화와 ML 과정에는
고정 난수 표본을 사용하였습니다. 팁 분석에서는 현금 팁이 기록되지 않는 데이터
특성을 고려하여 카드 결제만 사용하였고, 모델에서는 목표값 계산에 사용한 컬럼을
제외하여 데이터 누수를 방지하였습니다.

개선 시에는 여러 달의 데이터를 연결한 뒤 시간 순서대로 학습·평가 데이터를
분리하고, 로지스틱 회귀 외에 트리 기반 모델과 성능을 비교할 수 있습니다.
또한 택시 존 이름과 날씨·행사 데이터를 결합하면 위치와 외부 환경에 따른
운행 수요 및 팁 패턴을 더 구체적으로 분석할 수 있습니다.

## 11. 분석 한계

- 원본 데이터는 사업자가 제출한 운행 기록이므로 기록 오류 가능성이 있습니다.
- 팁 분석은 현금 팁이 없는 것이 아니라 데이터에 기록되지 않는 문제 때문에 카드 결제로 제한했습니다.
- 20% 기준은 공식 분류가 아니라 본 프로젝트에서 설정한 분석 기준입니다.
- 기본 코드는 재현성과 실행 시간을 위해 {analysis_scope_text}을 사용합니다.
- 무작위 학습·평가 분리는 교육용 구성이며 실제 운영 모델은 시간 순서 분리가 더 적절할 수 있습니다.
"""

    REPORT_FILE.write_text(report_content, encoding="utf-8")
    print(f"report.md 자동 생성 완료: {REPORT_FILE}")


# ==============================================================================
# [메인 실행] 전체 분석 순서
# ==============================================================================


def main() -> None:
    """데이터 준비부터 report.md 생성까지 전체 과정을 실행합니다."""
    create_directories()

    print_section("[문제 1] 데이터셋 준비")
    download_dataset()

    print_section("[문제 2] Pandas와 Polars 로딩 비교")
    analysis_df, loader_comparison, consistency_check = load_and_compare()

    print_section("[문제 3] 중복·결측치·비정상값 처리 및 기본 EDA")
    (
        cleaned_df,
        eda_df,
        cleaning_summary,
        missing_before,
        missing_after,
    ) = clean_analysis_data(analysis_df)
    card_tip_df = prepare_card_tip_data(cleaned_df)
    target_summary = print_basic_eda(eda_df, card_tip_df)

    print_section("[문제 4] 기술통계와 상관계수")
    descriptive_statistics, correlation_matrix = calculate_statistics(
        eda_df,
        card_tip_df,
    )

    print_section("[문제 5] Seaborn·Plotly 시각화")
    create_visualizations(eda_df, card_tip_df)

    print_section("[문제 6] 독립표본 t-test")
    t_test_result = perform_t_test(card_tip_df)

    print_section("[문제 7] ML Pipeline 학습·평가·저장")
    model_result = train_ml_pipeline(card_tip_df)

    print_section("[문제 8] report.md 자동 생성")
    generate_report(
        loader_comparison,
        consistency_check,
        cleaning_summary,
        missing_before,
        missing_after,
        descriptive_statistics,
        correlation_matrix,
        target_summary,
        t_test_result,
        model_result,
    )

    print_section("분석 완료")
    print(f"최종 보고서: {REPORT_FILE}")
    print(f"결과 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n분석 실행 중 오류가 발생했습니다: {error}")
        raise
