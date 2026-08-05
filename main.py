"""
프로그램명: NYC Yellow Taxi Question-Driven End-to-End EDA + ML Pipeline
작성자: 윤서준

프로그램 설명:
    NYC TLC Yellow Taxi 2026년 5월 Parquet 데이터를 대상으로 다음 과정을
    하나의 파이썬 파일에서 순서대로 수행합니다.

    1. Pandas와 Polars 전체 로딩 결과 비교
    2. 질문 중심(Question-Driven) EDA
    3. 결측치·중복·논리 오류·극단값 후보 진단
    4. 다양한 Seaborn 정적 차트와 Plotly 인터랙티브 차트 생성
    5. 기술통계·상관계수·Welch t-test 및 p-value 해석
    6. sklearn Pipeline 기반 고팁 여부 분류 모델 학습
    7. 정확도·F1·ROC-AUC 등 평가, 계수 분석, joblib 모델 저장
    8. EDA 근거와 모델링 판단을 report.md로 자동 작성

핵심 원칙:
    - 매우 큰 값은 곧바로 삭제하지 않습니다.
    - 논리적으로 불가능한 값과 드물지만 가능한 극단값을 구분합니다.
    - 극단값은 IQR·상위 분위수·물리적 관계를 이용해 '후보 플래그'로 남깁니다.
    - 결측치는 데이터 전체에서 임의로 미리 채우지 않고, ML Pipeline 내부에서
      학습 데이터 기준으로 처리하여 데이터 누수를 줄입니다.
    - 팁 분석은 현금 팁이 tip_amount에 기록되지 않는 문제를 피하기 위해
      신용카드 결제(payment_type == 1)만 사용합니다.

실행 방법:
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import gc
import io
import json
import math
import shutil
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==============================================================================
# [설정] 경로, 표본 크기, 분석 기준
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
INTERACTIVE_DIR = OUTPUT_DIR / "interactive"
MODEL_DIR = OUTPUT_DIR / "models"

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2026-05.parquet"
)
DATA_FILE = DATA_DIR / "yellow_tripdata_2026-05.parquet"

ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)
ZONE_LOOKUP_FILE = DATA_DIR / "taxi_zone_lookup.csv"

REPORT_FILE = BASE_DIR / "report.md"
PRESENTATION_FILE = BASE_DIR / "presentation_5min.md"

# 전체 Pandas 데이터를 읽은 뒤, 후속 EDA에 사용할 행 수입니다.
# None으로 변경하면 후속 Pandas EDA에도 전체 데이터를 사용합니다.
ANALYSIS_SAMPLE_SIZE: int | None = 500_000

# 시각화는 화면에 표현할 수 있는 정보량이 제한적이므로 별도 표본을 사용합니다.
# None으로 변경하면 전체 분석 코호트를 사용합니다.
PLOT_SAMPLE_SIZE: int | None = 80_000

# ML 학습에 사용할 최대 행 수입니다. None이면 전체 카드 결제 코호트를 사용합니다.
MODEL_SAMPLE_SIZE: int | None = 200_000

# t-test는 대규모 표본에서 아주 작은 차이도 유의해질 수 있으므로 그룹별 최대 수를 둡니다.
# None이면 각 그룹의 전체 데이터를 사용합니다.
TTEST_MAX_PER_GROUP: int | None = 150_000

RANDOM_STATE = 42
TEST_SIZE = 0.20
ALPHA = 0.05

# 결제 전 금액 대비 팁 비율이 20% 이상이면 고팁으로 정의합니다.
# 이는 본 프로젝트의 분류 기준이며 TLC 공식 기준은 아닙니다.
HIGH_TIP_THRESHOLD = 0.20

# t-test 민감도 분석에서 극단 팁 비율의 영향을 확인하기 위한 상단 분위수입니다.
TTEST_WINSOR_QUANTILE = 0.995

# 평균속도는 단독 삭제 기준이 아니라 검토용 품질 플래그로만 사용합니다.
SPEED_REVIEW_THRESHOLD_MPH = 100.0

# 요금 구성 합과 total_amount의 차이가 이 값을 넘으면 검토 대상으로 표시합니다.
TOTAL_COMPONENT_TOLERANCE = 0.10

# 범주별 고팁률을 해석할 때 너무 작은 표본을 제외하기 위한 최소 건수입니다.
MIN_CATEGORY_SUPPORT = 100

ANALYSIS_START = pd.Timestamp("2026-05-01 00:00:00")
ANALYSIS_END = pd.Timestamp("2026-06-01 00:00:00")

# 시간 순서로 학습/평가 데이터를 나누는 방식이 실제 미래 예측에 더 가깝습니다.
# 시간 분할이 불가능하면 자동으로 층화 무작위 분할로 전환합니다.
MODEL_SPLIT_MODE = "time"

# 완전히 동일한 레코드는 분석 표본에서 제거합니다.
DROP_EXACT_DUPLICATES = True


# ==============================================================================
# [설정] 컬럼 정의와 공식 코드 체계
# ==============================================================================

CORE_REQUIRED_COLUMNS = {
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

OPTIONAL_NUMERIC_COLUMNS = [
    "airport_fee",
    "cbd_congestion_fee",
]

VENDOR_LABELS = {
    1: "Creative Mobile Technologies",
    2: "Curb Mobility",
    6: "Myle Technologies",
    7: "Helix",
}

RATE_CODE_LABELS = {
    1: "Standard rate",
    2: "JFK",
    3: "Newark",
    4: "Nassau or Westchester",
    5: "Negotiated fare",
    6: "Group ride",
    99: "Null/unknown",
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

VALID_VENDOR_CODES = set(VENDOR_LABELS)
VALID_RATE_CODES = set(RATE_CODE_LABELS)
VALID_PAYMENT_CODES = set(PAYMENT_TYPE_LABELS)
VALID_STORE_FLAGS = {"Y", "N"}

# 수치값의 크기 자체가 의미를 갖는 연속형·이산형 변수입니다.
BASE_NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

# 숫자로 저장되어 있어도 크기와 순서가 의미 없는 코드형 범주 변수입니다.
BASE_CATEGORICAL_COLUMNS = [
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
]

DATETIME_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]

COLUMN_DESCRIPTIONS = {
    "VendorID": "운행 레코드를 제공한 TPEP 사업자 코드",
    "tpep_pickup_datetime": "택시 미터가 켜진 승차 일시",
    "tpep_dropoff_datetime": "택시 미터가 꺼진 하차 일시",
    "passenger_count": "운전자가 보고한 승객 수",
    "trip_distance": "미터기가 보고한 누적 운행거리(마일)",
    "RatecodeID": "운행 종료 시 적용된 최종 요금 체계 코드",
    "store_and_fwd_flag": "통신 문제로 차량에 저장 후 전송했는지 여부",
    "PULocationID": "승차 TLC Taxi Zone 코드",
    "DOLocationID": "하차 TLC Taxi Zone 코드",
    "payment_type": "결제 방법 코드",
    "fare_amount": "미터가 계산한 시간·거리 기반 기본요금",
    "extra": "기타 할증 및 추가요금",
    "mta_tax": "요금 체계에 따라 자동 부과되는 MTA 세금",
    "tip_amount": "카드 결제 팁 금액; 현금 팁은 포함되지 않음",
    "tolls_amount": "운행 중 지불한 통행료 합계",
    "improvement_surcharge": "개선 부담금",
    "total_amount": "승객에게 청구된 총 금액; 현금 팁 제외",
    "congestion_surcharge": "NYS 혼잡 할증료",
    "airport_fee": "라과디아·JFK 공항 승차 시 부과되는 공항 요금",
    "cbd_congestion_fee": "혼잡통행료 구역 관련 운행당 요금",
}

LEAKAGE_COLUMNS = {
    "tip_amount",
    "total_amount",
    "tip_rate",
    "tip_rate_pct",
    "high_tip",
}


# ==============================================================================
# [공통 함수] 출력, 저장, 자료형 변환
# ==============================================================================


def print_section(title: str) -> None:
    """실행 단계를 명확하게 구분하여 출력합니다."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def create_directories() -> None:
    """분석에 필요한 폴더를 생성합니다."""
    for directory in [
        DATA_DIR,
        OUTPUT_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        INTERACTIVE_DIR,
        MODEL_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    """NumPy·Pandas 값을 JSON 저장 가능한 기본형으로 변환합니다."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"JSON 변환을 지원하지 않는 자료형입니다: {type(value)}")


def save_json(data: dict[str, Any], path: Path) -> None:
    """딕셔너리를 UTF-8 JSON으로 저장합니다."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


def safe_float(value: Any, default: float = float("nan")) -> float:
    """숫자로 변환할 수 없는 값을 안전하게 기본값으로 바꿉니다."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """분모가 0이거나 결측인 경우 NaN을 반환하는 안전한 나눗셈입니다."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def format_markdown_value(value: Any) -> str:
    """Markdown 표에 들어갈 값을 보기 좋은 문자열로 바꿉니다."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.4f}"
        return str(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(
    dataframe: pd.DataFrame,
    include_index: bool = True,
    max_rows: int | None = 20,
) -> str:
    """tabulate 없이 DataFrame을 Markdown 표로 변환합니다."""
    if dataframe is None or dataframe.empty:
        return "표시할 결과가 없습니다."

    table_df = dataframe.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)

    if include_index:
        index_name = table_df.index.name or "index"
        table_df = table_df.reset_index().rename(columns={"index": index_name})

    headers = [str(column) for column in table_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in table_df.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(format_markdown_value(value) for value in row)
            + " |"
        )

    return "\n".join(lines)


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
    index: bool = False,
) -> Path:
    """표를 outputs/tables 폴더에 UTF-8 CSV로 저장합니다."""
    path = TABLE_DIR / filename
    dataframe.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def sample_rows(
    dataframe: pd.DataFrame,
    max_rows: int | None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """설정값에 따라 전체 또는 재현 가능한 무작위 표본을 반환합니다."""
    if max_rows is None or len(dataframe) <= max_rows:
        return dataframe.copy()

    return dataframe.sample(
        n=max_rows,
        random_state=random_state,
    ).copy()


def existing_columns(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> list[str]:
    """후보 목록 중 실제 DataFrame에 존재하는 컬럼만 반환합니다."""
    return [column for column in candidates if column in dataframe.columns]


def first_existing_value(
    dataframe: pd.DataFrame,
    column: str,
    default: Any = "",
) -> Any:
    """DataFrame의 첫 행 값을 안전하게 가져옵니다."""
    if dataframe.empty or column not in dataframe.columns:
        return default
    return dataframe.iloc[0][column]


# ==============================================================================
# [문제 1] 데이터 다운로드와 컬럼 표준화
# ==============================================================================


def download_file(url: str, destination: Path, label: str) -> bool:
    """파일이 없을 때 URL에서 다운로드하며, 실패 여부를 bool로 반환합니다."""
    if destination.exists() and destination.stat().st_size > 0:
        size_mb = destination.stat().st_size / 1024**2
        print(f"기존 {label} 파일 사용: {destination} ({size_mb:.1f} MB)")
        return True

    print(f"{label} 다운로드 시작: {url}")
    temporary_file = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
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
                            f"\r{label} 다운로드 진행률: {progress:6.2f}%",
                            end="",
                            flush=True,
                        )

        temporary_file.replace(destination)
        print(f"\n{label} 다운로드 완료: {destination}")
        return True

    except (urllib.error.URLError, TimeoutError, OSError) as error:
        temporary_file.unlink(missing_ok=True)
        print(f"경고: {label} 다운로드 실패 - {error}")
        return False


def prepare_input_files() -> None:
    """주 데이터는 필수, Taxi Zone Lookup은 선택적으로 준비합니다."""
    data_ready = download_file(DATA_URL, DATA_FILE, "Yellow Taxi Parquet")
    if not data_ready:
        raise RuntimeError(
            "Yellow Taxi 데이터가 없습니다. 인터넷 연결을 확인하거나 "
            f"파일을 직접 '{DATA_FILE}' 경로에 넣어주세요."
        )

    # Zone Lookup이 없어도 기본 EDA와 모델 학습은 가능하므로 실패 시 계속 진행합니다.
    download_file(ZONE_LOOKUP_URL, ZONE_LOOKUP_FILE, "Taxi Zone Lookup")


def standardize_pandas_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """연도·버전에 따라 달라질 수 있는 컬럼명의 대소문자를 통일합니다."""
    rename_map: dict[str, str] = {}

    if "Airport_fee" in dataframe.columns and "airport_fee" not in dataframe.columns:
        rename_map["Airport_fee"] = "airport_fee"

    return dataframe.rename(columns=rename_map)


def standardize_polars_columns(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Polars DataFrame의 컬럼명도 Pandas와 같은 형식으로 통일합니다."""
    rename_map: dict[str, str] = {}

    if "Airport_fee" in dataframe.columns and "airport_fee" not in dataframe.columns:
        rename_map["Airport_fee"] = "airport_fee"

    return dataframe.rename(rename_map) if rename_map else dataframe


def ensure_optional_pandas_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """선택 컬럼이 없을 때 NaN 컬럼을 추가하여 이후 코드의 호환성을 높입니다."""
    result = dataframe.copy()
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result


def ensure_optional_polars_columns(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Polars에서도 선택 컬럼이 없으면 Float64 null 컬럼을 추가합니다."""
    result = dataframe
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in result.columns:
            result = result.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(column)
            )
    return result


def validate_required_columns(columns: Iterable[str]) -> None:
    """분석에 반드시 필요한 컬럼이 존재하는지 확인합니다."""
    missing_columns = sorted(CORE_REQUIRED_COLUMNS - set(columns))
    if missing_columns:
        raise ValueError(
            "분석에 필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
        )


# ==============================================================================
# [문제 2] Pandas와 Polars 전체 로딩 비교
# ==============================================================================


def load_polars_profile() -> dict[str, Any]:
    """Polars로 전체 데이터를 읽고 구조·결측치·기술통계를 저장합니다."""
    start_time = time.perf_counter()
    dataframe = pl.read_parquet(DATA_FILE)
    dataframe = standardize_polars_columns(dataframe)
    dataframe = ensure_optional_polars_columns(dataframe)
    validate_required_columns(dataframe.columns)
    elapsed = time.perf_counter() - start_time

    null_count_row = dataframe.null_count().row(0)
    missing_by_column = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": [int(value) for value in null_count_row],
        }
    )
    missing_by_column["missing_rate_percent"] = (
        missing_by_column["missing_count"] / dataframe.height * 100
    )
    save_table(missing_by_column, "polars_missing_summary.csv")

    numeric_columns = [
        column
        for column in BASE_NUMERIC_COLUMNS
        if column in dataframe.columns
    ]
    if numeric_columns:
        polars_describe = dataframe.select(numeric_columns).describe()
        polars_describe.write_csv(TABLE_DIR / "polars_descriptive_statistics.csv")

    if "payment_type" in dataframe.columns:
        payment_summary = (
            dataframe.group_by("payment_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        payment_summary.write_csv(TABLE_DIR / "polars_payment_type_summary.csv")

    # Pandas의 ``duplicated().sum()``과 같은 기준으로, 중복 때문에 추가된 행 수를
    # 계산합니다. Polars ``is_duplicated``는 버전에 따라 중복 묶음 전체를 표시할 수
    # 있으므로 전체 행 수 - 고유 행 수 방식이 두 라이브러리 비교에 더 일관적입니다.
    duplicate_rows = int(dataframe.height - dataframe.unique().height)

    profile = {
        "library": "Polars",
        "rows": int(dataframe.height),
        "columns": int(dataframe.width),
        "missing_values": int(sum(null_count_row)),
        "duplicate_rows": duplicate_rows,
        "memory_mb": float(dataframe.estimated_size("mb")),
        "load_time_seconds": elapsed,
        "column_names": list(dataframe.columns),
        "schema": {column: str(dtype) for column, dtype in dataframe.schema.items()},
    }

    del dataframe
    gc.collect()
    return profile


def load_pandas_profile_and_sample() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pandas로 전체 데이터를 읽고 비교용 정보를 수집한 뒤 EDA 표본을 반환합니다."""
    start_time = time.perf_counter()
    full_df = pd.read_parquet(DATA_FILE, engine="pyarrow")
    full_df = standardize_pandas_columns(full_df)
    full_df = ensure_optional_pandas_columns(full_df)
    validate_required_columns(full_df.columns)
    elapsed = time.perf_counter() - start_time

    missing_summary = pd.DataFrame(
        {
            "column": full_df.columns,
            "missing_count": full_df.isna().sum().values,
            "missing_rate_percent": full_df.isna().mean().values * 100,
        }
    ).sort_values("missing_count", ascending=False)
    save_table(missing_summary, "pandas_full_missing_summary.csv")

    duplicate_rows = int(full_df.duplicated().sum())
    memory_mb = float(full_df.memory_usage(deep=True).sum() / 1024**2)

    profile = {
        "library": "Pandas",
        "rows": int(full_df.shape[0]),
        "columns": int(full_df.shape[1]),
        "missing_values": int(full_df.isna().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "memory_mb": memory_mb,
        "load_time_seconds": elapsed,
        "column_names": list(full_df.columns),
        "schema": {column: str(dtype) for column, dtype in full_df.dtypes.items()},
    }

    analysis_df = sample_rows(full_df, ANALYSIS_SAMPLE_SIZE)
    profile["analysis_rows"] = int(len(analysis_df))
    profile["analysis_is_sample"] = bool(len(analysis_df) < len(full_df))

    del full_df
    gc.collect()
    return analysis_df, profile


def load_and_compare() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pandas·Polars 전체 로딩 결과와 일관성을 비교합니다."""
    print("Polars 전체 로딩 및 프로파일링을 수행합니다.")
    polars_profile = load_polars_profile()

    print("Pandas 전체 로딩 및 프로파일링을 수행합니다.")
    analysis_df, pandas_profile = load_pandas_profile_and_sample()

    comparison = pd.DataFrame(
        [
            {
                "library": polars_profile["library"],
                "rows": polars_profile["rows"],
                "columns": polars_profile["columns"],
                "missing_values": polars_profile["missing_values"],
                "duplicate_rows": polars_profile["duplicate_rows"],
                "memory_mb": polars_profile["memory_mb"],
                "load_time_seconds": polars_profile["load_time_seconds"],
            },
            {
                "library": pandas_profile["library"],
                "rows": pandas_profile["rows"],
                "columns": pandas_profile["columns"],
                "missing_values": pandas_profile["missing_values"],
                "duplicate_rows": pandas_profile["duplicate_rows"],
                "memory_mb": pandas_profile["memory_mb"],
                "load_time_seconds": pandas_profile["load_time_seconds"],
            },
        ]
    )
    save_table(comparison, "loader_comparison.csv")

    consistency = {
        "same_row_count": polars_profile["rows"] == pandas_profile["rows"],
        "same_column_count": polars_profile["columns"] == pandas_profile["columns"],
        "same_column_order": (
            polars_profile["column_names"] == pandas_profile["column_names"]
        ),
        "same_total_missing": (
            polars_profile["missing_values"] == pandas_profile["missing_values"]
        ),
        "same_duplicate_count": (
            polars_profile["duplicate_rows"] == pandas_profile["duplicate_rows"]
        ),
        "full_rows": pandas_profile["rows"],
        "analysis_rows": pandas_profile["analysis_rows"],
        "analysis_is_sample": pandas_profile["analysis_is_sample"],
        "analysis_sample_size_setting": ANALYSIS_SAMPLE_SIZE,
    }
    save_json(consistency, TABLE_DIR / "loader_consistency_check.json")

    print("\n[Pandas·Polars 로딩 비교]")
    print(comparison.round(4).to_string(index=False))
    print("\n[일관성 확인]")
    print(json.dumps(consistency, ensure_ascii=False, indent=2, default=json_default))

    profiles = {
        "pandas": pandas_profile,
        "polars": polars_profile,
        "consistency": consistency,
    }
    return analysis_df, comparison, profiles


# ==============================================================================
# [문제 3] 데이터 구조와 머신러닝 관점의 컬럼 진단
# ==============================================================================


def infer_column_role(column: str, series: pd.Series) -> tuple[str, str, str, str]:
    """
    컬럼 의미와 자료형을 함께 고려하여 ML 역할·권장 전처리·누수 위험·비고를 반환합니다.

    숫자형이라고 해서 모두 연속형으로 처리하지 않습니다. VendorID나 LocationID처럼
    숫자가 단지 코드인 경우에는 범주형으로 판단합니다.
    """
    if column in DATETIME_COLUMNS:
        return (
            "datetime",
            "datetime 변환 후 시간·요일·운행시간 파생; 원시 시각은 필요에 따라 제외",
            "낮음",
            "임의 시각 대체는 잘못된 운행시간을 만들 수 있으므로 결측 표시 또는 행 제외",
        )

    if column in {"VendorID", "RatecodeID", "PULocationID", "DOLocationID", "payment_type"}:
        return (
            "categorical_code",
            "문자열 범주로 변환 → Unknown 대체 → One-Hot Encoding",
            "payment_type은 카드 전용 모델에서 상수이므로 제외",
            "숫자 크기와 순서에 의미가 없는 코드형 변수",
        )

    if column == "store_and_fwd_flag":
        return (
            "binary_categorical",
            "Unknown 대체 후 One-Hot Encoding 또는 Y/N 이진 변환",
            "낮음",
            "Y/N 외 값과 결측치를 별도로 확인",
        )

    if column in {"tip_amount", "total_amount"}:
        return (
            "target_source",
            "목표값 생성에만 사용하고 모델 입력에서는 제외",
            "매우 높음",
            "high_tip 계산에 직접 사용되므로 입력 시 데이터 누수 발생",
        )

    if column in BASE_NUMERIC_COLUMNS or pd.api.types.is_numeric_dtype(series):
        return (
            "numeric",
            "중앙값 대체(+결측 표시) → StandardScaler; 강한 왜도는 로그/Robust 처리 검토",
            "낮음",
            "0·음수·극단값은 도메인 의미를 확인한 뒤 처리",
        )

    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return (
            "categorical",
            "Unknown 대체 → One-Hot Encoding; 희소 범주는 통합 검토",
            "낮음",
            "고유값 수와 희소 범주 비율 확인 필요",
        )

    return (
        "unknown",
        "자료형·고유값·도메인 의미를 추가 확인",
        "확인 필요",
        "자동 판단이 어려운 컬럼",
    )


def build_column_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    """각 컬럼의 자료형·고유값·결측·치우침·ML 처리 권고를 한 표로 만듭니다."""
    rows: list[dict[str, Any]] = []
    total_rows = len(dataframe)

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        unique_count = int(series.nunique(dropna=False))
        unique_ratio = unique_count / total_rows if total_rows else np.nan
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / total_rows * 100 if total_rows else np.nan

        value_counts = series.value_counts(dropna=False)
        if not value_counts.empty:
            top_value = value_counts.index[0]
            top_count = int(value_counts.iloc[0])
            top_ratio = top_count / total_rows * 100 if total_rows else np.nan
        else:
            top_value = ""
            top_count = 0
            top_ratio = np.nan

        numeric = pd.api.types.is_numeric_dtype(series)
        zero_count = int((series == 0).sum()) if numeric else np.nan
        negative_count = int((series < 0).sum()) if numeric else np.nan
        mean_value = safe_float(series.mean()) if numeric and not non_null.empty else np.nan
        median_value = safe_float(series.median()) if numeric and not non_null.empty else np.nan
        std_value = safe_float(series.std()) if numeric and len(non_null) > 1 else np.nan
        skewness = safe_float(series.skew()) if numeric and len(non_null) > 2 else np.nan
        minimum = safe_float(series.min()) if numeric and not non_null.empty else np.nan
        maximum = safe_float(series.max()) if numeric and not non_null.empty else np.nan

        role, preprocessing, leakage_risk, note = infer_column_role(column, series)

        rows.append(
            {
                "column": column,
                "description": COLUMN_DESCRIPTIONS.get(column, "파생 또는 추가 확인 필요"),
                "dtype": str(series.dtype),
                "ml_role": role,
                "row_count": total_rows,
                "non_null_count": int(series.notna().sum()),
                "missing_count": missing_count,
                "missing_rate_percent": missing_rate,
                "unique_count_including_missing": unique_count,
                "unique_ratio": unique_ratio,
                "top_value": str(top_value),
                "top_count": top_count,
                "top_ratio_percent": top_ratio,
                "is_constant": unique_count <= 1,
                "is_near_constant_95pct": bool(top_ratio >= 95) if not pd.isna(top_ratio) else False,
                "zero_count": zero_count,
                "negative_count": negative_count,
                "mean": mean_value,
                "median": median_value,
                "std": std_value,
                "min": minimum,
                "max": maximum,
                "skewness": skewness,
                "recommended_preprocessing": preprocessing,
                "leakage_risk": leakage_risk,
                "ml_note": note,
            }
        )

    profile = pd.DataFrame(rows)
    save_table(profile, "column_ml_profile.csv")

    print("[컬럼별 머신러닝 관점 진단]")
    display_columns = [
        "column",
        "dtype",
        "ml_role",
        "missing_rate_percent",
        "unique_count_including_missing",
        "unique_ratio",
        "top_value",
        "top_ratio_percent",
        "recommended_preprocessing",
    ]
    print(profile[display_columns].round(4).to_string(index=False))
    return profile


# ==============================================================================
# [문제 4] 파생변수와 데이터 품질 플래그 생성
# ==============================================================================


def load_zone_lookup() -> pd.DataFrame | None:
    """Taxi Zone Lookup이 있으면 읽고, 없거나 형식이 다르면 None을 반환합니다."""
    if not ZONE_LOOKUP_FILE.exists():
        return None

    try:
        lookup = pd.read_csv(ZONE_LOOKUP_FILE)
    except Exception as error:  # 파일 인코딩·형식 문제를 분석 전체 실패로 만들지 않습니다.
        warnings.warn(f"Taxi Zone Lookup을 읽지 못했습니다: {error}")
        return None

    required = {"LocationID", "Borough", "Zone"}
    if not required.issubset(lookup.columns):
        warnings.warn("Taxi Zone Lookup에 LocationID, Borough, Zone 컬럼이 없습니다.")
        return None

    lookup = lookup.drop_duplicates(subset="LocationID").copy()
    return lookup


def attach_zone_information(
    dataframe: pd.DataFrame,
    lookup: pd.DataFrame | None,
) -> tuple[pd.DataFrame, set[int] | None]:
    """승·하차 LocationID를 Borough와 Zone 이름으로 변환합니다."""
    result = dataframe.copy()
    if lookup is None:
        return result, None

    valid_zone_ids = set(
        pd.to_numeric(lookup["LocationID"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    pickup_lookup = lookup[["LocationID", "Borough", "Zone"]].rename(
        columns={
            "LocationID": "PULocationID",
            "Borough": "PU_Borough",
            "Zone": "PU_Zone",
        }
    )
    dropoff_lookup = lookup[["LocationID", "Borough", "Zone"]].rename(
        columns={
            "LocationID": "DOLocationID",
            "Borough": "DO_Borough",
            "Zone": "DO_Zone",
        }
    )

    result = result.merge(pickup_lookup, on="PULocationID", how="left")
    result = result.merge(dropoff_lookup, on="DOLocationID", how="left")
    return result, valid_zone_ids


def create_derived_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """시간·속도·단위요금·팁 비율·경로 등 EDA와 ML용 파생변수를 만듭니다."""
    result = dataframe.copy()

    for column in DATETIME_COLUMNS:
        result[column] = pd.to_datetime(result[column], errors="coerce")

    result["trip_duration_min"] = (
        result["tpep_dropoff_datetime"] - result["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    result["pickup_date"] = result["tpep_pickup_datetime"].dt.date
    result["pickup_hour"] = result["tpep_pickup_datetime"].dt.hour
    result["pickup_dayofweek"] = result["tpep_pickup_datetime"].dt.dayofweek
    result["pickup_day_name"] = result["tpep_pickup_datetime"].dt.day_name()
    result["is_weekend"] = result["pickup_dayofweek"].isin([5, 6]).astype("Int64")
    result["day_type"] = np.where(
        result["pickup_dayofweek"].isin([5, 6]),
        "Weekend",
        "Weekday",
    )
    result["is_rush_hour"] = result["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype("Int64")

    result["average_speed_mph"] = safe_divide(
        result["trip_distance"],
        result["trip_duration_min"] / 60,
    )
    result["fare_per_mile"] = safe_divide(
        result["fare_amount"],
        result["trip_distance"],
    )
    result["fare_per_minute"] = safe_divide(
        result["fare_amount"],
        result["trip_duration_min"],
    )

    result["pre_tip_amount"] = result["total_amount"] - result["tip_amount"]
    result["tip_rate"] = safe_divide(
        result["tip_amount"],
        result["pre_tip_amount"],
    )
    result["tip_rate_pct"] = result["tip_rate"] * 100

    component_columns = existing_columns(
        result,
        [
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "congestion_surcharge",
            "airport_fee",
            "cbd_congestion_fee",
        ],
    )
    result["calculated_total_from_components"] = (
        result[component_columns].fillna(0).sum(axis=1)
    )
    result["total_component_gap"] = (
        result["total_amount"] - result["calculated_total_from_components"]
    )

    result["route_id"] = (
        result["PULocationID"].astype("Int64").astype("string")
        + "_"
        + result["DOLocationID"].astype("Int64").astype("string")
    )

    return result


def add_quality_flags(
    dataframe: pd.DataFrame,
    valid_zone_ids: set[int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    논리 오류와 검토가 필요한 값을 삭제하지 않고 bool 플래그로 추가합니다.

    매우 큰 거리·요금은 자동 삭제하지 않습니다. 물리적으로 불가능하거나 코드 체계와
    맞지 않는 값만 별도 플래그로 표시하고, 모델 코호트에서 필요한 조건만 선택합니다.
    """
    result = dataframe.copy()

    result["flag_missing_pickup_datetime"] = result["tpep_pickup_datetime"].isna()
    result["flag_missing_dropoff_datetime"] = result["tpep_dropoff_datetime"].isna()
    result["flag_pickup_outside_target_month"] = ~(
        result["tpep_pickup_datetime"].ge(ANALYSIS_START)
        & result["tpep_pickup_datetime"].lt(ANALYSIS_END)
    )
    result["flag_invalid_time_order"] = (
        result["tpep_dropoff_datetime"] <= result["tpep_pickup_datetime"]
    )
    result["flag_nonpositive_duration"] = result["trip_duration_min"].le(0)
    result["flag_zero_distance"] = result["trip_distance"].eq(0)
    result["flag_negative_distance"] = result["trip_distance"].lt(0)
    result["flag_negative_fare"] = result["fare_amount"].lt(0)
    result["flag_negative_tip"] = result["tip_amount"].lt(0)
    result["flag_negative_total"] = result["total_amount"].lt(0)
    result["flag_implausible_speed_review"] = result["average_speed_mph"].gt(
        SPEED_REVIEW_THRESHOLD_MPH
    )
    result["flag_total_component_mismatch"] = result["total_component_gap"].abs().gt(
        TOTAL_COMPONENT_TOLERANCE
    )

    result["flag_unknown_vendor_code"] = (
        result["VendorID"].notna()
        & ~result["VendorID"].isin(VALID_VENDOR_CODES)
    )
    result["flag_unknown_rate_code"] = (
        result["RatecodeID"].notna()
        & ~result["RatecodeID"].isin(VALID_RATE_CODES)
    )
    result["flag_unknown_payment_code"] = (
        result["payment_type"].notna()
        & ~result["payment_type"].isin(VALID_PAYMENT_CODES)
    )
    result["flag_unknown_store_flag"] = (
        result["store_and_fwd_flag"].notna()
        & ~result["store_and_fwd_flag"].isin(VALID_STORE_FLAGS)
    )

    if valid_zone_ids is None:
        result["flag_invalid_pickup_zone"] = False
        result["flag_invalid_dropoff_zone"] = False
    else:
        result["flag_invalid_pickup_zone"] = (
            result["PULocationID"].notna()
            & ~result["PULocationID"].isin(valid_zone_ids)
        )
        result["flag_invalid_dropoff_zone"] = (
            result["DOLocationID"].notna()
            & ~result["DOLocationID"].isin(valid_zone_ids)
        )

    flag_columns = [column for column in result.columns if column.startswith("flag_")]
    result["quality_flag_count"] = result[flag_columns].fillna(False).sum(axis=1)

    quality_summary = pd.DataFrame(
        {
            "quality_flag": flag_columns,
            "flagged_count": [int(result[column].fillna(False).sum()) for column in flag_columns],
        }
    )
    quality_summary["flagged_rate_percent"] = (
        quality_summary["flagged_count"] / len(result) * 100
    )
    quality_summary = quality_summary.sort_values("flagged_count", ascending=False)
    save_table(quality_summary, "quality_flag_summary.csv")

    return result, quality_summary


def calculate_iqr_outlier_summary(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IQR 기준 극단값 후보를 계산하고, 후보 여부를 플래그로만 추가합니다."""
    result = dataframe.copy()
    records: list[dict[str, Any]] = []

    for column in columns:
        if column not in result.columns:
            continue

        series = pd.to_numeric(result[column], errors="coerce").dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        if iqr == 0:
            flag = pd.Series(False, index=result.index)
        else:
            flag = (result[column] < lower_bound) | (result[column] > upper_bound)

        flag_column = f"flag_iqr_outlier__{column}"
        result[flag_column] = flag.fillna(False)

        records.append(
            {
                "variable": column,
                "count": int(series.shape[0]),
                "missing_count": int(result[column].isna().sum()),
                "q1": q1,
                "median": float(series.median()),
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "p999": float(series.quantile(0.999)),
                "max": float(series.max()),
                "outlier_count": int(flag.sum()),
                "outlier_rate_percent": float(flag.mean() * 100),
                "treatment": "삭제하지 않고 후보 플래그로 유지",
            }
        )

    summary = pd.DataFrame(records).sort_values(
        "outlier_rate_percent",
        ascending=False,
    )
    save_table(summary, "iqr_outlier_candidate_summary.csv")
    return result, summary


def prepare_analysis_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """중복 처리, Zone 조인, 파생변수, 품질·IQR 플래그를 순서대로 수행합니다."""
    working_df = dataframe.copy()
    rows_before = len(working_df)
    duplicate_count = int(working_df.duplicated().sum())

    if DROP_EXACT_DUPLICATES:
        working_df = working_df.drop_duplicates().copy()

    duplicate_info = {
        "rows_before": rows_before,
        "exact_duplicate_rows": duplicate_count,
        "duplicates_removed": duplicate_count if DROP_EXACT_DUPLICATES else 0,
        "rows_after_duplicate_step": len(working_df),
        "drop_exact_duplicates_setting": DROP_EXACT_DUPLICATES,
    }
    save_json(duplicate_info, TABLE_DIR / "duplicate_summary.json")

    working_df = create_derived_features(working_df)
    zone_lookup = load_zone_lookup()
    working_df, valid_zone_ids = attach_zone_information(working_df, zone_lookup)
    working_df, quality_summary = add_quality_flags(working_df, valid_zone_ids)

    iqr_columns = existing_columns(
        working_df,
        [
            "passenger_count",
            "trip_distance",
            "trip_duration_min",
            "average_speed_mph",
            "fare_amount",
            "tip_amount",
            "tolls_amount",
            "total_amount",
            "pre_tip_amount",
            "tip_rate_pct",
            "fare_per_mile",
            "fare_per_minute",
        ],
    )
    working_df, outlier_summary = calculate_iqr_outlier_summary(
        working_df,
        iqr_columns,
    )

    print("[중복 처리]")
    print(json.dumps(duplicate_info, ensure_ascii=False, indent=2))
    print("\n[데이터 품질 플래그 상위 결과]")
    print(quality_summary.head(20).round(4).to_string(index=False))
    print("\n[IQR 극단값 후보 상위 결과]")
    print(outlier_summary.head(20).round(4).to_string(index=False))

    return working_df, quality_summary, outlier_summary, duplicate_info


# ==============================================================================
# [문제 5] 결측치 구조와 결측의 머신러닝 의미 분석
# ==============================================================================


def analyze_missingness(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """결측치 비율, 동시 결측 패턴, Vendor별 결측 편향을 분석합니다."""
    base_columns = existing_columns(
        dataframe,
        list(CORE_REQUIRED_COLUMNS) + OPTIONAL_NUMERIC_COLUMNS,
    )

    missing_summary = pd.DataFrame(
        {
            "column": base_columns,
            "missing_count": [int(dataframe[column].isna().sum()) for column in base_columns],
        }
    )
    missing_summary["missing_rate_percent"] = (
        missing_summary["missing_count"] / len(dataframe) * 100
    )
    missing_summary = missing_summary.sort_values("missing_count", ascending=False)
    save_table(missing_summary, "missing_summary_analysis_sample.csv")

    # 결측치가 실제로 존재하는 상위 컬럼들의 동시 결측 패턴을 확인합니다.
    pattern_columns = (
        missing_summary.loc[missing_summary["missing_count"] > 0, "column"]
        .head(10)
        .tolist()
    )

    if pattern_columns:
        pattern_booleans = dataframe[pattern_columns].isna()
        missing_patterns = (
            pattern_booleans.value_counts(dropna=False)
            .head(20)
            .rename("count")
            .reset_index()
        )
        missing_patterns["ratio_percent"] = (
            missing_patterns["count"] / len(dataframe) * 100
        )

        def describe_pattern(row: pd.Series) -> str:
            missing_names = [
                column for column in pattern_columns if bool(row[column])
            ]
            return ", ".join(missing_names) if missing_names else "결측 없음"

        missing_patterns["missing_columns"] = missing_patterns.apply(
            describe_pattern,
            axis=1,
        )
        ordered_columns = ["missing_columns", "count", "ratio_percent"] + pattern_columns
        missing_patterns = missing_patterns[ordered_columns]
    else:
        missing_patterns = pd.DataFrame(
            [{"missing_columns": "결측 없음", "count": len(dataframe), "ratio_percent": 100.0}]
        )

    save_table(missing_patterns, "top_missing_patterns.csv")

    # 결측이 특정 Vendor에 집중되어 있다면 무작위 누락이 아닐 가능성이 있습니다.
    vendor_missing_columns = existing_columns(
        dataframe,
        [
            "passenger_count",
            "RatecodeID",
            "store_and_fwd_flag",
            "airport_fee",
            "cbd_congestion_fee",
        ],
    )

    if "VendorID" in dataframe.columns and vendor_missing_columns:
        vendor_group = dataframe.groupby("VendorID", dropna=False)
        vendor_rows: list[dict[str, Any]] = []

        for vendor, group in vendor_group:
            record: dict[str, Any] = {
                "VendorID": vendor,
                "row_count": len(group),
            }
            for column in vendor_missing_columns:
                record[f"{column}_missing_count"] = int(group[column].isna().sum())
                record[f"{column}_missing_rate_percent"] = float(
                    group[column].isna().mean() * 100
                )
            vendor_rows.append(record)

        missing_by_vendor = pd.DataFrame(vendor_rows).sort_values(
            "row_count",
            ascending=False,
        )
    else:
        missing_by_vendor = pd.DataFrame()

    save_table(missing_by_vendor, "missing_by_vendor.csv")

    print("[결측치 요약]")
    print(missing_summary.head(20).round(4).to_string(index=False))
    print("\n[상위 동시 결측 패턴]")
    print(missing_patterns.head(10).round(4).to_string(index=False))

    return missing_summary, missing_patterns, missing_by_vendor


# ==============================================================================
# [문제 6] 팁 분석 코호트와 Target 분포
# ==============================================================================


def prepare_card_tip_cohort(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    카드 결제 중 팁 비율을 계산할 수 있는 행을 선택하고 high_tip 목표값을 생성합니다.

    팁 비율이 100%를 넘는다는 이유만으로 삭제하지 않습니다. 결제 전 금액이 매우 작거나
    실제로 큰 팁을 준 운행일 수 있으므로 원본 코호트에 유지하고 분포·민감도 분석에서 확인합니다.
    """
    base_count = len(dataframe)
    card_mask = dataframe["payment_type"].eq(1)
    target_source_mask = (
        dataframe["tip_amount"].notna()
        & dataframe["total_amount"].notna()
        & dataframe["pre_tip_amount"].gt(0)
        & dataframe["tip_amount"].ge(0)
        & dataframe["tip_rate"].notna()
    )

    card_df = dataframe.loc[card_mask & target_source_mask].copy()
    card_df["high_tip"] = (
        card_df["tip_rate"] >= HIGH_TIP_THRESHOLD
    ).astype(int)

    if card_df.empty:
        raise ValueError(
            "카드 결제 팁 분석에 사용할 행이 없습니다. payment_type, tip_amount, "
            "total_amount 값을 확인해 주세요."
        )

    target_summary = (
        card_df["high_tip"]
        .value_counts(dropna=False)
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
    save_table(target_summary, "high_tip_target_summary.csv")

    tip_rate_summary = pd.DataFrame(
        [
            {
                "count": len(card_df),
                "mean_tip_rate_percent": card_df["tip_rate_pct"].mean(),
                "median_tip_rate_percent": card_df["tip_rate_pct"].median(),
                "std_tip_rate_percent": card_df["tip_rate_pct"].std(),
                "p90_tip_rate_percent": card_df["tip_rate_pct"].quantile(0.90),
                "p95_tip_rate_percent": card_df["tip_rate_pct"].quantile(0.95),
                "p99_tip_rate_percent": card_df["tip_rate_pct"].quantile(0.99),
                "p999_tip_rate_percent": card_df["tip_rate_pct"].quantile(0.999),
                "max_tip_rate_percent": card_df["tip_rate_pct"].max(),
                "skewness": card_df["tip_rate_pct"].skew(),
                "over_100_percent_count": int(card_df["tip_rate_pct"].gt(100).sum()),
                "over_100_percent_rate": float(card_df["tip_rate_pct"].gt(100).mean() * 100),
            }
        ]
    )
    save_table(tip_rate_summary, "tip_rate_distribution_summary.csv")

    cohort_info = {
        "analysis_rows": base_count,
        "credit_card_rows": int(card_mask.sum()),
        "target_calculable_card_rows": len(card_df),
        "excluded_card_rows_due_to_target_source": int(
            card_mask.sum() - len(card_df)
        ),
        "positive_class_count": int(card_df["high_tip"].sum()),
        "positive_class_rate_percent": float(card_df["high_tip"].mean() * 100),
        "high_tip_threshold": HIGH_TIP_THRESHOLD,
        "tip_rate_over_100_percent_count": int(card_df["tip_rate_pct"].gt(100).sum()),
    }
    save_json(cohort_info, TABLE_DIR / "card_tip_cohort_info.json")

    print("[고팁 목표값 분포]")
    print(target_summary.round(4).to_string(index=False))
    print("\n[팁 비율 분포 요약]")
    print(tip_rate_summary.round(4).to_string(index=False))

    return card_df, target_summary, cohort_info


# ==============================================================================
# [문제 7] 수치형 변수의 분포·왜도·단위·극단값 확인
# ==============================================================================


def build_numeric_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """평균·중앙값·표준편차·분위수·왜도·0·음수를 함께 계산합니다."""
    numeric_columns = existing_columns(
        dataframe,
        [
            "passenger_count",
            "trip_distance",
            "trip_duration_min",
            "average_speed_mph",
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "total_amount",
            "congestion_surcharge",
            "airport_fee",
            "cbd_congestion_fee",
            "pre_tip_amount",
            "tip_rate_pct",
            "fare_per_mile",
            "fare_per_minute",
            "total_component_gap",
        ],
    )

    records: list[dict[str, Any]] = []
    for column in numeric_columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        records.append(
            {
                "variable": column,
                "count": int(valid.shape[0]),
                "missing_count": int(series.isna().sum()),
                "missing_rate_percent": float(series.isna().mean() * 100),
                "mean": float(valid.mean()),
                "median": float(valid.median()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "p01": float(valid.quantile(0.01)),
                "q1": float(valid.quantile(0.25)),
                "q3": float(valid.quantile(0.75)),
                "p95": float(valid.quantile(0.95)),
                "p99": float(valid.quantile(0.99)),
                "p999": float(valid.quantile(0.999)),
                "max": float(valid.max()),
                "skewness": float(valid.skew()) if len(valid) > 2 else np.nan,
                "kurtosis": float(valid.kurt()) if len(valid) > 3 else np.nan,
                "zero_count": int(valid.eq(0).sum()),
                "zero_rate_percent": float(valid.eq(0).mean() * 100),
                "negative_count": int(valid.lt(0).sum()),
                "negative_rate_percent": float(valid.lt(0).mean() * 100),
            }
        )

    summary = pd.DataFrame(records).sort_values(
        "skewness",
        ascending=False,
    )
    save_table(summary, "numeric_distribution_summary.csv")

    print("[수치형 변수 분포 요약]")
    print(summary.round(4).to_string(index=False))
    return summary


# ==============================================================================
# [문제 8] 범주형 변수의 고유값·희소 범주·편중 확인
# ==============================================================================


def category_label(column: str, value: Any) -> str:
    """코드형 범주의 설명 라벨을 반환합니다."""
    if pd.isna(value):
        return "Missing"

    try:
        numeric_value = int(float(value))
    except (TypeError, ValueError):
        numeric_value = None

    if column == "VendorID" and numeric_value is not None:
        return VENDOR_LABELS.get(numeric_value, "Other/Unknown Vendor")
    if column == "RatecodeID" and numeric_value is not None:
        return RATE_CODE_LABELS.get(numeric_value, "Other/Unknown Rate")
    if column == "payment_type" and numeric_value is not None:
        return PAYMENT_TYPE_LABELS.get(numeric_value, "Other/Unknown Payment")

    return str(value)


def analyze_categorical_variables(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """범주별 빈도와 컬럼별 cardinality·편중·희소 범주 수를 계산합니다."""
    categorical_columns = existing_columns(
        dataframe,
        [
            "VendorID",
            "RatecodeID",
            "store_and_fwd_flag",
            "PULocationID",
            "DOLocationID",
            "payment_type",
            "pickup_hour",
            "pickup_dayofweek",
            "pickup_day_name",
            "day_type",
            "is_rush_hour",
            "PU_Borough",
            "DO_Borough",
        ],
    )

    distribution_records: list[dict[str, Any]] = []
    cardinality_records: list[dict[str, Any]] = []

    for column in categorical_columns:
        counts = dataframe[column].value_counts(dropna=False)
        total = len(dataframe)

        for rank, (value, count) in enumerate(counts.items(), start=1):
            distribution_records.append(
                {
                    "variable": column,
                    "rank": rank,
                    "value": str(value),
                    "label": category_label(column, value),
                    "count": int(count),
                    "ratio_percent": float(count / total * 100),
                    "is_rare_below_min_support": bool(count < MIN_CATEGORY_SUPPORT),
                }
            )

        top_count = int(counts.iloc[0]) if not counts.empty else 0
        rare_category_count = int((counts < MIN_CATEGORY_SUPPORT).sum())
        rare_row_count = int(counts[counts < MIN_CATEGORY_SUPPORT].sum())

        cardinality_records.append(
            {
                "variable": column,
                "unique_count_including_missing": int(len(counts)),
                "top_value": str(counts.index[0]) if not counts.empty else "",
                "top_count": top_count,
                "top_ratio_percent": float(top_count / total * 100) if total else np.nan,
                "rare_category_count": rare_category_count,
                "rare_row_count": rare_row_count,
                "rare_row_rate_percent": float(rare_row_count / total * 100) if total else np.nan,
                "recommended_encoding": (
                    "One-Hot Encoding"
                    if len(counts) <= 50
                    else "One-Hot 가능하나 희소 범주 통합·빈도 인코딩 검토"
                ),
            }
        )

    distributions = pd.DataFrame(distribution_records)
    cardinality = pd.DataFrame(cardinality_records).sort_values(
        "unique_count_including_missing",
        ascending=False,
    )
    save_table(distributions, "categorical_value_distribution.csv")
    save_table(cardinality, "categorical_cardinality_summary.csv")

    print("[범주형 변수 Cardinality 요약]")
    print(cardinality.round(4).to_string(index=False))
    return distributions, cardinality


# ==============================================================================
# [문제 9] 시간대·요일·경로 패턴 확인
# ==============================================================================


def build_temporal_and_route_summaries(
    dataframe: pd.DataFrame,
    card_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """일자·시간·요일·경로·승하차 지역별 집계표를 생성합니다."""
    valid_time_df = dataframe.loc[
        dataframe["tpep_pickup_datetime"].notna()
    ].copy()

    daily_summary = (
        valid_time_df.groupby("pickup_date", dropna=False)
        .agg(
            trip_count=("pickup_date", "size"),
            median_distance=("trip_distance", "median"),
            median_duration_min=("trip_duration_min", "median"),
            median_fare=("fare_amount", "median"),
            mean_total_amount=("total_amount", "mean"),
        )
        .reset_index()
        .sort_values("pickup_date")
    )

    hourly_summary = (
        valid_time_df.groupby("pickup_hour", dropna=False)
        .agg(
            trip_count=("pickup_hour", "size"),
            median_distance=("trip_distance", "median"),
            median_duration_min=("trip_duration_min", "median"),
            median_fare=("fare_amount", "median"),
        )
        .reset_index()
        .sort_values("pickup_hour")
    )

    dayofweek_summary = (
        valid_time_df.groupby(["pickup_dayofweek", "pickup_day_name"], dropna=False)
        .agg(
            trip_count=("pickup_dayofweek", "size"),
            median_distance=("trip_distance", "median"),
            median_duration_min=("trip_duration_min", "median"),
            median_fare=("fare_amount", "median"),
        )
        .reset_index()
        .sort_values("pickup_dayofweek")
    )

    hour_day_pivot = pd.pivot_table(
        valid_time_df,
        index="pickup_dayofweek",
        columns="pickup_hour",
        values="VendorID",
        aggfunc="count",
        fill_value=0,
    )

    route_summary = (
        dataframe.groupby(
            ["PULocationID", "DOLocationID", "route_id"],
            dropna=False,
        )
        .agg(
            trip_count=("route_id", "size"),
            median_distance=("trip_distance", "median"),
            median_duration_min=("trip_duration_min", "median"),
            median_fare=("fare_amount", "median"),
        )
        .reset_index()
        .sort_values("trip_count", ascending=False)
    )

    pickup_zone_summary = (
        dataframe.groupby(
            ["PULocationID"]
            + (["PU_Borough", "PU_Zone"] if "PU_Zone" in dataframe.columns else []),
            dropna=False,
        )
        .agg(
            trip_count=("PULocationID", "size"),
            median_fare=("fare_amount", "median"),
            median_distance=("trip_distance", "median"),
        )
        .reset_index()
        .sort_values("trip_count", ascending=False)
    )

    dropoff_zone_summary = (
        dataframe.groupby(
            ["DOLocationID"]
            + (["DO_Borough", "DO_Zone"] if "DO_Zone" in dataframe.columns else []),
            dropna=False,
        )
        .agg(
            trip_count=("DOLocationID", "size"),
            median_fare=("fare_amount", "median"),
            median_distance=("trip_distance", "median"),
        )
        .reset_index()
        .sort_values("trip_count", ascending=False)
    )

    card_hour_summary = (
        card_df.groupby("pickup_hour", dropna=False)
        .agg(
            card_trip_count=("high_tip", "size"),
            high_tip_rate=("high_tip", "mean"),
            median_tip_rate_pct=("tip_rate_pct", "median"),
        )
        .reset_index()
        .sort_values("pickup_hour")
    )
    card_hour_summary["high_tip_rate_percent"] = (
        card_hour_summary["high_tip_rate"] * 100
    )

    summaries = {
        "daily": daily_summary,
        "hourly": hourly_summary,
        "dayofweek": dayofweek_summary,
        "hour_day_pivot": hour_day_pivot,
        "routes": route_summary,
        "pickup_zones": pickup_zone_summary,
        "dropoff_zones": dropoff_zone_summary,
        "card_hour": card_hour_summary,
    }

    save_table(daily_summary, "daily_trip_summary.csv")
    save_table(hourly_summary, "hourly_trip_summary.csv")
    save_table(dayofweek_summary, "dayofweek_trip_summary.csv")
    save_table(hour_day_pivot.reset_index(), "hour_day_trip_pivot.csv")
    save_table(route_summary, "route_summary.csv")
    save_table(pickup_zone_summary, "pickup_zone_summary.csv")
    save_table(dropoff_zone_summary, "dropoff_zone_summary.csv")
    save_table(card_hour_summary, "high_tip_rate_by_hour.csv")

    print("[시간대별 운행 요약]")
    print(hourly_summary.round(4).to_string(index=False))
    print("\n[상위 운행 경로]")
    print(route_summary.head(20).round(4).to_string(index=False))

    return summaries


# ==============================================================================
# [문제 10] 설명변수와 Target의 관계 및 데이터 누수 점검
# ==============================================================================


def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """두 독립 집단 평균 차이의 표준화 효과크기 Cohen's d를 계산합니다."""
    group1 = pd.to_numeric(group1, errors="coerce").dropna()
    group2 = pd.to_numeric(group2, errors="coerce").dropna()

    if len(group1) < 2 or len(group2) < 2:
        return float("nan")

    variance1 = group1.var(ddof=1)
    variance2 = group2.var(ddof=1)
    pooled_variance = (
        ((len(group1) - 1) * variance1 + (len(group2) - 1) * variance2)
        / (len(group1) + len(group2) - 2)
    )
    if pooled_variance <= 0 or pd.isna(pooled_variance):
        return float("nan")

    return float((group1.mean() - group2.mean()) / math.sqrt(pooled_variance))


def analyze_numeric_target_relationships(
    card_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수치형 변수와 high_tip의 상관, 그룹 평균·중앙값, 효과크기를 계산합니다."""
    model_safe_numeric = existing_columns(
        card_df,
        [
            "passenger_count",
            "trip_distance",
            "trip_duration_min",
            "average_speed_mph",
            "fare_amount",
            "extra",
            "mta_tax",
            "tolls_amount",
            "improvement_surcharge",
            "congestion_surcharge",
            "airport_fee",
            "cbd_congestion_fee",
            "fare_per_mile",
            "fare_per_minute",
        ],
    )

    records: list[dict[str, Any]] = []
    for column in model_safe_numeric:
        valid = card_df[[column, "high_tip"]].dropna()
        low = valid.loc[valid["high_tip"] == 0, column]
        high = valid.loc[valid["high_tip"] == 1, column]

        correlation = (
            valid[column].corr(valid["high_tip"])
            if valid[column].nunique() > 1
            else np.nan
        )

        records.append(
            {
                "variable": column,
                "valid_count": len(valid),
                "correlation_with_high_tip": correlation,
                "low_tip_mean": low.mean(),
                "high_tip_mean": high.mean(),
                "mean_difference_high_minus_low": high.mean() - low.mean(),
                "low_tip_median": low.median(),
                "high_tip_median": high.median(),
                "cohens_d_high_minus_low": calculate_cohens_d(high, low),
                "ml_interpretation": (
                    "선형 연관성 참고; 비선형·상호작용 가능성은 별도 모델로 확인"
                ),
            }
        )

    relationship_summary = pd.DataFrame(records)
    relationship_summary["absolute_correlation"] = relationship_summary[
        "correlation_with_high_tip"
    ].abs()
    relationship_summary = relationship_summary.sort_values(
        "absolute_correlation",
        ascending=False,
    )
    save_table(relationship_summary, "numeric_target_relationships.csv")

    correlation_columns = model_safe_numeric + ["high_tip"]
    correlation_matrix = card_df[correlation_columns].corr(numeric_only=True)
    save_table(correlation_matrix, "model_safe_correlation_matrix.csv", index=True)

    print("[수치형 변수와 high_tip 관계]")
    print(relationship_summary.round(4).to_string(index=False))
    return relationship_summary, correlation_matrix


def analyze_categorical_target_relationships(
    card_df: pd.DataFrame,
) -> pd.DataFrame:
    """범주별 표본 수·고팁률·평균 팁 비율을 계산합니다."""
    categorical_columns = existing_columns(
        card_df,
        [
            "VendorID",
            "RatecodeID",
            "store_and_fwd_flag",
            "PULocationID",
            "DOLocationID",
            "pickup_hour",
            "pickup_dayofweek",
            "pickup_day_name",
            "day_type",
            "is_rush_hour",
            "PU_Borough",
            "DO_Borough",
        ],
    )

    records: list[pd.DataFrame] = []
    for column in categorical_columns:
        grouped = (
            card_df.groupby(column, dropna=False)
            .agg(
                count=("high_tip", "size"),
                high_tip_rate=("high_tip", "mean"),
                mean_tip_rate_pct=("tip_rate_pct", "mean"),
                median_tip_rate_pct=("tip_rate_pct", "median"),
            )
            .reset_index()
        )
        grouped.insert(0, "variable", column)
        grouped = grouped.rename(columns={column: "value"})
        grouped["value"] = grouped["value"].astype("string")
        grouped["high_tip_rate_percent"] = grouped["high_tip_rate"] * 100
        grouped["has_minimum_support"] = grouped["count"] >= MIN_CATEGORY_SUPPORT
        records.append(grouped)

    if records:
        result = pd.concat(records, ignore_index=True)
        result = result.sort_values(
            ["variable", "count"],
            ascending=[True, False],
        )
    else:
        result = pd.DataFrame()

    save_table(result, "categorical_target_relationships.csv")
    print("[범주형 변수별 high_tip 관계 - 최소 지지 건수 충족 상위]")
    supported = result.loc[result["has_minimum_support"]].copy()
    print(supported.head(50).round(4).to_string(index=False))
    return result


def analyze_missingness_target_signal(
    card_df: pd.DataFrame,
) -> pd.DataFrame:
    """결측 여부 자체가 high_tip과 연관되는지 확인합니다."""
    candidate_columns = existing_columns(
        card_df,
        [
            "passenger_count",
            "RatecodeID",
            "store_and_fwd_flag",
            "airport_fee",
            "cbd_congestion_fee",
        ],
    )

    records: list[dict[str, Any]] = []
    for column in candidate_columns:
        missing_mask = card_df[column].isna()
        if not missing_mask.any():
            continue

        missing_group = card_df.loc[missing_mask, "high_tip"]
        present_group = card_df.loc[~missing_mask, "high_tip"]
        records.append(
            {
                "variable": column,
                "missing_count": int(missing_mask.sum()),
                "missing_rate_percent": float(missing_mask.mean() * 100),
                "high_tip_rate_when_missing_percent": float(
                    missing_group.mean() * 100
                ),
                "high_tip_rate_when_present_percent": float(
                    present_group.mean() * 100
                ),
                "rate_difference_percentage_point": float(
                    (missing_group.mean() - present_group.mean()) * 100
                ),
                "recommended_action": (
                    "중앙값/Unknown 대체와 함께 결측 표시 변수를 유지하여 "
                    "결측 자체의 신호를 보존"
                ),
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result["absolute_rate_difference"] = result[
            "rate_difference_percentage_point"
        ].abs()
        result = result.sort_values("absolute_rate_difference", ascending=False)

    save_table(result, "missingness_target_signal.csv")
    return result


# ==============================================================================
# [문제 11] Seaborn 정적 차트와 Plotly 인터랙티브 차트
# ==============================================================================


def finite_quantile(series: pd.Series, quantile: float, default: float) -> float:
    """유한한 수치만 사용하여 분위수를 계산합니다."""
    valid = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return default
    return float(valid.quantile(quantile))


def create_static_visualizations(
    dataframe: pd.DataFrame,
    card_df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    temporal_summaries: dict[str, pd.DataFrame],
) -> list[Path]:
    """Question-Driven EDA에 맞는 여러 Seaborn·Matplotlib 정적 차트를 저장합니다."""
    sns.set_theme(style="whitegrid")
    saved_files: list[Path] = []

    # --------------------------------------------------------------------------
    # 1. 결측치 비율
    # --------------------------------------------------------------------------
    missing_plot = missing_summary.loc[
        missing_summary["missing_count"] > 0
    ].head(20)
    if not missing_plot.empty:
        plt.figure(figsize=(11, 6))
        sns.barplot(
            data=missing_plot,
            y="column",
            x="missing_rate_percent",
            hue="column",
            legend=False,
        )
        plt.title("Missing Rate by Column")
        plt.xlabel("Missing Rate (%)")
        plt.ylabel("Column")
        plt.tight_layout()
        path = FIGURE_DIR / "01_missing_rate_by_column.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 2. 데이터 품질 플래그
    # --------------------------------------------------------------------------
    quality_plot = quality_summary.loc[
        quality_summary["flagged_count"] > 0
    ].head(20)
    if not quality_plot.empty:
        plt.figure(figsize=(11, 7))
        sns.barplot(
            data=quality_plot,
            y="quality_flag",
            x="flagged_rate_percent",
            hue="quality_flag",
            legend=False,
        )
        plt.title("Data Quality Review Flags")
        plt.xlabel("Flagged Rate (%)")
        plt.ylabel("Quality Flag")
        plt.tight_layout()
        path = FIGURE_DIR / "02_quality_flag_rates.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 3. 팁 비율 Target 분포
    #    시각화에만 99.5% 분위수 상한을 적용하며 원본 데이터는 삭제하지 않습니다.
    # --------------------------------------------------------------------------
    target_plot_df = sample_rows(card_df, PLOT_SAMPLE_SIZE)
    tip_upper = finite_quantile(target_plot_df["tip_rate_pct"], 0.995, 100.0)
    target_plot_df = target_plot_df.loc[
        target_plot_df["tip_rate_pct"].between(0, tip_upper, inclusive="both")
    ].copy()

    if not target_plot_df.empty:
        plt.figure(figsize=(11, 6))
        plt.hist(
            target_plot_df["tip_rate_pct"],
            bins=50,
        )
        plt.axvline(
            target_plot_df["tip_rate_pct"].mean(),
            linestyle="--",
            label="Mean",
        )
        plt.axvline(
            target_plot_df["tip_rate_pct"].median(),
            linestyle=":",
            label="Median",
        )
        plt.axvline(
            HIGH_TIP_THRESHOLD * 100,
            linestyle="-.",
            label="High-tip threshold",
        )
        plt.title("Distribution of Credit-Card Tip Rate (Display up to P99.5)")
        plt.xlabel("Tip Rate (%)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        path = FIGURE_DIR / "03_tip_rate_distribution.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 4. 주요 수치형 변수 분포 Grid
    # --------------------------------------------------------------------------
    distribution_columns = existing_columns(
        dataframe,
        [
            "trip_distance",
            "trip_duration_min",
            "average_speed_mph",
            "fare_amount",
            "total_amount",
            "passenger_count",
        ],
    )
    numeric_plot_df = sample_rows(dataframe, PLOT_SAMPLE_SIZE)

    if distribution_columns:
        fig, axes = plt.subplots(3, 2, figsize=(15, 13))
        axes = axes.flatten()

        for axis, column in zip(axes, distribution_columns):
            lower = finite_quantile(numeric_plot_df[column], 0.001, -np.inf)
            upper = finite_quantile(numeric_plot_df[column], 0.995, np.inf)
            plot_series = numeric_plot_df.loc[
                numeric_plot_df[column].between(lower, upper, inclusive="both"),
                column,
            ]
            axis.hist(plot_series, bins=40)
            axis.set_title(f"Distribution of {column} (P0.1~P99.5)")
            axis.set_xlabel(column)
            axis.set_ylabel("Count")

        for axis in axes[len(distribution_columns):]:
            axis.set_visible(False)

        plt.suptitle("Distribution of Major Numeric Features", y=1.01)
        plt.tight_layout()
        path = FIGURE_DIR / "04_numeric_feature_distributions.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 5. IQR 극단값 후보 Boxplot Grid
    # --------------------------------------------------------------------------
    boxplot_columns = (
        outlier_summary.sort_values("outlier_rate_percent", ascending=False)[
            "variable"
        ]
        .head(6)
        .tolist()
        if not outlier_summary.empty
        else []
    )
    boxplot_columns = existing_columns(numeric_plot_df, boxplot_columns)

    if boxplot_columns:
        fig, axes = plt.subplots(3, 2, figsize=(15, 11))
        axes = axes.flatten()

        for axis, column in zip(axes, boxplot_columns):
            clean_values = pd.to_numeric(
                numeric_plot_df[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()
            axis.boxplot(clean_values, vert=False, showfliers=True)
            axis.set_title(f"Box Plot of {column}")
            axis.set_xlabel(column)

        for axis in axes[len(boxplot_columns):]:
            axis.set_visible(False)

        plt.suptitle("IQR Outlier Candidates: Review, Not Automatic Deletion", y=1.01)
        plt.tight_layout()
        path = FIGURE_DIR / "05_iqr_candidate_boxplots.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 6. 모델 입력 후보 수치형 변수 상관관계
    # --------------------------------------------------------------------------
    if not correlation_matrix.empty:
        plt.figure(figsize=(13, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
        )
        plt.title("Model-Safe Numeric Correlation Heatmap")
        plt.tight_layout()
        path = FIGURE_DIR / "06_model_safe_correlation_heatmap.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 7. 일별 운행량
    # --------------------------------------------------------------------------
    daily = temporal_summaries["daily"]
    if not daily.empty:
        plt.figure(figsize=(13, 5))
        plt.plot(daily["pickup_date"], daily["trip_count"], marker="o")
        plt.title("Daily Yellow Taxi Trip Count")
        plt.xlabel("Pickup Date")
        plt.ylabel("Trip Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = FIGURE_DIR / "07_daily_trip_count.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 8. 시간대별 운행량
    # --------------------------------------------------------------------------
    hourly = temporal_summaries["hourly"]
    if not hourly.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(hourly["pickup_hour"], hourly["trip_count"], marker="o")
        plt.title("Hourly Yellow Taxi Trip Count")
        plt.xlabel("Pickup Hour")
        plt.ylabel("Trip Count")
        plt.xticks(range(0, 24))
        plt.tight_layout()
        path = FIGURE_DIR / "08_hourly_trip_count.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 9. 평일·주말 팁 비율 그룹 비교
    # --------------------------------------------------------------------------
    if not target_plot_df.empty:
        plt.figure(figsize=(9, 6))
        sns.boxplot(
            data=target_plot_df,
            x="day_type",
            y="tip_rate_pct",
            order=["Weekday", "Weekend"],
            hue="day_type",
            legend=False,
            showfliers=False,
        )
        plt.title("Credit-Card Tip Rate: Weekday vs Weekend")
        plt.xlabel("Day Type")
        plt.ylabel("Tip Rate (%)")
        plt.tight_layout()
        path = FIGURE_DIR / "09_tip_rate_weekday_weekend.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 10. 거리와 기본요금의 관계
    # --------------------------------------------------------------------------
    relation_df = sample_rows(card_df, min(PLOT_SAMPLE_SIZE or len(card_df), 50_000))
    distance_upper = finite_quantile(relation_df["trip_distance"], 0.995, np.inf)
    fare_upper = finite_quantile(relation_df["fare_amount"], 0.995, np.inf)
    relation_df = relation_df.loc[
        relation_df["trip_distance"].between(0, distance_upper, inclusive="both")
        & relation_df["fare_amount"].between(0, fare_upper, inclusive="both")
    ].copy()

    if not relation_df.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=relation_df,
            x="trip_distance",
            y="fare_amount",
            hue="high_tip",
            alpha=0.35,
        )
        plt.title("Trip Distance, Fare Amount and High-Tip Class")
        plt.xlabel("Trip Distance (mile)")
        plt.ylabel("Fare Amount ($)")
        plt.tight_layout()
        path = FIGURE_DIR / "10_distance_fare_high_tip.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 11. 시간대별 고팁률
    # --------------------------------------------------------------------------
    card_hour = temporal_summaries["card_hour"]
    if not card_hour.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(
            card_hour["pickup_hour"],
            card_hour["high_tip_rate_percent"],
            marker="o",
        )
        plt.title("High-Tip Rate by Pickup Hour")
        plt.xlabel("Pickup Hour")
        plt.ylabel("High-Tip Rate (%)")
        plt.xticks(range(0, 24))
        plt.tight_layout()
        path = FIGURE_DIR / "11_high_tip_rate_by_hour.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 12. 상위 승차 지역
    # --------------------------------------------------------------------------
    pickup_zones = temporal_summaries["pickup_zones"].head(20).copy()
    if not pickup_zones.empty:
        if "PU_Zone" in pickup_zones.columns:
            pickup_zones["zone_label"] = (
                pickup_zones["PU_Borough"].astype("string")
                + " / "
                + pickup_zones["PU_Zone"].astype("string")
            )
        else:
            pickup_zones["zone_label"] = (
                "LocationID " + pickup_zones["PULocationID"].astype("Int64").astype("string")
            )

        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=pickup_zones,
            y="zone_label",
            x="trip_count",
            hue="zone_label",
            legend=False,
        )
        plt.title("Top 20 Pickup Zones by Trip Count")
        plt.xlabel("Trip Count")
        plt.ylabel("Pickup Zone")
        plt.tight_layout()
        path = FIGURE_DIR / "12_top_pickup_zones.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    return saved_files


def create_interactive_visualizations(
    card_df: pd.DataFrame,
    temporal_summaries: dict[str, pd.DataFrame],
) -> list[Path]:
    """Plotly 인터랙티브 차트를 HTML로 저장합니다."""
    saved_files: list[Path] = []

    # --------------------------------------------------------------------------
    # 1. 요일×시간 운행량 Heatmap
    # --------------------------------------------------------------------------
    pivot = temporal_summaries["hour_day_pivot"].copy()
    if not pivot.empty:
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = pivot.reindex(index=range(7), fill_value=0)
        pivot = pivot.reindex(columns=range(24), fill_value=0)

        figure = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=list(range(24)),
                y=day_labels,
                hovertemplate=(
                    "Day: %{y}<br>Hour: %{x}<br>Trips: %{z:,}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            title="Interactive Trip Count Heatmap: Day of Week × Hour",
            xaxis_title="Pickup Hour",
            yaxis_title="Day of Week",
        )
        path = INTERACTIVE_DIR / "01_hour_day_trip_heatmap.html"
        figure.write_html(path, include_plotlyjs=True, full_html=True)
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 2. 거리·요금·팁 관계 Scatter
    # --------------------------------------------------------------------------
    scatter_df = sample_rows(card_df, min(PLOT_SAMPLE_SIZE or len(card_df), 40_000))
    distance_upper = finite_quantile(scatter_df["trip_distance"], 0.995, np.inf)
    fare_upper = finite_quantile(scatter_df["fare_amount"], 0.995, np.inf)
    scatter_df = scatter_df.loc[
        scatter_df["trip_distance"].between(0, distance_upper, inclusive="both")
        & scatter_df["fare_amount"].between(0, fare_upper, inclusive="both")
    ].copy()

    if not scatter_df.empty:
        scatter_df["high_tip_label"] = scatter_df["high_tip"].map(
            {0: "Below 20%", 1: "20% or more"}
        )
        figure = px.scatter(
            scatter_df,
            x="trip_distance",
            y="fare_amount",
            color="high_tip_label",
            hover_data=[
                "pickup_hour",
                "day_type",
                "trip_duration_min",
                "tip_rate_pct",
                "PULocationID",
                "DOLocationID",
            ],
            opacity=0.45,
            title="Interactive Distance–Fare Relationship by High-Tip Class",
            labels={
                "trip_distance": "Trip Distance (mile)",
                "fare_amount": "Fare Amount ($)",
                "high_tip_label": "High-Tip Class",
            },
        )
        path = INTERACTIVE_DIR / "02_distance_fare_high_tip_scatter.html"
        figure.write_html(path, include_plotlyjs=True, full_html=True)
        saved_files.append(path)

    # --------------------------------------------------------------------------
    # 3. 상위 승차 지역별 운행량·중앙요금
    # --------------------------------------------------------------------------
    zone_df = temporal_summaries["pickup_zones"].head(30).copy()
    if not zone_df.empty:
        if "PU_Zone" in zone_df.columns:
            zone_df["zone_label"] = (
                zone_df["PU_Borough"].astype("string")
                + " / "
                + zone_df["PU_Zone"].astype("string")
            )
        else:
            zone_df["zone_label"] = (
                "LocationID " + zone_df["PULocationID"].astype("Int64").astype("string")
            )

        figure = px.bar(
            zone_df,
            x="zone_label",
            y="trip_count",
            hover_data={
                "median_fare": ":.2f",
                "median_distance": ":.2f",
                "PULocationID": True,
            },
            title="Interactive Top Pickup Zones",
            labels={
                "zone_label": "Pickup Zone",
                "trip_count": "Trip Count",
            },
        )
        figure.update_layout(xaxis_tickangle=-45)
        path = INTERACTIVE_DIR / "03_top_pickup_zones.html"
        figure.write_html(path, include_plotlyjs=True, full_html=True)
        saved_files.append(path)

    return saved_files


# ==============================================================================
# [문제 12] Welch t-test와 p-value·효과크기 해석
# ==============================================================================


def sample_ttest_group(series: pd.Series) -> pd.Series:
    """t-test 그룹별 최대 행 수 설정을 적용합니다."""
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if TTEST_MAX_PER_GROUP is None or len(clean) <= TTEST_MAX_PER_GROUP:
        return clean
    return clean.sample(n=TTEST_MAX_PER_GROUP, random_state=RANDOM_STATE)


def welch_mean_difference_ci(
    group1: pd.Series,
    group2: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Welch 방식으로 두 평균 차이의 신뢰구간을 계산합니다."""
    from scipy.stats import t as student_t

    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")

    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)
    standard_error = math.sqrt(var1 / n1 + var2 / n2)
    if standard_error == 0:
        difference = group1.mean() - group2.mean()
        return float(difference), float(difference)

    numerator = (var1 / n1 + var2 / n2) ** 2
    denominator = (
        (var1 / n1) ** 2 / (n1 - 1)
        + (var2 / n2) ** 2 / (n2 - 1)
    )
    degrees_of_freedom = numerator / denominator if denominator > 0 else n1 + n2 - 2
    alpha = 1 - confidence
    critical = student_t.ppf(1 - alpha / 2, degrees_of_freedom)

    difference = group1.mean() - group2.mean()
    margin = critical * standard_error
    return float(difference - margin), float(difference + margin)


def interpret_effect_size(effect_size: float) -> str:
    """Cohen's d 절댓값을 일반적인 참고 기준으로 해석합니다."""
    if pd.isna(effect_size):
        return "계산 불가"

    absolute = abs(effect_size)
    if absolute < 0.2:
        return "매우 작음"
    if absolute < 0.5:
        return "작음"
    if absolute < 0.8:
        return "중간"
    return "큼"


def perform_t_test(card_df: pd.DataFrame) -> dict[str, Any]:
    """
    평일·주말 카드 팁 비율의 평균 차이를 Welch t-test로 검정합니다.

    원자료 검정과 함께 상위 0.5% 극단값 영향을 줄인 민감도 검정도 수행합니다.
    이는 극단값을 원본에서 삭제하기 위한 것이 아니라 결론의 안정성을 확인하기 위함입니다.
    """
    valid = card_df.loc[
        card_df["day_type"].isin(["Weekday", "Weekend"])
        & card_df["tip_rate_pct"].ge(0)
        & card_df["tip_rate_pct"].notna()
    ].copy()

    weekday = sample_ttest_group(
        valid.loc[valid["day_type"] == "Weekday", "tip_rate_pct"]
    )
    weekend = sample_ttest_group(
        valid.loc[valid["day_type"] == "Weekend", "tip_rate_pct"]
    )

    if len(weekday) < 2 or len(weekend) < 2:
        raise ValueError("t-test를 수행하기 위한 평일·주말 표본이 부족합니다.")

    raw_test = ttest_ind(
        weekday,
        weekend,
        equal_var=False,
        nan_policy="omit",
    )
    raw_effect = calculate_cohens_d(weekday, weekend)
    raw_ci_low, raw_ci_high = welch_mean_difference_ci(weekday, weekend)

    pooled = pd.concat([weekday, weekend], ignore_index=True)
    winsor_upper = float(pooled.quantile(TTEST_WINSOR_QUANTILE))
    weekday_winsor = weekday.clip(upper=winsor_upper)
    weekend_winsor = weekend.clip(upper=winsor_upper)
    robust_test = ttest_ind(
        weekday_winsor,
        weekend_winsor,
        equal_var=False,
        nan_policy="omit",
    )
    robust_effect = calculate_cohens_d(weekday_winsor, weekend_winsor)

    p_value = float(raw_test.pvalue)
    if p_value < ALPHA:
        significance_interpretation = (
            f"p-value가 {ALPHA}보다 작으므로 평일과 주말의 평균 카드 팁 비율이 "
            "같다는 귀무가설을 기각합니다."
        )
    else:
        significance_interpretation = (
            f"p-value가 {ALPHA} 이상이므로 평일과 주말의 평균 카드 팁 비율이 "
            "다르다고 판단할 통계적 근거가 충분하지 않습니다."
        )

    result = {
        "test_name": "Welch independent two-sample t-test",
        "variable": "tip_rate_pct",
        "group1": "Weekday",
        "group2": "Weekend",
        "weekday_count": len(weekday),
        "weekend_count": len(weekend),
        "weekday_mean": float(weekday.mean()),
        "weekend_mean": float(weekend.mean()),
        "weekday_median": float(weekday.median()),
        "weekend_median": float(weekend.median()),
        "mean_difference_weekday_minus_weekend": float(
            weekday.mean() - weekend.mean()
        ),
        "mean_difference_ci95_low": raw_ci_low,
        "mean_difference_ci95_high": raw_ci_high,
        "t_statistic": float(raw_test.statistic),
        "p_value": p_value,
        "alpha": ALPHA,
        "is_statistically_significant": bool(p_value < ALPHA),
        "cohens_d": raw_effect,
        "effect_size_interpretation": interpret_effect_size(raw_effect),
        "significance_interpretation": significance_interpretation,
        "large_sample_caution": (
            "표본이 매우 크면 작은 평균 차이도 유의해질 수 있으므로 p-value뿐 아니라 "
            "평균 차이, 신뢰구간, Cohen's d를 함께 해석해야 합니다."
        ),
        "sensitivity_analysis": {
            "winsor_quantile": TTEST_WINSOR_QUANTILE,
            "winsor_upper_tip_rate_percent": winsor_upper,
            "weekday_winsor_mean": float(weekday_winsor.mean()),
            "weekend_winsor_mean": float(weekend_winsor.mean()),
            "t_statistic": float(robust_test.statistic),
            "p_value": float(robust_test.pvalue),
            "cohens_d": robust_effect,
            "effect_size_interpretation": interpret_effect_size(robust_effect),
        },
    }
    save_json(result, TABLE_DIR / "ttest_weekday_weekend_tip_rate.json")

    print("[Welch t-test 결과]")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return result


# ==============================================================================
# [문제 13] ML Pipeline: 전처리 + 고팁 분류 모델
# ==============================================================================


def select_stratified_model_sample(dataframe: pd.DataFrame) -> pd.DataFrame:
    """목표 클래스 비율을 유지하면서 ML용 최대 표본 수를 적용합니다."""
    if MODEL_SAMPLE_SIZE is None or len(dataframe) <= MODEL_SAMPLE_SIZE:
        return dataframe.copy()

    sampled, _ = train_test_split(
        dataframe,
        train_size=MODEL_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe["high_tip"],
    )
    return sampled.copy()


def prepare_model_dataframe(
    card_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    """
    모델이 실제로 학습할 코호트와 입력 컬럼을 준비합니다.

    고정 상한으로 장거리·고액 운행을 제거하지 않습니다. 다만 운행시간 계산이 불가능하거나
    음수 거리·요금처럼 정상 고팁 예측의 의미가 달라지는 행은 모델 코호트에서 제외합니다.
    원본 EDA 데이터에는 그대로 남아 있습니다.
    """
    cohort = card_df.loc[
        card_df["tpep_pickup_datetime"].notna()
        & card_df["tpep_dropoff_datetime"].notna()
        & card_df["tpep_pickup_datetime"].ge(ANALYSIS_START)
        & card_df["tpep_pickup_datetime"].lt(ANALYSIS_END)
        & card_df["trip_duration_min"].gt(0)
        & card_df["trip_distance"].ge(0)
        & card_df["fare_amount"].ge(0)
        & card_df["total_amount"].ge(0)
    ].copy()

    cohort = select_stratified_model_sample(cohort)

    numeric_features = existing_columns(
        cohort,
        [
            "passenger_count",
            "trip_distance",
            "trip_duration_min",
            "average_speed_mph",
            "fare_amount",
            "extra",
            "mta_tax",
            "tolls_amount",
            "improvement_surcharge",
            "congestion_surcharge",
            "airport_fee",
            "cbd_congestion_fee",
            "fare_per_mile",
            "fare_per_minute",
        ],
    )

    categorical_features = existing_columns(
        cohort,
        [
            "VendorID",
            "RatecodeID",
            "store_and_fwd_flag",
            "PULocationID",
            "DOLocationID",
            "pickup_hour",
            "pickup_dayofweek",
            "day_type",
            "is_rush_hour",
            "PU_Borough",
            "DO_Borough",
        ],
    )

    # 카드 결제만 사용하므로 payment_type은 모든 값이 1인 상수이며 모델에서 제외합니다.
    # tip_amount·total_amount·tip_rate는 목표값 생성에 직접 사용되어 데이터 누수 위험이 큽니다.
    excluded_features = sorted(
        LEAKAGE_COLUMNS
        | {
            "payment_type",
            "pre_tip_amount",
            "calculated_total_from_components",
            "total_component_gap",
            "quality_flag_count",
        }
    )

    for column in numeric_features:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )

    for column in categorical_features:
        # pandas StringDtype의 결측값(pd.NA)은 일부 sklearn 버전에서
        # ``boolean value of NA is ambiguous`` 오류를 만들 수 있습니다.
        # 따라서 문자열 값은 유지하되 결측 표현만 실제 np.nan으로 통일합니다.
        converted = cohort[column].astype("string")
        cohort[column] = converted.astype(object)
        cohort.loc[converted.isna(), column] = np.nan

    model_info = {
        "rows_before_model_filter": len(card_df),
        "rows_after_model_filter_and_sample": len(cohort),
        "model_sample_size_setting": MODEL_SAMPLE_SIZE,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_leakage_or_constant_features": excluded_features,
        "extreme_value_policy": (
            "장거리·고액 값은 유지하며, 음수·비정상 시간 순서처럼 모델 의미가 불명확한 "
            "행만 모델 코호트에서 제외"
        ),
    }
    save_json(model_info, TABLE_DIR / "model_cohort_and_feature_policy.json")

    return cohort, numeric_features, categorical_features, model_info


def split_model_data(
    model_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict[str, Any],
]:
    """시간 분할을 우선 적용하고 불가능할 때 층화 무작위 분할을 사용합니다."""
    if MODEL_SPLIT_MODE == "time":
        sorted_df = model_df.sort_values("tpep_pickup_datetime").copy()
        split_index = int(len(sorted_df) * (1 - TEST_SIZE))
        split_index = min(max(split_index, 1), len(sorted_df) - 1)

        train_df = sorted_df.iloc[:split_index].copy()
        test_df = sorted_df.iloc[split_index:].copy()

        time_split_valid = (
            train_df["high_tip"].nunique() == 2
            and test_df["high_tip"].nunique() == 2
            and len(test_df) >= 10
        )

        if time_split_valid:
            x_train = train_df[feature_columns]
            x_test = test_df[feature_columns]
            y_train = train_df["high_tip"]
            y_test = test_df["high_tip"]
            split_info = {
                "split_method": "time_ordered_holdout",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": train_df["tpep_pickup_datetime"].min(),
                "train_end": train_df["tpep_pickup_datetime"].max(),
                "test_start": test_df["tpep_pickup_datetime"].min(),
                "test_end": test_df["tpep_pickup_datetime"].max(),
                "reason": (
                    "과거 운행으로 학습하고 이후 운행을 평가하여 실제 미래 예측 상황에 가깝게 구성"
                ),
            }
            return x_train, x_test, y_train, y_test, split_info

    x = model_df[feature_columns]
    y = model_df["high_tip"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    split_info = {
        "split_method": "stratified_random_holdout",
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "reason": (
            "시간 분할에서 한 클래스가 사라지는 경우 목표 클래스 비율을 유지하는 층화 분할로 전환"
        ),
    }
    return x_train, x_test, y_train, y_test, split_info


def build_ml_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """결측치 처리·스케일링·원-핫 인코딩·로지스틱 회귀를 하나로 묶습니다."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_model_evaluation_charts(
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> list[Path]:
    """혼동행렬·ROC·Precision-Recall 차트를 저장합니다."""
    saved: list[Path] = []

    # 혼동행렬
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Below 20%", "20% or more"],
        cmap="Blues",
        ax=axis,
    )
    axis.set_title("Confusion Matrix: High-Tip Classifier")
    plt.tight_layout()
    path = FIGURE_DIR / "13_model_confusion_matrix.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved.append(path)

    # ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    path = FIGURE_DIR / "14_model_roc_curve.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    saved.append(path)

    # Precision-Recall Curve
    precision_values, recall_values, _ = precision_recall_curve(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(recall_values, precision_values, label=f"AP = {average_precision:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    path = FIGURE_DIR / "15_model_precision_recall_curve.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    saved.append(path)

    return saved


def extract_logistic_coefficients(model_pipeline: Pipeline) -> pd.DataFrame:
    """전처리 후 생성된 전체 변수명과 로지스틱 회귀 계수를 추출합니다."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "direction": np.where(
                coefficients > 0,
                "고팁 확률 증가 방향",
                "고팁 확률 감소 방향",
            ),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    save_table(coefficient_table, "model_logistic_coefficients.csv")
    return coefficient_table


def train_ml_pipeline(
    card_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, list[Path]]:
    """ML 코호트를 만들고 Pipeline을 학습·평가·저장합니다."""
    model_df, numeric_features, categorical_features, model_info = (
        prepare_model_dataframe(card_df)
    )

    if model_df["high_tip"].nunique() < 2:
        raise ValueError("모델 학습 데이터에 high_tip 클래스가 하나만 존재합니다.")

    feature_columns = numeric_features + categorical_features
    x_train, x_test, y_train, y_test, split_info = split_model_data(
        model_df,
        feature_columns,
    )

    pipeline = build_ml_pipeline(numeric_features, categorical_features)
    start_time = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_time = time.perf_counter() - start_time

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    majority_class = int(y_train.value_counts().idxmax())
    baseline_predictions = np.full(len(y_test), majority_class)

    metrics = {
        "baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Below 20%", "20% or more"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_text": classification_report(
            y_test,
            predictions,
            target_names=["Below 20%", "20% or more"],
            digits=4,
            zero_division=0,
        ),
        "train_positive_rate_percent": float(y_train.mean() * 100),
        "test_positive_rate_percent": float(y_test.mean() * 100),
        "fit_time_seconds": fit_time,
        "split_info": split_info,
        "model_info": model_info,
        "feature_columns": feature_columns,
        "model_name": "LogisticRegression(class_weight='balanced')",
        "model_file": str(MODEL_DIR / "yellow_taxi_high_tip_pipeline.joblib"),
        "target_definition": (
            f"tip_amount / (total_amount - tip_amount) >= {HIGH_TIP_THRESHOLD:.2f}"
        ),
    }

    model_path = MODEL_DIR / "yellow_taxi_high_tip_pipeline.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    save_json(metrics, TABLE_DIR / "model_metrics.json")

    coefficient_table = extract_logistic_coefficients(pipeline)
    model_chart_files = save_model_evaluation_charts(
        y_test,
        predictions,
        probabilities,
    )

    print("[ML Pipeline 평가 지표]")
    for key in [
        "baseline_accuracy",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ]:
        print(f"{key}: {metrics[key]:.4f}")
    print("\n[Classification Report]")
    print(metrics["classification_report_text"])
    print(f"모델 저장: {model_path}")

    return metrics, coefficient_table, model_chart_files


# ==============================================================================
# [문제 14] Question-Driven EDA Findings와 report.md 자동 생성
# ==============================================================================


def markdown_image(path: Path, alt_text: str) -> str:
    """파일이 존재할 때만 Markdown 이미지 문법을 반환합니다."""
    if not path.exists():
        return ""
    relative = path.relative_to(BASE_DIR).as_posix()
    return f"![{alt_text}]({relative})"


def markdown_link(path: Path, label: str) -> str:
    """파일이 존재할 때만 상대경로 Markdown 링크를 반환합니다."""
    if not path.exists():
        return ""
    relative = path.relative_to(BASE_DIR).as_posix()
    return f"[{label}]({relative})"


def join_top_items(
    dataframe: pd.DataFrame,
    label_column: str,
    value_column: str,
    count: int = 3,
    value_format: str = ".4f",
) -> str:
    """상위 결과를 보고서 문장에 넣기 위한 짧은 문자열로 변환합니다."""
    if dataframe.empty:
        return "확인 가능한 결과 없음"

    items: list[str] = []
    for _, row in dataframe.head(count).iterrows():
        value = row[value_column]
        if isinstance(value, (float, np.floating)):
            formatted = format(float(value), value_format)
        else:
            formatted = str(value)
        items.append(f"{row[label_column]}({formatted})")
    return ", ".join(items)


def select_supported_category_extremes(
    categorical_target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """최소 지지 건수를 충족하는 범주 중 고팁률 상·하위 범주를 선택합니다."""
    if categorical_target.empty:
        return pd.DataFrame(), pd.DataFrame()

    supported = categorical_target.loc[
        categorical_target["has_minimum_support"]
    ].copy()
    supported = supported.loc[
        ~supported["variable"].isin(["PULocationID", "DOLocationID"])
    ].copy()

    highest = supported.sort_values(
        "high_tip_rate_percent",
        ascending=False,
    ).head(10)
    lowest = supported.sort_values(
        "high_tip_rate_percent",
        ascending=True,
    ).head(10)
    return highest, lowest


def generate_report(
    loader_comparison: pd.DataFrame,
    profiles: dict[str, Any],
    column_profile: pd.DataFrame,
    duplicate_info: dict[str, Any],
    missing_summary: pd.DataFrame,
    missing_patterns: pd.DataFrame,
    missing_by_vendor: pd.DataFrame,
    quality_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    categorical_cardinality: pd.DataFrame,
    target_summary: pd.DataFrame,
    cohort_info: dict[str, Any],
    temporal_summaries: dict[str, pd.DataFrame],
    numeric_target: pd.DataFrame,
    categorical_target: pd.DataFrame,
    missing_target_signal: pd.DataFrame,
    ttest_result: dict[str, Any],
    model_metrics: dict[str, Any],
    coefficient_table: pd.DataFrame,
    static_files: list[Path],
    interactive_files: list[Path],
) -> None:
    """관찰 근거·가능한 원인·ML 처리 방향을 포함한 상세 report.md를 생성합니다."""
    full_rows = int(profiles["pandas"]["rows"])
    analysis_rows = int(profiles["pandas"]["analysis_rows"])
    analysis_is_sample = bool(profiles["pandas"]["analysis_is_sample"])

    role_counts = (
        column_profile["ml_role"].value_counts().rename_axis("ml_role").reset_index(name="count")
    )
    missing_with_action = missing_summary.merge(
        column_profile[
            [
                "column",
                "ml_role",
                "recommended_preprocessing",
                "ml_note",
            ]
        ],
        on="column",
        how="left",
    )
    missing_with_action = missing_with_action.loc[
        missing_with_action["missing_count"] > 0
    ]

    top_skew = numeric_summary.sort_values("skewness", ascending=False).head(8)
    top_outliers = outlier_summary.sort_values(
        "outlier_rate_percent",
        ascending=False,
    ).head(8)
    top_quality = quality_summary.loc[
        quality_summary["flagged_count"] > 0
    ].head(10)

    peak_hour_row = (
        temporal_summaries["hourly"].sort_values("trip_count", ascending=False).head(1)
    )
    peak_day_row = (
        temporal_summaries["dayofweek"].sort_values("trip_count", ascending=False).head(1)
    )
    top_route_row = temporal_summaries["routes"].head(1)

    top_numeric_rel = numeric_target.sort_values(
        "absolute_correlation",
        ascending=False,
    ).head(8)
    highest_categories, lowest_categories = select_supported_category_extremes(
        categorical_target
    )

    positive_rate = cohort_info["positive_class_rate_percent"]
    high_tip_count = cohort_info["positive_class_count"]
    target_total = cohort_info["target_calculable_card_rows"]

    top_positive_coefficients = coefficient_table.loc[
        coefficient_table["coefficient"] > 0
    ].sort_values("coefficient", ascending=False).head(15)
    top_negative_coefficients = coefficient_table.loc[
        coefficient_table["coefficient"] < 0
    ].sort_values("coefficient", ascending=True).head(15)

    sample_statement = (
        f"전체 {full_rows:,}행 중 재현 가능한 무작위 표본 {analysis_rows:,}행을 후속 EDA에 사용했습니다."
        if analysis_is_sample
        else f"전체 {full_rows:,}행을 후속 EDA에 사용했습니다."
    )

    report = f"""# NYC Yellow Taxi 2026년 5월 Question-Driven EDA 및 ML 보고서

## 1. 분석 목적과 범위

본 분석은 NYC Yellow Taxi 운행 데이터를 단순 요약하는 데서 끝내지 않고, 각 EDA 단계에서
**무엇을 확인했는지, 왜 그런 현상이 나타날 수 있는지, 머신러닝 전처리에 어떻게 반영할지**를
연결하는 것을 목적으로 합니다.

- 전체 데이터 규모: **{full_rows:,}행 × {profiles['pandas']['columns']}열**
- 후속 EDA 규모: **{analysis_rows:,}행**
- 표본 사용 여부: **{'예' if analysis_is_sample else '아니오'}**
- 고팁 분류 목표: 결제 전 금액 대비 카드 팁 비율 **{HIGH_TIP_THRESHOLD * 100:.0f}% 이상**
- 분석 원칙: 큰 거리·긴 시간·큰 요금은 자동 삭제하지 않고 검토 플래그로 유지

{sample_statement}

> 표본 설정은 실행 가능성을 위한 것이며, Pandas·Polars의 행 수·열 수·결측치·중복 비교는
> 전체 Parquet 파일을 기준으로 수행합니다.

---

## 2. 실습 평가기준 충족 현황

| 평가 항목 | 구현 내용 |
| --- | --- |
| Pandas·Polars | 동일 Parquet 전체 로딩, 행·열·결측·중복·메모리·시간 비교 |
| 결측·중복·EDA | 컬럼 ML 프로파일, 결측 패턴, Vendor별 결측, 완전 중복 처리, 품질 플래그 |
| Seaborn | 분포·Boxplot·그룹 비교·상관관계·시간대·지역 차트 다수 생성 |
| Plotly | 요일×시간 Heatmap, 거리–요금 Scatter, 상위 승차 지역 인터랙티브 차트 |
| 기술통계 | 평균·중앙값·표준편차·분위수·왜도·첨도·0·음수 비율 산출 |
| 상관계수 | 모델 입력 가능 수치형 변수와 high_tip 상관관계 계산 |
| t-test | 평일·주말 팁 비율 Welch t-test, p-value·신뢰구간·Cohen's d 해석 |
| ML Pipeline | Imputer + StandardScaler + OneHotEncoder + LogisticRegression |
| 모델 평가·저장 | Accuracy·F1·ROC-AUC·PR-AUC·혼동행렬, joblib 저장 |
| 자동화 | 본 report.md와 presentation_5min.md 자동 생성 |

---

## 3. Pandas와 Polars 로딩 비교

{dataframe_to_markdown(loader_comparison, include_index=False, max_rows=None)}

### Finding

- 두 라이브러리의 행 수 일치: **{profiles['consistency']['same_row_count']}**
- 두 라이브러리의 열 수 일치: **{profiles['consistency']['same_column_count']}**
- 컬럼 순서 일치: **{profiles['consistency']['same_column_order']}**
- 전체 결측치 수 일치: **{profiles['consistency']['same_total_missing']}**

### Evidence

Pandas와 Polars 결과가 일치해야 이후 분석 차이가 라이브러리 로딩 오류가 아니라 분석 로직에서
발생한 것이라고 판단할 수 있습니다. 로딩 시간은 실행 환경, 디스크 캐시, 메모리 상태에 따라 달라지므로
한 번의 결과만으로 항상 어느 라이브러리가 빠르다고 일반화하지 않습니다.

### Possible Action

대용량 집계는 Polars Lazy 또는 DuckDB로 확장하고, 세밀한 시각화·sklearn 학습은 Pandas 표본을
사용하는 혼합 구조를 고려할 수 있습니다.

---

# Question-Driven EDA

## Question 1. 데이터는 어떤 구조이며 각 컬럼은 ML에서 어떤 역할인가?

### Finding

원본 컬럼을 단순히 dtype만으로 나누지 않고, 날짜·연속형·코드형 범주·Binary·목표값 생성 컬럼으로
구분했습니다. `VendorID`, `RatecodeID`, `PULocationID`, `DOLocationID`, `payment_type`은 숫자로 저장되어도
크기 관계가 없는 범주 코드입니다. `tip_amount`와 `total_amount`는 목표값 계산에 직접 사용되므로 모델 입력 시
데이터 누수 위험이 큽니다.

### Evidence

{dataframe_to_markdown(role_counts, include_index=False, max_rows=None)}

{dataframe_to_markdown(column_profile[[
    'column', 'dtype', 'ml_role', 'missing_rate_percent',
    'unique_count_including_missing', 'unique_ratio', 'top_value',
    'top_ratio_percent', 'recommended_preprocessing', 'leakage_risk'
]], include_index=False, max_rows=30)}

### Possible Action

- 날짜는 시간·요일·운행시간으로 파생합니다.
- 코드형 숫자는 문자열 범주로 바꾼 뒤 One-Hot Encoding합니다.
- 연속형은 중앙값 대체와 결측 표시, 스케일링을 적용합니다.
- 목표값 생성 컬럼은 학습 입력에서 제외합니다.
- 고유값 비율이 거의 1인 컬럼은 식별자 가능성을 검토하고 그대로 학습하지 않습니다.

---

## Question 2. 평균·중앙값·분위수는 어떤 분포 특성을 보여주는가?

### Finding

왜도가 큰 변수 상위 항목은 **{join_top_items(top_skew, 'variable', 'skewness', 5)}**입니다.
평균과 중앙값 차이가 크고 P99·최댓값 차이가 큰 변수는 오른쪽 꼬리가 길 가능성이 높습니다.
택시 거리·요금 자료에서는 실제 장거리 운행과 기록 오류가 함께 존재할 수 있으므로 최대값만으로 삭제하지 않습니다.

### Evidence

{dataframe_to_markdown(numeric_summary, include_index=False, max_rows=25)}

{markdown_image(FIGURE_DIR / '04_numeric_feature_distributions.png', '주요 수치형 변수 분포')}

### Possible Action

- 선형 모델에서는 강한 왜도 변수에 `log1p`, RobustScaler, 구간화 등을 비교합니다.
- 트리 모델은 스케일에 덜 민감하지만 극단값 오류 여부는 여전히 검증해야 합니다.
- 평균만 보고 대표값을 판단하지 않고 중앙값·P95·P99를 함께 봅니다.

---

## Question 3. 결측치는 어디에 있으며 단순 누락인가?

### Finding

결측 비율이 높은 컬럼은 **{join_top_items(missing_with_action, 'column', 'missing_rate_percent', 5)}**입니다.
결측이 Vendor별로 다르게 나타난다면 완전 무작위 결측이 아니라 수집 시스템 또는 사업자 차이와 연결될 수 있습니다.

### Evidence

{dataframe_to_markdown(missing_with_action[[
    'column', 'missing_count', 'missing_rate_percent', 'ml_role',
    'recommended_preprocessing', 'ml_note'
]], include_index=False, max_rows=20)}

#### 상위 동시 결측 패턴

{dataframe_to_markdown(missing_patterns, include_index=False, max_rows=15)}

#### Vendor별 결측률

{dataframe_to_markdown(missing_by_vendor, include_index=False, max_rows=15)}

{markdown_image(FIGURE_DIR / '01_missing_rate_by_column.png', '컬럼별 결측률')}

### Possible Action

- 수치형: Pipeline에서 중앙값 대체와 `add_indicator=True`로 결측 여부를 보존합니다.
- 범주형: 최빈값으로 특정 사업자·지역을 임의 생성하지 않고 `Unknown` 범주를 사용합니다.
- 목표값 생성에 필요한 `tip_amount`, `total_amount` 결측은 가짜 정답을 만들 수 있으므로 대체하지 않습니다.
- 결측 여부별 고팁률이 다르면 결측 자체가 예측 신호일 수 있으므로 표시 변수를 유지합니다.

#### 결측 여부와 Target 관계

{dataframe_to_markdown(missing_target_signal, include_index=False, max_rows=15)}

---

## Question 4. Target인 팁 비율과 high_tip은 어떤 분포인가?

### Finding

카드 결제 중 목표값을 계산할 수 있는 행은 **{target_total:,}개**이며, 고팁 행은
**{high_tip_count:,}개({positive_rate:.2f}%)**입니다. 클래스 비율이 한쪽으로 치우치면 Accuracy만으로
모델을 평가할 수 없으므로 F1, Recall, Balanced Accuracy, ROC-AUC, Average Precision을 함께 봅니다.

### Evidence

{dataframe_to_markdown(target_summary, include_index=False, max_rows=None)}

{markdown_image(FIGURE_DIR / '03_tip_rate_distribution.png', '카드 팁 비율 분포')}

### Possible Action

- `class_weight='balanced'`를 사용해 소수 클래스 오류에 더 큰 가중치를 줍니다.
- 팁 비율 100% 초과 값도 자동 삭제하지 않고 별도 건수와 원인을 확인합니다.
- 목표 임계값 20%는 프로젝트 정의이므로 서비스 목적에 따라 15%·25% 등으로 민감도 분석할 수 있습니다.

---

## Question 5. 수치형 변수 중 극단값 후보가 많은 변수는 무엇인가?

### Finding

IQR 후보 비율이 높은 항목은 **{join_top_items(top_outliers, 'variable', 'outlier_rate_percent', 5)}**입니다.
IQR은 분포에서 드문 값을 찾는 통계 규칙이지 오류 판정 규칙이 아닙니다. 장거리·장시간·고액 운행은
실제일 수 있으므로 삭제하지 않고 `flag_iqr_outlier__컬럼명`으로 유지했습니다.

### Evidence

{dataframe_to_markdown(top_outliers, include_index=False, max_rows=12)}

{markdown_image(FIGURE_DIR / '05_iqr_candidate_boxplots.png', 'IQR 극단값 후보 Boxplot')}

### Possible Action

- 논리 오류: 음수 거리, 하차≤승차 등은 정상 운행 모델 코호트에서 제외합니다.
- 희귀하지만 가능한 값: 원본 유지, 로그 변환·RobustScaler·트리 모델을 검토합니다.
- 변수 조합 오류: 거리와 시간을 이용한 속도, 요금 구성 합과 total_amount 차이로 추가 검증합니다.
- 이상탐지 목적에서는 극단값을 제거하지 말고 핵심 탐지 대상으로 사용합니다.

---

## Question 6. 데이터 품질 플래그는 무엇을 보여주는가?

### Finding

플래그 비율이 높은 항목은 **{join_top_items(top_quality, 'quality_flag', 'flagged_rate_percent', 5)}**입니다.
이 플래그는 삭제 목록이 아니라 검토 목록입니다. 하나의 행에 여러 플래그가 동시에 나타날 수 있습니다.

### Evidence

{dataframe_to_markdown(top_quality, include_index=False, max_rows=15)}

{markdown_image(FIGURE_DIR / '02_quality_flag_rates.png', '데이터 품질 플래그 비율')}

### Possible Action

정상 운행 예측, 이상탐지, 정산 검증 등 분석 목적별로 코호트를 별도로 구성해야 합니다.
예를 들어 음수 요금은 정상 팁 예측에서는 제외할 수 있지만 환불·분쟁 탐지에서는 중요한 데이터입니다.

---

## Question 7. 범주형 변수는 얼마나 다양하고 특정 값에 편중되는가?

### Finding

고유 범주 수가 많은 변수는 **{join_top_items(categorical_cardinality, 'variable', 'unique_count_including_missing', 5, '.0f')}**입니다.
LocationID는 CustomerID와 달리 동일 지역 코드가 반복되므로 패턴 학습이 가능합니다. 다만 희소 지역을 모두
독립 One-Hot 컬럼으로 만들면 차원이 커지고 작은 표본의 고팁률이 불안정해질 수 있습니다.

### Evidence

{dataframe_to_markdown(categorical_cardinality, include_index=False, max_rows=20)}

{markdown_image(FIGURE_DIR / '12_top_pickup_zones.png', '상위 승차 지역')}

### Possible Action

- 낮은 cardinality: One-Hot Encoding
- 높은 cardinality: 최소 빈도 기준 통합, 빈도 인코딩, 교차검증 기반 Target Encoding 검토
- 범주별 Target 비율은 반드시 표본 수와 함께 해석
- Taxi Zone Lookup을 조인해 Borough·Zone 단위로 의미를 강화

---

## Question 8. 어떤 수치형 변수가 high_tip과 관계가 강한가?

### Finding

절대 상관계수가 큰 변수는 **{join_top_items(top_numeric_rel, 'variable', 'correlation_with_high_tip', 5)}**입니다.
상관계수가 낮아도 비선형 관계나 시간·지역과의 상호작용이 존재할 수 있습니다. 상관관계는 인과관계를 의미하지 않습니다.

### Evidence

{dataframe_to_markdown(top_numeric_rel[[
    'variable', 'valid_count', 'correlation_with_high_tip',
    'low_tip_mean', 'high_tip_mean', 'mean_difference_high_minus_low',
    'cohens_d_high_minus_low'
]], include_index=False, max_rows=15)}

{markdown_image(FIGURE_DIR / '06_model_safe_correlation_heatmap.png', '모델 안전 변수 상관관계')}

{markdown_image(FIGURE_DIR / '10_distance_fare_high_tip.png', '거리·요금·고팁 교차 탐색')}

### Possible Action

- 선형 모델은 기본 해석 기준으로 사용합니다.
- 비선형성 확인을 위해 RandomForest·Gradient Boosting 모델을 비교할 수 있습니다.
- 모델 입력 시 `tip_amount`, `total_amount`, `tip_rate`는 Target 누수로 제외합니다.

---

## Question 9. 범주형 변수에 따라 고팁률 차이가 나타나는가?

### Finding

최소 {MIN_CATEGORY_SUPPORT:,}건 이상인 범주만 비교했습니다. 표본이 매우 작은 범주의 0% 또는 100% 고팁률은
우연에 의한 불안정한 값일 수 있기 때문입니다.

#### 고팁률 상위 범주

{dataframe_to_markdown(highest_categories[[
    'variable', 'value', 'count', 'high_tip_rate_percent',
    'median_tip_rate_pct'
]], include_index=False, max_rows=10)}

#### 고팁률 하위 범주

{dataframe_to_markdown(lowest_categories[[
    'variable', 'value', 'count', 'high_tip_rate_percent',
    'median_tip_rate_pct'
]], include_index=False, max_rows=10)}

{markdown_image(FIGURE_DIR / '11_high_tip_rate_by_hour.png', '시간대별 고팁률')}

### Possible Action

범주별 차이는 요금 체계, 지역, 시간대가 동시에 작용한 결과일 수 있으므로 단일 범주만으로 원인을 단정하지 않습니다.
모델에서는 여러 변수를 함께 사용하고, 계수 또는 SHAP 등으로 조건부 관계를 확인합니다.

---

## Question 10. 시간대·요일·경로를 함께 보면 어떤 패턴이 나타나는가?

### Finding

- 운행량이 가장 많은 시간: **{first_existing_value(peak_hour_row, 'pickup_hour', '확인 불가')}시**
- 운행량이 가장 많은 요일: **{first_existing_value(peak_day_row, 'pickup_day_name', '확인 불가')}**
- 가장 많이 관찰된 경로: **{first_existing_value(top_route_row, 'route_id', '확인 불가')}**

단일 변수만 볼 때보다 시간대×요일, 거리×요금×고팁 클래스, 승차×하차 지역을 함께 보면 운영 패턴과
Target 차이를 더 구체적으로 확인할 수 있습니다.

### Evidence

{markdown_image(FIGURE_DIR / '07_daily_trip_count.png', '일별 운행량')}

{markdown_image(FIGURE_DIR / '08_hourly_trip_count.png', '시간대별 운행량')}

- {markdown_link(INTERACTIVE_DIR / '01_hour_day_trip_heatmap.html', '요일×시간 운행량 인터랙티브 Heatmap')}
- {markdown_link(INTERACTIVE_DIR / '02_distance_fare_high_tip_scatter.html', '거리–요금–고팁 인터랙티브 Scatter')}
- {markdown_link(INTERACTIVE_DIR / '03_top_pickup_zones.html', '상위 승차 지역 인터랙티브 차트')}

### Possible Action

- 향후 여러 달 데이터를 연결할 때 월·계절·공휴일 변수를 추가합니다.
- 동일 월 내부 무작위 분할보다 시간 순서 분할을 사용해 미래 일반화 성능을 확인합니다.
- 경로 단위 표본이 충분할 때 출발·도착 상호작용 특성을 추가합니다.

---

# 통계 분석

## 4. 평일과 주말 카드 팁 비율 Welch t-test

### 가설

- 귀무가설: 평일과 주말의 평균 카드 팁 비율은 같다.
- 대립가설: 평일과 주말의 평균 카드 팁 비율은 다르다.

### 결과

| 항목 | 값 |
| --- | ---: |
| 평일 표본 수 | {ttest_result['weekday_count']:,} |
| 주말 표본 수 | {ttest_result['weekend_count']:,} |
| 평일 평균 팁 비율 | {ttest_result['weekday_mean']:.4f}% |
| 주말 평균 팁 비율 | {ttest_result['weekend_mean']:.4f}% |
| 평균 차이(평일-주말) | {ttest_result['mean_difference_weekday_minus_weekend']:.4f}%p |
| 평균 차이 95% CI | [{ttest_result['mean_difference_ci95_low']:.4f}, {ttest_result['mean_difference_ci95_high']:.4f}] |
| t-statistic | {ttest_result['t_statistic']:.4f} |
| p-value | {ttest_result['p_value']:.6e} |
| Cohen's d | {ttest_result['cohens_d']:.4f} |
| 효과크기 해석 | {ttest_result['effect_size_interpretation']} |

**p-value 해석:** {ttest_result['significance_interpretation']}

**주의:** {ttest_result['large_sample_caution']}

상위 {100 * (1 - TTEST_WINSOR_QUANTILE):.1f}% 극단 팁 비율의 영향을 줄인 민감도 분석 p-value는
**{ttest_result['sensitivity_analysis']['p_value']:.6e}**, Cohen's d는
**{ttest_result['sensitivity_analysis']['cohens_d']:.4f}**입니다. 원자료와 민감도 분석의 방향이 같은지 확인해
결론이 일부 극단값에만 좌우되는지 판단합니다.

{markdown_image(FIGURE_DIR / '09_tip_rate_weekday_weekend.png', '평일·주말 팁 비율 비교')}

---

# ML Pipeline

## 5. 전처리와 모델 구성

### 수치형 Pipeline

```text
SimpleImputer(strategy='median', add_indicator=True)
→ StandardScaler
```

중앙값은 평균보다 극단값 영향을 덜 받으며, `add_indicator=True`는 원래 결측이었던 행을 별도 신호로 보존합니다.

### 범주형 Pipeline

```text
SimpleImputer(strategy='constant', fill_value='Unknown')
→ OneHotEncoder(handle_unknown='ignore', min_frequency=20)
```

범주 결측을 최빈값으로 채우면 실제 존재하지 않은 Vendor·지역을 만들 수 있으므로 `Unknown`으로 분리합니다.
`handle_unknown='ignore'`는 미래 데이터에 새로운 범주가 등장해도 예측 오류가 나지 않게 합니다.

### 분류 모델

```text
LogisticRegression(class_weight='balanced')
```

로지스틱 회귀는 고팁 확률에 대한 기본선 모델이며 계수를 통해 방향을 해석할 수 있습니다.

### 데이터 누수 방지

다음 컬럼은 목표값 생성에 직접 또는 강하게 연결되므로 모델 입력에서 제외했습니다.

```text
{', '.join(model_metrics['model_info']['excluded_leakage_or_constant_features'])}
```

카드 결제만 사용하므로 `payment_type`은 모두 1인 상수이며 정보력이 없어 제외했습니다.

---

## 6. 학습·평가 분할

{dataframe_to_markdown(pd.DataFrame([model_metrics['split_info']]), include_index=False, max_rows=None)}

시간 분할을 우선 사용하여 과거 운행으로 학습하고 이후 운행을 평가했습니다. 시간 분할에서 한 클래스가 사라지면
층화 무작위 분할로 자동 전환합니다.

---

## 7. 모델 평가 결과

| 지표 | 값 |
| --- | ---: |
| Baseline Accuracy | {model_metrics['baseline_accuracy']:.4f} |
| Accuracy | {model_metrics['accuracy']:.4f} |
| Balanced Accuracy | {model_metrics['balanced_accuracy']:.4f} |
| Precision | {model_metrics['precision']:.4f} |
| Recall | {model_metrics['recall']:.4f} |
| F1-score | {model_metrics['f1_score']:.4f} |
| ROC-AUC | {model_metrics['roc_auc']:.4f} |
| Average Precision | {model_metrics['average_precision']:.4f} |
| 학습 시간 | {model_metrics['fit_time_seconds']:.2f}초 |

```text
{model_metrics['classification_report_text']}
```

{markdown_image(FIGURE_DIR / '13_model_confusion_matrix.png', '모델 혼동행렬')}

{markdown_image(FIGURE_DIR / '14_model_roc_curve.png', 'ROC Curve')}

{markdown_image(FIGURE_DIR / '15_model_precision_recall_curve.png', 'Precision Recall Curve')}

### 평가 해석 기준

- Accuracy가 Baseline보다 높은지 확인합니다.
- 클래스 불균형 때문에 Accuracy만 보지 않고 Balanced Accuracy와 F1을 함께 봅니다.
- Recall은 실제 고팁 운행을 얼마나 찾았는지, Precision은 고팁 예측 중 실제 고팁 비율을 나타냅니다.
- ROC-AUC와 Average Precision은 임계값 0.5 하나에만 의존하지 않는 순위 성능을 보여줍니다.

---

## 8. 로지스틱 회귀 계수 해석

계수는 다른 변수가 같다는 조건에서 고팁 로그오즈가 증가·감소하는 방향을 나타냅니다. 범주 빈도, 스케일링,
다중공선성 영향을 받으므로 인과관계로 해석하면 안 됩니다.

### 고팁 확률 증가 방향 상위 변수

{dataframe_to_markdown(top_positive_coefficients[[
    'transformed_feature', 'coefficient', 'absolute_coefficient', 'direction'
]], include_index=False, max_rows=15)}

### 고팁 확률 감소 방향 상위 변수

{dataframe_to_markdown(top_negative_coefficients[[
    'transformed_feature', 'coefficient', 'absolute_coefficient', 'direction'
]], include_index=False, max_rows=15)}

---

# 종합 결론

## 9. 주요 Finding

1. Yellow Taxi 데이터는 날짜·거리·요금·코드형 범주가 섞여 있어 dtype만으로 전처리를 결정하면 안 됩니다.
2. 결측은 단순 누락뿐 아니라 Vendor·수집 방식 차이일 수 있으므로 결측 여부 자체를 모델 신호로 보존했습니다.
3. 거리·시간·요금의 큰 값은 실제 희귀 운행일 수 있어 자동 삭제하지 않고 IQR·속도·요금 합계 플래그로 검토했습니다.
4. 현금 팁은 기록되지 않으므로 팁 분석과 high_tip 모델은 카드 결제만 사용했습니다.
5. Target 생성 컬럼을 입력에서 제외하고 시간 분할을 우선 적용해 데이터 누수와 비현실적 평가를 줄였습니다.
6. 모델 평가는 Accuracy뿐 아니라 F1, Balanced Accuracy, ROC-AUC, Average Precision으로 다면 평가했습니다.

## 10. 개선 방향

- 한 달이 아닌 여러 달을 사용해 계절성과 월별 분포 변화를 검증합니다.
- 공휴일·날씨·이벤트·공항·Borough 정보와 결합합니다.
- 로지스틱 회귀와 RandomForest·Gradient Boosting 성능을 비교합니다.
- 확률 Calibration과 서비스 비용에 따른 최적 임계값을 선택합니다.
- PSI·결측률·범주 변화·성능 저하를 모니터링하는 AIOps 단계를 추가합니다.
- 이상탐지가 목적이라면 현재 품질 플래그와 극단값을 삭제하지 않고 별도 탐지 모델의 입력으로 사용합니다.

---

## 11. 생성 파일

### 정적 차트

{chr(10).join(f'- `{path.relative_to(BASE_DIR).as_posix()}`' for path in static_files)}

### 인터랙티브 차트

{chr(10).join(f'- `{path.relative_to(BASE_DIR).as_posix()}`' for path in interactive_files)}

### 모델

- `{Path(model_metrics['model_file']).relative_to(BASE_DIR).as_posix()}`

### 주요 표

- `outputs/tables/column_ml_profile.csv`
- `outputs/tables/missing_summary_analysis_sample.csv`
- `outputs/tables/iqr_outlier_candidate_summary.csv`
- `outputs/tables/quality_flag_summary.csv`
- `outputs/tables/numeric_distribution_summary.csv`
- `outputs/tables/categorical_cardinality_summary.csv`
- `outputs/tables/numeric_target_relationships.csv`
- `outputs/tables/categorical_target_relationships.csv`
- `outputs/tables/ttest_weekday_weekend_tip_rate.json`
- `outputs/tables/model_metrics.json`
- `outputs/tables/model_logistic_coefficients.csv`

## 12. 분석 한계

- 후속 EDA가 표본 설정일 경우 희귀 운행과 희소 범주의 비율이 전체와 조금 다를 수 있습니다.
- Taxi 기록은 기술 제공업체가 수집한 자료이므로 극단값이 실제 운행인지 입력 오류인지 원본만으로 확정할 수 없습니다.
- 본 모델은 인과 추론 모델이 아니라 고팁 여부의 예측 패턴을 찾는 분류 모델입니다.
- high_tip 20% 기준은 프로젝트 정의이므로 실제 서비스에서는 비즈니스 목적에 맞춰 재설정해야 합니다.
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"report.md 생성 완료: {REPORT_FILE}")


def generate_presentation(
    profiles: dict[str, Any],
    cohort_info: dict[str, Any],
    ttest_result: dict[str, Any],
    model_metrics: dict[str, Any],
    numeric_target: pd.DataFrame,
) -> None:
    """5분 발표용 핵심 흐름을 Markdown으로 생성합니다."""
    top_numeric = join_top_items(
        numeric_target.sort_values("absolute_correlation", ascending=False),
        "variable",
        "correlation_with_high_tip",
        3,
    )

    content = f"""# NYC Yellow Taxi 분석 5분 발표안

## 1분: 데이터와 목표

- 2026년 5월 Yellow Taxi 전체 데이터: **{profiles['pandas']['rows']:,}행**
- Pandas와 Polars 전체 로딩 결과를 비교하여 행·열·결측치 일관성을 확인했습니다.
- 통계 주제: 평일과 주말의 카드 팁 비율 평균 비교
- ML 주제: 카드 팁 비율 20% 이상 여부 예측
- 카드 결제 분석 행: **{cohort_info['target_calculable_card_rows']:,}개**

## 1분: Question-Driven EDA

- 컬럼을 날짜·연속형·코드형 범주·목표 생성 변수로 구분했습니다.
- 결측치 비율뿐 아니라 Vendor별 결측과 동시 결측 패턴도 확인했습니다.
- 큰 거리·요금은 삭제하지 않고 IQR·속도·요금합 불일치 플래그로 관리했습니다.
- high_tip과 수치형 관계 상위: {top_numeric}

## 1분: 통계 검정

- 평일 평균 팁 비율: **{ttest_result['weekday_mean']:.4f}%**
- 주말 평균 팁 비율: **{ttest_result['weekend_mean']:.4f}%**
- p-value: **{ttest_result['p_value']:.6e}**
- Cohen's d: **{ttest_result['cohens_d']:.4f} ({ttest_result['effect_size_interpretation']})**
- 큰 표본에서는 작은 차이도 유의할 수 있어 효과크기와 민감도 분석을 함께 봤습니다.

## 1분: ML Pipeline

```text
수치형: 중앙값 대체 + 결측 표시 + 표준화
범주형: Unknown 대체 + One-Hot Encoding
모델: class_weight='balanced' LogisticRegression
```

- `tip_amount`, `total_amount`, `tip_rate`는 Target 누수 방지를 위해 제외했습니다.
- 시간 순서 분할을 우선 적용해 미래 예측 상황에 가깝게 평가했습니다.

## 1분: 결과와 개선

- Accuracy: **{model_metrics['accuracy']:.4f}**
- F1-score: **{model_metrics['f1_score']:.4f}**
- Balanced Accuracy: **{model_metrics['balanced_accuracy']:.4f}**
- ROC-AUC: **{model_metrics['roc_auc']:.4f}**
- 개선 방향: 여러 달 데이터, 날씨·공휴일 결합, 비선형 모델 비교, 확률 임계값 최적화
"""
    PRESENTATION_FILE.write_text(content, encoding="utf-8")
    print(f"5분 발표안 생성 완료: {PRESENTATION_FILE}")


def save_run_manifest(
    profiles: dict[str, Any],
    static_files: list[Path],
    interactive_files: list[Path],
    model_metrics: dict[str, Any],
) -> None:
    """실행 설정과 주요 산출물 목록을 JSON으로 저장합니다."""
    manifest = {
        "data_file": DATA_FILE,
        "analysis_period": {
            "start": ANALYSIS_START,
            "end_exclusive": ANALYSIS_END,
        },
        "settings": {
            "analysis_sample_size": ANALYSIS_SAMPLE_SIZE,
            "plot_sample_size": PLOT_SAMPLE_SIZE,
            "model_sample_size": MODEL_SAMPLE_SIZE,
            "ttest_max_per_group": TTEST_MAX_PER_GROUP,
            "random_state": RANDOM_STATE,
            "high_tip_threshold": HIGH_TIP_THRESHOLD,
            "model_split_mode": MODEL_SPLIT_MODE,
        },
        "profiles": profiles,
        "static_figures": [path.relative_to(BASE_DIR) for path in static_files],
        "interactive_figures": [
            path.relative_to(BASE_DIR) for path in interactive_files
        ],
        "model_file": model_metrics["model_file"],
        "report_file": REPORT_FILE,
        "presentation_file": PRESENTATION_FILE,
    }
    save_json(manifest, OUTPUT_DIR / "run_manifest.json")


# ==============================================================================
# [메인 실행] 평가기준과 Question-Driven EDA를 한 번에 수행
# ==============================================================================


def main() -> None:
    """전체 End-to-End 분석 파이프라인을 순서대로 실행합니다."""
    pd.set_option("display.float_format", "{:.4f}".format)
    warnings.filterwarnings("once")
    create_directories()

    print_section("[문제 1] 데이터 파일 준비")
    prepare_input_files()

    print_section("[문제 2] Pandas와 Polars 전체 로딩 비교")
    raw_analysis_df, loader_comparison, profiles = load_and_compare()

    print_section("[문제 3] 데이터 구조와 컬럼별 ML 진단")
    column_profile = build_column_profile(raw_analysis_df)

    print_section("[문제 4] 파생변수·중복·품질 플래그·극단값 후보")
    analysis_df, quality_summary, outlier_summary, duplicate_info = (
        prepare_analysis_dataframe(raw_analysis_df)
    )
    del raw_analysis_df
    gc.collect()

    print_section("[문제 5] 결측치 구조와 결측 편향")
    missing_summary, missing_patterns, missing_by_vendor = analyze_missingness(
        analysis_df
    )

    print_section("[문제 6] 카드 팁 코호트와 Target 분포")
    card_df, target_summary, cohort_info = prepare_card_tip_cohort(analysis_df)

    print_section("[문제 7] 수치형 변수 분포·왜도·분위수")
    numeric_summary = build_numeric_summary(analysis_df)

    print_section("[문제 8] 범주형 변수 분포·Cardinality·희소 범주")
    _, categorical_cardinality = analyze_categorical_variables(analysis_df)

    print_section("[문제 9] 시간대·요일·경로·지역 패턴")
    temporal_summaries = build_temporal_and_route_summaries(
        analysis_df,
        card_df,
    )

    print_section("[문제 10] 설명변수와 Target 관계·누수 점검")
    numeric_target, correlation_matrix = analyze_numeric_target_relationships(
        card_df
    )
    categorical_target = analyze_categorical_target_relationships(card_df)
    missing_target_signal = analyze_missingness_target_signal(card_df)

    print_section("[문제 11] Seaborn·Plotly 다양한 시각화")
    static_files = create_static_visualizations(
        analysis_df,
        card_df,
        missing_summary,
        quality_summary,
        numeric_summary,
        outlier_summary,
        correlation_matrix,
        temporal_summaries,
    )
    interactive_files = create_interactive_visualizations(
        card_df,
        temporal_summaries,
    )
    print(f"정적 차트 {len(static_files)}개 생성")
    print(f"인터랙티브 차트 {len(interactive_files)}개 생성")

    print_section("[문제 12] Welch t-test와 p-value 해석")
    ttest_result = perform_t_test(card_df)

    print_section("[문제 13] sklearn Pipeline 학습·평가·joblib 저장")
    model_metrics, coefficient_table, model_chart_files = train_ml_pipeline(card_df)
    static_files.extend(model_chart_files)

    print_section("[문제 14] report.md와 5분 발표안 자동 생성")
    generate_report(
        loader_comparison=loader_comparison,
        profiles=profiles,
        column_profile=column_profile,
        duplicate_info=duplicate_info,
        missing_summary=missing_summary,
        missing_patterns=missing_patterns,
        missing_by_vendor=missing_by_vendor,
        quality_summary=quality_summary,
        numeric_summary=numeric_summary,
        outlier_summary=outlier_summary,
        categorical_cardinality=categorical_cardinality,
        target_summary=target_summary,
        cohort_info=cohort_info,
        temporal_summaries=temporal_summaries,
        numeric_target=numeric_target,
        categorical_target=categorical_target,
        missing_target_signal=missing_target_signal,
        ttest_result=ttest_result,
        model_metrics=model_metrics,
        coefficient_table=coefficient_table,
        static_files=static_files,
        interactive_files=interactive_files,
    )
    generate_presentation(
        profiles=profiles,
        cohort_info=cohort_info,
        ttest_result=ttest_result,
        model_metrics=model_metrics,
        numeric_target=numeric_target,
    )
    save_run_manifest(
        profiles,
        static_files,
        interactive_files,
        model_metrics,
    )

    print_section("분석 완료")
    print(f"상세 보고서: {REPORT_FILE}")
    print(f"5분 발표안: {PRESENTATION_FILE}")
    print(f"결과 폴더: {OUTPUT_DIR}")
    print(f"저장 모델: {model_metrics['model_file']}")


if __name__ == "__main__":
    main()
