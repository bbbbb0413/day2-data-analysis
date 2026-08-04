"""
================================================================================
[데이터 탐색] NYC Yellow Taxi 데이터 기본정보 출력
파일명 : taxi_data_overview.py
작성일 : 2026-08-04
================================================================================

■ 프로그램 개요
  대용량 택시 운행 기록(409만 행 × 20열)을 읽어 "분석을 시작하기 전에 반드시
  확인해야 할 기본정보"를 7단계로 나누어 출력한다.

    0) 파일 검증        : 존재/크기/확장자 확인 + parquet은 전체를 읽기 전에
                          메타데이터(행수·열수)만 먼저 조회
    1) 로딩             : dtype·parse_dates를 명시해 읽고 소요 시간을 측정
    2) 크기 / 메모리    : shape, 셀 수, 실제 메모리 점유량, 파일 대비 팽창률
    3) 미리보기         : 앞 5행
    4) 컬럼 정보        : dtype · 결측수 · 결측% · 고유값수를 한 표로 정리
    5) 기술통계         : 수치형 describe (평균 vs 중앙값 비교용)
    6) 범주형 분포      : 주요 코드성 컬럼의 value_counts
    7) 데이터 품질 점검 : 기간 이탈·음수 요금·0거리·시간 역전 등 이상치 집계

■ 설계 원칙
  - CSV/parquet 모두 지원 : 같은 데이터를 두 포맷으로 읽어 비교할 수 있게
                            확장자를 보고 자동으로 읽기 방식을 전환한다.
  - 타입 명시 로딩        : CSV에는 타입 정보가 저장되지 않아 날짜가 문자열로
                            읽힌다. DATE_COLS/INT32_COLS를 넘겨 parquet과
                            동일한 DataFrame이 되도록 복원한다.
  - 조기 실패            : 409만 행을 다 읽은 뒤 컬럼이 없다고 죽지 않도록,
                            헤더만 먼저 읽어 필수 컬럼을 검증한다.
  - 화면 출력만          : 파일을 생성하지 않는 읽기 전용 스크립트다.

■ 실행 방법
  python taxi_data_overview.py                      # 기본 CSV 사용
  python taxi_data_overview.py yellow.parquet       # 다른 파일 지정
  (pandas 필요: pip install pandas pyarrow)

■ 변경 내역
  2026-08-04  v1.0  최초 작성
================================================================================
"""

import sys
import time
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# 전역 설정 (경로·컬럼·상수를 한 곳에서만 관리해 중복을 없앤다)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent        # 실행 위치와 무관하게 동작시키기 위함
DEFAULT_PATH = BASE_DIR / "yellow_tripdata_2026-05.csv"

# CSV는 타입 정보를 담지 못하므로 읽을 때 직접 알려줘야 한다.
DATE_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
INT32_COLS = ["VendorID", "PULocationID", "DOLocationID"]  # 값 범위가 작아 int64는 낭비

# 코드값(범주형)으로 다뤄야 하는 컬럼 — 숫자지만 평균을 내면 의미가 없다.
CATEGORICAL_COLS = ["VendorID", "payment_type", "RatecodeID", "passenger_count"]

REQUIRED_COLS = DATE_COLS + ["trip_distance", "fare_amount", "total_amount"]

EXPECTED_MONTH = "2026-05"                        # 이 파일이 담고 있어야 할 기간
LINE = "=" * 88


# ----------------------------------------------------------------------------
# 1. 공통 유틸
# ----------------------------------------------------------------------------
def print_step(title):
    """단계 구분 헤더를 통일된 형식으로 출력한다."""
    print(f"\n{LINE}\n{title}\n{LINE}")


def human_bytes(n):
    """바이트 수를 KB/MB/GB 단위 문자열로 바꾼다."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024


def validate_file(path):
    """
    전체 로딩 전에 파일 사용 가능 여부를 검증한다(조기 실패).

    - 존재 여부 / 빈 파일 여부 확인
    - 지원 확장자(.csv, .parquet) 확인
    - CSV는 헤더만 읽어 필수 컬럼 존재 여부까지 확인

    Raises: FileNotFoundError, ValueError, KeyError
    """
    if not path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"파일이 비어 있습니다: {path}")
    if path.suffix.lower() not in (".csv", ".parquet"):
        raise ValueError(f"지원하지 않는 확장자입니다(.csv/.parquet만 가능): {path.suffix}")

    if path.suffix.lower() == ".csv":
        try:
            header = pd.read_csv(path, nrows=0)     # 헤더 1줄만 읽으므로 즉시 끝난다
        except UnicodeDecodeError:
            raise ValueError("CSV 인코딩을 확인하세요. UTF-8 파일이 필요합니다.")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV에 헤더가 없습니다.")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV 파싱 오류: {e}")

        missing = [c for c in REQUIRED_COLS if c not in header.columns]
        if missing:
            raise KeyError(f"필수 컬럼이 없습니다: {missing} / 실제 컬럼: {list(header.columns)}")


def peek_parquet_metadata(path):
    """
    parquet 파일을 '읽지 않고' 푸터의 메타데이터만 조회해 출력한다.

    CSV라면 전체를 스캔해야 알 수 있는 행 수를 즉시 알 수 있다.
    pyarrow가 없으면 경고만 남기고 넘어간다(스크립트가 죽지 않도록).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[안내] pyarrow가 없어 메타데이터 조회를 건너뜁니다.")
        return

    md = pq.ParquetFile(path).metadata
    print(f"  행 수      : {md.num_rows:,}   (파일 전체를 읽지 않고 확인)")
    print(f"  열 수      : {md.num_columns}")
    print(f"  Row group  : {md.num_row_groups}개")
    print(f"  생성 도구  : {md.created_by}")


def load_data(path):
    """
    확장자에 따라 CSV/parquet을 읽어 동일한 DataFrame을 만든다.

    ★ 핵심: CSV에는 타입 정보가 없어 날짜가 문자열(object)로 읽히고 .dt 접근자가
      막힌다. parse_dates/dtype을 명시해 parquet과 같은 상태로 복원한다.

    Returns: (DataFrame, 로딩 소요 초)
    """
    start = time.perf_counter()
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)                  # 타입이 파일에 저장돼 있어 지정 불필요
    else:
        df = pd.read_csv(
            path,
            parse_dates=DATE_COLS,                  # 문자열 → datetime64 복원
            dtype={c: "int32" for c in INT32_COLS},  # int64 기본값 대비 메모리 절반
            low_memory=False,                       # 컬럼별 타입 추론이 흔들리지 않게
        )
    return df, time.perf_counter() - start


# ----------------------------------------------------------------------------
# 2. 정보 출력 단계별 함수
# ----------------------------------------------------------------------------
def report_size(df, path, load_sec):
    """크기·메모리·로딩 시간을 출력한다."""
    file_size = path.stat().st_size
    mem = df.memory_usage(deep=True).sum()          # deep=True: 문자열 실제 크기까지 반영

    print(f"  파일 경로   : {path}")
    print(f"  파일 크기   : {human_bytes(file_size)}")
    print(f"  로딩 시간   : {load_sec:.2f}초")
    print(f"  행 × 열     : {df.shape[0]:,} × {df.shape[1]}")
    print(f"  전체 셀 수  : {df.shape[0] * df.shape[1]:,}")
    print(f"  메모리 점유 : {human_bytes(mem)}  (파일 대비 {mem / file_size:.1f}배)")


def report_columns(df):
    """dtype·결측·고유값수를 한 표로 정리한다. 결측치 파악의 출발점."""
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "결측수": df.isna().sum(),
            "결측%": (df.isna().mean() * 100).round(2),
            "고유값수": df.nunique(),
        }
    )
    print(info.to_string())

    # 결측이 있는 컬럼들이 '같은 행'에서 동시에 비는지 확인한다.
    # 개수가 똑같다면 우연이 아니라 구조적 원인(제출 포맷 차이 등)일 가능성이 크다.
    na_cols = info.index[info["결측수"] > 0].tolist()
    if na_cols:
        counts = set(info.loc[na_cols, "결측수"])
        print(f"\n  결측 컬럼 {len(na_cols)}개: {na_cols}")
        if len(counts) == 1:
            print(f"  → 결측 개수가 모두 {counts.pop():,}건으로 동일합니다.")
            print("    랜덤 결측이 아니라 특정 조건의 행이 통째로 빠진 구조적 패턴일 수 있으니,")
            print("    fillna() 전에 '어떤 행에서 비는지'부터 확인하세요.")
    else:
        print("\n  결측치 없음")


def report_describe(df):
    """수치형 기술통계. 평균과 중앙값 차이로 이상치 영향을 가늠한다."""
    desc = df.describe().T
    print(desc.to_string(float_format=lambda x: f"{x:,.2f}"))

    # 평균이 중앙값에서 크게 벗어난 컬럼 = 소수 극단값에 평균이 끌려간 컬럼.
    # 판단 기준으로 표준편차를 쓰면, 이상치 때문에 표준편차 자체가 같이 부풀려져
    # 어떤 컬럼도 걸리지 않는다. 그래서 '평균 ÷ 중앙값' 비율로 본다.
    numeric = df.select_dtypes("number").columns
    target = desc.loc[desc.index.intersection(numeric)]
    target = target[target["50%"] > 0]               # 중앙값 0이면 비율 계산 불가
    ratio = (target["mean"] / target["50%"]).sort_values(ascending=False)
    skewed = ratio[(ratio > 1.5) | (ratio < 0.67)]

    if not skewed.empty:
        print("\n  [주의] 평균이 중앙값과 1.5배 이상 차이 나는 컬럼")
        for col, r in skewed.items():
            print(f"     {col:<22} 평균 {target.loc[col, 'mean']:>10,.2f}"
                  f" vs 중앙값 {target.loc[col, '50%']:>8,.2f}  ({r:.1f}배)")
        print("        분포가 한쪽으로 쏠려 있습니다. 평균만 보고 판단하지 마세요.")


def report_categorical(df):
    """코드값 컬럼의 분포. 숫자지만 평균이 아니라 빈도로 봐야 하는 컬럼들."""
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).head(8)
        ratio = (vc / len(df) * 100).round(1)
        print(f"\n  -- {col} (고유값 {df[col].nunique()}개, 상위 {len(vc)}개)")
        for value, cnt in vc.items():
            print(f"     {str(value):>8} : {cnt:>10,}건  ({ratio[value]:>5.1f}%)")


def report_quality(df):
    """분석 전에 걸러야 할 이상 레코드를 집계한다."""
    pickup, dropoff = DATE_COLS
    checks = {}

    if pickup in df.columns:
        month = pd.Period(EXPECTED_MONTH)
        out_of_range = (df[pickup] < month.start_time) | (df[pickup] > month.end_time)
        checks[f"{EXPECTED_MONTH} 기간 밖의 승차시각"] = out_of_range.sum()
    if pickup in df.columns and dropoff in df.columns:
        checks["하차시각 <= 승차시각 (시간 역전)"] = (df[dropoff] <= df[pickup]).sum()
    if "trip_distance" in df.columns:
        checks["이동거리 = 0"] = (df["trip_distance"] == 0).sum()
        checks["이동거리 > 100 mile"] = (df["trip_distance"] > 100).sum()
    if "fare_amount" in df.columns:
        checks["요금 < 0 (환불·정정)"] = (df["fare_amount"] < 0).sum()
    if "total_amount" in df.columns:
        checks["총액 <= 0"] = (df["total_amount"] <= 0).sum()

    total = len(df)
    for label, cnt in checks.items():
        print(f"  {label:<34} : {cnt:>9,}건  ({cnt / total * 100:>5.2f}%)")

    if pickup in df.columns:
        print(f"\n  승차시각 범위 : {df[pickup].min()}  ~  {df[pickup].max()}")
        print(f"  하차시각 범위 : {df[dropoff].min()}  ~  {df[dropoff].max()}")


# ----------------------------------------------------------------------------
# 3. 메인 (0~7단계를 순서대로 실행)
# ----------------------------------------------------------------------------
def main():
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_PATH

    print_step(f"[0] 파일 검증 : {path.name}")
    validate_file(path)
    print("  검증 통과 (파일 존재 · 확장자 · 필수 컬럼 확인 완료)")
    if path.suffix.lower() == ".parquet":
        peek_parquet_metadata(path)

    print(f"\n  로딩 중... (대용량 파일은 시간이 걸립니다)")
    df, load_sec = load_data(path)

    print_step("[1] 데이터 크기 / 메모리")
    report_size(df, path, load_sec)

    print_step("[2] 미리보기 (앞 5행)")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.head())

    print_step("[3] 컬럼 정보 (타입 · 결측 · 고유값)")
    report_columns(df)

    print_step("[4] 수치형 기술통계")
    with pd.option_context("display.width", 250):
        report_describe(df)

    print_step("[5] 범주형 분포")
    report_categorical(df)

    print_step("[6] 데이터 품질 점검")
    report_quality(df)

    print(f"\n{LINE}\n분석 완료 — 총 {len(df):,}행 확인\n{LINE}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n[오류] {e}", file=sys.stderr)
        print("실행 예: python taxi_data_overview.py yellow_tripdata_2026-05.csv",
              file=sys.stderr)
        sys.exit(1)
    except MemoryError:
        print("\n[오류] 메모리가 부족합니다. read_csv(..., nrows=100000)으로 일부만 읽어보세요.",
              file=sys.stderr)
        sys.exit(1)
