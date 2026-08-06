"""
================================================================================
[데이터 준비 + 기본 EDA] NYC Yellow Taxi 2026-05
파일명 : taxi_eda_pipeline.py
작성일 : 2026-08-05
================================================================================

■ 프로그램 개요
  같은 parquet을 pandas와 polars 양쪽으로 읽어 결과를 비교한 뒤,
  『결측치_중복_처리기준.md』에서 확정한 기준을 그대로 코드로 옮겨
  결측치·중복을 처리하고 기본 EDA를 수행한다.

    [1] 로딩 비교      : pandas vs polars — 속도·메모리·dtype·결측수·중복수
    [2] 결측치 처리    : 문서 §2 기준 (삭제·대체 금지, 소스 분리 + sentinel 변환)
    [3] 중복 처리      : 문서 §3 기준 (유형별 선택 제거, keep='first' 금지)
    [4] 이상치 처리    : 문서 §4 기준 (기간·소요시간·거리·금액)
    [5] 기본 EDA       : 정제 데이터의 분포·시간대·존·요금 구조
    [6] 검증 요약      : 문서에 기록된 기대값과 실측값 대조

■ 설계 원칙
  - 기준의 출처를 코드에 남긴다
      모든 임계값 위에 "문서 몇 절의 어떤 근거로 정했는지"를 주석으로 붙였다.
      숫자만 남으면 3개월 뒤에 아무도 왜 6시간인지 설명하지 못한다.

  - 삭제보다 표시(flag)를 우선한다
      결측 955,371행은 특정 사업자·심야·주말에 쏠려 있어(문서 §2.4) 지우면
      표본이 편향된다. record_source 컬럼으로 나누기만 하고 행은 남긴다.

  - 중복은 '지우기 전에 정체를 밝힌다'
      drop_duplicates(keep='first')는 이 데이터에서 정상 운행을 지우고
      취소 기록을 남긴다(문서 §3.5). 그룹의 금액 관계를 먼저 판정한다.

  - 두 라이브러리를 순차 로딩한다
      동시에 들고 있으면 최대 RSS가 4GB를 넘고, 서로의 메모리 측정값이
      오염된다. polars → 지표 추출 → 해제 → pandas 순으로 각각 측정한다.

  - 입력은 parquet만 받는다
      CSV는 타입 정보가 없어 날짜가 문자열로 읽히고, 그러면 소요시간 계산이
      막혀 [4]단계 기준을 적용할 수 없다. 원본을 parquet으로 고정해
      "읽는 방법에 따라 결과가 달라지는" 변수를 아예 없앴다.

  - 기본은 읽기 전용
      --save를 명시할 때만 정제 결과를 outputs/에 parquet으로 저장한다.

■ 입력 파일
  data/raw/yellow_tripdata_2026-05.parquet   (66.5MB, 4,090,836행 × 20열)

■ 실행 방법
  python taxi_eda_pipeline.py               # 기본 입력으로 전체 파이프라인
  python taxi_eda_pipeline.py <다른.parquet>  # 파일 직접 지정
  python taxi_eda_pipeline.py --rss         # 최대 RSS 정밀 측정(프로세스 분리)
  python taxi_eda_pipeline.py --save        # 정제 결과 parquet 저장
  (pip install -r requirements.txt)

■ 참고 문서
  결측치_중복_처리기준.md  — 아래 모든 임계값의 근거

■ 변경 내역
  2026-08-05  v1.0  최초 작성
================================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ============================================================================
# 전역 설정 — 문서에서 확정한 기준값을 한 곳에 모은다
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
# 입력은 parquet 하나로 고정한다.
#   - 타입 정보가 파일에 들어 있어 parse_dates/dtype 지정 없이 원본 상태로 복원된다
#     (CSV로 읽으면 날짜가 문자열이 되어 소요시간 계산 자체가 불가능하다)
#   - 컬럼 단위 저장이라 필요한 컬럼만 읽는 지연 평가(scan_parquet)가 가능하다
#   - 66.5MB로 압축돼 반복 실행이 빠르다
RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_PATH = RAW_DIR / "yellow_tripdata_2026-05.parquet"
OUT_DIR = BASE_DIR / "outputs"

DATE_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
PU, DO = DATE_COLS

# --- 중복 판정 키 (문서 §3.2) -------------------------------------------------
# 키를 넓히거나 좁히며 관측한 결과:
#     시각 2개만          → 33,925건 (우연 충돌 섞임)
#     시각 2개 + 존 2개   → 29,401건 ★ 채택
#     + 거리 + 총액       →      0건 (중복 후보의 금액이 서로 달라 탐지 불가)
# 즉 "탐지가 가능하면서 우연이 아닌" 유일한 경계가 이 4개 키다.
# 우연 가능성은 귀무모형으로 배제했다(문서 §3.3): 무작위 재배치 시 기대 2건인데
# 실제 29,401건 → 약 14,700배. 우연으로 설명할 수 없다.
DUP_KEY = [PU, DO, "PULocationID", "DOLocationID"]

# --- 중복 유형 판정 임계 (문서 §3.4) ------------------------------------------
# ① 상쇄쌍  : 그룹 두 행의 total_amount 합이 0 → 취소·환불로 장부상 상쇄
#             (음수 행 14,877건 중 13,301건에 정확히 반대 금액의 짝이 존재,
#              음수 행의 60.8%가 payment_type=4 Dispute)
#             부동소수 오차만 흡수하면 되므로 허용오차는 1센트 미만으로 잡는다.
VOID_TOL = 0.01
# ② 이중계상 : 20개 컬럼 중 total_amount만 정확히 2배 → 큰 쪽이 이중 계상된 행
#             (16,086쌍 전부 양쪽 payment_type=0. 동일 포맷 단독행의
#              total/fare 중앙값 1.231 대비 큰 쪽은 2.785로 완전히 이탈)
DOUBLE_LO, DOUBLE_HI = 1.99, 2.01

# --- 이상치 기준 (문서 §4) ----------------------------------------------------
EXPECTED_MONTH = "2026-05"
# 소요시간 상한 6시간: 상위 99.9%가 7,750초(2.15시간)이므로 매우 관대한 상한이다.
# 장거리 공항 운행을 살리려는 선택이며, 최댓값 166시간 같은 오기록만 걸러낸다.
DUR_MIN_SEC, DUR_MAX_SEC = 1, 6 * 3600
# 거리 상한 100마일: 최댓값 307,491마일(지구 12바퀴)은 명백한 오기록.
DIST_MIN, DIST_MAX = 0.01, 100.0

# ################################################################################
# 🆕🆕🆕 [NEW] 속력 기반 이상치 필터 (2026-08-05 추가) 🆕🆕🆕
# 거리·시간을 "각각" 자르는 것만으로는 "거리 99mile + 시간 2초" 같은
# 물리적으로 불가능한 "조합"은 못 걸러냄! 그래서 속력(distance/time)을 새로
# 계산해서 추가로 거름. 뉴욕은 정체가 심해서 하한선은 안 둠(정체=저속은 정상이라
# 지우면 안 됨), GPS 오류로 튀는 상한선(200km/h)만 잡음.
# ################################################################################
SPEED_MAX_KMH = 200.0

# --- 위장 결측 (문서 §2.6) ----------------------------------------------------
# isna()에 안 잡히지만 실질적으로 결측인 값들. 합계 약 114만 건으로,
# 그냥 두면 payment_type 평균 같은 집계가 조용히 오염된다.
SENTINELS = {
    "payment_type": [0],          # TLC 코드북에 없는 값 (955,371건, 전부 결측행)
    "RatecodeID": [99],           # TLC 정의상 "Unknown" (140,897건)
    "passenger_count": [0],       # 0명 탑승은 물리적으로 불가 (12,533건)
    "PULocationID": [264, 265],   # 264=Unknown, 265=구역 외 (6,486건)
    "DOLocationID": [264, 265],   # 〃 (21,726건)
}

# 결측이 100% 동시 발생하는 5개 컬럼 (문서 §2.2 — 일부만 결측인 행이 0건)
NA_GROUP_COLS = ["passenger_count", "RatecodeID", "store_and_fwd_flag",
                 "congestion_surcharge", "Airport_fee"]
# 그 중 결측 여부 판정의 대표 컬럼. 5개가 완전히 같은 행에서 비므로 하나면 충분하다.
NA_FLAG_COL = "congestion_surcharge"

CATEGORICAL_COLS = ["VendorID", "payment_type", "RatecodeID", "passenger_count"]

# --- 문서에 기록된 기대값 (문서 §0, §3.4, §5) — [6]단계에서 실측과 대조 --------
EXPECTED = {
    "총 행 수": 4_090_836,
    "결측 행 수": 955_371,
    "완전중복": 0,
    "중복키 그룹": 29_401,
    "상쇄쌍(그룹)": 13_301,
    "이중계상(행)": 16_086,
    "최종 행 수": 3_884_062,  # 🆕 2026-08-05 속력(speed) 규칙 추가로 -560행 (기존 3,884,622)
}

LINE = "=" * 88


# ============================================================================
# 1. 공통 유틸
# ============================================================================
def print_step(title):
    """단계 구분 헤더를 통일된 형식으로 출력한다."""
    print(f"\n{LINE}\n{title}\n{LINE}")


def human_bytes(n):
    """바이트 수를 KB/MB/GB 단위 문자열로 바꾼다."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024


def peak_rss_mb():
    """현재 프로세스가 지금까지 점유한 최대 물리메모리(MB).

    resource는 표준 라이브러리라 psutil 없이 쓸 수 있다.
    macOS는 ru_maxrss를 바이트로, Linux는 KB로 돌려주므로 단위를 보정한다.

    ★ 한 프로세스에서 두 라이브러리를 순차로 재면 안 된다
      ru_maxrss는 '프로세스가 지금까지 찍은 최댓값'이라 절대 내려가지 않는다.
      polars를 해제한 뒤 pandas를 재도 앞서 찍힌 최댓값이 그대로 남아
      두 값이 똑같이 나온다(= 비교가 성립하지 않는다).
      정확히 재려면 프로세스를 분리해야 한다 → measure_peak_rss() 참고.
    """
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def measure_peak_rss(path, lib):
    """라이브러리 하나만 로딩하는 자식 프로세스를 띄워 최대 RSS를 측정한다.

    이 파일 자신을 --bench-rss 모드로 다시 실행한다. 측정용 로딩 코드를 따로
    쓰지 않고 load_with_polars/load_with_pandas를 그대로 재사용하기 위함이다.
    (측정 대상과 실제 코드가 다르면 측정값에 의미가 없다)
    """
    import subprocess
    proc = subprocess.run(
        [sys.executable, __file__, str(path), "--bench-rss", lib],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return float("nan")


def resolve_path(args):
    """입력 경로를 정한다: 인자로 받은 경로가 있으면 그것, 없으면 기본 parquet."""
    return Path(args[0]).resolve() if args else DEFAULT_PATH


def warm_page_cache(path):
    """파일을 한 번 통독해 OS 페이지 캐시에 올린다.

    ★ 왜 필요한가
      먼저 측정하는 쪽만 디스크에서 실제로 읽고(cold), 나중 쪽은 메모리에서
      읽는다(warm). 실측에서 pandas가 5.98초 → 0.11초로 54배 널뛰었는데,
      이는 라이브러리 성능이 아니라 캐시 상태의 차이였다.
      두 라이브러리를 같은 조건에 두려면 측정 전에 캐시를 데워야 한다.
    """
    with open(path, "rb") as f:
        while f.read(1 << 24):          # 16MB씩 — 전체를 메모리에 담지 않는다
            pass


def validate_file(path):
    """전체 로딩 전에 파일을 검증한다(조기 실패).

    409만 행을 다 읽은 뒤 "컬럼이 없다"고 죽지 않도록, parquet 푸터의
    메타데이터만 조회해 행수·열수·필수 컬럼을 먼저 확인한다.
    메타데이터는 파일 끝에 따로 저장돼 있어 데이터를 한 바이트도 읽지 않는다.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {path}\n"
            f"       원본 parquet을 {RAW_DIR} 에 두세요.")
    if path.stat().st_size == 0:
        raise ValueError(f"파일이 비어 있습니다: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(
            f"parquet만 지원합니다(입력: {path.suffix}). "
            "CSV는 타입 정보가 없어 날짜가 문자열로 읽히고 소요시간 계산이 불가능합니다.")

    import pyarrow.parquet as pq
    md = pq.ParquetFile(path)
    cols = md.schema_arrow.names
    required = DUP_KEY + ["trip_distance", "fare_amount", "total_amount"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise KeyError(f"필수 컬럼이 없습니다: {missing} / 실제 컬럼: {cols}")

    print(f"  행 수     : {md.metadata.num_rows:,}  (파일을 읽지 않고 메타데이터로 확인)")
    print(f"  열 수     : {md.metadata.num_columns}")
    print(f"  Row group : {md.metadata.num_row_groups}개")


# ============================================================================
# 2. [1단계] pandas vs polars 로딩 비교
# ============================================================================
def load_with_polars(path):
    """polars로 읽고 비교 지표만 추출한 뒤 DataFrame은 버린다.

    ★ 인자가 없는 이유
      parquet은 컬럼별 타입이 파일에 저장돼 있어 날짜·정수 타입이 그대로 복원된다.
      CSV처럼 parse_dates/dtype을 지정할 필요가 없고, 지정할 여지도 없다.
      그래서 이 단계에서 두 라이브러리의 차이는 '설정'이 아니라 순수한 구현 차이다.

    ★ 반환값에 DataFrame을 포함하지 않는 이유
      pandas와 동시에 들고 있으면 최대 RSS가 4GB를 넘고, 나중에 측정하는 쪽의
      메모리 수치가 앞선 로딩에 오염된다. 지표만 뽑고 즉시 해제한다.
    """
    t = time.perf_counter()
    df = pl.read_parquet(path)
    load_sec = time.perf_counter() - t

    # 결측 수: polars는 null_count()가 컬럼별 1행 DataFrame으로 나온다.
    nulls = dict(zip(df.columns, df.null_count().row(0)))

    # 완전중복: n_unique()는 "서로 다른 행의 수"이므로 전체에서 빼면 중복 수가 된다.
    n_exact_dup = len(df) - df.n_unique()

    # ★ 부분키 중복에서 두 라이브러리의 정의가 다르다 (아래 표에서 대조)
    #   polars is_duplicated() : 그룹 구성원 '전부'를 True로 표시  → 58,802
    #   pandas duplicated()    : 그룹의 '첫 행을 뺀 나머지'만 True → 29,401
    #   같은 데이터인데 숫자가 2배 차이 나므로, 무엇을 세는지 반드시 확인해야 한다.
    dup_all_members = int(df.select(DUP_KEY).is_duplicated().sum())

    metrics = {
        "라이브러리": f"polars {pl.__version__}",
        "로딩 시간(초)": load_sec,
        "DataFrame 크기(MB)": df.estimated_size("mb"),
        "shape": df.shape,
        "dtype 요약": _dtype_summary([str(d) for d in df.dtypes]),
        "결측 총합": sum(nulls.values()),
        "결측 컬럼수": sum(1 for v in nulls.values() if v > 0),
        "완전중복": n_exact_dup,
        "중복키 대상행(전 구성원)": dup_all_members,
    }
    del df                      # 다음 로딩 전에 반드시 해제
    return metrics, nulls


def load_with_pandas(path):
    """pandas로 읽는다. 이후 모든 처리는 이 DataFrame으로 진행한다.

    parquet이라 타입 지정이 필요 없다. 날짜는 datetime64로, 코드성 컬럼은
    저장 당시의 int32로 그대로 복원되므로 [4]단계의 소요시간 계산이 바로 된다.
    """
    t = time.perf_counter()
    df = pd.read_parquet(path)
    load_sec = time.perf_counter() - t

    metrics = {
        "라이브러리": f"pandas {pd.__version__}",
        "로딩 시간(초)": load_sec,
        # deep=True라야 object 컬럼(문자열)의 실제 크기가 반영된다.
        "DataFrame 크기(MB)": df.memory_usage(deep=True).sum() / 1024**2,
        "shape": df.shape,
        "dtype 요약": _dtype_summary([str(d) for d in df.dtypes]),
        "결측 총합": int(df.isna().sum().sum()),
        "결측 컬럼수": int((df.isna().sum() > 0).sum()),
        "완전중복": int(df.duplicated().sum()),
        # pandas는 첫 행을 제외하므로 polars 수치의 절반이 나온다(정의 차이).
        "중복키 대상행(첫행 제외)": int(df.duplicated(subset=DUP_KEY).sum()),
    }
    return df, metrics


def _dtype_summary(dtypes):
    """dtype 목록을 '종류: 개수' 문자열로 압축한다."""
    s = pd.Series(dtypes).value_counts()
    return ", ".join(f"{k} {v}개" for k, v in s.items())


def report_loading(pl_metrics, pd_metrics, pl_nulls, df, path, rss=None):
    """두 로딩 결과를 나란히 출력하고, 일치 여부를 검증한다."""
    print(f"  파일 : {path}")
    print(f"  크기 : {human_bytes(path.stat().st_size)}\n")

    keys = ["라이브러리", "로딩 시간(초)", "DataFrame 크기(MB)",
            "shape", "dtype 요약", "결측 총합", "결측 컬럼수", "완전중복"]
    print(f"  {'항목':<22}{'polars':>34}   {'pandas':>34}")
    print("  " + "-" * 92)
    for k in keys:
        a, b = pl_metrics[k], pd_metrics[k]
        fmt = (lambda v: f"{v:,.2f}") if isinstance(a, float) else (lambda v: str(v))
        print(f"  {k:<22}{fmt(a):>34}   {fmt(b):>34}")

    speed = pd_metrics["로딩 시간(초)"] / max(pl_metrics["로딩 시간(초)"], 1e-9)
    size = pd_metrics["DataFrame 크기(MB)"] / max(pl_metrics["DataFrame 크기(MB)"], 1e-9)
    print(f"\n  → polars가 로딩 {speed:.1f}배 빠르고, DataFrame이 {size:.2f}배 작다.")

    # 최대 RSS는 프로세스 누적 최댓값이라 한 프로세스에서 순차 측정하면
    # 두 값이 같아진다(peak_rss_mb() 주석 참고). --rss를 줄 때만 프로세스를
    # 분리해 정확히 잰다. 파일을 두 번 더 읽으므로 기본값은 꺼둔다.
    if rss:
        # 로딩만이 아니라 위 지표 계산(n_unique·중복 판정 등)까지 포함한 값이다.
        # 두 라이브러리가 같은 작업을 하므로 비교는 성립한다.
        print(f"\n  [최대 RSS — 로딩+지표계산, 프로세스 분리 측정]")
        print(f"     polars {rss['polars']:>10,.1f} MB   /   pandas {rss['pandas']:>10,.1f} MB"
              f"   ({rss['pandas']/max(rss['polars'], 1e-9):.1f}배)")
    else:
        print("     (최대 RSS 비교는 --rss 옵션 — 프로세스를 분리해야 정확히 측정된다)")

    # ---- 결과 일치 검증 ----------------------------------------------------
    # 이 검증이 통과해야 "이후 분석은 도구 선택과 무관한 데이터 자체의 성질"이라고
    # 말할 수 있다. shape·결측·완전중복이 다르면 파서 설정부터 다시 봐야 한다.
    pd_nulls = df.isna().sum().to_dict()
    same_shape = pl_metrics["shape"] == pd_metrics["shape"]
    same_null = all(int(pl_nulls[c]) == int(pd_nulls[c]) for c in pd_nulls)
    same_dup = pl_metrics["완전중복"] == pd_metrics["완전중복"]
    print(f"\n  [일치 검증] shape {'OK' if same_shape else '불일치'}"
          f" / 컬럼별 결측수 {'OK' if same_null else '불일치'}"
          f" / 완전중복 {'OK' if same_dup else '불일치'}")

    # ---- 정의가 달라 숫자가 갈리는 지점 ------------------------------------
    print(f"\n  [주의] 부분키 중복은 두 라이브러리의 '세는 방식'이 다르다")
    print(f"     polars  df.select(KEY).is_duplicated().sum() = "
          f"{pl_metrics['중복키 대상행(전 구성원)']:>9,}  (그룹 구성원 전부)")
    print(f"     pandas  df.duplicated(subset=KEY).sum()      = "
          f"{pd_metrics['중복키 대상행(첫행 제외)']:>9,}  (첫 행 제외)")
    print("     같은 데이터인데 2배 차이. '중복 N건'이라고 말할 때 무엇을 센 것인지")
    print("     명시하지 않으면 보고서끼리 숫자가 안 맞는다.")

    # ---- 결측 의미론 차이 (문서 §1) ----------------------------------------
    # pandas는 '연산 실패로 생긴 NaN'과 '원본에 값이 없는 결측'을 구분하지 못한다.
    # 파생 컬럼을 만든 뒤 결측을 집계하면 0/0 같은 계산 실패가 섞여 들어간다.
    # → pandas를 쓸 때는 반드시 '파생 컬럼 생성 전에' 결측 집계를 끝내야 한다.
    demo_pd = pd.Series([1.0, np.nan, None])
    demo_pl = pl.Series([1.0, float("nan"), None])
    print(f"\n  [결측 의미론] 입력 [1.0, NaN, None]")
    print(f"     pandas  isna()   = {demo_pd.isna().tolist()}   ← NaN과 결측을 구분 못 함")
    print(f"     polars  is_null()= {demo_pl.is_null().to_list()}")
    print(f"     polars  is_nan() = {demo_pl.is_nan().to_list()}   ← 둘을 구분함")


def demo_polars_lazy(path):
    """polars 지연 평가(scan_parquet) 시연.

    ★ 왜 따로 보여주는가
      read_parquet은 20개 컬럼 전부를 메모리에 올린다. scan_parquet은 쿼리를
      먼저 분석해 실제로 쓰이는 컬럼만 읽고, 필터를 파일 스캔 단계로 내려보낸다.
      아래 질의는 20개 중 3개 컬럼만 필요하므로 나머지는 디스크에서 건드리지도 않는다.
      "존별 평균 요금" 하나 때문에 550MB를 통째로 올리는 것은 낭비다.
    """
    t = time.perf_counter()
    out = (
        pl.scan_parquet(path)
        .filter(pl.col("trip_distance") > 0)          # 필터가 스캔 단계로 내려간다
        .group_by("PULocationID")
        .agg(pl.col("total_amount").mean().alias("평균총액"),
             pl.len().alias("건수"))
        .sort("건수", descending=True)
        .head(5)
        .collect()                                     # 여기서 비로소 실행
    )
    print(f"\n  [polars 지연평가] scan_parquet → filter → group_by 까지 "
          f"{time.perf_counter() - t:.3f}초 (컬럼 3개만 읽음)")
    print("   " + str(out).replace("\n", "\n   "))


# ============================================================================
# 3. [2단계] 결측치 처리 — 문서 §2
# ============================================================================
def analyze_missing(df):
    """결측의 '구조'를 먼저 확인한다. 처리 방식은 여기서 결정된다."""
    na = df[NA_GROUP_COLS].isna()
    all_na = na.all(axis=1)      # 5개가 모두 결측
    any_na = na.any(axis=1)      # 하나라도 결측
    partial = int((any_na & ~all_na).sum())

    print("  [컬럼별 결측]")
    info = pd.DataFrame({
        "결측수": df.isna().sum(),
        "결측%": (df.isna().mean() * 100).round(2),
    })
    print(info[info["결측수"] > 0].to_string())

    # ★ 핵심 판정: '일부만 결측인 행'이 0건인가?
    #   0건이라면 결측이 우연히 흩어진 것(MCAR)일 수 없다. 값을 못 받은 게 아니라
    #   애초에 그 필드를 제출하지 않는 다른 소스의 레코드라는 뜻이다.
    #   → 이 판정 결과에 따라 dropna/fillna를 쓸지 말지가 갈린다.
    print(f"\n  5개 컬럼 동시 결측 : {int(all_na.sum()):,}행")
    print(f"  일부만 결측        : {partial:,}행")
    if partial == 0 and all_na.any():
        print("  → 완전한 구조적 결측(MNAR). 랜덤 결측이 아니므로 삭제·대체 모두 부적절.")

    # ---- 결측행이 '다른 소스'라는 증거 (문서 §2.3) --------------------------
    if "payment_type" in df.columns and all_na.any():
        pt0 = (df.loc[all_na, "payment_type"] == 0).mean()
        print(f"\n  결측행의 payment_type=0 비율 : {pt0:.1%}"
              "   (0은 TLC 코드북에 없는 값)")
        # int()로 변환하지 않으면 numpy 스칼라가 [np.int32(6)] 형태로 출력된다.
        na_v = {int(v) for v in df.loc[all_na, "VendorID"].dropna().unique()}
        ok_v = {int(v) for v in df.loc[~all_na, "VendorID"].dropna().unique()}
        print(f"  결측행에만 있는 VendorID     : {sorted(na_v - ok_v) or '없음'}")
        print(f"  정상행에만 있는 VendorID     : {sorted(ok_v - na_v) or '없음'}")

    # ---- dropna()를 쓰면 무엇이 사라지는가 (문서 §2.4) ----------------------
    # 결측이 시간대·요일에 쏠려 있으면 dropna()는 '심야·주말을 골라 버리는' 처리가 된다.
    # 아래 편향이 확인되면 삭제는 선택지에서 제외한다.
    if all_na.any():
        print("\n  [dropna() 시 사라지는 것 — 요일·시간대 편향]")
        by_dow = all_na.groupby(df[PU].dt.dayofweek).mean()
        names = ["월", "화", "수", "목", "금", "토", "일"]
        print("     요일별 결측 비중 : "
              + " ".join(f"{names[i]} {by_dow.get(i, 0):.1%}" for i in range(7)))
        by_hour = all_na.groupby(df[PU].dt.hour).mean()
        hi, lo = by_hour.idxmax(), by_hour.idxmin()
        print(f"     시간대 최고 {hi:2d}시 {by_hour[hi]:.1%}  /  최저 {lo:2d}시 {by_hour[lo]:.1%}")
        print(f"     삭제 시 표본 : {len(df):,} → {int((~all_na).sum()):,}"
              f" ({(~all_na).mean():.2%} 잔존)")
        print(f"     평균 이동거리: {df['trip_distance'].mean():.3f}"
              f" → {df.loc[~all_na, 'trip_distance'].mean():.3f} (통계량이 실제로 이동)")

    # ---- fillna(0)이 틀린 이유 (문서 §2.5) ---------------------------------
    # total_amount에서 congestion_surcharge를 뺀 나머지 항목 합을 빼면
    # '총액에는 반영됐지만 컬럼에는 안 적힌 금액' = 결측된 실제 값이 나온다.
    # 잔차가 2.50에 몰려 있으면, 결측은 "0원"이 아니라 "부과됐는데 기록 안 된 값"이다.
    parts = ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
             "improvement_surcharge", "Airport_fee", "cbd_congestion_fee"]
    parts = [c for c in parts if c in df.columns]
    if all_na.any() and "total_amount" in df.columns:
        resid = (df.loc[all_na, "total_amount"] - df.loc[all_na, parts].fillna(0).sum(axis=1)).round(2)
        top = resid.value_counts().head(3)
        print(f"\n  [fillna(0) 검증 — 총액 역산으로 구한 실제 결측값]")
        for val, cnt in top.items():
            print(f"     잔차 {val:>6.2f} : {cnt:>9,}건 ({cnt/len(resid):5.1%})")
        real_mean = df[NA_FLAG_COL].mean()          # 결측을 제외한 실측 평균
        zero_mean = df[NA_FLAG_COL].fillna(0).mean()  # 0으로 채웠을 때의 평균
        print(f"     {NA_FLAG_COL} 평균 : 실측 {real_mean:.3f} → fillna(0) 시 {zero_mean:.3f}"
              f" ({(zero_mean/real_mean - 1):+.1%} 왜곡)")
        print("     → 0으로 채우면 실제로 부과된 요금이 0원으로 기록된다. 대체 금지.")

    return all_na


def prepare_missing(df, all_na):
    """문서 §2.7 기준을 적용한다: 삭제·대체 없이 '구분'만 한다."""
    # ---- 기준 ⑤ : 위장 결측(sentinel)을 NaN으로 명시 변환 --------------------
    # payment_type=0, RatecodeID=99 등은 isna()에 안 잡히지만 의미상 결측이다.
    # 변환하지 않으면 payment_type의 최빈값·평균 같은 집계가 조용히 오염된다.
    # 주의: 원본 값을 잃지 않도록 변환 전 건수를 기록해 둔다.
    print("  [기준 ⑤] 위장 결측 → NaN 변환")
    converted = 0
    for col, bad in SENTINELS.items():
        if col not in df.columns:
            continue
        hit = df[col].isin(bad).sum()
        if hit:
            # object가 아닌 수치형 컬럼이므로 NaN으로 두면 dtype이 유지된다.
            df[col] = df[col].where(~df[col].isin(bad), np.nan)
            print(f"     {col:<18} {str(bad):<12} → {hit:>9,}건")
            converted += hit
    print(f"     합계 {converted:,}건 (isna()만으로는 놓쳤을 결측)")

    # ---- 기준 ① : 행 삭제 대신 소스 구분 플래그 -----------------------------
    # 결측행은 '값이 없는 불량 데이터'가 아니라 '필드 구성이 다른 소스의 데이터'다.
    # 지우면 심야·주말·특정 사업자가 통째로 사라지므로(§2.4) 표시만 한다.
    # 이후 승객수·요율 분석은 record_source == "full" 부분집합에서만 수행한다(기준 ④).
    df["record_source"] = np.where(all_na, "partial", "full")
    print(f"\n  [기준 ①] record_source 부여 — 행 삭제 없음")
    print(df["record_source"].value_counts().to_frame("행 수").assign(
        비율=lambda d: (d["행 수"] / len(df)).map("{:.2%}".format)).to_string())

    # ---- 기준 ② : 결측값을 채우지 않는다 ------------------------------------
    print("\n  [기준 ②] fillna 미적용 — 결측은 결측으로 유지")
    print("     요금 총액(total_amount, fare_amount)은 결측 0건이므로 전체 사용 가능(기준 ③)")
    print("     승객수·요율 분석은 record_source == 'full' 로 한정(기준 ④)")
    return df


# ============================================================================
# 4. [3단계] 중복 처리 — 문서 §3
# ============================================================================
def analyze_and_dedup(df):
    """중복을 '지우기 전에' 유형부터 판정한다.

    이 데이터에서 drop_duplicates()를 그냥 쓰면 안 되는 이유가 두 가지다.
      1) 20개 컬럼 완전중복은 0건 → 아무 것도 안 지워지고 '중복 없음'으로 오판
      2) 부분키에 keep='first'를 쓰면 정상 운행을 지우고 취소 기록을 남김(§3.5)
    """
    # ---- 완전중복 확인 (문서 §3.1) ------------------------------------------
    n_exact = int(df.duplicated().sum())
    print(f"  [완전중복] 전체 컬럼 일치 : {n_exact:,}건")
    if n_exact == 0:
        print("     → drop_duplicates()는 0행을 지운다. 부분키로 봐야 중복이 보인다.")

    # ---- 부분키 중복 규모 (문서 §3.2) ---------------------------------------
    dup_mask = df.duplicated(subset=DUP_KEY, keep=False)   # keep=False: 구성원 전부
    n_rows = int(dup_mask.sum())
    if n_rows == 0:
        print("  [부분키 중복] 없음 — 중복 처리 단계를 건너뛴다.")
        return df, {"상쇄쌍": 0, "이중계상": 0, "제거행": 0, "그룹": 0}

    dup = df.loc[dup_mask, DUP_KEY + ["total_amount", "fare_amount", "payment_type"]].copy()
    # ngroup()으로 그룹에 정수 ID를 부여하면 이후 head/tail 연산이 간단해진다.
    dup["gid"] = dup.groupby(DUP_KEY, dropna=False).ngroup()
    # 그룹 내부를 total_amount 오름차순으로 정렬 → head=작은 값, tail=큰 값.
    # 상쇄쌍이면 (음수, 양수), 이중계상이면 (정상, 2배)가 각각 (lo, hi)로 잡힌다.
    dup = dup.sort_values(["gid", "total_amount"])
    lo_idx = dup.groupby("gid").head(1).index
    hi_idx = dup.groupby("gid").tail(1).index
    n_groups = len(lo_idx)
    print(f"\n  [부분키 중복] 키 {DUP_KEY}")
    print(f"     대상 {n_rows:,}행 / {n_groups:,}그룹"
          f" (그룹당 평균 {n_rows/n_groups:.2f}행)")

    lo_tot = df.loc[lo_idx, "total_amount"].to_numpy()
    hi_tot = df.loc[hi_idx, "total_amount"].to_numpy()

    # ---- 유형 판정 (문서 §3.4) ----------------------------------------------
    # ① 상쇄쌍   : 두 행의 합이 0 → 운행이 취소·환불되어 장부상 지워진 쌍
    is_void = np.abs(lo_tot + hi_tot) < VOID_TOL
    # ② 이중계상 : 큰 쪽이 작은 쪽의 정확히 2배 → 같은 운행의 금액이 두 번 더해진 행
    #    0으로 나누는 것을 피하려고 분모의 0을 NaN으로 바꾼다(비교 시 자동 False).
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = hi_tot / np.where(lo_tot == 0, np.nan, lo_tot)
    is_double = ~is_void & (ratio > DOUBLE_LO) & (ratio < DOUBLE_HI)
    is_other = ~(is_void | is_double)

    print(f"\n  [유형 분류]")
    print(f"     ① 상쇄쌍(합계≈0, 취소·환불)   : {int(is_void.sum()):>7,}그룹"
          f" ({is_void.mean():6.1%})")
    print(f"     ② 이중계상(총액 정확히 2배)   : {int(is_double.sum()):>7,}그룹"
          f" ({is_double.mean():6.1%})")
    print(f"     ③ 기타                        : {int(is_other.sum()):>7,}그룹"
          f" ({is_other.mean():6.1%})")

    # ---- keep='first'가 왜 위험한지 실측으로 보여준다 (문서 §3.5) ------------
    # 파일에 저장된 행 순서상 상쇄쌍은 취소 행(음수)이 먼저 오는 경우가 많다.
    # 따라서 keep='first'는 '실제 운행'을 지우고 '취소 기록'을 남긴다.
    naive_drop = df.index[df.duplicated(subset=DUP_KEY, keep="first")]
    n_pos_dropped = int((df.loc[naive_drop, "total_amount"] > 0).sum())
    n_neg_left = int((df.drop(index=naive_drop)["total_amount"] < 0).sum())
    print(f"\n  [대조] drop_duplicates(keep='first')를 썼다면")
    print(f"     지워지는 양수 total_amount : {n_pos_dropped:>9,}건  ← 정상 운행")
    print(f"     남는 음수 total_amount     : {n_neg_left:>9,}건  ← 취소 기록")
    print("     정상 운행을 지우고 취소 기록을 남긴다. 순서 기반 제거는 사용 금지.")

    # ---- 유형별 선택 제거 (문서 §3.6) ---------------------------------------
    # ① 상쇄쌍  → 양쪽 모두 제거. 취소된 운행은 '발생하지 않은 운행'으로 본다.
    #             (한쪽만 남기면 매출 집계가 그 구간에서 음수가 된다)
    # ② 이중계상 → 큰 쪽만 제거, 작은 쪽 유지. 작은 쪽이 동일 포맷 단독행의
    #             total/fare 분포(중앙값 1.231)와 일치하기 때문(§3.4).
    # ③ 기타    → 유지. 서로 다른 운행일 가능성이 높고 규모가 무시 가능하다.
    drop_idx = lo_idx[is_void].union(hi_idx[is_void])      # ① 양쪽
    drop_idx = drop_idx.union(hi_idx[is_double])           # ② 큰 쪽만
    before = len(df)
    df = df.drop(index=drop_idx)

    print(f"\n  [적용] {before:,} → {len(df):,}행  (-{len(drop_idx):,},"
          f" {len(drop_idx)/before:.2%})")
    print(f"     음수 total_amount : {int((df['total_amount'] < 0).sum()):,}건 잔존"
          " (원인 제거 방식이라 이상치 필터보다 정확)")

    stats = {"상쇄쌍": int(is_void.sum()), "이중계상": int(is_double.sum()),
             "제거행": len(drop_idx), "그룹": n_groups}
    return df, stats


# ============================================================================
# 5. [4단계] 이상치 처리 — 문서 §4
# ============================================================================
def filter_outliers(df):
    """물리적으로 불가능하거나 명백한 오기록을 제거한다.

    중복 처리를 '먼저' 끝낸 뒤에 실행해야 한다. 상쇄쌍을 제거하면 음수 금액이
    89% 줄어들기 때문에, 순서를 바꾸면 이상치 필터가 취소 쌍의 한쪽만 지워
    짝이 깨진 반쪽짜리 기록이 남는다.
    """
    month = pd.Period(EXPECTED_MONTH)
    dur = (df[DO] - df[PU]).dt.total_seconds()

    # ############################################################################
    # 🆕🆕🆕 [NEW] 파생변수 speed_kmh 추가 (2026-08-05) 🆕🆕🆕
    # 속력(km/h) = 거리(mile→km 환산) / 소요시간(초→시간 환산)
    # trip_distance는 안 지우고 그대로 둔 채로 speed_kmh를 "새 컬럼"으로 추가함
    # (원본 대체 아님! 파생변수는 추가하는 거지 원본을 없애는 게 아님)
    # duration<=0이면 나누기 에러(inf) 나니까 그 경우만 NaN으로 비워둠
    # → duration<=0인 행은 바로 아래 "소요시간" 규칙에서 이미 걸러지니까
    #   여기서 NaN이어도 아무 문제 없음
    # ############################################################################
    df["speed_kmh"] = np.where(dur > 0, df["trip_distance"] * 1.60934 / (dur / 3600), np.nan)

    rules = [
        # 기간: 승차시각으로만 판정한다. 하차가 6월로 넘어간 785건은
        #       자정을 넘긴 정상 심야운행이므로 하차시각으로 자르면 안 된다.
        ("기간 이탈(승차시각 기준)",
         df[PU].between(month.start_time, month.end_time)),

        # 소요시간: 0초 이하는 물리적으로 불가(52,063건).
        #           상한 6시간은 99.9분위(2.15시간)의 3배 수준으로 매우 관대하다.
        ("소요시간 0초 이하 / 6시간 초과",
         dur.between(DUR_MIN_SEC, DUR_MAX_SEC)),

        # 거리: 0마일 113,031건은 미터기만 켠 기록. 상한 100마일은
        #       최댓값 307,491마일(지구 12바퀴) 같은 오기록을 걸러낸다.
        ("이동거리 0 / 100mile 초과",
         df["trip_distance"].between(DIST_MIN, DIST_MAX)),

        # 금액: 상쇄쌍 제거 후에도 남은 잔여 음수·0원 건.
        ("총액 0 이하 / 요금 음수",
         (df["total_amount"] > 0) & (df["fare_amount"] >= 0)),

        # ==== 🆕🆕🆕 [NEW] 여기부터 속력 기반 규칙 (2026-08-05 추가) 🆕🆕🆕 ===========
        # 거리·시간을 각각 봐서는 못 잡는 "거리는 짧은데 시간도 짧음" 류의
        # 조합형 오기록을 잡는다! speed_kmh가 NaN인 행(duration<=0)은 위
        # 소요시간 규칙에서 이미 제거되므로 여기서는 통과시켜 중복 제거를 피한다.
        ("속력 200km/h 초과",
         df["speed_kmh"].isna() | (df["speed_kmh"] <= SPEED_MAX_KMH)),
        # ================================================================================
    ]

    keep = pd.Series(True, index=df.index)
    prev = len(df)
    for label, cond in rules:
        keep &= cond                    # 누적 적용 — 중복 카운트를 피하기 위함
        now = int(keep.sum())
        print(f"  {label:<30} {now:>10,}행  (-{prev - now:,})")
        prev = now

    out = df[keep]
    print(f"\n  이상치 제거 후 : {len(out):,}행")

    # ---- 편향 자동 점검 -----------------------------------------------------
    # 결측 처리에서 dropna()를 거부한 이유(§2.4)와 똑같은 위험이 이상치 필터에도 있다.
    # "물리적으로 불가능한 값"이 사실은 '특정 사업자의 기록 방식'일 수 있고,
    # 그러면 필터가 그 사업자를 통째로 지운다. 조용히 넘어가면 발견하지 못하므로
    # 필터 전후 VendorID 구성을 비교해 전멸한 사업자를 경고한다.
    wiped = set(df["VendorID"].dropna().unique()) - set(out["VendorID"].dropna().unique())
    if wiped:
        print("\n  [경고] 이상치 필터로 '전량 삭제된' 사업자가 있다")
        for v in sorted(wiped):
            n = int((df["VendorID"] == v).sum())
            sub = df[df["VendorID"] == v]
            sub_dur = (sub[DO] - sub[PU]).dt.total_seconds()
            print(f"     VendorID={int(v)} : {n:,}건 전량 제거"
                  f"  (소요시간<=0 인 비율 {(sub_dur <= 0).mean():.1%})")
        print("     → 값이 이상한 게 아니라 '하차시각을 기록하지 않는 사업자'일 수 있다.")
        print("       소요시간 기반 분석에서만 제외하고 그 외 분석에는 살리려면,")
        print("       삭제 대신 duration_valid 플래그로 다루는 편이 안전하다.")
        print("       (문서 §4 기준의 보완 사항 — §6 한계 5번 참고)")
    return out


# ============================================================================
# 6. [5단계] 기본 EDA
# ============================================================================
def eda(df, raw_stats):
    """정제된 데이터로 기본 EDA를 수행한다.

    ★ 모든 집계에서 결측 소스를 어떻게 다뤘는지 명시한다.
      record_source='partial'은 승객수·요율 컬럼이 아예 없으므로, 해당 컬럼을
      쓰는 지표는 'full' 한정으로 계산한다(문서 §2.7 기준 ④).
      반대로 요금·거리·시각은 결측이 0건이라 전체를 쓴다(기준 ③).
    """
    full = df[df["record_source"] == "full"]

    # ---- 5-1. 수치형 기술통계 (전체 사용 — 기준 ③) --------------------------
    print("  [5-1] 수치형 기술통계 (전체 행)")
    cols = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
    desc = df[cols].describe(percentiles=[.25, .5, .75, .95, .99]).T
    print(desc.to_string(float_format=lambda x: f"{x:,.2f}"))
    # 평균÷중앙값으로 쏠림을 본다. 표준편차는 이상치에 같이 부풀려져 기준이 안 된다.
    print("\n     평균/중앙값 비율 (1.5 초과 = 오른쪽 꼬리가 긴 분포):")
    for c in cols:
        med = desc.loc[c, "50%"]
        if med > 0:
            r = desc.loc[c, "mean"] / med
            mark = "  ← 평균만 보면 안 됨" if r > 1.5 else ""
            print(f"       {c:<16} {r:>5.2f}배{mark}")

    # ---- 5-2. 소스별 비교 (결측 처리 판단의 사후 검증) -----------------------
    # 여기서 두 소스의 프로파일이 크게 다르면, dropna()를 안 한 판단이 옳았다는
    # 사후 증거가 된다. 같았다면 굳이 나눌 필요가 없었다는 뜻이기도 하다.
    print("\n  [5-2] record_source별 프로파일 — 결측행을 남긴 판단의 검증")
    prof = df.groupby("record_source").agg(
        건수=("total_amount", "size"),
        평균거리=("trip_distance", "mean"),
        평균요금=("fare_amount", "mean"),
        평균팁=("tip_amount", "mean"),
        평균총액=("total_amount", "mean"),
    )
    prof["비율"] = (prof["건수"] / len(df)).map("{:.1%}".format)
    print(prof.to_string(float_format=lambda x: f"{x:,.2f}"))
    if len(prof) > 1:
        print("     → 두 소스의 평균 팁·거리가 크게 다르다. 섞어서 평균 내면 왜곡되고,")
        print("       한쪽을 지우면 표본이 편향된다. 구분 보존이 유일한 안전한 선택.")

    # ---- 5-3. 시간대별 운행 패턴 (전체 사용) --------------------------------
    print("\n  [5-3] 시간대별 운행량·평균요금")
    by_hour = df.groupby(df[PU].dt.hour).agg(
        건수=("total_amount", "size"), 평균요금=("fare_amount", "mean"),
        평균거리=("trip_distance", "mean"))
    peak, low = by_hour["건수"].idxmax(), by_hour["건수"].idxmin()
    for h in range(0, 24, 3):
        if h not in by_hour.index:      # 운행이 한 건도 없는 시간대 방어
            continue
        bar = "█" * int(by_hour.loc[h, "건수"] / by_hour["건수"].max() * 34)
        print(f"     {h:02d}시 {int(by_hour.loc[h, '건수']):>8,}  "
              f"요금 {by_hour.loc[h, '평균요금']:>6.2f}  {bar}")
    print(f"     최다 {peak}시({by_hour.loc[peak, '건수']:,}건) / "
          f"최소 {low}시({by_hour.loc[low, '건수']:,}건)")

    # ---- 5-4. 요일별 패턴 ----------------------------------------------------
    print("\n  [5-4] 요일별 운행량·평균요금")
    names = ["월", "화", "수", "목", "금", "토", "일"]
    by_dow = df.groupby(df[PU].dt.dayofweek).agg(
        건수=("total_amount", "size"), 평균요금=("fare_amount", "mean"),
        평균팁=("tip_amount", "mean"))
    for i, row in by_dow.iterrows():
        # row는 여러 dtype이 섞인 Series라 건수가 float으로 승격된다. int로 되돌린다.
        print(f"     {names[i]}요일 {int(row['건수']):>8,}건  평균요금 {row['평균요금']:>6.2f}"
              f"  평균팁 {row['평균팁']:>5.2f}")

    # ---- 5-5. 승하차 존 TOP (전체 사용 — 단, sentinel 변환으로 264/265는 제외됨)
    print("\n  [5-5] 승차 존 TOP 10 (Unknown/구역외는 §2.6에서 NaN 처리되어 자동 제외)")
    top_pu = df["PULocationID"].value_counts().head(10)
    for zone, cnt in top_pu.items():
        print(f"     존 {int(zone):>3d} : {cnt:>9,}건 ({cnt/len(df):5.2%})")

    # ---- 5-6. 범주형 분포 (full 한정 — 기준 ④) ------------------------------
    print(f"\n  [5-6] 범주형 분포 — record_source='full' {len(full):,}행 한정")
    print("        (partial 소스는 해당 컬럼 자체가 없어 분모에 넣으면 오염됨)")
    for col in CATEGORICAL_COLS:
        if col not in full.columns:
            continue
        vc = full[col].value_counts(dropna=False).head(5)
        print(f"\n     -- {col} (고유값 {full[col].nunique()}개)")
        for value, cnt in vc.items():
            label = "결측" if pd.isna(value) else str(value)
            print(f"        {label:>8} : {cnt:>9,}건 ({cnt/len(full):5.1%})")

    # ---- 5-7. 요금 구조 ------------------------------------------------------
    print("\n  [5-7] 거리-요금 관계")
    corr = df[["trip_distance", "fare_amount", "total_amount", "tip_amount"]].corr()
    print(corr.to_string(float_format=lambda x: f"{x:6.3f}"))
    # 마일당 요금은 단거리에서 기본요금 때문에 급등한다. 구간별로 나눠 봐야 의미가 있다.
    # ★ 평균이 아니라 중앙값을 쓴다: 0.01마일에 요금 8달러 같은 행이 섞이면
    #   비율의 평균이 800/mile로 튀어 구간 대표값 역할을 못 한다.
    bins = [0, 1, 2, 5, 10, 100]
    df_b = df.assign(거리구간=pd.cut(df["trip_distance"], bins),
                     마일당=df["fare_amount"] / df["trip_distance"])
    by_bin = df_b.groupby("거리구간", observed=True).agg(
        건수=("fare_amount", "size"), 평균요금=("fare_amount", "mean"),
        마일당요금_중앙=("마일당", "median"))
    print("\n     거리 구간별 요금 (마일당 요금은 중앙값)")
    print(by_bin.to_string(float_format=lambda x: f"{x:,.2f}"))

    # ---- 5-8. 정제 효과 요약 -------------------------------------------------
    print("\n  [5-8] 정제 전후 통계량 변화")
    print(f"     {'지표':<14}{'원본':>14}{'정제 후':>14}   해석")
    for label, key, now in [
        ("행 수", "행수", len(df)),
        ("평균 이동거리", "평균거리", df["trip_distance"].mean()),
        ("중앙 이동거리", "중앙거리", df["trip_distance"].median()),
        ("평균 요금", "평균요금", df["fare_amount"].mean()),
        ("평균 총액", "평균총액", df["total_amount"].mean()),
    ]:
        before = raw_stats[key]
        fmt = ",.0f" if key == "행수" else ",.3f"
        note = ""
        if key == "평균거리":
            note = "극단 이상치 제거 효과"
        elif key == "평균요금":
            note = "거의 불변 = 과도한 삭제가 아님"
        print(f"     {label:<14}{before:>14{fmt}}{now:>14{fmt}}   {note}")


# ============================================================================
# 7. [6단계] 문서 기대값과 대조
# ============================================================================
def verify(observed):
    """문서에 기록된 수치와 이번 실행 결과가 일치하는지 확인한다.

    다른 월·다른 파일로 돌리면 당연히 불일치한다. 그 경우 '기준 데이터가 아님'을
    바로 알 수 있도록, 틀렸다고 단정하지 않고 차이만 표시한다.
    """
    print(f"  {'항목':<16}{'문서 기대값':>14}{'이번 실행':>14}   판정")
    all_ok = True
    for k, exp in EXPECTED.items():
        got = observed.get(k)
        if got is None:
            continue
        ok = got == exp
        all_ok &= ok
        print(f"  {k:<16}{exp:>14,}{got:>14,}   {'일치' if ok else '차이'}")
    if all_ok:
        print("\n  전 항목 일치 — 문서의 분석이 이 코드로 재현된다.")
    else:
        print("\n  일부 항목이 다르다. 기준 데이터(2026-05)가 아닌 파일을 읽었는지 확인하라.")


# ============================================================================
# 8. 메인
# ============================================================================
def main():
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    save = "--save" in argv
    want_rss = "--rss" in argv

    # --bench-rss <lib> : measure_peak_rss()가 띄우는 자식 프로세스 전용 모드.
    # 라이브러리 하나만 로딩하고 최대 RSS를 찍은 뒤 즉시 종료한다.
    if "--bench-rss" in argv:
        lib = argv[argv.index("--bench-rss") + 1]
        p = resolve_path(args)
        load_with_polars(p) if lib == "polars" else load_with_pandas(p)
        print(f"{peak_rss_mb():.1f}")
        return

    path = resolve_path(args)

    print_step(f"[0] 파일 검증 : {path.name}")
    validate_file(path)
    print("  검증 통과 (존재 · 확장자 · 필수 컬럼)")

    # ------------------------------------------------------------------ [1]
    print_step("[1] pandas vs polars 로딩 비교")
    print("  polars 먼저 읽고 지표만 추출한 뒤 해제한다.")
    print("  (동시에 들고 있으면 최대 RSS가 4GB를 넘고 두 로딩이 서로 간섭한다)")
    print("  측정 전 페이지 캐시를 데운다 — 먼저 재는 쪽만 디스크를 읽으면 불공정하다.\n")
    warm_page_cache(path)
    pl_metrics, pl_nulls = load_with_polars(path)
    df, pd_metrics = load_with_pandas(path)

    rss = None
    if want_rss:
        # 파일을 두 번 더 읽으므로 명시적으로 요청했을 때만 수행한다.
        rss = {lib: measure_peak_rss(path, lib) for lib in ("polars", "pandas")}
    report_loading(pl_metrics, pd_metrics, pl_nulls, df, path, rss)
    demo_polars_lazy(path)

    # 정제 전 기준값을 미리 저장해 둔다([5-8]에서 전후 비교에 사용)
    raw_stats = {
        "행수": len(df),
        "평균거리": df["trip_distance"].mean(),
        "중앙거리": df["trip_distance"].median(),
        "평균요금": df["fare_amount"].mean(),
        "평균총액": df["total_amount"].mean(),
    }
    observed = {"총 행 수": len(df), "완전중복": pd_metrics["완전중복"]}

    # ------------------------------------------------------------------ [2]
    print_step("[2] 결측치 처리  (문서 §2)")
    all_na = analyze_missing(df)
    observed["결측 행 수"] = int(all_na.sum())
    print()
    df = prepare_missing(df, all_na)

    # ------------------------------------------------------------------ [3]
    print_step("[3] 중복 처리  (문서 §3)")
    df, dup_stats = analyze_and_dedup(df)
    observed["중복키 그룹"] = dup_stats["그룹"]
    observed["상쇄쌍(그룹)"] = dup_stats["상쇄쌍"]
    observed["이중계상(행)"] = dup_stats["이중계상"]

    # ------------------------------------------------------------------ [4]
    print_step("[4] 이상치 처리  (문서 §4)")
    df = filter_outliers(df)
    observed["최종 행 수"] = len(df)
    print(f"  최종 보존율 : {len(df)/raw_stats['행수']:.2%}")

    # ------------------------------------------------------------------ [5]
    print_step("[5] 기본 EDA")
    eda(df, raw_stats)

    # ------------------------------------------------------------------ [6]
    print_step("[6] 문서 기대값 대조")
    verify(observed)

    # ------------------------------------------------------------------ 저장
    if save:
        # 입력과 같은 parquet으로 저장한다. record_source 같은 파생 컬럼과
        # datetime 타입이 그대로 보존되어, 다음 분석 스크립트가 타입 지정 없이
        # 이어받을 수 있다. data/raw는 원본 전용이므로 outputs/에 따로 쓴다.
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"yellow_{EXPECTED_MONTH}_clean.parquet"
        df.to_parquet(out, index=False)
        print(f"\n  저장 완료 : {out}  ({human_bytes(out.stat().st_size)})")

    print(f"\n{LINE}\n완료 — 정제 {len(df):,}행\n{LINE}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n[오류] {e}", file=sys.stderr)
        print(f"실행 예: python taxi_eda_pipeline.py {DEFAULT_PATH.name}", file=sys.stderr)
        sys.exit(1)
    except MemoryError:
        print("\n[오류] 메모리 부족. pandas 로딩에만 약 1.6GB가 필요합니다.\n"
              "       polars 지연 평가(scan_parquet)로 필요한 컬럼만 읽는 방식을 검토하세요.",
              file=sys.stderr)
        sys.exit(1)
