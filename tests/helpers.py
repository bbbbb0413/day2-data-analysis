"""테스트가 공유하는 설정과 입력 생성기.

409만 행을 읽지 않고 로직만 검증한다. 단계를 (DataFrame, Config) -> StepResult
순수 함수로 만든 진짜 이유가 이것이다 — 파일 IO와 출력이 섞여 있으면 작은
입력으로 로직만 떼어 검증할 수 없다.

여기서 잡고 싶은 사고:
  - 리팩터링하다 중복·이상치 판정 기준이 미묘하게 바뀌는 것
  - keep='first' 같은 '편한 방법'이 슬쩍 들어오는 것
  - sentinel 목록에서 값이 빠지는 것
  - 리포트 섹션이 예외 없이 조용히 비는 것

## 네이밍 규칙

**테스트 함수 이름은 한글로 쓰고, 프로덕션 코드 식별자는 ASCII만 쓴다.**

테스트 이름은 다른 코드가 호출하지 않는 '실행되는 문서'다. 무엇을 보장하는지
그대로 적는 편이 읽기 쉽다(PEP 3131이 허용하며 pytest는 `test_` 접두사로 찾는다).
반대로 프로덕션 식별자를 한글로 쓰면 로케일이 깨진 콘솔의 traceback, 비한국어
협업자, `pytest -k` 필터 입력에서 비용만 생긴다. `taxi_pipeline/`과
`run_pipeline.py`에는 한글 식별자가 0건이며 이 상태를 유지한다.

## 실행

    .venv/bin/python -m pytest tests/ -v
    .venv/bin/python tests/run_all.py      # pytest가 없는 환경
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxi_pipeline.config import load_config    # noqa: E402

CFG = load_config(ROOT / "config" / "pipeline.toml", root=ROOT)


def row(pickup, dropoff, pu=100, do=200, dist=1.0, fare=10.0, total=15.0,
        vendor=2, payment=1, passengers=1.0, rate=1.0, cong=2.5, air=0.0):
    """테스트용 행 하나. 필요한 컬럼만 채운다."""
    return {
        "VendorID": vendor,
        "tpep_pickup_datetime": pd.Timestamp(pickup),
        "tpep_dropoff_datetime": pd.Timestamp(dropoff),
        "passenger_count": passengers,
        "trip_distance": dist,
        "RatecodeID": rate,
        "store_and_fwd_flag": "N",
        "PULocationID": pu,
        "DOLocationID": do,
        "payment_type": payment,
        "fare_amount": fare,
        "extra": 0.0,
        "mta_tax": 0.5,
        "tip_amount": 0.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "total_amount": total,
        "congestion_surcharge": cong,
        "Airport_fee": air,
        "cbd_congestion_fee": 0.0,
    }


def frame(rows):
    """행 dict 목록을 DataFrame으로 만든다."""
    return pd.DataFrame(rows)
