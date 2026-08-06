"""단계 레지스트리 — 파이프라인의 목차.

여기 리스트만 고치면 순서 변경·단계 추가가 끝난다. runner는 이 목록을 돌 뿐
어떤 단계가 있는지 모른다. 새 단계(예: 좌표 보정, 요금 재계산)를 붙일 때
runner를 건드릴 필요가 없다는 뜻이다.

순서에 의존성이 있는 곳은 한 군데다:
  duplicates → outliers
  상쇄쌍을 먼저 지워야 음수 금액이 89% 줄어든다. 순서를 바꾸면 이상치 필터가
  취소 쌍의 한쪽만 지워 짝이 깨진 반쪽짜리 기록이 남는다.
"""

from __future__ import annotations

from .base import Artifact, Step, StepFn, StepResult
from .duplicates import deduplicate
from .features import engineer_features  # 🆕 [2026-08-06/유길선]
from .feature_validation import validate_features  # 🆕 [2026-08-06/유길선]
from .loaders import compare_loaders
from .model import train_model
from .missing import analyze_missing, prepare_missing
from .outliers import filter_outliers
from .statistics import statistics_step
from .visualize import visualize_step

PIPELINE: list[Step] = [
    # 가장 앞에 두는 이유: 이후 모든 분석이 "도구와 무관한 데이터의 성질"임을
    # 먼저 확인해야 뒤의 결과를 신뢰할 수 있다.
    Step("compare_loaders", compare_loaders,
         "Pandas·Polars 로딩 결과 비교", mutates=False),
    Step("analyze_missing", analyze_missing,
         "결측 구조 진단 (삭제·대체가 왜 안 되는지 근거 수집)", mutates=False),
    Step("prepare_missing", prepare_missing,
         "위장 결측 변환 + record_source 플래그 (행 삭제 없음)"),
    Step("deduplicate", deduplicate,
         "중복 유형 판정 후 선택 제거 (상쇄쌍 양쪽 / 이중계상 큰 쪽)"),
    Step("filter_outliers", filter_outliers,
         "기간·소요시간·거리·금액 이상치 제거"),
    # ################################################################################
    # 🆕🆕🆕 [신규 2026-08-06 / 유길선 —통계분석에 파생변수를 포함시키려면 statistics 전에
    # 어딘가 끼워 넣어야 해서 최소한으로 여기만 건드림. 위치를 바꾸고 싶으면 
    # 상의 후 옮기면 됨 — filter_outliers 이후(정제 끝난 데이터 필요) ~ visualize/
    # statistics 이전(새 컬럼을 봐야 함) 사이 어디든 상관없음. 🆕🆕🆕
    # ################################################################################
    Step("engineer_features", engineer_features,
         "파생변수 추가: is_rush_hour · is_airport_trip (speed_kmh는 이상치 단계에서 이미 생성)"),
    # 🆕 [2026-08-06/유길선] statistics_step()의 "검정은 하나만 한다" 원칙을 지키려고
    # 새 파생변수 검증(t-test 2개)은 별도 단계로 분리했다. DataFrame은 안 바꾼다.
    Step("validate_features", validate_features,
         "파생변수 보조 검증: is_airport_trip·is_rush_hour 그룹 차이 t-test", mutates=False),
    # 시각화는 정제가 끝난 데이터를 그린다. DataFrame을 바꾸지 않으므로
    # profile과 순서를 바꿔도 결과가 같다.
    Step("visualize", visualize_step,
         "Seaborn 정적 7개 + Plotly 인터랙티브 3개 차트 생성", mutates=False),
    # 통계분석도 정제 데이터를 읽기만 한다. visualize와 서로 독립이라
    # 순서를 바꿔도 결과가 같다(상관계수를 각자 계산하는 이유 — 문서 03 §S-3).
    Step("statistics", statistics_step,
         "기술통계·상관계수·t-test와 p-value 해석", mutates=False),
    # 모델도 정제 데이터를 읽기만 한다. 학습·평가 후 joblib으로 저장한다.
    Step("model", train_model,
         "Pipeline으로 전처리+모델 학습, 평가 지표 출력, joblib 저장", mutates=False),
]

STEPS: dict[str, Step] = {s.name: s for s in PIPELINE}

__all__ = ["PIPELINE", "STEPS", "Artifact", "Step", "StepFn", "StepResult"]
