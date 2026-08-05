"""파이프라인 단계의 등록 순서를 정의합니다."""

from __future__ import annotations

from .base import Artifact, Step, StepFn, StepResult
from .duplicates import deduplicate
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
