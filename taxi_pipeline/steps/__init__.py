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

from .base import Step, StepFn, StepResult
from .duplicates import deduplicate
from .missing import analyze_missing, prepare_missing
from .outliers import filter_outliers
from .profile import profile

PIPELINE: list[Step] = [
    Step("analyze_missing", analyze_missing,
         "결측 구조 진단 (삭제·대체가 왜 안 되는지 근거 수집)", mutates=False),
    Step("prepare_missing", prepare_missing,
         "위장 결측 변환 + record_source 플래그 (행 삭제 없음)"),
    Step("deduplicate", deduplicate,
         "중복 유형 판정 후 선택 제거 (상쇄쌍 양쪽 / 이중계상 큰 쪽)"),
    Step("filter_outliers", filter_outliers,
         "기간·소요시간·거리·금액 이상치 제거"),
    Step("profile", profile,
         "정제 결과 기본 EDA (지표만 반환)", mutates=False),
]

STEPS: dict[str, Step] = {s.name: s for s in PIPELINE}

__all__ = ["PIPELINE", "STEPS", "Step", "StepFn", "StepResult"]
