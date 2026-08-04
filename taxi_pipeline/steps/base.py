"""단계(Step)의 계약 — 파이프라인이 조립 가능해지는 지점.

모든 단계가 같은 모양을 갖도록 강제한다.

    (DataFrame, Config) -> StepResult(DataFrame, metrics, notes)

이 계약이 자동화에 주는 것:
  - 순서 변경·부분 실행이 가능하다. runner가 단계를 리스트로 다루기만 하면 된다.
  - 각 단계를 독립적으로 테스트할 수 있다. 작은 DataFrame을 넣고 metrics를 검사한다.
  - 지표가 반환값이라 로그를 파싱할 필요가 없다. 게이트·대시보드가 그대로 쓴다.

부수효과(파일 쓰기·print)를 단계 안에 두지 않는 것이 핵심이다.
저장은 runner가, 출력은 report가 담당한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ..config import Config


@dataclass
class StepResult:
    """단계 하나의 산출물.

    df      : 다음 단계로 넘길 DataFrame (분석만 하는 단계는 입력을 그대로 반환)
    metrics : 기계가 읽는 지표. 품질 게이트와 대시보드가 이것만 본다.
    notes   : 사람이 읽는 근거 문장. 리포트에 그대로 실린다.
              "왜 이렇게 처리했는가"를 코드 주석이 아니라 산출물에 남기기 위함이다.
    """

    df: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# 단계 함수의 타입. 이 시그니처를 따르면 어떤 함수든 파이프라인에 꽂을 수 있다.
StepFn = Callable[[pd.DataFrame, Config], StepResult]


@dataclass(frozen=True)
class Step:
    """등록된 단계 하나.

    name        : CLI에서 --steps 로 지정하는 이름, 산출물 파일명에도 쓰인다
    fn          : 실제 처리 함수
    description : 리포트·도움말에 표시할 한 줄 설명
    mutates     : DataFrame을 바꾸는가. False면 분석 전용(체크포인트 저장 불필요)
    """

    name: str
    fn: StepFn
    description: str
    mutates: bool = True

    def __call__(self, df: pd.DataFrame, cfg: Config) -> StepResult:
        return self.fn(df, cfg)
