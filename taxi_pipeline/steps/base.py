"""파이프라인 단계와 결과에 사용하는 공통 타입을 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import Config


@dataclass
class Artifact:
    """단계가 만든 파일 산출물(차트·모델 등). **저장 위치는 runner가 정한다.**

    단계가 직접 파일을 쓰지 않게 하려고 이 형태를 쓴다.
    단계는 "이런 이름의 산출물이 있고, 저장하려면 이 함수를 부르면 된다"만 알려주고,
    실제 경로(run_id 디렉터리)는 runner가 결정한다.

    이렇게 하면
      - 단계는 여전히 순수 함수로 남아 작은 입력으로 테스트할 수 있다
      - 실행마다 산출물이 자동으로 격리된다 (덮어쓰기 사고 방지)
      - 저장 경로 규칙이 바뀌어도 단계 코드를 고치지 않는다

    name    : 파일명 (경로 아님). 예: "distance_distribution.png"
    save    : 경로를 받아 파일을 쓰는 함수
    caption : 리포트에 그림과 함께 실릴 설명. 그림만 넣으면 무슨 뜻인지 알 수 없다.
    kind    : "figure" | "model" 등. 리포트가 표시 방식을 정할 때 쓴다.
    """

    name: str
    save: Callable[[Path], None]
    caption: str = ""
    kind: str = "figure"


@dataclass
class StepResult:
    """단계 하나의 산출물.

    df        : 다음 단계로 넘길 DataFrame (분석만 하는 단계는 입력을 그대로 반환)
    metrics   : 기계가 읽는 지표. 품질 게이트와 대시보드가 이것만 본다.
    notes     : 사람이 읽는 근거 문장. 리포트에 그대로 실린다.
                "왜 이렇게 처리했는가"를 코드 주석이 아니라 산출물에 남기기 위함이다.
    artifacts : 파일로 저장할 산출물. runner가 run_id 디렉터리에 기록한다.
    """

    df: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)


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
        """runner가 단계를 함수처럼 호출할 수 있게 한다."""
        return self.fn(df, cfg)
