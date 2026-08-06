"""파이프라인 단계와 결과에 사용할 공통 타입을 정의한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import Config


@dataclass
class Artifact:
    """단계에서 생성한 파일 산출물과 저장 함수를 관리한다."""

    name: str
    save: Callable[[Path], None]
    caption: str = ""
    kind: str = "figure"


@dataclass
class StepResult:
    """단계 실행 후 반환할 데이터와 지표를 저장한다."""

    df: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)


# 모든 단계 함수는 같은 입력과 반환 형식을 사용한다.
StepFn = Callable[[pd.DataFrame, Config], StepResult]


@dataclass(frozen=True)
class Step:
    """파이프라인에 등록할 단계 정보를 저장한다."""

    name: str
    fn: StepFn
    description: str
    mutates: bool = True

    def __call__(self, df: pd.DataFrame, cfg: Config) -> StepResult:
        """등록된 처리 함수를 호출한다."""
        return self.fn(df, cfg)
