"""TOML 설정 파일을 파이프라인 설정 객체로 변환합니다."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Columns:
    """컬럼의 '역할'을 이름과 분리한다.

    코드가 "tpep_pickup_datetime"을 직접 참조하면 컬럼명이 바뀔 때마다
    모든 파일을 고쳐야 한다. 역할로 참조하면 설정 한 줄만 바꾸면 된다.
    """

    pickup: str
    dropoff: str
    vendor: str
    missing_group: list[str]
    missing_flag: str
    amount_parts: list[str]
    categorical: list[str]


@dataclass(frozen=True)
class DuplicateRules:
    """중복 판정 기준 (문서 §3).

    key            : 중복으로 볼 컬럼 조합. 좁으면 우연 충돌, 넓히면 탐지 불가.
    void_tolerance : 두 행의 금액 합이 이만큼 이내면 상쇄쌍(취소·환불)으로 본다.
    double_ratio_* : 총액 비율이 이 범위면 이중계상으로 본다.
    """

    key: list[str]
    void_tolerance: float
    double_ratio_min: float
    double_ratio_max: float


@dataclass(frozen=True)
class OutlierRules:
    """이상치 판정 기준 (문서 §4).

    duration_policy 가 'drop'이면 소요시간 이상 행을 제거하고,
    'flag'면 duration_valid 컬럼만 달고 행은 남긴다. 소요시간 규칙이
    특정 사업자를 통째로 지우기 때문에 목적에 따라 선택해야 한다.
    """

    duration_min_sec: int
    duration_max_sec: int
    distance_min: float
    distance_max: float
    duration_policy: str          # "drop" | "flag"

    def __post_init__(self):
        """설정 오타를 로딩 시점에 잡는다.

        파이프라인 한복판에서 알 수 없는 정책 문자열로 실패하는 것보다
        시작하자마자 명확히 죽는 편이 낫다.
        """
        if self.duration_policy not in ("drop", "flag"):
            raise ValueError(
                f"duration_policy는 'drop' 또는 'flag'만 가능합니다: {self.duration_policy}")


@dataclass(frozen=True)
class VisualizeConfig:
    """차트 생성 설정 (문서 02_시각화_근거.md)."""

    font_candidates: list[str]
    dpi: int
    figsize: list[int]
    distance_xlim: float


@dataclass(frozen=True)
class StatisticsConfig:
    """통계분석 설정 (문서 03_통계분석_근거.md)."""

    alpha: float
    equal_var: bool
    describe_columns: list[str]
    percentiles: list[float]
    correlation_columns: list[str]
    long_trip_threshold: float


@dataclass(frozen=True)
class ModelConfig:
    """ML 파이프라인 설정 (문서 04_ML파이프라인_근거.md)."""

    target_tip_rate: float
    payment_type: int
    train_sample: int
    test_size: float
    random_state: int
    min_frequency: float


@dataclass(frozen=True)
class Expectations:
    """품질 게이트 기대값.

    exact : 기준 데이터에서 재현돼야 하는 정확한 값. 로직 회귀를 잡는다.
    range : 월이 달라도 통하는 일반 규칙. 데이터 자체의 이상을 잡는다.
    check_exact=False 로 두면 exact 검사를 건너뛴다(다른 월 처리 시).
    """

    check_exact: bool
    exact: dict[str, int]
    range: dict[str, dict[str, float]]


@dataclass(frozen=True)
class Paths:
    """입출력 경로. 모두 프로젝트 루트 기준 절대경로로 해석된 값이다.

    스케줄러가 임의의 작업 디렉터리에서 실행해도 경로가 흔들리지 않게 한다.
    """

    root: Path
    raw: Path
    interim: Path
    processed: Path
    runs: Path


@dataclass(frozen=True)
class Source:
    """원본 데이터 출처. 입력 파일이 없을 때 여기서 받는다."""

    url_template: str
    auto_download: bool
    timeout_sec: int


@dataclass(frozen=True)
class Config:
    """파이프라인 전체 설정.

    모든 단계가 이 객체 하나만 받는다. frozen=True로 굳혀 실행 도중
    설정이 바뀌는 일이 없게 한다 — 중간에 값이 달라지면 리포트의
    "이 기준으로 처리했다"는 서술이 거짓이 된다.
    """

    name: str
    month: str
    paths: Paths
    source: Source
    columns: Columns
    visualize: VisualizeConfig
    statistics: StatisticsConfig
    model: ModelConfig
    duplicates: DuplicateRules
    outliers: OutlierRules
    sentinels: dict[str, list[int]]
    expectations: Expectations
    source_file: Path
    digest: str = field(compare=False)     # 설정 내용 해시 (매니페스트 기록용)


def load_config(path: str | Path, root: Path | None = None) -> Config:
    """TOML을 읽어 Config를 만든다.

    root: 상대경로의 기준점. 기본은 설정 파일의 상위 폴더(= 프로젝트 루트).
          스케줄러가 임의의 작업 디렉터리에서 실행해도 경로가 흔들리지 않는다.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")

    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    root = Path(root).resolve() if root else path.parent.parent
    p = raw["paths"]
    paths = Paths(
        root=root,
        raw=root / p["raw"],
        interim=root / p["interim"],
        processed=root / p["processed"],
        runs=root / p["runs"],
    )

    # [source]는 나중에 추가된 섹션이라, 없는 설정 파일도 그대로 돌아가게 기본값을 둔다.
    # auto_download 기본을 False로 두는 이유: 설정에 명시하지 않은 파이프라인이
    # 갑자기 66MB를 내려받는 것은 놀라운 동작이다. 받으려면 명시해야 한다.
    src = raw.get("source", {})
    source = Source(
        url_template=src.get("url_template", ""),
        auto_download=bool(src.get("auto_download", False)),
        timeout_sec=int(src.get("timeout_sec", 120)),
    )

    exp = raw["expectations"]
    return Config(
        name=raw["project"]["name"],
        month=raw["project"]["month"],
        paths=paths,
        source=source,
        columns=Columns(**raw["columns"]),
        visualize=VisualizeConfig(**raw["visualize"]),
        statistics=StatisticsConfig(**raw["statistics"]),
        model=ModelConfig(**raw["model"]),
        duplicates=DuplicateRules(**raw["duplicates"]),
        outliers=OutlierRules(**raw["outliers"]),
        sentinels=raw["sentinels"],
        expectations=Expectations(
            check_exact=exp.get("check_exact", True),
            exact=exp.get("exact", {}),
            range=exp.get("range", {}),
        ),
        source_file=path,
        # sort_keys=True: dict 순서가 바뀌어도 같은 내용이면 같은 해시가 나오도록
        digest=hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
    )
