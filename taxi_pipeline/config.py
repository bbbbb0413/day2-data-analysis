"""TOML 설정을 파이프라인 설정 객체로 변환한다."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Columns:
    """컬럼별 역할을 설정값으로 관리한다."""

    pickup: str
    dropoff: str
    vendor: str
    missing_group: list[str]
    missing_flag: str
    amount_parts: list[str]
    categorical: list[str]


@dataclass(frozen=True)
class DuplicateRules:
    """중복 판정에 사용할 기준을 저장한다."""

    key: list[str]
    void_tolerance: float
    double_ratio_min: float
    double_ratio_max: float


@dataclass(frozen=True)
class OutlierRules:
    """이상치 판정 기준과 소요시간 처리 방식을 저장한다."""

    duration_min_sec: int
    duration_max_sec: int
    distance_min: float
    distance_max: float
    speed_max_kmh: float          # 속력(km/h) 상한. 거리·시간 조합형 오기록을 잡는다.
    duration_policy: str          # "drop" 또는 "flag"를 사용한다.

    def __post_init__(self):
        """duration_policy 값이 올바른지 확인한다."""
        if self.duration_policy not in ("drop", "flag"):
            raise ValueError(
                f"duration_policy는 'drop' 또는 'flag'만 가능합니다: {self.duration_policy}")


@dataclass(frozen=True)
class VisualizeConfig:
    """차트 생성 설정을 저장한다."""

    font_candidates: list[str]
    dpi: int
    figsize: list[int]
    distance_xlim: float


@dataclass(frozen=True)
class StatisticsConfig:
    """통계분석 설정을 저장한다."""

    alpha: float
    equal_var: bool
    describe_columns: list[str]
    percentiles: list[float]
    correlation_columns: list[str]
    long_trip_threshold: float


@dataclass(frozen=True)
class ModelConfig:
    """모델 학습 설정을 저장한다."""

    target_tip_rate: float
    payment_type: int
    train_sample: int
    test_size: float
    random_state: int
    min_frequency: float


@dataclass(frozen=True)
class Expectations:
    """품질 게이트의 정확값과 허용 범위를 저장한다."""

    check_exact: bool
    exact: dict[str, int]
    range: dict[str, dict[str, float]]


@dataclass(frozen=True)
class Paths:
    """프로젝트에서 사용하는 입출력 경로를 저장한다."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    runs: Path


@dataclass(frozen=True)
class Source:
    """원본 데이터 다운로드 설정을 저장한다."""

    url_template: str
    auto_download: bool
    timeout_sec: int


@dataclass(frozen=True)
class Config:
    """파이프라인 전체 설정을 저장한다."""

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
    digest: str = field(compare=False)     # 설정 내용 해시이다.


def load_config(path: str | Path, root: Path | None = None) -> Config:
    """TOML 파일을 읽어 Config 객체를 만든다."""
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

    # source 설정이 없으면 자동 다운로드를 끈 기본값을 사용한다.
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
        # 키 순서가 달라도 같은 내용이면 같은 해시를 만든다.
        digest=hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
    )
