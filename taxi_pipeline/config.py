"""설정 로딩 — 모든 임계값은 코드가 아니라 config/pipeline.toml 에 있다.

자동화 관점에서 중요한 점 두 가지:
  1) 설정을 dataclass로 굳혀 오타를 로딩 시점에 잡는다.
     cfg.outliers.duration_max_sec 를 잘못 쓰면 즉시 AttributeError가 난다.
     dict로 들고 다니면 오타가 KeyError로 파이프라인 한복판에서 터진다.
  2) 설정 해시를 남긴다. 산출물이 어떤 기준으로 만들어졌는지 나중에 증명해야
     하는데, 파일 경로만 기록하면 그 사이 파일이 바뀌었는지 알 수 없다.

TOML을 쓰는 이유: Python 3.11부터 tomllib가 표준 라이브러리라 의존성이 늘지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Columns:
    pickup: str
    dropoff: str
    vendor: str
    missing_group: list[str]
    missing_flag: str
    amount_parts: list[str]
    categorical: list[str]


@dataclass(frozen=True)
class DuplicateRules:
    key: list[str]
    void_tolerance: float
    double_ratio_min: float
    double_ratio_max: float


@dataclass(frozen=True)
class OutlierRules:
    duration_min_sec: int
    duration_max_sec: int
    distance_min: float
    distance_max: float
    duration_policy: str          # "drop" | "flag"

    def __post_init__(self):
        if self.duration_policy not in ("drop", "flag"):
            raise ValueError(
                f"duration_policy는 'drop' 또는 'flag'만 가능합니다: {self.duration_policy}")


@dataclass(frozen=True)
class Expectations:
    check_exact: bool
    exact: dict[str, int]
    range: dict[str, dict[str, float]]


@dataclass(frozen=True)
class Paths:
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
    name: str
    month: str
    paths: Paths
    source: Source
    columns: Columns
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
