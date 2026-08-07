"""parquet 입출력과 실행 결과 저장을 처리한다."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .observability import jsonable

log = logging.getLogger(__name__)


def file_digest(path: Path, chunk: int = 1 << 22) -> str:
    """파일 내용으로 SHA-256 해시 앞 12자리를 만든다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:12]


def make_run_id(config_digest: str, input_digest: str) -> str:
    """실행 시각과 설정, 입력 해시로 실행 ID를 만든다."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{config_digest}_{input_digest}"


@dataclass
class RunManifest:
    """한 번의 실행 정보를 저장한다."""

    run_id: str
    started_at: str
    config_file: str
    config_digest: str
    input_file: str
    input_digest: str
    input_bytes: int
    status: str = "running"                    # running, success, failed 중 하나이다.
    finished_at: str | None = None
    duration_sec: float | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    # 저장에 실패한 차트·모델. 산출물이 왜 비었는지 추적하는 용도다.
    artifact_failures: list[dict[str, str]] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """매니페스트를 JSON 문자열로 변환한다."""
        return json.dumps(jsonable(asdict(self)), ensure_ascii=False, indent=2)


class RunStore:
    """실행별 산출물 디렉터리를 관리한다."""

    def __init__(self, base: Path, run_id: str):
        """실행 결과를 저장할 디렉터리를 만든다."""
        self.dir = base / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    @property
    def log_file(self) -> Path:
        """실행 로그 파일 경로를 반환한다."""
        return self.dir / "pipeline.log"

    def write_json(self, name: str, payload: Any) -> Path:
        """데이터를 JSON 파일로 저장한다."""
        target = self.dir / name
        target.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
                          encoding="utf-8")
        log.debug("기록 %s", target)
        return target

    def write_text(self, name: str, text: str) -> Path:
        """문자열을 텍스트 파일로 저장한다."""
        target = self.dir / name
        target.write_text(text, encoding="utf-8")
        return target


def read_parquet(path: Path) -> pd.DataFrame:
    """parquet 파일을 DataFrame으로 읽는다."""
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"parquet만 지원합니다(입력: {path.suffix}): {path}")
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    df = pd.read_parquet(path)
    log.info("읽기 완료 %s (%s행 × %s열)", path.name, f"{len(df):,}", df.shape[1])
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """DataFrame을 parquet 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("저장 완료 %s (%s행, %.1f MB)",
             path.name, f"{len(df):,}", path.stat().st_size / 1024**2)
    return path


def peek_metadata(path: Path) -> dict[str, Any]:
    """parquet 파일의 메타데이터를 읽는다."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return {
        "num_rows": pf.metadata.num_rows,
        "num_columns": pf.metadata.num_columns,
        "num_row_groups": pf.metadata.num_row_groups,
        "created_by": pf.metadata.created_by,
        "columns": list(pf.schema_arrow.names),
    }
