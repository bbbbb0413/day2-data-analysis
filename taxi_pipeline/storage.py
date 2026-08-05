"""입출력과 실행 기록(lineage) — '이 산출물이 무엇으로 만들어졌는가'를 남긴다.

자동화에서 가장 자주 겪는 사고가 "지난주 결과가 왜 이렇게 나왔는지 모르겠다"이다.
입력 파일 해시·설정 해시·행 수·소요 시간을 매니페스트로 남기면,
산출물만 보고도 어떤 입력에 어떤 기준을 적용한 결과인지 되짚을 수 있다.
"""

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
    """파일 내용 해시(sha256 앞 12자).

    경로만 기록하면 그 사이 파일이 교체됐는지 알 수 없다. 내용 해시가 있어야
    "같은 입력인데 결과가 달라졌다"를 판별할 수 있다.
    66MB 기준 0.2초 내외라 매 실행 계산해도 부담이 없다.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:12]


def make_run_id(config_digest: str, input_digest: str) -> str:
    """실행 ID = 시각 + 설정해시 + 입력해시.

    시각을 앞에 두어 디렉터리 정렬만으로 시간순이 되고,
    뒤의 두 해시로 '같은 입력·같은 기준의 재실행'인지 눈으로 구분된다.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{config_digest}_{input_digest}"


@dataclass
class RunManifest:
    """실행 한 건의 요약. runs/<run_id>/manifest.json 으로 저장된다."""

    run_id: str
    started_at: str
    config_file: str
    config_digest: str
    input_file: str
    input_digest: str
    input_bytes: int
    status: str = "running"                    # running | success | failed
    finished_at: str | None = None
    duration_sec: float | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """매니페스트를 JSON 문자열로 만든다."""
        return json.dumps(jsonable(asdict(self)), ensure_ascii=False, indent=2)


class RunStore:
    """실행별 산출물 디렉터리를 관리한다.

    구조:
        outputs/runs/<run_id>/
            manifest.json   실행 요약 (입력 해시·단계별 소요·게이트 결과)
            metrics.json    단계별 지표 전체 (대시보드·회귀 비교용)
            report.md       사람이 읽는 근거 리포트
            pipeline.log    실행 로그
    """

    def __init__(self, base: Path, run_id: str):
        """실행 디렉터리를 만든다. 이미 있으면 그대로 쓴다."""
        self.dir = base / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    @property
    def log_file(self) -> Path:
        """실행 로그 경로. 산출물과 같은 디렉터리에 둬 함께 보관된다."""
        return self.dir / "pipeline.log"

    def write_json(self, name: str, payload: Any) -> Path:
        """dict를 JSON으로 저장한다. numpy·dataclass는 jsonable이 변환한다."""
        target = self.dir / name
        target.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
                          encoding="utf-8")
        log.debug("기록 %s", target)
        return target

    def write_text(self, name: str, text: str) -> Path:
        """텍스트를 그대로 저장한다(리포트 등)."""
        target = self.dir / name
        target.write_text(text, encoding="utf-8")
        return target


def read_parquet(path: Path) -> pd.DataFrame:
    """parquet을 읽는다.

    parquet만 받는 이유: CSV는 타입 정보가 없어 날짜가 문자열로 읽히고,
    그러면 소요시간 기준을 계산할 수 없다. 입력 포맷을 고정해
    "읽는 방법에 따라 결과가 달라지는" 변수를 없앤다.
    """
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"parquet만 지원합니다(입력: {path.suffix}): {path}")
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    df = pd.read_parquet(path)
    log.info("읽기 완료 %s (%s행 × %s열)", path.name, f"{len(df):,}", df.shape[1])
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """parquet으로 저장한다.

    입력과 같은 포맷을 쓰면 record_source 같은 파생 컬럼과 datetime 타입이
    그대로 보존되어, 다음 단계·다음 스크립트가 타입 지정 없이 이어받는다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("저장 완료 %s (%s행, %.1f MB)",
             path.name, f"{len(df):,}", path.stat().st_size / 1024**2)
    return path


def peek_metadata(path: Path) -> dict[str, Any]:
    """데이터를 읽지 않고 parquet 푸터 메타데이터만 조회한다.

    409만 행을 다 읽은 뒤 "컬럼이 없다"고 죽지 않도록, 전체 로딩 전에
    행수·열수·컬럼 목록을 먼저 확인하는 용도(조기 실패).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return {
        "num_rows": pf.metadata.num_rows,
        "num_columns": pf.metadata.num_columns,
        "num_row_groups": pf.metadata.num_row_groups,
        "created_by": pf.metadata.created_by,
        "columns": list(pf.schema_arrow.names),
    }
