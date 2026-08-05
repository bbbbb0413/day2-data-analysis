"""로깅과 메트릭 수집 — 자동화 환경에서 '무슨 일이 있었는지' 남기는 계층.

print를 걷어낸 이유:
  스케줄러(cron·Airflow·GitHub Actions)는 stdout을 그냥 삼키거나 통째로
  로그 파일에 붓는다. 시각·심각도·단계 이름이 없으면 실패를 찾을 수 없고,
  print는 로그 레벨을 조절할 수도, 파일로 나눠 담을 수도 없다.

  사람이 읽을 근거 문장(notes)은 별도로 모아 리포트로 렌더링한다.
  로그는 '기계가 읽는 실행 기록', 리포트는 '사람이 읽는 분석 결과'로 분리한다.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class _JsonFormatter(logging.Formatter):
    """로그 수집기(Loki·CloudWatch 등)가 파싱할 수 있는 한 줄 JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 한 줄 JSON으로 바꾼다."""
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if extra := getattr(record, "extra_fields", None):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", fmt: str = "text",
                  log_file: Path | None = None) -> None:
    """루트 로거를 구성한다.

    fmt="text" : 사람이 터미널에서 볼 때
    fmt="json" : 스케줄러·로그 수집기에 넘길 때
    log_file   : 지정하면 콘솔과 파일에 동시에 남긴다(실행 기록 보존용).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()                      # 재실행 시 핸들러 중복 방지

    formatter: logging.Formatter = (
        _JsonFormatter() if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
                               datefmt="%H:%M:%S")
    )
    console = logging.StreamHandler(sys.stderr)   # stdout은 리포트 전용으로 비워 둔다
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)


def jsonable(obj: Any) -> Any:
    """numpy/pandas 타입을 JSON이 쓸 수 있는 형태로 바꾼다.

    이 변환이 없으면 metrics.json 덤프가 'Object of type int64 is not JSON
    serializable'로 죽는다. 파이프라인 마지막 단계에서 터지는 가장 흔한 사고다.
    """
    # dataclass를 먼저 처리한다. 빠뜨리면 마지막 str(obj) 분기로 떨어져
    # manifest.json 전체가 "RunManifest(run_id=...)" 라는 문자열 한 줄로 저장된다.
    # (파일은 정상적으로 생기고 예외도 안 나서 한참 뒤에야 발견된다)
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if np.isnan(f) else f          # NaN은 JSON에 없다
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return jsonable(obj.to_dict())
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return str(obj)
