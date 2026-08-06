"""콘솔 로그와 JSON 직렬화를 처리한다."""

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
    """로그 레코드를 한 줄 JSON으로 출력하는 포매터이다."""

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 한 줄 JSON으로 변환한다."""
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
    """로그 형식과 출력 위치를 설정한다."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()                      # 재설정할 때 기존 핸들러를 제거한다.

    formatter: logging.Formatter = (
        _JsonFormatter() if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
                               datefmt="%H:%M:%S")
    )
    console = logging.StreamHandler(sys.stderr)   # 일반 출력과 로그를 분리한다.
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)


def jsonable(obj: Any) -> Any:
    """NumPy와 Pandas 값을 JSON으로 저장할 수 있는 형태로 변환한다."""
    # dataclass는 dict로 변환한 뒤 다시 처리한다.
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
        return None if np.isnan(f) else f          # JSON에서는 NaN을 null로 저장한다.
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
