"""원본 parquet 파일을 내려받고 상태를 확인한다."""

from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

# 다운로드 진행 로그를 8MB마다 남긴다.
_PROGRESS_STEP = 8 * 1024 * 1024
_CHUNK = 1 << 20                      # 1MB 단위로 읽는다.


def build_url(cfg: Config) -> str:
    """설정한 월을 적용해 다운로드 URL을 만든다."""
    return cfg.source.url_template.format(month=cfg.month)


def _verify(path: Path, expected_bytes: int | None) -> int:
    """다운로드한 parquet 파일의 크기와 행 수를 확인한다."""
    size = path.stat().st_size
    if size == 0:
        raise ValueError("받은 파일이 비어 있습니다")
    if expected_bytes and size != expected_bytes:
        raise ValueError(
            f"크기가 다릅니다: 기대 {expected_bytes:,} / 실제 {size:,} bytes "
            "(전송이 중간에 끊겼을 수 있습니다)")

    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path)
    except Exception as e:
        # HTTP 응답이 parquet이 아니면 여기서 확인된다.
        raise ValueError(f"parquet으로 읽을 수 없습니다: {e}") from e

    if meta.metadata.num_rows == 0:
        raise ValueError("행이 0개입니다")
    return meta.metadata.num_rows


def download(url: str, target: Path, timeout: int = 120) -> Path:
    """파일을 임시 경로에 내려받고 검증 후 최종 경로로 옮긴다."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    log.info("다운로드 시작 %s", url)
    started = time.perf_counter()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "taxi-pipeline"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if total:
                log.info("크기 %.1f MB", total / 1024**2)

            done = next_mark = 0
            with open(tmp, "wb") as f:
                while chunk := resp.read(_CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if done >= next_mark:
                        pct = f" ({done / total:.0%})" if total else ""
                        log.info("  받는 중 %.1f MB%s", done / 1024**2, pct)
                        next_mark = done + _PROGRESS_STEP

        rows = _verify(tmp, total or None)
        elapsed = time.perf_counter() - started
        # 검증을 통과한 임시 파일만 최종 경로로 바꾼다.
        tmp.replace(target)
        log.info("다운로드 완료 %s (%.1f MB, %s행, %.1f초)",
                 target.name, target.stat().st_size / 1024**2, f"{rows:,}", elapsed)
        return target

    except urllib.error.HTTPError as e:
        raise ValueError(f"다운로드 실패 HTTP {e.code} {e.reason}: {url}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"네트워크 오류: {e.reason} ({url})") from e
    finally:
        # 실패하면 남아 있는 임시 파일을 삭제한다.
        if tmp.exists():
            tmp.unlink(missing_ok=True)
            log.warning("불완전한 임시 파일을 삭제했습니다: %s", tmp.name)


def ensure_input(cfg: Config, *, force: bool = False) -> Path:
    """입력 파일이 있으면 사용하고 없으면 설정에 따라 내려받는다."""
    target = cfg.paths.raw

    if target.is_file() and not force:
        log.info("입력 파일 있음, 다운로드 건너뜀 (%.1f MB) %s",
                 target.stat().st_size / 1024**2, target)
        return target

    url = build_url(cfg)

    if not cfg.source.auto_download and not force:
        raise FileNotFoundError(
            f"입력 파일이 없습니다: {target}\n"
            f"       auto_download=false 로 설정돼 있어 자동으로 받지 않습니다.\n"
            f"       직접 받으려면: python run_pipeline.py fetch\n"
            f"       또는 아래에서 내려받아 위 경로에 두세요:\n"
            f"       {url}")

    if force and target.is_file():
        log.info("--force-download: 기존 파일을 새로 받습니다")
    return download(url, target, timeout=cfg.source.timeout_sec)
