"""오케스트레이션 — 단계를 순서대로 돌리고 기록을 남긴다.

runner는 '무엇을 하는지' 모른다. 단계 목록을 받아 돌리고, 시간을 재고,
지표를 모으고, 실패하면 어디서 멈췄는지 기록할 뿐이다.
분석 로직이 runner에 스며들지 않게 하는 것이 이 구조의 핵심이다.

자동화에 필요한 것들이 여기 모여 있다.
  - 단계별 소요 시간과 행 수 변화 기록
  - --checkpoint : 중간 산출물 저장 (긴 파이프라인의 재시작 지점)
  - --steps      : 일부 단계만 실행 (디버깅·부분 재처리)
  - 품질 게이트 실패 시 종료 코드 1
  - 매니페스트 : 입력 해시·설정 해시·산출물 경로
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .fetch import ensure_input
from .quality import evaluate, failures
from .report import render_report
from .steps import PIPELINE, STEPS, Step
from .storage import (RunManifest, RunStore, file_digest, make_run_id,
                      peek_metadata, read_parquet, write_parquet)

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """실행 한 건의 결과. CLI가 종료 코드를 정할 때 쓴다."""

    run_id: str
    manifest: RunManifest
    metrics: dict[str, dict] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)
    # 게이트 결과를 결과 객체에 담아 CLI가 재평가하지 않게 한다.
    # 두 번 평가하면 로그에 같은 내용이 두 번 찍히고, 그 사이 값이 달라질 여지도 생긴다.
    gates: list = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    output_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.manifest.status == "success" and not self.gate_failures


def select_steps(names: list[str] | None) -> list[Step]:
    """실행할 단계를 고른다.

    이름을 주면 그 단계만, 안 주면 전체. 정의된 순서는 유지한다
    (사용자가 --steps outliers,duplicates 로 줘도 의존 순서가 깨지지 않게).
    """
    if not names:
        return list(PIPELINE)
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise ValueError(f"알 수 없는 단계: {unknown} / 사용 가능: {list(STEPS)}")
    return [s for s in PIPELINE if s.name in names]


def run_pipeline(
    cfg: Config,
    *,
    steps: list[str] | None = None,
    checkpoint: bool = False,
    save_output: bool = True,
    input_path: Path | None = None,
    force_download: bool = False,
) -> RunResult:
    """파이프라인을 실행한다."""
    started = time.perf_counter()

    # ---- 입력 확보: 없으면 설정된 출처에서 내려받는다 ------------------------
    # --input으로 파일을 직접 지정한 경우엔 건드리지 않는다. 사용자가 명시한
    # 파일을 두고 다른 것을 받아오면 안 된다.
    if input_path:
        src = input_path
        if not src.is_file():
            raise FileNotFoundError(f"지정한 입력 파일이 없습니다: {src}")
    else:
        src = ensure_input(cfg, force=force_download)

    # ---- 조기 실패: 전체를 읽기 전에 메타데이터로 먼저 검증 ------------------
    # 409만 행을 다 읽은 뒤 "컬럼이 없다"고 죽으면 시간과 메모리를 버린다.
    meta = peek_metadata(src)
    required = set(cfg.duplicates.key) | {"trip_distance", "fare_amount", "total_amount"}
    if missing := sorted(required - set(meta["columns"])):
        raise KeyError(f"필수 컬럼이 없습니다: {missing}")
    log.info("입력 검증 통과 %s (%s행 × %s열, row group %s개)",
             src.name, f"{meta['num_rows']:,}", meta["num_columns"], meta["num_row_groups"])

    digest = file_digest(src)
    run_id = make_run_id(cfg.digest, digest)
    store = RunStore(cfg.paths.runs, run_id)

    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        config_file=str(cfg.source_file),
        config_digest=cfg.digest,
        input_file=str(src),
        input_digest=digest,
        input_bytes=src.stat().st_size,
    )
    log.info("run_id=%s (설정 %s · 입력 %s)", run_id, cfg.digest, digest)

    result = RunResult(run_id=run_id, manifest=manifest)

    try:
        df = read_parquet(src)
        rows_before_all = len(df)

        for step in select_steps(steps):
            t0 = time.perf_counter()
            rows_in = len(df)
            log.info("─ %s 시작 — %s", step.name, step.description)

            out = step(df, cfg)
            df = out.df
            elapsed = time.perf_counter() - t0

            result.metrics[step.name] = out.metrics
            result.notes[step.name] = out.notes
            manifest.steps.append({
                "name": step.name,
                "description": step.description,
                "duration_sec": round(elapsed, 3),
                "rows_in": rows_in,
                "rows_out": len(df),
                "rows_delta": len(df) - rows_in,
            })
            log.info("─ %s 완료 (%.2f초, %s → %s행)",
                     step.name, elapsed, f"{rows_in:,}", f"{len(df):,}")

            # 중간 산출물: 긴 파이프라인에서 특정 단계부터 재실행할 때의 시작점.
            # 매 단계 70MB씩 쓰므로 기본은 꺼 둔다.
            if checkpoint and step.mutates:
                write_parquet(df, cfg.paths.interim / f"{step.name}.parquet")

        # ---- 산출물 저장 -----------------------------------------------------
        if save_output:
            out_path = cfg.paths.processed / f"yellow_{cfg.month}_clean.parquet"
            write_parquet(df, out_path)
            result.output_path = out_path
            manifest.outputs["processed"] = str(out_path)

        # ---- 품질 게이트 -----------------------------------------------------
        gates = evaluate(result.metrics, cfg)
        result.gates = gates
        result.gate_failures = failures(gates)
        manifest.gate_failures = result.gate_failures
        manifest.status = "success" if not result.gate_failures else "failed"

        # ---- 기록 -------------------------------------------------------------
        # 소요 시간을 리포트 렌더링 '전에' 채운다. finally에서만 채우면
        # 리포트에는 아직 None인 값이 실려 "소요 -"로 남는다.
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        manifest.duration_sec = round(time.perf_counter() - started, 3)

        manifest.outputs["metrics"] = str(store.write_json("metrics.json", result.metrics))
        report = render_report(cfg, manifest, result.metrics, result.notes, gates,
                               rows_before_all)
        manifest.outputs["report"] = str(store.write_text("report.md", report))

    except Exception:
        manifest.status = "failed"
        log.exception("파이프라인 실패 (run_id=%s)", run_id)
        raise
    finally:
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        manifest.duration_sec = round(time.perf_counter() - started, 3)
        store.write_json("manifest.json", manifest)
        log.info("총 소요 %.2f초 · 기록 %s", manifest.duration_sec, store.dir)

    return result
