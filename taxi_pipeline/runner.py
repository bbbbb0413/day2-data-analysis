"""등록된 파이프라인 단계를 순서대로 실행하고 결과를 기록한다."""

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
    """파이프라인 한 번의 실행 결과를 저장한다."""

    run_id: str
    manifest: RunManifest
    metrics: dict[str, dict] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)
    # 단계별 파일 산출물 정보를 저장한다.
    artifacts: dict[str, list[dict]] = field(default_factory=dict)
    # 품질 게이트 결과를 저장하여 중복 평가를 막는다.
    gates: list = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    output_path: Path | None = None

    @property
    def ok(self) -> bool:
        """실행이 성공하고 품질 게이트를 통과했는지 반환한다."""
        return self.manifest.status == "success" and not self.gate_failures


def select_steps(names: list[str] | None) -> list[Step]:
    """지정한 단계만 등록 순서대로 반환하며, 지정하지 않으면 전체 단계를 반환한다."""
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

    # 입력 파일을 준비한다.
    if input_path:
        src = input_path
        if not src.is_file():
            raise FileNotFoundError(f"지정한 입력 파일이 없습니다: {src}")
    else:
        src = ensure_input(cfg, force=force_download)

    # parquet 메타데이터로 필수 컬럼을 확인한다.
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

            # 단계별 산출물을 실행 폴더에 저장한다.
            if out.artifacts:
                saved = []
                for art in out.artifacts:
                    # 차트는 figures 폴더에 저장하고 나머지 산출물은 실행 폴더에 저장한다.
                    rel = f"figures/{art.name}" if art.kind in ("figure", "plotly") \
                        else art.name
                    target = store.dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    art.save(target)
                    saved.append({"name": art.name, "kind": art.kind,
                                  "caption": art.caption, "path": rel})
                    log.info("  산출물 저장 %s", rel)
                result.artifacts[step.name] = saved

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

            # 변경 단계의 중간 상태를 parquet으로 저장한다.
            if checkpoint and step.mutates:
                write_parquet(df, cfg.paths.interim / f"{step.name}.parquet")

        # 최종 정제 데이터를 저장한다.
        if save_output:
            out_path = cfg.paths.processed / f"yellow_{cfg.month}_clean.parquet"
            write_parquet(df, out_path)
            result.output_path = out_path
            manifest.outputs["processed"] = str(out_path)

        # 품질 게이트를 검사한다.
        gates = evaluate(result.metrics, cfg)
        result.gates = gates
        result.gate_failures = failures(gates)
        manifest.gate_failures = result.gate_failures
        manifest.status = "success" if not result.gate_failures else "failed"

        # 리포트를 생성하기 전에 실행 시간을 기록한다.
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        manifest.duration_sec = round(time.perf_counter() - started, 3)

        manifest.outputs["metrics"] = str(store.write_json("metrics.json", result.metrics))
        report = render_report(cfg, manifest, result.metrics, result.notes,
                               gates, rows_before_all, result.artifacts)
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
