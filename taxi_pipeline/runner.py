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


@dataclass
class _Run:
    """실행 한 번에 걸쳐 함께 넘겨야 하는 상태를 묶는다."""

    store: RunStore
    result: RunResult
    started: float                      # time.perf_counter() 기준점이다.

    @property
    def manifest(self) -> RunManifest:
        """진행 중인 실행의 매니페스트를 반환한다."""
        return self.result.manifest


def _resolve_input(cfg: Config, input_path: Path | None, force_download: bool) -> Path:
    """입력 parquet을 준비하고 필수 컬럼이 있는지 확인한다."""
    if input_path:
        src = input_path
        if not src.is_file():
            raise FileNotFoundError(f"지정한 입력 파일이 없습니다: {src}")
    else:
        src = ensure_input(cfg, force=force_download)

    # 전체를 읽기 전에 parquet 메타데이터만으로 스키마를 검증한다.
    meta = peek_metadata(src)
    required = set(cfg.duplicates.key) | {"trip_distance", "fare_amount", "total_amount"}
    if missing := sorted(required - set(meta["columns"])):
        raise KeyError(f"필수 컬럼이 없습니다: {missing}")
    log.info("입력 검증 통과 %s (%s행 × %s열, row group %s개)",
             src.name, f"{meta['num_rows']:,}", meta["num_columns"], meta["num_row_groups"])
    return src


def _begin_run(cfg: Config, src: Path, started: float) -> _Run:
    """run_id를 만들고 실행 기록 디렉터리와 매니페스트를 준비한다."""
    digest = file_digest(src)
    run_id = make_run_id(cfg.digest, digest)
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
    return _Run(store=RunStore(cfg.paths.runs, run_id),
                result=RunResult(run_id=run_id, manifest=manifest),
                started=started)


def _persist_artifacts(run: _Run, step_name: str, artifacts: list) -> None:
    """단계가 만든 파일 산출물을 실행 폴더에 저장하고 경로를 기록한다.

    차트 하나가 저장에 실패해도 파이프라인 전체를 죽이지 않는다. 지표와 분석은
    이미 끝난 상태라 그것까지 버릴 이유가 없다. 대신 실패를 매니페스트에 남겨
    산출물이 왜 비었는지 나중에 추적할 수 있게 한다.
    """
    saved = []
    for art in artifacts:
        # 차트는 figures 폴더에 저장하고 나머지 산출물은 실행 폴더에 저장한다.
        rel = f"figures/{art.name}" if art.kind in ("figure", "plotly") else art.name
        target = run.store.dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            art.save(target)
        except Exception as e:
            log.error("  산출물 저장 실패 %s — %s: %s", rel, type(e).__name__, e)
            run.manifest.artifact_failures.append(
                {"step": step_name, "name": art.name, "error": f"{type(e).__name__}: {e}"})
            continue
        saved.append({"name": art.name, "kind": art.kind,
                      "caption": art.caption, "path": rel})
        log.info("  산출물 저장 %s", rel)
    run.result.artifacts[step_name] = saved


def _execute_steps(df: pd.DataFrame, cfg: Config, steps: list[Step],
                   run: _Run, checkpoint: bool) -> pd.DataFrame:
    """등록된 단계를 순서대로 실행하고 지표·근거·산출물을 모은다."""
    for step in steps:
        t0 = time.perf_counter()
        rows_in = len(df)
        log.info("─ %s 시작 — %s", step.name, step.description)

        out = step(df, cfg)
        df = out.df
        elapsed = time.perf_counter() - t0

        run.result.metrics[step.name] = out.metrics
        run.result.notes[step.name] = out.notes
        if out.artifacts:
            _persist_artifacts(run, step.name, out.artifacts)

        run.manifest.steps.append({
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
    return df


def _save_processed(df: pd.DataFrame, cfg: Config, run: _Run) -> None:
    """최종 정제 데이터를 parquet으로 저장한다."""
    out_path = cfg.paths.processed / f"yellow_{cfg.month}_clean.parquet"
    write_parquet(df, out_path)
    run.result.output_path = out_path
    run.manifest.outputs["processed"] = str(out_path)


def _evaluate_gates(cfg: Config, run: _Run) -> None:
    """품질 게이트를 검사하고 실행 상태를 확정한다."""
    gates = evaluate(run.result.metrics, cfg)
    run.result.gates = gates
    run.result.gate_failures = failures(gates)
    run.manifest.gate_failures = run.result.gate_failures
    run.manifest.status = "success" if not run.result.gate_failures else "failed"


def _stamp_duration(run: _Run) -> None:
    """소요시간을 한 번만 계산한다.

    리포트가 이 값을 본문에 싣기 때문에 리포트를 만들기 전에 확정해야 하고,
    확정한 뒤에는 다시 계산하지 않는다. 예전에는 여기와 finally에서 각각
    계산해서 리포트에 찍힌 소요시간과 manifest.json의 값이 서로 달랐다.
    """
    if run.manifest.duration_sec is None:
        run.manifest.finished_at = datetime.now(timezone.utc).isoformat()
        run.manifest.duration_sec = round(time.perf_counter() - run.started, 3)


def _write_outputs(cfg: Config, run: _Run, rows_before: int) -> None:
    """지표 JSON과 마크다운 리포트를 실행 폴더에 쓴다."""
    _stamp_duration(run)
    res = run.result
    run.manifest.outputs["metrics"] = str(
        run.store.write_json("metrics.json", res.metrics))
    report = render_report(cfg, run.manifest, res.metrics, res.notes,
                           res.gates, rows_before, res.artifacts)
    run.manifest.outputs["report"] = str(run.store.write_text("report.md", report))


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
    src = _resolve_input(cfg, input_path, force_download)
    run = _begin_run(cfg, src, started)

    try:
        df = read_parquet(src)
        rows_before_all = len(df)

        df = _execute_steps(df, cfg, select_steps(steps), run, checkpoint)
        if save_output:
            _save_processed(df, cfg, run)

        _evaluate_gates(cfg, run)
        _write_outputs(cfg, run, rows_before_all)

    except Exception:
        run.manifest.status = "failed"
        log.exception("파이프라인 실패 (run_id=%s)", run.result.run_id)
        raise
    finally:
        # 성공 경로에서는 _write_outputs()가 이미 확정했다. 실패했을 때만 여기서 찍힌다.
        _stamp_duration(run)
        run.store.write_json("manifest.json", run.manifest)
        log.info("총 소요 %.2f초 · 기록 %s", run.manifest.duration_sec, run.store.dir)

    return run.result
