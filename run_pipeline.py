#!/usr/bin/env python
"""
================================================================================
NYC Yellow Taxi 정제·EDA 파이프라인 — CLI 진입점
파일명 : run_pipeline.py
작성일 : 2026-08-05
================================================================================

■ 사용법
  python run_pipeline.py                          전체 실행
  python run_pipeline.py fetch                     원본만 내려받기
  python run_pipeline.py --steps deduplicate       특정 단계만
  python run_pipeline.py --checkpoint              단계별 중간 산출물 저장
  python run_pipeline.py --config config/2026-06.toml   다른 월 설정으로
  python run_pipeline.py --log-format json         스케줄러·로그 수집기용
  python run_pipeline.py --dry-run                 실행 계획만 출력
  python run_pipeline.py --force-download          원본을 새로 받아 실행
  python run_pipeline.py compare-loaders           pandas vs polars 로딩 비교

■ 입력 확보
  data/raw/에 파일이 없으면 config의 [source] url_template에서 자동으로 받는다.
  이미 있으면 건드리지 않는다. 새 서버·새 컨테이너에서도 저장소만 있으면
  바로 돌아가게 하기 위함이다.

■ 종료 코드 (자동화가 읽는 값)
  0  성공 — 모든 품질 게이트 통과
  1  품질 게이트 실패 — 산출물은 생겼지만 신뢰할 수 없다
  2  실행 오류 — 파일 없음·설정 오류·예외

  cron·Airflow·GitHub Actions 모두 종료 코드로 성공/실패를 판단한다.
  0이 아니면 알림이 울리도록 스케줄러에 걸면 된다.

■ 스케줄 등록 예시
  # 매월 5일 새벽 3시
  0 3 5 * * cd /path/to/day2-data-analysis && \
      .venv/bin/python run_pipeline.py --log-format json >> logs/cron.log 2>&1

■ 참고 문서
  결측치_중복_처리기준.md  — 모든 임계값의 근거
  README.md               — 구조 설명
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # 설치 없이 실행 가능하게

from taxi_pipeline import __version__, load_config          # noqa: E402
from taxi_pipeline.fetch import build_url, ensure_input      # noqa: E402
from taxi_pipeline.observability import setup_logging       # noqa: E402
from taxi_pipeline.report import render_console_summary     # noqa: E402
from taxi_pipeline.runner import run_pipeline, select_steps # noqa: E402

log = logging.getLogger("cli")

EXIT_OK, EXIT_GATE_FAILED, EXIT_ERROR = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="NYC Yellow Taxi 정제·EDA 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", nargs="?", default="run",
                   choices=["run", "fetch", "compare-loaders", "list-steps"],
                   help="run(기본) | fetch | compare-loaders | list-steps")
    p.add_argument("--config", default=str(ROOT / "config" / "pipeline.toml"),
                   help="설정 파일 경로")
    p.add_argument("--input", help="입력 parquet 경로 (설정값보다 우선)")
    p.add_argument("--steps", help="실행할 단계를 쉼표로 구분 (미지정 시 전체)")
    p.add_argument("--checkpoint", action="store_true",
                   help="단계별 중간 산출물을 data/interim에 저장")
    p.add_argument("--no-save", action="store_true",
                   help="최종 parquet을 저장하지 않는다(검증만 할 때)")
    p.add_argument("--dry-run", action="store_true",
                   help="실행 계획만 출력하고 종료")
    p.add_argument("--force-download", action="store_true",
                   help="입력 파일이 이미 있어도 원본을 다시 받는다")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-format", default="text", choices=["text", "json"],
                   help="json은 로그 수집기용")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def cmd_list_steps() -> int:
    """등록된 단계를 보여준다. 파이프라인 구성 확인용."""
    from taxi_pipeline.steps import PIPELINE

    print("등록된 단계 (실행 순서):\n")
    for i, s in enumerate(PIPELINE, 1):
        kind = "변환" if s.mutates else "분석"
        print(f"  {i}. {s.name:<18} [{kind}] {s.description}")
    return EXIT_OK


def cmd_compare_loaders(cfg, input_path: Path) -> int:
    """pandas와 polars로 같은 파일을 읽어 결과가 일치하는지 확인한다.

    파이프라인 본체에는 넣지 않았다. 매 실행 두 번 읽는 것은 낭비이고,
    이 비교는 '도구를 바꿔도 결론이 같은가'를 한 번 확인하는 성격이기 때문이다.
    """
    import time

    import pandas as pd
    import polars as pl

    # 먼저 재는 쪽만 디스크를 읽으면 불공정하므로 페이지 캐시를 데운다.
    with open(input_path, "rb") as f:
        while f.read(1 << 24):
            pass

    t = time.perf_counter()
    ldf = pl.read_parquet(input_path)
    pl_sec = time.perf_counter() - t
    pl_nulls = dict(zip(ldf.columns, ldf.null_count().row(0)))
    pl_exact = len(ldf) - ldf.n_unique()
    pl_key = int(ldf.select(cfg.duplicates.key).is_duplicated().sum())
    pl_size = ldf.estimated_size("mb")
    pl_types = {c: str(t) for c, t in zip(ldf.columns, ldf.dtypes)}
    pl_shape = ldf.shape
    # pandas 측정 전에 해제한다. 동시에 들고 있으면 메모리가 서로 간섭한다.
    del ldf

    t = time.perf_counter()
    pdf = pd.read_parquet(input_path)
    pd_sec = time.perf_counter() - t
    pd_nulls = pdf.isna().sum().to_dict()
    pd_exact = int(pdf.duplicated().sum())
    pd_key = int(pdf.duplicated(subset=cfg.duplicates.key).sum())
    pd_size = pdf.memory_usage(deep=True).sum() / 1024**2

    print(f"\n입력 : {input_path}  ({input_path.stat().st_size / 1024**2:,.1f} MB)\n")
    print(f"{'항목':<24}{'polars':>18}{'pandas':>18}")
    print("-" * 60)
    print(f"{'로딩 시간(초)':<24}{pl_sec:>18.3f}{pd_sec:>18.3f}")
    print(f"{'DataFrame 크기(MB)':<24}{pl_size:>18,.1f}{pd_size:>18,.1f}")
    print(f"{'shape':<24}{str(pl_shape):>18}{str(pdf.shape):>18}")
    print(f"{'결측 총합':<24}{sum(pl_nulls.values()):>18,}{int(sum(pd_nulls.values())):>18,}")
    print(f"{'완전중복':<24}{pl_exact:>18,}{pd_exact:>18,}")
    print(f"\n  → polars가 로딩 {pd_sec / max(pl_sec, 1e-9):.1f}배 빠르고 "
          f"DataFrame이 {pd_size / pl_size:.2f}배 작다.")

    same_null = all(int(pl_nulls[c]) == int(pd_nulls[c]) for c in pd_nulls)
    print(f"\n  [일치 검증] 컬럼별 결측수 {'OK' if same_null else '불일치'}"
          f" / 완전중복 {'OK' if pl_exact == pd_exact else '불일치'}")

    # 부분키 중복은 두 라이브러리의 '세는 대상'이 다르다.
    print(f"\n  [주의] 부분키 중복 정의 차이")
    print(f"     polars is_duplicated().sum() = {pl_key:>9,}  (그룹 구성원 전부)")
    print(f"     pandas duplicated().sum()    = {pd_key:>9,}  (첫 행 제외)")

    # 결측이 있는 정수 컬럼에서 pandas는 float64로 승격시킨다.
    print(f"\n  [타입 복원 차이] 결측이 있는 정수 컬럼")
    for col in pdf.columns:
        p, l = str(pdf[col].dtype), pl_types[col]
        if p.startswith("float") and l.startswith("Int"):
            print(f"     {col:<24} pandas {p:<10} vs polars {l:<8}"
                  f"  (결측 {int(pd_nulls[col]):,})")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, root=ROOT)
    except Exception as e:
        setup_logging(args.log_level, args.log_format)
        log.error("설정 로딩 실패: %s", e)
        return EXIT_ERROR

    input_path = Path(args.input).resolve() if args.input else cfg.paths.raw
    steps = [s.strip() for s in args.steps.split(",")] if args.steps else None

    # 로그 파일은 실행 디렉터리에 함께 남긴다. run_id를 미리 계산해야 하는데
    # 입력 해시는 파일을 읽어야 나오므로, 여기서는 설정 해시만으로 임시 경로를 잡고
    # runner가 실제 run_id 디렉터리를 만든다.
    setup_logging(args.log_level, args.log_format)

    if args.command == "list-steps":
        return cmd_list_steps()

    if args.command == "fetch":
        # 파이프라인 없이 입력만 확보한다. 스케줄러에서 다운로드 태스크를
        # 따로 두거나, 처음 저장소를 받은 사람이 준비만 할 때 쓴다.
        try:
            path = ensure_input(cfg, force=args.force_download)
        except (FileNotFoundError, ValueError) as e:
            log.error("%s", e)
            return EXIT_ERROR
        print(f"입력 준비 완료: {path} ({path.stat().st_size / 1024**2:,.1f} MB)")
        return EXIT_OK

    if args.command == "compare-loaders":
        if not input_path.is_file():
            log.error("입력 파일이 없습니다: %s (먼저 `run_pipeline.py fetch`)", input_path)
            return EXIT_ERROR
        return cmd_compare_loaders(cfg, input_path)

    # ---- dry-run: 무엇을 할지만 보여준다 -------------------------------------
    if args.dry_run:
        print(f"설정      : {cfg.source_file}  (sha {cfg.digest})")
        exists = "있음" if input_path.is_file() else "없음"
        print(f"입력      : {input_path}  [{exists}]")
        if not input_path.is_file():
            print(f"  다운로드: {build_url(cfg)}"
                  f"  (auto_download={cfg.source.auto_download})")
        print(f"기간 기준 : {cfg.month}")
        print(f"소요시간 정책 : {cfg.outliers.duration_policy}")
        print(f"저장      : {'안 함' if args.no_save else cfg.paths.processed}")
        print("\n실행할 단계:")
        for i, s in enumerate(select_steps(steps), 1):
            print(f"  {i}. {s.name:<18} {s.description}")
        print(f"\n품질 게이트: exact {len(cfg.expectations.exact)}개"
              f"(검사 {'ON' if cfg.expectations.check_exact else 'OFF'})"
              f", range {len(cfg.expectations.range)}개")
        return EXIT_OK

    # ---- 실행 ----------------------------------------------------------------
    try:
        result = run_pipeline(
            cfg,
            steps=steps,
            checkpoint=args.checkpoint,
            save_output=not args.no_save,
            input_path=Path(args.input).resolve() if args.input else None,
            force_download=args.force_download,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        log.error("%s", e)
        return EXIT_ERROR
    except Exception:
        log.exception("예기치 못한 오류")
        return EXIT_ERROR

    print()
    print(render_console_summary(result.metrics, result.gates))
    print(f"리포트: {result.manifest.outputs.get('report', '-')}")

    if result.gate_failures:
        log.error("품질 게이트 %d개 실패 — 산출물을 신뢰할 수 없다",
                  len(result.gate_failures))
        return EXIT_GATE_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
