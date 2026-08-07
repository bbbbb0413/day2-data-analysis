"""결측 구조 진단과 결측 처리 단계(steps/missing.py) 테스트."""

from __future__ import annotations

from helpers import CFG, frame, row

from taxi_pipeline.steps.missing import analyze_missing, prepare_missing


def test_구조적_결측을_감지한다():
    """5개 컬럼이 같은 행에서만 비면 structural로 판정해야 한다."""
    rows = [row("2026-05-01 00:00", "2026-05-01 00:10")]
    bad = row("2026-05-01 01:00", "2026-05-01 01:10")
    for col in CFG.columns.missing_group:
        bad[col] = None
    rows.append(bad)

    res = analyze_missing(frame(rows), CFG)
    assert res.metrics["missing_rows"] == 1
    assert res.metrics["partial_missing_rows"] == 0
    assert res.metrics["is_structural"] is True


def test_일부만_결측이면_구조적이_아니다():
    """한 컬럼만 비는 행이 섞이면 structural=False여야 한다."""
    rows = [row("2026-05-01 00:00", "2026-05-01 00:10")]
    partial = row("2026-05-01 01:00", "2026-05-01 01:10")
    partial["RatecodeID"] = None            # 5개 중 하나만 결측
    rows.append(partial)

    res = analyze_missing(frame(rows), CFG)
    assert res.metrics["partial_missing_rows"] == 1
    assert res.metrics["is_structural"] is False


def test_결측_처리는_행을_지우지_않는다():
    """prepare_missing은 절대 행 수를 줄이면 안 된다 (문서 §2.7 기준 ①)."""
    rows = [row("2026-05-01 00:00", "2026-05-01 00:10")]
    bad = row("2026-05-01 01:00", "2026-05-01 01:10")
    for col in CFG.columns.missing_group:
        bad[col] = None
    rows.append(bad)

    res = prepare_missing(frame(rows), CFG)
    assert len(res.df) == 2
    assert res.metrics["rows_dropped"] == 0
    assert set(res.df["record_source"]) == {"full", "partial"}


def test_위장_결측이_NaN으로_바뀐다():
    """payment_type=0, RatecodeID=99 등은 NaN이 되어야 한다 (문서 §2.6)."""
    rows = [
        row("2026-05-01 00:00", "2026-05-01 00:10", payment=0),
        row("2026-05-01 01:00", "2026-05-01 01:10", rate=99),
        row("2026-05-01 02:00", "2026-05-01 02:10", passengers=0),
        row("2026-05-01 03:00", "2026-05-01 03:10", pu=264),
    ]
    res = prepare_missing(frame(rows), CFG)
    assert res.df["payment_type"].isna().sum() == 1
    assert res.df["RatecodeID"].isna().sum() == 1
    assert res.df["passenger_count"].isna().sum() == 1
    assert res.df["PULocationID"].isna().sum() == 1
    assert res.metrics["sentinels_total"] == 4
