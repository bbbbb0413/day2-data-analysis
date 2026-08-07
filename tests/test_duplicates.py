"""중복 유형 판정과 선택 제거(steps/duplicates.py) 테스트."""

from __future__ import annotations

from helpers import CFG, frame, row

from taxi_pipeline.steps.duplicates import _classify, deduplicate


def test_상쇄쌍은_양쪽_모두_제거된다():
    """합이 0인 쌍은 두 행 다 사라져야 한다 (문서 §3.6 ①)."""
    rows = [
        row("2026-05-01 00:02", "2026-05-01 00:12", fare=-12.1, total=-17.85),
        row("2026-05-01 00:02", "2026-05-01 00:12", fare=12.1, total=17.85),
        row("2026-05-01 05:00", "2026-05-01 05:10"),      # 무관한 정상 행
    ]
    res = deduplicate(frame(rows), CFG)
    assert res.metrics["void_pairs"] == 1
    assert res.metrics["rows_dropped"] == 2
    assert len(res.df) == 1
    assert (res.df["total_amount"] > 0).all()


def test_이중계상은_큰_쪽만_제거된다():
    """총액이 정확히 2배인 쌍은 큰 쪽만 사라져야 한다 (문서 §3.6 ②)."""
    rows = [
        row("2026-05-01 03:39", "2026-05-01 04:04", total=41.48, payment=0),
        row("2026-05-01 03:39", "2026-05-01 04:04", total=82.96, payment=0),
    ]
    res = deduplicate(frame(rows), CFG)
    assert res.metrics["double_rows"] == 1
    assert res.metrics["rows_dropped"] == 1
    assert len(res.df) == 1
    assert res.df["total_amount"].iloc[0] == 41.48      # 작은 쪽이 남는다


def test_keep_first_였다면_정상운행이_지워진다():
    """naive 비교 지표가 'keep=first는 위험'을 실제로 보여줘야 한다 (문서 §3.5)."""
    rows = [
        row("2026-05-01 00:02", "2026-05-01 00:12", fare=-12.1, total=-17.85),
        row("2026-05-01 00:02", "2026-05-01 00:12", fare=12.1, total=17.85),
    ]
    res = deduplicate(frame(rows), CFG)
    naive = res.metrics["naive_keep_first"]
    # 음수가 먼저 오므로 keep='first'는 양수(정상 운행)를 지운다
    assert naive["positive_rows_dropped"] == 1
    assert naive["negative_rows_left"] == 1


def test_서로_다른_운행은_유지된다():
    """상쇄쌍도 2배도 아니면 건드리지 않는다 (문서 §3.6 ③)."""
    rows = [
        row("2026-05-01 00:02", "2026-05-01 00:12", total=16.02),
        row("2026-05-01 00:02", "2026-05-01 00:12", total=20.16),
    ]
    res = deduplicate(frame(rows), CFG)
    assert res.metrics["other_groups"] == 1
    assert res.metrics["rows_dropped"] == 0
    assert len(res.df) == 2


def test_중복이_없으면_아무것도_안_한다():
    rows = [
        row("2026-05-01 00:02", "2026-05-01 00:12"),
        row("2026-05-01 01:02", "2026-05-01 01:12"),
    ]
    res = deduplicate(frame(rows), CFG)
    assert res.metrics["duplicate_groups"] == 0
    assert len(res.df) == 2


def test_분류_함수는_빈_입력을_견딘다():
    """중복이 하나도 없을 때 _classify가 죽지 않아야 한다."""
    df = frame([row("2026-05-01 00:02", "2026-05-01 00:12")])
    lo, hi, void, double = _classify(df, CFG)
    assert len(lo) == 0 and len(hi) == 0
    assert void.size == 0 and double.size == 0
