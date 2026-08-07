"""이상치 필터(steps/outliers.py) 테스트."""

from __future__ import annotations

from dataclasses import replace

from helpers import CFG, frame, row

from taxi_pipeline.steps.outliers import filter_outliers


def test_기간_이탈은_승차시각으로만_판정한다():
    """하차가 다음 달로 넘어간 심야운행은 살아야 한다 (문서 §4)."""
    rows = [
        row("2026-05-31 23:50", "2026-06-01 00:10"),   # 정상 심야운행
        row("2026-04-30 23:50", "2026-04-30 23:59"),   # 기간 이탈
    ]
    res = filter_outliers(frame(rows), CFG)
    assert len(res.df) == 1
    assert res.df["tpep_pickup_datetime"].iloc[0].day == 31


def test_소요시간_0이하는_제거된다():
    rows = [
        row("2026-05-01 00:00", "2026-05-01 00:10"),
        row("2026-05-01 01:00", "2026-05-01 01:00"),   # 0초
        row("2026-05-01 02:00", "2026-05-01 01:50"),   # 역전
    ]
    res = filter_outliers(frame(rows), CFG)
    assert len(res.df) == 1


def test_flag_정책이면_소요시간_이상행이_남는다():
    """duration_policy='flag'는 삭제 대신 표시만 해야 한다 (문서 §6-5)."""
    cfg = replace(CFG, outliers=replace(CFG.outliers, duration_policy="flag"))
    rows = [
        row("2026-05-01 00:00", "2026-05-01 00:10"),
        row("2026-05-01 01:00", "2026-05-01 01:00"),   # 0초
    ]
    res = filter_outliers(frame(rows), cfg)
    assert len(res.df) == 2                              # 지우지 않았다
    assert res.df["duration_valid"].tolist() == [True, False]


def test_사업자_전멸을_경고한다():
    """특정 VendorID가 필터로 전량 사라지면 지표에 남아야 한다 (문서 §4 보완)."""
    rows = [
        row("2026-05-01 00:00", "2026-05-01 00:10", vendor=2),
        row("2026-05-01 01:00", "2026-05-01 01:00", vendor=7),   # 0초, vendor 7뿐
    ]
    res = filter_outliers(frame(rows), CFG)
    wiped = res.metrics["vendors_wiped_out"]
    assert len(wiped) == 1
    assert wiped[0]["vendor"] == 7
    assert wiped[0]["nonpositive_duration_ratio"] == 1.0
    assert any("경고" in n for n in res.notes)


def test_거리와_금액_이상치가_제거된다():
    rows = [
        row("2026-05-01 00:00", "2026-05-01 00:10"),
        row("2026-05-01 01:00", "2026-05-01 01:10", dist=0.0),      # 거리 0
        row("2026-05-01 02:00", "2026-05-01 02:10", dist=500.0),    # 거리 과다
        row("2026-05-01 03:00", "2026-05-01 03:10", total=0.0),     # 총액 0
        row("2026-05-01 04:00", "2026-05-01 04:10", fare=-5.0),     # 요금 음수
    ]
    res = filter_outliers(frame(rows), CFG)
    assert len(res.df) == 1
    assert res.metrics["rows_dropped"] == 4


def test_입력_DataFrame을_제자리에서_바꾸지_않는다():
    """speed_kmh를 입력 df에 붙이면 --steps 부분 실행에서 입력이 오염된다."""
    df = frame([row("2026-05-01 00:00", "2026-05-01 00:10")])
    before = list(df.columns)
    res = filter_outliers(df, CFG)
    assert list(df.columns) == before          # 입력은 그대로
    assert "speed_kmh" in res.df.columns       # 결과에만 붙는다
