"""파생변수 생성(steps/features.py) 테스트."""

from __future__ import annotations

import pandas as pd
from helpers import CFG, frame, row

from taxi_pipeline.steps.features import engineer_features
from taxi_pipeline.steps.missing import prepare_missing


def test_러시아워는_승차시각_기준으로_판정된다():
    """아침 8시·저녁 17시는 True, 낮 13시는 False여야 한다."""
    rows = [
        row("2026-05-01 08:00", "2026-05-01 08:10"),   # 아침 러시아워
        row("2026-05-01 17:00", "2026-05-01 17:10"),   # 저녁 러시아워
        row("2026-05-01 13:00", "2026-05-01 13:10"),   # 러시아워 아님
    ]
    res = engineer_features(frame(rows), CFG)
    assert res.df["is_rush_hour"].tolist() == [True, True, False]


def test_공항트립은_요율_또는_요금으로_판정되고_partial행은_결측이다():
    """RatecodeID 공항코드나 Airport_fee>0이면 True, partial 행(§2.7)은 NA여야 한다."""
    rows = [
        row("2026-05-01 00:00", "2026-05-01 00:10", rate=2),      # JFK 요율
        row("2026-05-01 01:00", "2026-05-01 01:10", air=1.75),    # 공항 부가금
        row("2026-05-01 02:00", "2026-05-01 02:10"),              # 일반 트립
    ]
    partial = row("2026-05-01 03:00", "2026-05-01 03:10")
    for col in CFG.columns.missing_group:
        partial[col] = None
    rows.append(partial)

    df = prepare_missing(frame(rows), CFG).df
    res = engineer_features(df, CFG)
    assert res.df["is_airport_trip"].tolist()[:3] == [True, True, False]
    assert pd.isna(res.df["is_airport_trip"].iloc[3])   # False가 아니라 결측이어야 함


def test_입력_DataFrame을_제자리에서_바꾸지_않는다():
    """prepare_missing과 같은 계약을 지켜야 한다."""
    df = frame([row("2026-05-01 08:00", "2026-05-01 08:10")])
    before = list(df.columns)
    res = engineer_features(df, CFG)
    assert list(df.columns) == before
    assert {"is_rush_hour", "is_airport_trip"} <= set(res.df.columns)
