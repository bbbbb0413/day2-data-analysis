"""ML Pipeline(steps/model.py) 테스트 — 누수 방지와 전처리 위치를 고정한다."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
from helpers import CFG, frame, row
from sklearn.pipeline import Pipeline

from taxi_pipeline.steps.model import (CATEGORICAL, LEAKAGE, NUMERIC,
                                       _build_pipeline, train_model)


def test_누수_컬럼이_피처에_없다():
    """total_amount는 팁을 포함한 합계라 넣으면 정답을 입력하는 것과 같다.

    이 목록이 흔들리면 모델 성능이 비현실적으로 높아지므로 고정한다.
    """
    assert set(LEAKAGE) == {"tip_amount", "total_amount", "payment_type"}
    for col in LEAKAGE:
        assert col not in NUMERIC and col not in CATEGORICAL


def test_전처리가_Pipeline_안에_있다():
    """전처리를 Pipeline 밖에서 하면 테스트셋 정보가 학습에 새어 들어간다."""
    pipe = _build_pipeline(CFG, NUMERIC, CATEGORICAL)
    assert isinstance(pipe, Pipeline)
    assert [n for n, _ in pipe.steps] == ["preprocess", "classifier"]

    # 전처리 안에 결측 대체·스케일링·원-핫이 모두 들어 있어야 한다
    pre = pipe.named_steps["preprocess"]
    inner = {n for name, trans, _ in pre.transformers for n, _ in trans.steps}
    assert {"imputer", "scaler", "onehot"} <= inner


def test_표본이_적으면_학습을_건너뛴다():
    """데이터가 말라붙었을 때 예외로 죽지 않고 건너뛰어야 한다."""
    rows = [row("2026-05-01 00:00", "2026-05-01 00:10") for _ in range(5)]
    res = train_model(frame(rows), CFG)
    assert res.metrics.get("skipped") is True
    assert res.artifacts == []


def test_모델이_joblib으로_저장되고_다시_불러진다():
    """저장한 Pipeline이 그대로 다시 예측 가능해야 한다."""
    pipe = _build_pipeline(CFG, NUMERIC, CATEGORICAL)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.joblib"
        joblib.dump(pipe, p)
        assert p.stat().st_size > 0
        assert isinstance(joblib.load(p), Pipeline)
