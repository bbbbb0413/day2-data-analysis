"""통계 해석 로직(steps/statistics.py) 테스트.

p-value와 효과크기를 어떻게 문장으로 옮기는지가 리포트의 결론을 좌우하므로
경계값과 해석 문장을 고정해 둔다.
"""

from __future__ import annotations

import inspect

from taxi_pipeline.steps import statistics as st
from taxi_pipeline.steps.statistics import P_MIN, _effect_label, _interpret, fmt_p


def test_효과크기_구간이_경계에서_바뀐다():
    """Cohen's d 구간 경계는 해석 문장을 좌우하므로 고정해 둔다."""
    assert _effect_label(0.19) == "무시 가능"
    assert _effect_label(0.2) == "작음"
    assert _effect_label(0.5) == "중간"
    assert _effect_label(-0.9) == "큼"          # 부호와 무관하게 크기로 판정


def test_p값_언더플로는_하한으로_표기된다():
    """scipy가 0.0을 돌려줄 때 'p = 0'으로 쓰면 사실과 다르다."""
    assert fmt_p(0.0) == f"p < {P_MIN:.3g}"
    assert fmt_p(0.03) == "p = 0.03"


def test_큰_표본에서_작은_차이는_유의하지만_무시_가능으로_해석된다():
    """p-value만 보면 '유의'인데 효과크기가 작으면 해석 문장이 달라져야 한다."""
    r = {"group_a": {"label": "A", "n": 500_000, "mean": 0.2447, "std": 0.13},
         "group_b": {"label": "B", "n": 2_000_000, "mean": 0.2426, "std": 0.13},
         "mean_diff": 0.0021, "t_statistic": 9.47, "p_value": 2.8e-21,
         "significant": True, "cohens_d": 0.016, "effect_size": "무시 가능"}
    text = _interpret(r, 0.05)
    assert "실질적 의미는 없다" in text
    assert "2,500,000건으로 커서" in text        # 표본 크기를 원인으로 지목


def test_효과크기가_중간이면_의미_있는_차이로_해석된다():
    r = {"group_a": {"label": "장거리", "n": 477_000, "mean": 0.1716, "std": 0.12},
         "group_b": {"label": "단거리", "n": 2_182_779, "mean": 0.2585, "std": 0.12},
         "mean_diff": -0.0869, "t_statistic": -450.96, "p_value": 0.0,
         "significant": True, "cohens_d": -0.703, "effect_size": "중간"}
    text = _interpret(r, 0.05)
    assert "실질적으로 의미 있는 차이" in text
    assert "p < " in text                        # 언더플로 하한 표기


def test_유의하지_않으면_차이_근거_없음으로_해석된다():
    r = {"group_a": {"label": "A", "n": 100, "mean": 0.10, "std": 0.1},
         "group_b": {"label": "B", "n": 100, "mean": 0.11, "std": 0.1},
         "mean_diff": -0.01, "t_statistic": -0.7, "p_value": 0.48,
         "significant": False, "cohens_d": -0.1, "effect_size": "무시 가능"}
    assert "근거가 없다" in _interpret(r, 0.05)


def test_ttest는_ttest_ind를_쓴다():
    """검정 구현이 조용히 다른 것으로 바뀌지 않게 고정한다.

    직접 구현한 t 통계량은 자유도 계산을 틀리기 쉽다.
    """
    assert "stats.ttest_ind" in inspect.getsource(st.statistics_step)
