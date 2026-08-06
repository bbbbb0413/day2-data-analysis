"""파생변수 검증 단계"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from .base import StepResult
# 표기 규칙(fmt_p)과 해석 방식(_cohens_d/_effect_label)을 그대로 재사용한다.
# 각자 새로 정의하면 리포트 전체의 p-value·효과크기 표기가 갈라진다.
from .statistics import _cohens_d, _effect_label, fmt_p

log = logging.getLogger(__name__)


def _compare(a: pd.Series, b: pd.Series, label_a: str, label_b: str, alpha: float) -> dict:
    """두 집단을 Welch's t-test(scipy.stats.ttest_ind)로 비교한다."""
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    d = _cohens_d(a, b)
    return {
        "group_a": {"label": label_a, "n": int(len(a)), "mean": float(a.mean())},
        "group_b": {"label": label_b, "n": int(len(b)), "mean": float(b.mean())},
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "cohens_d": d,
        "effect_size": _effect_label(d),
    }


def validate_features(df: pd.DataFrame, cfg: Config) -> StepResult:
    """engineer_features가 만든 파생변수를 보조적으로 검증한다. DataFrame은 바꾸지 않는다."""
    st = cfg.statistics
    notes: list[str] = []
    metrics: dict = {}

    # ---- ① is_airport_trip: 팁 비율 차이 ---------------------------------
    # statistics_step()과 동일하게 카드결제로 한정한다. 현금·무료·분쟁 결제는
    # 팁이 100% 0으로 기록돼 섞으면 '결제수단 맞히기'가 되기 때문이다(§S 재사용).
    if "is_airport_trip" in df.columns:
        card = df[df["payment_type"] == 1]
        rate = card["tip_amount"] / card["fare_amount"].replace(0, np.nan)
        ok = rate.notna() & (rate < 2)
        rate, air = rate[ok], card.loc[ok, "is_airport_trip"]
        known = air.notna()                     # partial 소스(§2.7)는 결측이라 제외
        rate, air = rate[known], air[known]

        if air.any() and (~air).any():
            r = _compare(rate[air], rate[~air], "공항 트립", "일반 트립", st.alpha)
            notes.append(
                f"[파생변수 검증] is_airport_trip — 공항 트립({r['group_a']['n']:,}건, "
                f"평균 팁 비율 {r['group_a']['mean']:.2%}) vs 일반 트립"
                f"({r['group_b']['n']:,}건, 평균 {r['group_b']['mean']:.2%}), "
                f"t={r['t_statistic']:.2f}, {fmt_p(r['p_value'])}, "
                f"Cohen's d={r['cohens_d']:+.3f}({r['effect_size']}). "
                f"{'통계적으로 유의한 차이다' if r['significant'] else '유의한 차이라 보기 어렵다'}. "
                f"(공식 가설검정이 아니라 파생변수 타당성을 보는 보조 분석이다.)")
            metrics["airport_trip_tip_rate"] = r

    # ---- ② is_rush_hour: 속력 차이 -----------------------------------------
    if "is_rush_hour" in df.columns and "speed_kmh" in df.columns:
        speed = df["speed_kmh"].dropna()
        rush = df.loc[speed.index, "is_rush_hour"]

        if rush.any() and (~rush).any():
            r = _compare(speed[rush], speed[~rush], "출퇴근시간대", "그 외 시간대", st.alpha)
            notes.append(
                f"[파생변수 검증] is_rush_hour — 출퇴근시간대({r['group_a']['n']:,}건, "
                f"평균 속력 {r['group_a']['mean']:.1f}km/h) vs 그 외 시간대"
                f"({r['group_b']['n']:,}건, 평균 {r['group_b']['mean']:.1f}km/h), "
                f"t={r['t_statistic']:.2f}, {fmt_p(r['p_value'])}, "
                f"Cohen's d={r['cohens_d']:+.3f}({r['effect_size']}). "
                f"{'통계적으로 유의한 차이다' if r['significant'] else '유의한 차이라 보기 어렵다'}. "
                f"출퇴근시간대에 실제로 느려진다면 is_rush_hour가 정체 상황을 잘 반영한다는 뜻이다.")
            metrics["rush_hour_speed"] = r

    log.info("파생변수 검증 완료: %d개 비교", len(metrics))
    return StepResult(df=df, metrics=metrics, notes=notes)
