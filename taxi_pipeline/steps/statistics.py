"""기술통계, 상관계수, t-test를 계산한다."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)

# p-value가 0으로 계산될 때 사용할 하한을 정의한다.
P_MIN = float(np.finfo(float).tiny)

# Cohen's d 해석 구간을 정의한다.
_EFFECT_BANDS = ((0.2, "무시 가능"), (0.5, "작음"), (0.8, "중간"))


def fmt_p(p: float) -> str:
    """p-value를 문자열로 변환한다."""
    return f"p < {P_MIN:.3g}" if p == 0 else f"p = {p:.3g}"


def _effect_label(d: float) -> str:
    """Cohen's d를 관례적 구간명으로 바꾼다."""
    for bound, label in _EFFECT_BANDS:
        if abs(d) < bound:
            return label
    return "큼"


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """두 집단의 Cohen's d를 계산한다."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else float("nan")


def _interpret(r: dict, alpha: float) -> str:
    """p-value와 효과크기를 함께 해석한다."""
    a, b, d = r["group_a"], r["group_b"], r["cohens_d"]
    head = (f"{a['label']}({a['n']:,}건, 평균 {a['mean'] * 100:.2f}%) vs "
            f"{b['label']}({b['n']:,}건, 평균 {b['mean'] * 100:.2f}%) — "
            f"차이 {r['mean_diff'] * 100:.2f}%p, t = {r['t_statistic']:.2f}, "
            f"{fmt_p(r['p_value'])}, Cohen's d = {d:+.3f}({r['effect_size']}).")

    if not r["significant"]:
        return f"{head} p ≥ {alpha}이므로 차이가 있다고 볼 근거가 없다."
    if abs(d) < 0.2:
        return (f"{head} 통계적으로는 유의하지만 효과크기가 무시 가능한 수준이라 "
                f"실질적 의미는 없다. 표본이 {a['n'] + b['n']:,}건으로 커서 "
                f"미세한 차이도 유의해진 것이다.")
    if abs(d) < 0.5:
        return f"{head} 유의하며 효과크기는 작다. 해석 시 크기를 함께 밝혀야 한다."
    return f"{head} 유의하며 효과크기도 중간 이상이라 실질적으로 의미 있는 차이다."


def statistics_step(df: pd.DataFrame, cfg: Config) -> StepResult:
    """기술통계·상관계수·t-test를 수행한다. DataFrame은 바꾸지 않는다."""
    st = cfg.statistics
    notes: list[str] = []

    # 기술통계를 계산한다.
    cols = [c for c in st.describe_columns if c in df.columns]
    desc = df[cols].describe(percentiles=st.percentiles).T
    describe = {c: {k: float(v) for k, v in row.items()} for c, row in desc.iterrows()}

    # 평균과 중앙값의 차이가 큰 컬럼을 확인한다.
    skewed = {c: v["mean"] / v["50%"] for c, v in describe.items()
              if v.get("50%", 0) > 0 and v["mean"] / v["50%"] > 1.5}
    if skewed:
        notes.append(
            "[기술통계] 평균이 중앙값의 1.5배를 넘는 컬럼: "
            + ", ".join(f"{c} {r:.2f}배" for c, r in skewed.items())
            + ". 오른쪽 꼬리가 길어 평균을 대표값으로 쓰면 전형적인 운행을 "
              "설명하지 못한다.")

    # 범주형 분포는 full 소스만 사용하여 집계한다.
    full = df[df["record_source"] == "full"] if "record_source" in df.columns else df
    categorical = {}
    for col in cfg.columns.categorical:
        if col not in full.columns:
            continue
        vc = full[col].value_counts(dropna=False).head(6)
        categorical[col] = {("결측" if pd.isna(k) else str(k)): int(v)
                            for k, v in vc.items()}
    if categorical:
        notes.append(
            f"[범주형 분포] record_source='full' {len(full):,}행 한정으로 셌다. "
            f"partial 소스는 승객수·요율 컬럼 자체가 없어 분모에 넣으면 "
            f"'결측이 많은 컬럼'처럼 보이게 된다.")

    # Pearson과 Spearman 상관계수를 계산한다.
    ccols = [c for c in st.correlation_columns if c in df.columns]
    pearson = df[ccols].corr(method="pearson")
    spearman = df[ccols].corr(method="spearman")
    gaps = {f"{a}~{b}": float(abs(pearson.loc[a, b] - spearman.loc[a, b]))
            for i, a in enumerate(ccols) for b in ccols[i + 1:]}
    worst = max(gaps.items(), key=lambda kv: kv[1]) if gaps else ("", 0.0)
    notes.append(
        f"[상관계수] 피어슨과 스피어만이 가장 크게 어긋나는 쌍은 {worst[0]}"
        f"(차이 {worst[1]:.3f})다. 팁이 0인 건이 대량이라 순위 계산에서 동점이 "
        f"생긴 결과다. 상관은 인과가 아니다.")

    # 카드결제 데이터로 장거리와 단거리의 팁 비율을 비교한다.
    card = df[df["payment_type"] == 1]
    rate = card["tip_amount"] / card["fare_amount"].replace(0, np.nan)
    ok = rate.notna() & (rate < 2)      # 계산할 수 없는 값과 극단값을 제외
    rate, dist = rate[ok], card.loc[ok, "trip_distance"]

    thr = st.long_trip_threshold
    long, short = rate[dist >= thr], rate[dist < thr]

    # 두 집단의 분산이 다를 수 있으므로 Welch t-test를 사용한다.
    t_stat, p_value = stats.ttest_ind(long, short, equal_var=st.equal_var)
    d = _cohens_d(long, short)

    test = {
        "name": "장거리 vs 단거리 — 팁 비율",
        "question": "이동거리가 길면 팁 비율이 낮아지는가?",
        "population": f"카드결제 {len(rate):,}건 (payment_type=1)",
        "threshold_mile": thr,
        "group_a": {"label": f"장거리(≥{thr:g}mi)", "n": len(long),
                    "mean": float(long.mean()), "std": float(long.std(ddof=1))},
        "group_b": {"label": f"단거리(<{thr:g}mi)", "n": len(short),
                    "mean": float(short.mean()), "std": float(short.std(ddof=1))},
        "mean_diff": float(long.mean() - short.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < st.alpha),
        "cohens_d": d,
        "effect_size": _effect_label(d),
    }
    test["interpretation"] = _interpret(test, st.alpha)
    notes.append(f"[t-test] {test['interpretation']}")
    notes.append(
        "[한계] 같은 기사·차량이 하루에 여러 번 운행하므로 관측이 완전히 독립이 "
        "아니다. t-test는 독립성을 가정하며, 위반 시 p-value가 실제보다 작게 나온다. "
        "데이터에 기사·차량 식별자가 없어 보정할 수 없다.")

    log.info("t-test: t=%.2f p=%.3g d=%+.3f (%s)",
             t_stat, p_value, d, test["effect_size"])

    metrics = {
        "describe": describe,
        "categorical": categorical,
        "categorical_base_rows": len(full),
        "correlation": {
            "pearson": {a: {b: float(pearson.loc[a, b]) for b in ccols} for a in ccols},
            "spearman": {a: {b: float(spearman.loc[a, b]) for b in ccols} for a in ccols},
            "max_gap": {"pair": worst[0], "gap": worst[1]},
        },
        "ttest": test,
        "cohens_d": abs(d),
        "alpha": st.alpha,
    }
    return StepResult(df=df, metrics=metrics, notes=notes)
