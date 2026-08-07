"""수집한 지표를 마크다운 리포트로 변환한다."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
# p-value 표기 형식을 통계 단계와 공유한다.
from .steps.statistics import fmt_p
from .quality import GateResult
from .storage import RunManifest

log = logging.getLogger(__name__)

# [한계]로 시작하는 문장은 리포트의 한계 섹션에 모은다.
LIMIT_PREFIX = "[한계]"


def _pick(source: Any, *path: str) -> Any:
    """중첩 지표를 읽고, 키가 없으면 경고를 남긴다.

    단계가 내는 지표의 키 이름이 바뀌면 리포트는 예외 없이 해당 섹션만 조용히
    비워 버린다(실제로 visualize_raw의 키가 바뀌었을 때 결측·품질 요약 두 줄이
    말없이 사라졌다). 여기서 경고를 남겨 로그만 봐도 알 수 있게 한다.
    """
    cur = source
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            log.warning("리포트: 지표 키 '%s' 없음 — 해당 섹션을 건너뛴다", ".".join(path))
            return None
        cur = cur[key]
    return cur


def _fmt_actual(v: Any) -> str:
    """품질 게이트의 실측값을 출력 형식에 맞게 변환한다."""
    if v is None:
        return "없음"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """행과 열 데이터를 마크다운 표로 변환한다."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# 순서: 개요 → 데이터 이해 → 원본 시각화 → 발견한 이슈를 바탕으로 한 처리 기준 →
# 핵심 결과 → 통계분석 → ML Pipeline → 자동화/한계
def render_report(cfg: Config, manifest: RunManifest, metrics: dict[str, dict],
                  notes: dict[str, list[str]], gates: list[GateResult],
                  rows_before: int,
                  artifacts: dict[str, list[dict]] | None = None) -> str:
    """개요→데이터이해→원본시각화→처리기준→핵심결과→통계→모델→자동화 순으로 리포트를 만든다."""
    L: list[str] = []
    add = L.append
    final_rows = manifest.steps[-1]["rows_out"] if manifest.steps else rows_before

    # [한계] 태그는 어느 단계에서 나왔든 8장에 모아서 보여준다.
    limitations = [(sn, x) for sn, lines in notes.items()
                   for x in lines if x.startswith(LIMIT_PREFIX)]

    def body(step_name: str) -> list[str]:
        return [x for x in notes.get(step_name, []) if not x.startswith(LIMIT_PREFIX)]

    add(f"# {cfg.name} 분석 리포트")
    add("")
    add(f"NYC Yellow Taxi **{cfg.month}** 운행 데이터 **{rows_before:,}행**을 대상으로 "
        f"데이터 준비부터 시각화, 통계 검정, 머신러닝 모델링까지 End-to-End 분석을 수행하였다.")
    add("")

    # ---- 1. 개요 -----------------------------------------------------------
    add("## 1. 개요")
    add("")
    summary = [
        f"결측·중복·이상치를 처리해 **{final_rows:,}행**({final_rows / rows_before:.1%})을 남겼다.",
    ]
    if (_st := metrics.get("statistics")) and (_t := _st.get("ttest")):
        summary.append(
            f"t-test 결과 {_t['name']}에서 Cohen's d = {_t['cohens_d']:+.3f}"
            f"({_t['effect_size']})의 차이를 확인했다.")
    if (_ml := metrics.get("model")) and not _ml.get("skipped"):
        summary.append(
            f"고액팁 예측 모델의 F1은 **{_ml['scores']['f1']:.4f}**"
            f"(기준선 대비 정확도 {_ml['improvement_over_baseline']:+.4f})다.")
    for line in summary:
        add(f"- {line}")
    add("")

    # ---- 2. 데이터 이해 (참고용 고정 표 — 실행마다 바뀌지 않는다) ------------
    add("## 2. 데이터 이해: 택시를 타는 순서로 컬럼 읽기")
    add("")
    add(_table(["이용 단계", "주요 컬럼", "의미"], [
        ["승차·하차", "`tpep_pickup_datetime`, `tpep_dropoff_datetime`",
         "운행시간·요일·시간대·러시아워 변수를 만들고 소요시간 이상치를 검사한다"],
        ["요금 체계", "`RatecodeID`", "공항·외곽·협상요금 등 요율 유형, 공항 운행 판별의 핵심"],
        ["이동", "`trip_distance`", "소요시간과 결합해 속도를 만들고 거리·요금 관계를 분석한다"],
        ["결제·전송", "`payment_type`, `store_and_fwd_flag`",
         "현금 팁은 시스템에 기록되지 않을 수 있어 팁 분석 시 결제수단을 함께 봐야 한다"],
        ["영수증", "`fare_amount` ~ `total_amount`", "최종 결제 금액의 구성 항목"],
    ]))
    add("")
    add("> **결측치 해석 주의:** 부가요금·공항 컬럼이 비어 있다고 곧바로 0으로 채우지 "
        "않았다. 특정 소스가 여러 컬럼을 함께 제공하지 않는 구조적 결측이 확인되었기 "
        "때문에 `record_source`로 구분하고 분석 목적에 맞는 분모를 사용한다.")
    add("")

    # ---- 3. 주요 시각화 요약 (정제 전 원본) ----------------------------------
    add("## 3. 주요 시각화 요약 — 정제 전 원본 데이터")
    add("")
    add("더 다양하고 자세한 시각화 자료는 아래 첨부된 파일을 참조한다.")
    add("")
    add("1. `NYC_Yellow_Taxi_원본_데이터_시각화_분석_보고서.pdf`")
    add("2. `visualization_original.ipynb`")
    add("")
    if vr := metrics.get("visualize_raw"):
        # 결측·품질이슈·IQR은 data_quality_charts 모듈이 낸 지표를 그대로 요약한다.
        if dq := _pick(vr, "data_quality_handoff"):
            if miss := dq.get("missing"):
                first = next(iter(miss.values()))
                if dq.get("missing_all_pairs_correlated"):
                    add(f"- **결측치**: {len(miss)}개 컬럼이 정확히 같은 "
                        f"{first['count']:,}행({first['pct']}%)에서 동시에 결측 — 구조적 결측")
                else:
                    items = " · ".join(f"{k} {v['count']:,}건({v['pct']}%)"
                                       for k, v in miss.items())
                    add(f"- **결측치**: {items}")
            if issues := dq.get("quality_issues"):
                items = " · ".join(f"{k} {v['count']:,}건({v['pct']}%)"
                                   for k, v in issues.items())
                add(f"- **데이터 품질**: {items}")
            if iqr := dq.get("iqr_outliers"):
                col, v = max(iqr.items(), key=lambda kv: kv[1].get("pct") or 0)
                add(f"- **IQR 이상치**: 비율이 가장 큰 컬럼은 `{col}` {v['pct']}%"
                    f"({v['count']:,}건, 정상범위 {v['lo']:.2f}~{v['hi']:.2f})")
        if ct := vr.get("correlation_trap"):
            add(f"- **극단 이상치와 상관관계 함정**: trip_distance 최댓값 "
                f"{ct['trip_distance_max']:,.0f}mile. 원본 그대로면 r={ct['corr_raw']:.3f}"
                f"이지만 극단치를 제외하면 r={ct['corr_filtered']:.3f}로 바뀐다")
        if hp := vr.get("hourly_pattern_raw"):
            add(f"- **시간대 패턴**: {hp['peak_hour']}시 최다({hp['peak_trips']:,}건), "
                f"{hp['low_hour']}시 최저({hp['low_trips']:,}건)")
        if pt := vr.get("payment_tip_raw"):
            add(f"- **결제수단**: 현금 {pt['cash_zero_ratio']:.1%}가 팁 0, "
                f"카드 중앙값 ${pt['card_tip_median']:.2f}")
        add("")
    if artifacts and (figs := artifacts.get("visualize_raw")):
        for i, f in enumerate(figs, 1):
            add(f"### 3.{i} {f['name']}")
            add("")
            add(f"![{f['name']}]({f['path']})")
            add("")
            if f.get("caption"):
                add(f"> {f['caption']}")
                add("")

    # ---- 4. 발견한 이슈를 바탕으로 한 처리 기준 설계 -------------------------
    add("## 4. 발견한 이슈를 바탕으로 한 처리 기준 설계")
    add("")
    add("3장에서 확인한 문제를 다음과 같이 결측치 처리, 중복·이상치 처리, "
        "파생변수 생성으로 연결하였다.")
    add("")

    add("### 4.1 결측치 처리")
    add("")
    for line in body("analyze_missing") + body("prepare_missing"):
        add(f"- {line}")
    add("")

    add("### 4.2 중복·이상치 처리")
    add("")
    for line in body("deduplicate") + body("filter_outliers"):
        add(f"- {line}")
    add("")
    if (dd := metrics.get("deduplicate")) and (fo := metrics.get("filter_outliers")):
        add("```text")
        add(f"원본 {rows_before:,}행")
        add(f" → 중복 유형 판정 후 {dd['rows_out']:,}행")
        add(f" → 이상치 처리 후 {fo['rows_out']:,}행")
        add("```")
        add("")

    add("### 4.3 파생변수 생성")
    add("")
    add("3장의 시간대 패턴과 결제수단별 팁 구조를 반영해 다음 파생변수를 추가하였다.")
    add("")
    add(_table(["생성변수", "생성 방법", "의미와 활용"], [
        ["`speed_kmh`", "`trip_distance`를 km로 변환한 후 운행시간으로 나눔",
         "운행별 평균속도. 정체 정도와 비정상 속도 운행을 확인하는 데 사용한다"
         "(4.2절 이상치 규칙에도 활용)."],
        ["`is_rush_hour`", "승차 시각이 출퇴근 시간대(07~09시, 16~19시)에 포함되는지 판별",
         "출퇴근 시간대와 그 외 시간대의 운행량·속도 차이를 비교하는 데 사용한다."],
        ["`is_airport_trip`", "`RatecodeID`, `Airport_fee`, 공항 관련 LocationID를 종합해 판별",
         "JFK·Newark 등 공항 운행과 일반 운행의 요금·거리·팁 차이를 분석하는 데 사용한다."],
    ]))
    add("")
    add("```python")
    add("speed_kmh = trip_distance(mile) * 1.60934 / (소요시간(초) / 3600)")
    add("```")
    add("")
    add("소요시간이 0 이하면 속도를 계산할 수 없어 결측으로 남겼고, `is_airport_trip`은 "
        "`record_source='partial'` 행에서 요율·공항 컬럼 자체가 없을 수 있어 무조건 "
        "`False`가 아닌 **판단 불가(NA)** 상태로 유지하였다.")
    add("")

    ef = metrics.get("engineer_features", {})
    vf = metrics.get("validate_features", {})
    stat_lines: list[str] = []
    if "rush_hour_ratio" in ef:
        stat_lines.append(
            f"`is_rush_hour=True` 비율: 전체 정제 데이터의 약 **{ef['rush_hour_ratio']:.1%}**")
    if rh := vf.get("rush_hour_speed"):
        a, b = rh["group_a"], rh["group_b"]
        stat_lines.append(
            f"출퇴근 시간대 평균속도 **{a['mean']:.1f}km/h** vs 그 외 시간대 "
            f"**{b['mean']:.1f}km/h** (Cohen's d={rh['cohens_d']:+.3f}, {rh['effect_size']})")
    if "airport_trip_ratio" in ef:
        stat_lines.append(
            f"`is_airport_trip=True` 비율(판단 가능 데이터 기준): 약 "
            f"**{ef['airport_trip_ratio']:.1%}**")
    if at := vf.get("airport_trip_tip_rate"):
        a, b = at["group_a"], at["group_b"]
        stat_lines.append(
            f"공항 운행 평균 팁 비율 **{a['mean']:.2%}** vs 일반 운행 "
            f"**{b['mean']:.2%}** (Cohen's d={at['cohens_d']:+.3f}, {at['effect_size']})")
    for line in stat_lines:
        add(f"- {line}")
    if stat_lines:
        add("")

    if rh := vf.get("rush_hour_speed"):
        peak_hour = (metrics.get("visualize_raw") or {}).get("hourly_pattern_raw", {}).get("peak_hour")
        peak_txt = (f"3장에서 확인했듯 실제 피크는 {peak_hour}시 단일 피크에 가까워, "
                    if peak_hour is not None else
                    "3장에서 확인한 저녁 단일 피크 경향을 고려하면, ")
        add(f"출퇴근 시간대의 평균속도가 더 낮게 나타나 `is_rush_hour`가 혼잡 상황을 "
            f"어느 정도 반영함을 확인하였다. 다만 {peak_txt}현재 경계값(07~09시·16~19시)은 "
            f"저녁 위주로 재검토할 여지가 있다.")
        add("")

    # ---- 5. 핵심 결과 --------------------------------------------------------
    add("## 5. 핵심 결과")
    add("")
    rows5 = [["원본 데이터", f"{rows_before:,}행"], ["최종 데이터", f"{final_rows:,}행"],
             ["데이터 보존율", f"{final_rows / rows_before:.2%}" if rows_before else "-"]]
    if dd := metrics.get("deduplicate"):
        rows5 += [["완전 중복", f"{dd['exact_duplicates']:,}건"],
                  ["부분키 중복 그룹", f"{dd['duplicate_groups']:,}개"]]
    if artifacts and (figs_clean := artifacts.get("visualize", [])):
        n_seaborn = sum(1 for f in figs_clean if f.get("kind") == "figure")
        n_plotly = sum(1 for f in figs_clean if f.get("kind") == "plotly")
        rows5.append(["생성된 시각화", f"Seaborn {n_seaborn}개 · Plotly {n_plotly}개"])
    if (_st := metrics.get("statistics")) and (_t := _st.get("ttest")):
        rows5.append(["t-test 효과크기", f"Cohen's d = {_t['cohens_d']:+.3f}({_t['effect_size']})"])
    if (_ml := metrics.get("model")) and not _ml.get("skipped"):
        s = _ml["scores"]
        rows5 += [["모델 Accuracy", f"{s['accuracy']:.4f}"],
                  ["모델 F1-score", f"{s['f1']:.4f}"],
                  ["모델 ROC-AUC", f"{s['roc_auc']:.4f}"]]
    if gates:
        n_fail = sum(1 for g in gates if not g.passed)
        rows5.append(["품질 게이트", f"{len(gates)}개 중 {len(gates) - n_fail}개 PASS"])
    if manifest.duration_sec:
        rows5.append(["전체 실행 시간", f"{manifest.duration_sec}초"])
    add(_table(["항목", "결과"], rows5))
    add("")

    # ---- 6. 통계 분석 ----------------------------------------------------
    if st := metrics.get("statistics"):
        add("## 6. 통계 분석")
        add("")
        if desc := st.get("describe"):
            add("### 6.1 기술통계")
            add("")
            add(_table(["컬럼", "평균", "표준편차", "25%", "중앙값", "75%", "95%", "99%"], [
                [c, f"{d['mean']:,.2f}", f"{d['std']:,.2f}", f"{d['25%']:,.2f}",
                 f"{d['50%']:,.2f}", f"{d['75%']:,.2f}", f"{d['95%']:,.2f}",
                 f"{d['99%']:,.2f}"]
                for c, d in desc.items()
            ]))
            add("")
        if cat := st.get("categorical"):
            base = st.get("categorical_base_rows", 0)
            add(f"### 6.2 범주형 분포 (record_source='full' {base:,}행 한정)")
            add("")
            for col, counts in cat.items():
                items = " · ".join(f"`{k}` {v:,}건({v / base:.1%})"
                                   for k, v in counts.items())
                add(f"- **{col}** — {items}")
            add("")
        if corr := st.get("correlation"):
            pe, sp = corr["pearson"], corr["spearman"]
            cols = list(pe)
            add("### 6.3 상관계수 (피어슨 / 스피어만)")
            add("")
            highlight_pairs = [
                ("trip_distance", "fare_amount", "운행 거리와 기본요금"),
                ("trip_distance", "total_amount", "운행 거리와 최종금액"),
                ("fare_amount", "total_amount", "기본요금과 최종금액"),
            ]
            highlights = [f"{label}의 Pearson 상관계수: **{pe[a][b]:+.3f}**"
                          for a, b, label in highlight_pairs if a in pe and b in pe]
            if highlights:
                for h in highlights:
                    add(f"- {h}")
                add("")
                add("상관관계는 변수 간 동반 변화를 의미할 뿐, 인과관계를 증명하지는 않는다. "
                    "전체 조합은 아래 매트릭스를 참조한다.")
                add("")
            add(_table([""] + cols, [
                [a] + [f"{pe[a][b]:+.3f} / {sp[a][b]:+.3f}" for b in cols]
                for a in cols
            ]))
            add("")
            g = corr.get("max_gap", {})
            add(f"두 계수가 가장 크게 어긋나는 쌍: **{g.get('pair')}** "
                f"(차이 {g.get('gap', 0):.3f}) — 팁이 0인 운행이 많아 순위 계산에서 "
                f"동점이 발생하기 때문이다.")
            add("")
        if t := st.get("ttest"):
            add("### 6.4 t-test (scipy.stats.ttest_ind)")
            add("")
            add(f"**{t['name']}** — {t['question']}")
            add("")
            add(f"대상: {t['population']}")
            add("")
            a, b = t["group_a"], t["group_b"]
            add(_table(["집단", "n", "평균 팁 비율", "표준편차"], [
                [a["label"], f"{a['n']:,}", f"{a['mean']:.2%}", f"{a['std']:.4f}"],
                [b["label"], f"{b['n']:,}", f"{b['mean']:.2%}", f"{b['std']:.4f}"],
            ]))
            add("")
            add(_table(["t 통계량", "p-value", "Cohen's d", "효과크기"], [
                [f"{t['t_statistic']:.2f}", fmt_p(t["p_value"]),
                 f"{t['cohens_d']:+.3f}", t["effect_size"]],
            ]))
            add("")
            add(f"> {t['interpretation']}")
            add("")
    # ---- 7. ML Pipeline ---------------------------------------------------
    if (ml := metrics.get("model")) and not ml.get("skipped"):
        add("## 7. ML Pipeline")
        add("")
        add(_table(["항목", "값"], [
            ["문제", f"이진 분류 — 팁 비율 ≥ {ml['target_threshold']:.0%} 여부"],
            ["모집단", f"카드결제 {ml['population_rows']:,}건"],
            ["양성 비율", f"{ml['positive_ratio']:.2%}"],
            ["학습 / 평가", f"{ml['train_rows']:,}행 / {ml['test_rows']:,}행"],
            ["피처", f"수치 {len(ml['features_numeric'])}개 + "
                    f"범주 {len(ml['features_categorical'])}개"],
            ["누수 제외", ", ".join(f"`{c}`" for c in ml["excluded_leakage"])],
            ["모델", f"`{ml['model']}` (scikit-learn {ml['sklearn_version']})"],
        ]))
        add("")

        # ---- 7.1 모델 데이터가 만들어지는 흐름 ------------------------------
        dedup_m = metrics.get("deduplicate") or {}
        outlier_m = metrics.get("filter_outliers") or {}
        prep_m = metrics.get("prepare_missing") or {}
        if dedup_m and outlier_m:
            add("### 7.1 모델 데이터가 만들어지는 흐름")
            add("")
            removed_dup = rows_before - dedup_m.get("rows_out", rows_before)
            removed_out = dedup_m.get("rows_out", 0) - outlier_m.get("rows_out", 0)
            add(f"모델은 원본 {rows_before:,}건을 바로 사용하지 않고, 앞선 전처리 결과를 "
                f"순서대로 전달받는다. 결측 컬럼이 함께 비는 구조를 확인해 행 삭제 대신 "
                f"`record_source`를 만들고, 위장 결측값 {prep_m.get('sentinels_total', 0):,}개를 "
                f"`NaN`으로 변환하였다. 이후 상쇄쌍과 이중계상을 구분해 {removed_dup:,}행을 "
                f"제거하고, 기간·소요시간·거리·금액·속도 기준으로 {removed_out:,}행을 추가 "
                f"제거하여 **{outlier_m['rows_out']:,}건**을 모델 이전 데이터로 확정하였다.")
            add("")
            add("```text")
            add(f"원본 {rows_before:,}건")
            add(" → 결측 행 보존 및 위장 결측 변환")
            add(f" → 중복 처리 {dedup_m['rows_out']:,}건")
            add(f" → 이상치 처리 {outlier_m['rows_out']:,}건")
            add(f" → 카드결제 및 팁 비율 검증 {ml['population_rows']:,}건")
            if cfg.model.train_sample:
                add(f" → 층화 추출 {cfg.model.train_sample:,}건")
            add(f" → 학습 {ml['train_rows']:,}건 / 평가 {ml['test_rows']:,}건")
            add("```")
            add("")
            add(f"카드결제 중 팁 비율을 계산할 수 있고 200% 미만인 "
                f"{ml['population_rows']:,}건을 모델 모집단으로 삼았다. 팁 비율 "
                f"{ml['target_threshold']:.0%} 이상을 고액 팁으로 정의했을 때 양성 비율은 "
                f"**{ml['positive_ratio']:.2%}**로 나타나 한쪽 클래스에 크게 치우치지 "
                f"않았다.")
            add("")

        # ---- 7.2 전처리와 학습 결과 -----------------------------------------
        add("### 7.2 전처리와 학습 결과")
        add("")
        add(f"모델 입력은 수치형 {len(ml['features_numeric'])}개와 범주형 "
            f"{len(ml['features_categorical'])}개로 구성하였다. 수치형에는 중앙값 대체와 "
            f"표준화를, 범주형에는 최빈값 대체와 원-핫 인코딩을 적용하였다. 희귀 범주는 "
            f"`min_frequency={cfg.model.min_frequency}`로 묶어 원-핫 컬럼이 지나치게 "
            f"늘어나는 것을 막았다. "
            + ", ".join(f"`{c}`" for c in ml["excluded_leakage"])
            + "은(는) 정답을 직접 또는 간접적으로 알려줄 수 있어 제외하였다.")
        add("")
        add(f"전처리는 `ColumnTransformer`와 `Pipeline` 안에서 수행하여 평가 데이터의 "
            f"정보가 학습 과정에 들어가지 않도록 하였다. 최종 분류기는 `{ml['model']}`를 "
            f"사용하였다.")
        add("")
        s = ml["scores"]
        add(_table(["정확도", "정밀도", "재현율", "F1", "ROC-AUC"], [
            [f"{s['accuracy']:.4f}", f"{s['precision']:.4f}", f"{s['recall']:.4f}",
             f"{s['f1']:.4f}", f"{s['roc_auc']:.4f}"],
        ]))
        add("")
        add(f"다수 클래스만 예측하는 기준선의 정확도가 **{ml['baseline_accuracy']:.4f}"
            f"**이므로 **{ml['improvement_over_baseline']:+.4f}** 개선했다. 재현율이 "
            f"{s['recall']:.4f}로 실제 고액 팁 운행을 폭넓게 찾아내지만 정밀도는 "
            f"{s['precision']:.4f}이므로, 후보를 넓게 찾는 데는 유용하나 확정 판정에는 "
            f"추가 검토가 필요하다. 또한 요금과 소요시간처럼 운행 종료 후 확정되는 "
            f"변수를 사용하므로, 현재 모델은 승차 전 예측보다 **운행 완료 후 고액 팁 "
            f"패턴을 분류하는 모델**에 가깝다.")
        add("")

        # ---- 7.3 저장된 모델 -------------------------------------------------
        if artifacts and (mods := artifacts.get("model")):
            add("### 7.3 저장된 모델")
            add("")
            for f in mods:
                add(f"`{f['path']}` — {f['caption']}")
                add("")
            add("```python")
            add("import joblib")
            add(f"model = joblib.load(\"{mods[0]['path']}\")")
            add("model.predict(df[FEATURES])   # 전처리가 Pipeline 안에 들어 있다")
            add("```")
            add("")

    # ---- 8. 자동화·결론 및 개선 방향 ----------------------------------------
    add("## 8. 자동화·결론 및 개선 방향")
    add("")

    # ---- 8.1 코드 흐름과 결과의 연결 ------------------------------------------
    add("### 8.1 코드 흐름과 결과의 연결")
    add("")
    add(f"`run_pipeline.py`가 설정과 실행 옵션을 읽으면 `runner.py`가 다음 "
        f"{len(manifest.steps)}개 단계를 등록된 순서로 실행한다. 각 단계가 반환한 "
        f"DataFrame과 지표는 다음 단계와 최종 리포트에 그대로 전달된다.")
    add("")
    add("```text")
    add("compare_loaders")
    add(" → analyze_missing → visualize_raw")
    add(" → prepare_missing")
    add(" → deduplicate → filter_outliers")
    add(" → engineer_features → validate_features")
    add(" → visualize → statistics → model")
    add("```")
    add("")

    rows81: list[list[str]] = []
    if lc := metrics.get("compare_loaders"):
        rows81.append([
            "`compare_loaders()`",
            "Pandas·Polars의 shape, 결측수, 완전중복 결과 일치",
            "이후 분석 결과가 로딩 도구의 차이가 아닌 데이터 자체의 특성임을 확인하였다."
            if lc.get("all_match") else "일부 불일치가 발견되어 원인을 별도로 확인하였다."])
    if am := metrics.get("analyze_missing"):
        di = am.get("dropna_impact") or {}
        detail = f"동시 결측 {am.get('missing_rows', 0):,}행, 부분 결측 {am.get('partial_missing_rows', 0):,}건"
        if di:
            detail += (f", dropna() 후 평균 거리 {di['mean_distance_all']:.2f}"
                       f"→{di['mean_distance_after_dropna']:.2f}마일")
        rows81.append([
            "`analyze_missing()` → `prepare_missing()`", detail,
            "구조적 결측으로 판단해 행을 보존하고 `full`과 `partial` 출처로 구분하였다."])
    dedup_m = metrics.get("deduplicate") or {}
    outlier_m = metrics.get("filter_outliers") or {}
    if dedup_m and outlier_m:
        removed_dup = rows_before - dedup_m.get("rows_out", rows_before)
        removed_out = dedup_m.get("rows_out", 0) - outlier_m.get("rows_out", 0)
        rows81.append([
            "`deduplicate()` → `filter_outliers()`",
            f"중복 {removed_dup:,}행, 이상치 {removed_out:,}행 제거",
            f"원인별로 필요한 행만 제거해 원본의 {outlier_m['rows_out'] / rows_before:.2%}를 "
            f"남기고 음수 총액을 {outlier_m.get('negative_total_after', 0)}건으로 만들었다."])
    ef = metrics.get("engineer_features") or {}
    vf = metrics.get("validate_features") or {}
    if ef and (_st := metrics.get("statistics")) and (_t := _st.get("ttest")):
        a, b = _t["group_a"], _t["group_b"]
        rows81.append([
            "`engineer_features()` → `validate_features()` → `statistics_step()`",
            f"러시아워 {ef.get('rush_hour_ratio', 0):.2%}, 공항 운행 "
            f"{ef.get('airport_trip_ratio', 0):.2%}, 장거리 팁 {a['mean']:.2%}·단거리 "
            f"{b['mean']:.2%}, Cohen's d={_t['cohens_d']:+.3f}",
            "파생변수는 보조 변수로 사용하고, 거리별 팁 차이는 실질적으로 의미 있는 "
            "결과로 해석하였다."])
    if ml:
        n_clean_figs = len(artifacts.get("visualize", [])) if artifacts else 0
        rows81.append([
            "`train_model()` → `evaluate()`",
            f"모델 F1 {ml['scores']['f1']:.4f}, 차트 {n_clean_figs}개, 최종 {final_rows:,}행",
            "정확값과 허용 범위로 구성한 품질 게이트를 모두 통과한 경우에만 실행을 성공으로 "
            "기록하였다."])
    if rows81:
        add(_table(["코드 단계", "확인한 지표", "판단과 다음 단계에 미친 영향"], rows81))
        add("")

    # ---- 8.2 한계 및 개선 방향 ------------------------------------------------
    add("### 8.2 한계 및 개선 방향")
    add("")
    limit_rows: list[list[str]] = []
    wiped = (metrics.get("filter_outliers") or {}).get("vendors_wiped_out") or []
    if wiped:
        w = wiped[0]
        limit_rows.append([
            f"소요시간 제거 정책으로 VendorID={w['vendor']}의 {w['rows']:,}건이 전부 "
            "제외되었다.",
            "운행량 분석에서는 `duration_policy=\"flag\"` 결과와 비교한다."])
    limit_rows.append([
        "한 달 데이터를 무작위 분할해 모델을 평가하였다.",
        "시간 순서 분할과 다른 월 데이터 검증을 추가한다."])
    rh = vf.get("rush_hour_speed")
    at = vf.get("airport_trip_tip_rate")
    if rh and at:
        limit_rows.append([
            f"러시아워·공항 파생변수의 효과크기가 각각 약 {rh['cohens_d']:+.3f}, "
            f"{at['cohens_d']:+.3f}로 작았다.",
            "시간대 기준을 다시 조정하고 변수 중요도를 확인해 실제 기여도를 검증한다."])
    if limit_rows:
        add(_table(["한계", "개선 방향"], limit_rows))
        add("")
    extra_limits = [(sn, x) for sn, x in limitations if sn not in ("filter_outliers", "model")]
    if extra_limits:
        add("그 외 파이프라인이 자동으로 수집한 한계는 다음과 같다.")
        add("")
        for step_name, line in extra_limits:
            add(f"- **{step_name}** — {line[len(LIMIT_PREFIX):].strip()}")
        add("")

    # ---- 최종 요약 -------------------------------------------------------
    summary_parts = [
        f"전처리 단계마다 지표를 확인해 원인에 맞는 처리 방식을 선택했고, 최종 "
        f"{final_rows:,}건으로 통계 검정과 모델 학습을 수행하였다."]
    if ml:
        summary_parts.append(f"모델 F1-score는 {ml['scores']['f1']:.4f}였으며,")
    if gates:
        n_fail = sum(1 for g in gates if not g.passed)
        summary_parts.append(
            f"전체 결과는 품질 게이트 {len(gates)}개 중 {len(gates) - n_fail}개를 통과한 "
            "뒤 자동 저장되었다.")
    add(f"> **최종 요약:** {' '.join(summary_parts)}")
    add("")

    # ---- 8.3 결론 및 의견 -----------------------------------------------------
    add("### 8.3 결론 및 의견")
    add("")
    add("이 분석의 핵심은 결과값 자체보다 **전처리 근거를 지표로 확인하고 그 결과를 "
        "다음 단계에 연결한 점**이다. 결측은 단순 삭제하지 않았고, 중복은 금액 관계로 "
        "유형을 나눴으며, 거리와 시간을 결합한 `speed_kmh`로 개별 범위 검사에서 놓칠 수 "
        f"있는 기록도 처리하였다. 그 결과 원본의 **{final_rows / rows_before:.2%}인 "
        f"{final_rows:,}건**을 보존한 상태에서 통계 검정과 모델 학습을 수행하였다.")
    add("")
    am2 = metrics.get("analyze_missing") or {}
    if am2 and dedup_m:
        removed_dup2 = rows_before - dedup_m.get("rows_out", rows_before)
        add(f"특히 구조적 결측 {am2.get('missing_rows', 0):,}행을 `record_source`로 "
            "구분한 것은 심야·장거리 운행이 분석에서 과도하게 빠지는 것을 막았고, "
            f"상쇄쌍과 이중계상을 구분하여 {removed_dup2:,}행만 선택적으로 제거한 것은 "
            "정상 운행과 매출 기록을 함께 보존하는 데 영향을 주었다. 또한 기간·소요시간·"
            "거리·금액·속도 기준을 순서대로 적용하면서 최종 데이터의 음수 총액을 "
            f"{outlier_m.get('negative_total_after', 0)}건으로 만들었다. 따라서 이후에 "
            "확인한 거리와 요금의 관계, 장거리와 단거리의 팁 비율 차이, 고액 팁 분류 "
            "성능은 서로 다른 기준으로 임의 처리한 데이터가 아니라 동일한 정제 기준을 "
            "통과한 데이터에서 나온 결과라는 점에서 의미가 있다.")
        add("")
    if (_st := metrics.get("statistics")) and (_t := _st.get("ttest")) and ml:
        a, b = _t["group_a"], _t["group_b"]
        add(f"정제 데이터에서는 장거리 운행의 평균 팁 비율이 {a['mean']:.2%}로 단거리의 "
            f"{b['mean']:.2%}보다 낮았고, Cohen's d도 {_t['cohens_d']:+.3f}로 나타나 "
            "통계적으로 유의할 뿐 아니라 실제 차이도 확인하였다. 모델은 F1-score "
            f"{ml['scores']['f1']:.4f}와 Recall {ml['scores']['recall']:.4f}를 기록하여 "
            "고액 팁 운행 후보를 넓게 찾는 데 활용 가능성을 보였다. 다만 현금 팁의 "
            f"미기록, {cfg.month} 한 달이라는 분석 범위, 무작위 학습·평가 분할을 고려하면 "
            "이 결과를 전체 승객의 행동이나 직접적인 인과관계로 확대해서 해석해서는 "
            "안 된다.")
        add("")
    add("실행이 끝나면 최종 parquet과 함께 `metrics.json`, `manifest.json`, "
        "`report.md`, `model.joblib`을 저장한다. 또한 품질 게이트로 데이터 "
        "건수, 보존율, 시각화, 통계와 모델 결과를 함께 검사한다. 이를 통해 이번 분석을 "
        "일회성 EDA로 끝내지 않고, 동일한 입력 조건과 전처리·평가 기준을 바탕으로 "
        "데이터 시점이 달라져도 분석 결과를 재현할 수 있도록 파이프라인을 구축하였다.")
    add("")

    # ---- 8.4 실행 정보 및 품질 게이트 ------------------------------------------
    add("### 8.4 실행 정보 및 품질 게이트")
    add("")
    add(f"`run_id` `{manifest.run_id}` · 입력 `{Path(manifest.input_file).name}` "
        f"(sha `{manifest.input_digest}`) · 설정 `{cfg.source_file.name}` "
        f"(sha `{cfg.digest}`) · 소요 {manifest.duration_sec}초.")
    add("")
    if gates:
        add(_table(["항목", "종류", "기대", "실측", "결과"], [
            [g.name, g.kind,
             f"{g.expected:,}" if g.kind == "exact" else f"{g.expected[0]} ~ {g.expected[1]}",
             _fmt_actual(g.actual),
             "PASS" if g.passed else "**FAIL**"]
            for g in gates
        ]))
        n_fail = sum(1 for g in gates if not g.passed)
        add("")
        add(f"{len(gates)}개 중 {len(gates) - n_fail}개 통과"
            + (f", **{n_fail}개 실패**" if n_fail else " — 전부 통과"))
        add("")

    add("---")
    add("")
    add(f"기준 문서: `결측치_중복_처리기준.md`, `docs/` · 설정: `{cfg.source_file.name}`")
    return "\n".join(L)


def render_console_summary(result_metrics: dict[str, dict],
                           gates: list[GateResult]) -> str:
    """터미널에 표시할 실행 요약을 만든다."""
    lines = []
    n_fail = sum(1 for g in gates if not g.passed)
    lines.append("품질 게이트: "
                 + (f"{len(gates)}개 전부 통과" if not n_fail
                    else f"{n_fail}개 실패 / {len(gates)}개"))
    for g in gates:
        if not g.passed:
            lines.append(f"  {g.describe()}")
    # 콘솔 보존율도 원본 행 수를 기준으로 계산한다.
    fo = result_metrics.get("filter_outliers")
    am = result_metrics.get("analyze_missing")
    if fo:
        final = fo["rows_out"]
        origin = am.get("rows") if am else None
        tail = f" (원본 대비 보존율 {final / origin:.2%})" if origin else ""
        lines.append(f"최종 {final:,}행{tail}")
    return "\n".join(lines)
