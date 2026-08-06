"""수집한 지표를 마크다운 리포트로 변환한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
# p-value 표기 형식을 통계 단계와 공유한다.
from .steps.statistics import fmt_p
from .quality import GateResult
from .storage import RunManifest

_DOW = ["월", "화", "수", "목", "금", "토", "일"]
# [한계]로 시작하는 문장은 리포트의 한계 섹션에 모은다.
LIMIT_PREFIX = "[한계]"


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


def render_report(cfg: Config, manifest: RunManifest, metrics: dict[str, dict],
                  notes: dict[str, list[str]], gates: list[GateResult],
                  rows_before: int,
                  artifacts: dict[str, list[dict]] | None = None) -> str:
    """한 번의 실행 결과를 마크다운 리포트로 만든다."""
    L: list[str] = []
    add = L.append

    add(f"# {cfg.name} 실행 리포트")
    add("")

    # 분석 대상과 핵심 결과를 먼저 정리한다.
    add("## 개요")
    add("")
    final_rows = manifest.steps[-1]["rows_out"] if manifest.steps else rows_before
    summary = [
        f"NYC Yellow Taxi **{cfg.month}** 운행 기록 **{rows_before:,}행**을 정제하고 "
        f"EDA·통계 검정·분류 모델까지 수행한 자동 실행 결과다.",
        f"결측·중복·이상치를 처리해 **{final_rows:,}행**"
        f"({final_rows / rows_before:.1%})을 남겼다.",
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
    add("모든 처리에는 근거가 있으며 기준은 `결측치_중복_처리기준.md`와 `docs/`에 있다. "
        "이 리포트는 파이프라인이 자동 생성한다.")
    add("")

    add("## 실행 정보")
    add("")
    add(_table(["항목", "값"], [
        ["run_id", f"`{manifest.run_id}`"],
        ["실행 시각", manifest.started_at],
        ["입력", f"`{Path(manifest.input_file).name}` (sha `{manifest.input_digest}`)"],
        ["설정", f"`{cfg.source_file.name}` (sha `{cfg.digest}`)"],
        ["상태", "**성공**" if manifest.status == "success" else "**실패**"],
        ["소요", f"{manifest.duration_sec}초" if manifest.duration_sec else "-"],
    ]))
    add("")

    # 품질 게이트 결과를 분석 내용보다 먼저 표시한다.
    add("## 품질 게이트")
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
    else:
        add("설정된 게이트 없음")
    add("")

    # 단계별 처리 결과를 정리한다.
    add("## 데이터 준비")
    add("")
    add("### 단계별 처리")
    add("")
    add(_table(["단계", "설명", "행 변화", "증감", "소요"], [
        [s["name"], s["description"],
         f"{s['rows_in']:,} → {s['rows_out']:,}",
         f"{s['rows_delta']:+,}" if s["rows_delta"] else "-",
         f"{s['duration_sec']}초"]
        for s in manifest.steps
    ]))
    final = manifest.steps[-1]["rows_out"] if manifest.steps else rows_before
    add("")
    add(f"원본 {rows_before:,}행 → 최종 {final:,}행 "
        f"(**{final / rows_before:.2%} 보존**)" if rows_before else "")
    add("")

    # 일반 근거와 한계 문장을 분리한다.
    limitations: list[str] = []
    add("### 처리 근거")
    add("")
    for step_name, lines in notes.items():
        body = [x for x in lines if not x.startswith(LIMIT_PREFIX)]
        limitations += [(step_name, x) for x in lines if x.startswith(LIMIT_PREFIX)]
        if not body:
            continue
        add(f"**{step_name}**")
        add("")
        for line in body:
            add(f"- {line}")
        add("")

    # 로딩 비교 결과를 표시한다.
    if (lc := metrics.get("compare_loaders")) and not lc.get("skipped"):
        add("### Pandas · Polars 로딩 비교")
        add("")
        add(_table(["항목", f"polars {lc['polars_version']}",
                    f"pandas {lc['pandas_version']}"], [
            ["DataFrame 크기", f"{lc['polars_size_mb']:,.0f} MB",
             f"{lc['pandas_size_mb']:,.0f} MB"],
            ["부분키 중복 계산", f"{lc['partial_key_polars']:,} (구성원 전부)",
             f"{lc['partial_key_pandas']:,} (첫 행 제외)"],
        ]))
        add("")
        add(f"shape·컬럼별 결측수·완전중복이 "
            f"**{'모두 일치' if lc['all_match'] else '불일치'}**한다. "
            f"이후 분석은 도구 선택과 무관한 데이터 자체의 성질이다.")
        add("")
        if promoted := lc.get("dtype_promoted"):
            add("결측이 있는 정수 컬럼의 타입 복원이 다르다.")
            add("")
            add(_table(["컬럼", "pandas", "polars", "결측"], [
                [f"`{p['column']}`", p["pandas"], p["polars"], f"{p['nulls']:,}"]
                for p in promoted
            ]))
            add("")

    # 차트와 함께 캡션을 표시한다.
    if artifacts and (figs := artifacts.get("visualize")):
        add("## 시각화")
        add("")
        for i, f in enumerate(figs, 1):
            add(f"### {i}. {f['name']}")
            add("")
            add(f"![{f['name']}]({f['path']})")
            add("")
            if f.get("caption"):
                add(f"> {f['caption']}")
                add("")
            # 인터랙티브 차트의 HTML 링크를 함께 표시한다.
            if f.get("kind") == "plotly":
                html = f["path"].rsplit(".", 1)[0] + ".html"
                add(f"**[인터랙티브 버전 열기]({html})** — 셀에 마우스를 올리면 값이 표시됩니다")
                add("")

    # 통계분석 결과를 표시한다.
    if st := metrics.get("statistics"):
        add("## 통계분석")
        add("")

        # 평균과 중앙값을 함께 표시한다.
        if desc := st.get("describe"):
            add("### 기술통계")
            add("")
            add(_table(["컬럼", "평균", "표준편차", "25%", "중앙값", "75%", "95%", "99%"], [
                [c, f"{d['mean']:,.2f}", f"{d['std']:,.2f}", f"{d['25%']:,.2f}",
                 f"{d['50%']:,.2f}", f"{d['75%']:,.2f}", f"{d['95%']:,.2f}",
                 f"{d['99%']:,.2f}"]
                for c, d in desc.items()
            ]))
            add("")

        # 코드형 컬럼은 빈도로 표시한다.
        if cat := st.get("categorical"):
            base = st.get("categorical_base_rows", 0)
            add(f"### 범주형 분포 (record_source='full' {base:,}행 한정)")
            add("")
            for col, counts in cat.items():
                items = " · ".join(f"`{k}` {v:,}건({v / base:.1%})"
                                   for k, v in counts.items())
                add(f"- **{col}** — {items}")
            add("")

        # Pearson과 Spearman 값을 나란히 표시한다.
        if corr := st.get("correlation"):
            pe, sp = corr["pearson"], corr["spearman"]
            cols = list(pe)
            add("### 상관계수 (피어슨 / 스피어만)")
            add("")
            add(_table([""] + cols, [
                [a] + [f"{pe[a][b]:+.3f} / {sp[a][b]:+.3f}" for b in cols]
                for a in cols
            ]))
            add("")
            g = corr.get("max_gap", {})
            add(f"두 계수가 가장 크게 어긋나는 쌍: **{g.get('pair')}** "
                f"(차이 {g.get('gap', 0):.3f})")
            add("")

        # 검정값과 해석 문장을 함께 표시한다.
        if t := st.get("ttest"):
            add("### t-test (scipy.stats.ttest_ind)")
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

    # 모델 학습 결과를 표시한다.
    if (ml := metrics.get("model")) and not ml.get("skipped"):
        add("## ML Pipeline")
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

        # 모델 지표와 기준선을 함께 표시한다.
        s = ml["scores"]
        add("### 평가 지표")
        add("")
        add(_table(["정확도", "정밀도", "재현율", "F1", "ROC-AUC"], [
            [f"{s['accuracy']:.4f}", f"{s['precision']:.4f}", f"{s['recall']:.4f}",
             f"{s['f1']:.4f}", f"{s['roc_auc']:.4f}"],
        ]))
        add("")
        add(f"다수 클래스만 예측하는 기준선의 정확도가 **{ml['baseline_accuracy']:.4f}"
            f"**이므로 **{ml['improvement_over_baseline']:+.4f}** 개선했다. "
            f"기준선을 함께 보지 않으면 정확도 {s['accuracy']:.2f}가 좋은 값인지 "
            f"판단할 수 없다.")
        add("")

        # 저장된 모델 경로와 사용 예시를 표시한다.
        if artifacts and (mods := artifacts.get("model")):
            add("### 저장된 모델")
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

    # 단계별 한계 문장을 한곳에 모은다.
    if limitations:
        add("## 한계")
        add("")
        add("이 결과를 해석할 때 함께 고려해야 할 조건들이다.")
        add("")
        for step_name, line in limitations:
            add(f"- **{step_name}** — {line[len(LIMIT_PREFIX):].strip()}")
        add("")

    add("---")
    add("")
    add(f"기준 문서: `결측치_중복_처리기준.md`, `docs/` · 설정: `{cfg.source_file.name}`")
    return "\n".join(L)


# ################################################################################
# 🆕🆕🆕 [신규 2026-08-06 / 유길선] render_report_revised() 🆕🆕🆕
# render_report()는 건드리지 않고 그대로 둔다(기존 report.md는 계속 그 함수가 만듦).
# 이 함수는 별도로 report_revised.md를 만든다 — 순서를 "개요 → 데이터 이해 →
# 원본 시각화 → 발견한 이슈를 바탕으로 한 처리 기준 → 핵심 결과 → 통계분석 →
# ML Pipeline → 자동화/한계"로 재구성한 버전. 팀 논의 후 원본을 대체할지 정한다.
#
# 살린 것: 품질 게이트 상세 표, 실행 정보(run_id·해시), 단계별 처리 근거(notes)
# 자동 수집, [한계] 자동 수집, 차트 실제 임베드 — 위치만 새 순서에 맞게 옮겼다.
# ################################################################################
def render_report_revised(cfg: Config, manifest: RunManifest, metrics: dict[str, dict],
                          notes: dict[str, list[str]], gates: list[GateResult],
                          rows_before: int,
                          artifacts: dict[str, list[dict]] | None = None) -> str:
    """개편된 순서(개요→데이터이해→원본시각화→처리기준→핵심결과→통계→모델→자동화)로 리포트를 만든다."""
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
    if vr := metrics.get("visualize_raw"):
        if mp := vr.get("missing_pattern"):
            add(f"- **결측치**: 5개 컬럼이 정확히 같은 {mp['rows_all_missing']:,}행"
                f"({mp['missing_ratio']:.2%})에서 결측 — 구조적 결측")
        if qi := vr.get("quality_issues"):
            items = " · ".join(f"{k} {v:,}건" for k, v in qi["issues"].items())
            add(f"- **데이터 품질**: {items}")
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
    for line in body("engineer_features") + body("validate_features"):
        add(f"- {line}")
    add("")

    # ---- 5. 핵심 결과 --------------------------------------------------------
    add("## 5. 핵심 결과")
    add("")
    rows5 = [["원본 데이터", f"{rows_before:,}행"], ["최종 데이터", f"{final_rows:,}행"],
             ["데이터 보존율", f"{final_rows / rows_before:.2%}" if rows_before else "-"]]
    if dd := metrics.get("deduplicate"):
        rows5 += [["완전 중복", f"{dd['exact_duplicates']:,}건"],
                  ["부분키 중복 그룹", f"{dd['duplicate_groups']:,}개"]]
    if artifacts:
        n_fig = len(artifacts.get("visualize", [])) + len(artifacts.get("visualize_raw", []))
        if n_fig:
            rows5.append(["생성된 시각화", f"{n_fig}개"])
    if (_st := metrics.get("statistics")) and (_t := _st.get("ttest")):
        rows5.append(["t-test 효과크기", f"Cohen's d = {_t['cohens_d']:+.3f}({_t['effect_size']})"])
    if (_ml := metrics.get("model")) and not _ml.get("skipped"):
        s = _ml["scores"]
        rows5 += [["모델 Accuracy", f"{s['accuracy']:.4f}"],
                  ["모델 F1-score", f"{s['f1']:.4f}"],
                  ["모델 ROC-AUC", f"{s['roc_auc']:.4f}"]]
    if gates:
        n_fail = sum(1 for g in gates if not g.passed)
        rows5.append(["품질 게이트", f"{len(gates) - n_fail}개 중 {len(gates)}개 PASS"])
    if manifest.duration_sec:
        rows5.append(["전체 실행 시간", f"{manifest.duration_sec}초"])
    add(_table(["항목", "결과"], rows5))
    add("")

    # ---- 6. 통계 분석 (render_report와 동일 로직) ----------------------------
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
            add(_table([""] + cols, [
                [a] + [f"{pe[a][b]:+.3f} / {sp[a][b]:+.3f}" for b in cols]
                for a in cols
            ]))
            add("")
            g = corr.get("max_gap", {})
            add(f"두 계수가 가장 크게 어긋나는 쌍: **{g.get('pair')}** "
                f"(차이 {g.get('gap', 0):.3f})")
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
        if vf := metrics.get("validate_features"):
            add("### 6.5 파생변수 보조 검증")
            add("")
            for line in body("validate_features"):
                add(f"- {line}")
            add("")

    # ---- 7. ML Pipeline (render_report와 동일 로직) --------------------------
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
        s = ml["scores"]
        add("### 7.1 평가 지표")
        add("")
        add(_table(["정확도", "정밀도", "재현율", "F1", "ROC-AUC"], [
            [f"{s['accuracy']:.4f}", f"{s['precision']:.4f}", f"{s['recall']:.4f}",
             f"{s['f1']:.4f}", f"{s['roc_auc']:.4f}"],
        ]))
        add("")
        add(f"다수 클래스만 예측하는 기준선의 정확도가 **{ml['baseline_accuracy']:.4f}"
            f"**이므로 **{ml['improvement_over_baseline']:+.4f}** 개선했다.")
        add("")
        if artifacts and (mods := artifacts.get("model")):
            add("### 7.2 저장된 모델")
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

    # ---- 8. 자동화·품질 검증·한계 --------------------------------------------
    add("## 8. 자동화·품질 검증 및 한계")
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
    if limitations:
        add("이 결과를 해석할 때 함께 고려해야 할 조건들이다.")
        add("")
        for step_name, line in limitations:
            add(f"- **{step_name}** — {line[len(LIMIT_PREFIX):].strip()}")
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
