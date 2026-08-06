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
