"""리포트 렌더러(report.py) 테스트.

리포트는 "예외 없이 조용히 비는" 방식으로 고장난다. 지표 키 이름이 바뀌었을 때
결측·품질 요약 두 줄이 말없이 사라졌고, 게이트 문장의 인자가 뒤바뀐 것도
전부 통과일 때는 드러나지 않았다. 렌더러 테스트가 0건이어서 둘 다 못 잡았다.

그래서 여기서는 "무엇이 나와야 하는가"를 문자열로 고정한다.
"""

from __future__ import annotations

from helpers import CFG

from taxi_pipeline.quality import GateResult
from taxi_pipeline.report import render_report
from taxi_pipeline.storage import RunManifest

ROWS_BEFORE = 4_090_836
FINAL_ROWS = 3_884_062


def _manifest() -> RunManifest:
    """단계 기록이 채워진 매니페스트를 만든다."""
    m = RunManifest(
        run_id="20260807T000000Z_cfg_input", started_at="2026-08-07T00:00:00+00:00",
        config_file=str(CFG.source_file), config_digest=CFG.digest,
        input_file="yellow_tripdata_2026-05.parquet", input_digest="deadbeef",
        input_bytes=69_000_000,
    )
    m.duration_sec = 45.5
    m.steps = [
        {"name": "analyze_missing", "description": "결측 진단",
         "duration_sec": 0.4, "rows_in": ROWS_BEFORE, "rows_out": ROWS_BEFORE,
         "rows_delta": 0},
        {"name": "filter_outliers", "description": "이상치 제거",
         "duration_sec": 1.0, "rows_in": 4_048_148, "rows_out": FINAL_ROWS,
         "rows_delta": -164_086},
    ]
    return m


def _metrics() -> dict:
    """리포트가 읽는 지표를 실제 구조와 같은 모양으로 만든다."""
    return {
        "visualize_raw": {
            "data_quality_handoff": {
                "total_rows": ROWS_BEFORE,
                "missing": {c: {"count": 955_371, "pct": 23.35}
                            for c in CFG.columns.missing_group},
                "missing_all_pairs_correlated": True,
                "quality_issues": {"fare_amount 음수": {"count": 14_231, "pct": 0.35}},
                "iqr_outliers": {
                    "trip_distance": {"count": 457_656, "pct": 11.19,
                                      "q1": 1.04, "q3": 3.81, "lo": -3.12, "hi": 7.97},
                    "extra": {"count": 45_763, "pct": 1.12,
                              "q1": 0.0, "q3": 2.5, "lo": -3.75, "hi": 6.25},
                },
            },
            "correlation_trap": {"corr_raw": 0.008, "corr_filtered": 0.853,
                                 "trip_distance_max": 307_491.0},
            "hourly_pattern_raw": {"peak_hour": 18, "peak_trips": 283_663,
                                   "low_hour": 4, "low_trips": 32_504},
            "payment_tip_raw": {"cash_zero_ratio": 1.0, "card_tip_median": 3.3},
        },
        "deduplicate": {"exact_duplicates": 0, "duplicate_groups": 29_401,
                        "rows_out": 4_048_148},
        "filter_outliers": {"rows_out": FINAL_ROWS},
        "statistics": {
            "describe": {"trip_distance": {"mean": 3.52, "std": 4.25, "25%": 1.04,
                                           "50%": 1.93, "75%": 3.81, "95%": 10.9,
                                           "99%": 19.2}},
            "categorical": {"payment_type": {"1": 2_000_000, "2": 900_000}},
            "categorical_base_rows": 3_021_070,
            "correlation": {
                "pearson": {"trip_distance": {"trip_distance": 1.0}},
                "spearman": {"trip_distance": {"trip_distance": 1.0}},
                "max_gap": {"pair": "tip_amount~total_amount", "gap": 0.31},
            },
            "ttest": {
                "name": "장거리 vs 단거리 — 팁 비율",
                "question": "이동거리가 길면 팁 비율이 낮아지는가?",
                "population": "카드결제 2,659,779건",
                "group_a": {"label": "장거리(≥5mi)", "n": 477_000,
                            "mean": 0.1716, "std": 0.12},
                "group_b": {"label": "단거리(<5mi)", "n": 2_182_779,
                            "mean": 0.2585, "std": 0.12},
                "t_statistic": -450.87, "p_value": 0.0,
                "cohens_d": -0.703, "effect_size": "중간",
                "interpretation": "유의하며 효과크기도 중간 이상이다.",
            },
        },
        "model": {
            "target_threshold": 0.25, "population_rows": 2_659_779,
            "positive_ratio": 0.5613, "train_rows": 400_000, "test_rows": 100_000,
            "features_numeric": ["trip_distance"], "features_categorical": ["VendorID"],
            "excluded_leakage": ["tip_amount", "total_amount", "payment_type"],
            "model": "HistGradientBoostingClassifier", "sklearn_version": "1.7.2",
            "scores": {"accuracy": 0.7693, "precision": 0.79, "recall": 0.85,
                       "f1": 0.8206, "roc_auc": 0.7877},
            "baseline_accuracy": 0.5614, "improvement_over_baseline": 0.2079,
        },
    }


def _artifacts() -> dict:
    """원본 EDA 차트 8개가 저장된 상태를 흉내낸다."""
    names = ["raw_dq_missing_matrix", "raw_dq_missing_bar", "raw_dq_missing_heatmap",
             "raw_dq_quality_issues", "raw_dq_iqr_boxplot", "raw_03_correlation_trap",
             "raw_04_hourly_pattern", "raw_05_payment_tip"]
    return {
        "visualize_raw": [{"name": f"{n}.png", "kind": "figure",
                           "caption": f"{n} 설명", "path": f"figures/{n}.png"}
                          for n in names],
        "visualize": [{"name": "01_numeric_distributions.png", "kind": "figure",
                       "caption": "", "path": "figures/01_numeric_distributions.png"},
                      {"name": "08_demand_heatmap.png", "kind": "plotly",
                       "caption": "", "path": "figures/08_demand_heatmap.png"}],
        "model": [{"name": "model.joblib", "kind": "model",
                   "caption": "전처리 + 분류기", "path": "model.joblib"}],
    }


def _gates(n_fail: int = 0, total: int = 14) -> list[GateResult]:
    """total개 중 n_fail개가 실패한 게이트 목록을 만든다."""
    return [GateResult(name=f"gate_{i}", passed=i >= n_fail,
                       expected=100, actual=100, kind="exact")
            for i in range(total)]


def _render(metrics=None, notes=None, gates=None, artifacts=None) -> str:
    """기본 fixture로 리포트를 렌더한다."""
    return render_report(
        CFG, _manifest(),
        _metrics() if metrics is None else metrics,
        {} if notes is None else notes,
        _gates() if gates is None else gates,
        ROWS_BEFORE,
        _artifacts() if artifacts is None else artifacts,
    )


def test_여덟_개_장이_모두_나온다():
    """장 구성이 조용히 빠지면 리포트만 읽는 사람은 알 수 없다."""
    out = _render()
    for i, title in enumerate(
            ["개요", "데이터 이해", "주요 시각화 요약", "발견한 이슈",
             "핵심 결과", "통계 분석", "ML Pipeline", "자동화"], 1):
        assert f"## {i}. " in out, f"{i}장이 없다"
        assert title in out, f"{i}장 제목에 '{title}'이 없다"


def test_원본_EDA_차트_여덟_개가_임베드된다():
    """차트가 링크만 남고 그림이 안 보이면 리포트 자립성이 깨진다."""
    out = _render()
    assert out.count("![") == 8
    assert out.count("](figures/raw_") == 8


def test_데이터품질_요약_세_줄이_나온다():
    """data_quality_handoff 키를 읽지 못해 두 줄이 사라진 적이 있다(회귀 방지)."""
    out = _render()
    assert "**결측치**: 5개 컬럼이 정확히 같은 955,371행(23.35%)" in out
    assert "**데이터 품질**: fare_amount 음수 14,231건(0.35%)" in out
    # IQR 이상치는 비율이 가장 큰 컬럼을 골라야 한다 (11.19% > 1.12%)
    assert "**IQR 이상치**: 비율이 가장 큰 컬럼은 `trip_distance` 11.19%" in out


def test_지표_키가_없으면_섹션만_비고_예외는_안_난다():
    """--steps 부분 실행처럼 지표가 없을 때도 리포트는 만들어져야 한다."""
    metrics = _metrics()
    del metrics["visualize_raw"]["data_quality_handoff"]
    out = render_report(CFG, _manifest(), metrics, {}, _gates(), ROWS_BEFORE, None)
    assert "**결측치**" not in out
    assert "## 1. 개요" in out            # 나머지는 정상 렌더


def test_지표가_아예_없어도_렌더된다():
    """빈 metrics·notes·artifacts로도 죽지 않아야 한다."""
    out = render_report(CFG, _manifest(), {}, {}, [], ROWS_BEFORE, None)
    assert "## 1. 개요" in out and "## 5. 핵심 결과" in out


def test_게이트_실패_문장은_전체수가_앞에_온다():
    """'14개 중 12개 PASS'여야 한다. 인자가 뒤바뀌어 '12개 중 14개'로 나온 적이 있다."""
    out = _render(gates=_gates(n_fail=2))
    assert "| 품질 게이트 | 14개 중 12개 PASS |" in out


def test_게이트_전부_통과면_같은_수가_찍힌다():
    out = _render(gates=_gates(n_fail=0))
    assert "| 품질 게이트 | 14개 중 14개 PASS |" in out


def _last_chapter(out: str) -> str:
    """8장 본문만 잘라낸다.

    `"## 8."`으로 자르면 하위 절 `### 8.1`에도 걸리므로 줄 시작을 함께 맞춘다.
    """
    return out[out.index("\n## 8. "):]


def test_한계_문장은_마지막_장에_모인다():
    """[한계] 접두사가 붙은 note는 어느 단계에서 나왔든 8장에 모여야 한다."""
    notes = {"statistics": ["일반 근거 문장이다.",
                            "[한계] 관측이 완전히 독립이 아니다."]}
    tail = _last_chapter(_render(notes=notes))
    assert "관측이 완전히 독립이 아니다" in tail
    assert "[한계]" not in tail            # 접두사는 떼고 싣는다


def test_큐레이션된_한계는_자동수집에서_중복되지_않는다():
    """filter_outliers·model의 한계는 8.2 표가 이미 다루므로 불릿으로 또 싣지 않는다."""
    notes = {"filter_outliers": ["[한계] 소요시간 규칙이 사업자를 통째로 제거했다."]}
    tail = _last_chapter(_render(notes=notes))
    assert "소요시간 규칙이 사업자를 통째로 제거했다" not in tail
    # 대신 지표에서 만든 큐레이션 행이 그 내용을 담는다
    assert "한 달 데이터를 무작위 분할" in tail


def test_실행_이력이_마지막_장에_남는다():
    """run_id·입력·설정 해시가 없으면 결과를 재현할 수 없다."""
    tail = _last_chapter(_render())
    assert "20260807T000000Z_cfg_input" in tail
    assert CFG.digest in tail and "deadbeef" in tail
    assert "45.5초" in tail


def test_참조_자료_파일명이_실제_파일과_일치한다():
    """리포트가 안내하는 첨부 파일이 저장소에 실제로 있어야 한다."""
    from helpers import ROOT

    out = _render()
    for name in ("NYC_Yellow_Taxi_원본_데이터_시각화_분석_보고서.pdf",
                 "visualization_original.ipynb"):
        assert f"`{name}`" in out, f"리포트가 {name}을 안내하지 않는다"
        assert (ROOT / name).exists(), f"{name}이 저장소에 없다"
