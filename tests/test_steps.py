"""단계 단위 테스트 — 409만 행을 읽지 않고 로직만 검증한다.

단계를 (DataFrame, Config) -> StepResult 순수 함수로 만든 진짜 이유가 이것이다.
파일 IO와 출력이 섞여 있으면 작은 입력으로 로직만 떼어 검증할 수 없다.

여기서 잡고 싶은 사고:
  - 리팩터링하다 중복 판정 기준이 미묘하게 바뀌는 것
  - keep='first' 같은 '편한 방법'이 슬쩍 들어오는 것
  - sentinel 목록에서 값이 빠지는 것

실행: .venv/bin/python -m pytest tests/ -v
      (pytest 없이) .venv/bin/python tests/test_steps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from taxi_pipeline.config import load_config                       # noqa: E402
from taxi_pipeline.steps.duplicates import _classify, deduplicate  # noqa: E402
from taxi_pipeline.steps.features import engineer_features         # noqa: E402  🆕 [2026-08-06/유길선]
from taxi_pipeline.steps.missing import analyze_missing, prepare_missing  # noqa: E402
from taxi_pipeline.steps.outliers import filter_outliers           # noqa: E402

CFG = load_config(ROOT / "config" / "pipeline.toml", root=ROOT)


def _row(pickup, dropoff, pu=100, do=200, dist=1.0, fare=10.0, total=15.0,
         vendor=2, payment=1, passengers=1.0, rate=1.0, cong=2.5, air=0.0):
    """테스트용 행 하나. 필요한 컬럼만 채운다."""
    return {
        "VendorID": vendor,
        "tpep_pickup_datetime": pd.Timestamp(pickup),
        "tpep_dropoff_datetime": pd.Timestamp(dropoff),
        "passenger_count": passengers,
        "trip_distance": dist,
        "RatecodeID": rate,
        "store_and_fwd_flag": "N",
        "PULocationID": pu,
        "DOLocationID": do,
        "payment_type": payment,
        "fare_amount": fare,
        "extra": 0.0,
        "mta_tax": 0.5,
        "tip_amount": 0.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "total_amount": total,
        "congestion_surcharge": cong,
        "Airport_fee": air,
        "cbd_congestion_fee": 0.0,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 결측
def test_구조적_결측을_감지한다():
    """5개 컬럼이 같은 행에서만 비면 structural로 판정해야 한다."""
    rows = [_row("2026-05-01 00:00", "2026-05-01 00:10")]
    bad = _row("2026-05-01 01:00", "2026-05-01 01:10")
    for col in CFG.columns.missing_group:
        bad[col] = None
    rows.append(bad)

    res = analyze_missing(_frame(rows), CFG)
    assert res.metrics["missing_rows"] == 1
    assert res.metrics["partial_missing_rows"] == 0
    assert res.metrics["is_structural"] is True


def test_일부만_결측이면_구조적이_아니다():
    """한 컬럼만 비는 행이 섞이면 structural=False여야 한다."""
    rows = [_row("2026-05-01 00:00", "2026-05-01 00:10")]
    partial = _row("2026-05-01 01:00", "2026-05-01 01:10")
    partial["RatecodeID"] = None            # 5개 중 하나만 결측
    rows.append(partial)

    res = analyze_missing(_frame(rows), CFG)
    assert res.metrics["partial_missing_rows"] == 1
    assert res.metrics["is_structural"] is False


def test_결측_처리는_행을_지우지_않는다():
    """prepare_missing은 절대 행 수를 줄이면 안 된다 (문서 §2.7 기준 ①)."""
    rows = [_row("2026-05-01 00:00", "2026-05-01 00:10")]
    bad = _row("2026-05-01 01:00", "2026-05-01 01:10")
    for col in CFG.columns.missing_group:
        bad[col] = None
    rows.append(bad)

    res = prepare_missing(_frame(rows), CFG)
    assert len(res.df) == 2
    assert res.metrics["rows_dropped"] == 0
    assert set(res.df["record_source"]) == {"full", "partial"}


def test_위장_결측이_NaN으로_바뀐다():
    """payment_type=0, RatecodeID=99 등은 NaN이 되어야 한다 (문서 §2.6)."""
    rows = [
        _row("2026-05-01 00:00", "2026-05-01 00:10", payment=0),
        _row("2026-05-01 01:00", "2026-05-01 01:10", rate=99),
        _row("2026-05-01 02:00", "2026-05-01 02:10", passengers=0),
        _row("2026-05-01 03:00", "2026-05-01 03:10", pu=264),
    ]
    res = prepare_missing(_frame(rows), CFG)
    assert res.df["payment_type"].isna().sum() == 1
    assert res.df["RatecodeID"].isna().sum() == 1
    assert res.df["passenger_count"].isna().sum() == 1
    assert res.df["PULocationID"].isna().sum() == 1
    assert res.metrics["sentinels_total"] == 4


# ---------------------------------------------------------------- 중복
def test_상쇄쌍은_양쪽_모두_제거된다():
    """합이 0인 쌍은 두 행 다 사라져야 한다 (문서 §3.6 ①)."""
    rows = [
        _row("2026-05-01 00:02", "2026-05-01 00:12", fare=-12.1, total=-17.85),
        _row("2026-05-01 00:02", "2026-05-01 00:12", fare=12.1, total=17.85),
        _row("2026-05-01 05:00", "2026-05-01 05:10"),      # 무관한 정상 행
    ]
    res = deduplicate(_frame(rows), CFG)
    assert res.metrics["void_pairs"] == 1
    assert res.metrics["rows_dropped"] == 2
    assert len(res.df) == 1
    assert (res.df["total_amount"] > 0).all()


def test_이중계상은_큰_쪽만_제거된다():
    """총액이 정확히 2배인 쌍은 큰 쪽만 사라져야 한다 (문서 §3.6 ②)."""
    rows = [
        _row("2026-05-01 03:39", "2026-05-01 04:04", total=41.48, payment=0),
        _row("2026-05-01 03:39", "2026-05-01 04:04", total=82.96, payment=0),
    ]
    res = deduplicate(_frame(rows), CFG)
    assert res.metrics["double_rows"] == 1
    assert res.metrics["rows_dropped"] == 1
    assert len(res.df) == 1
    assert res.df["total_amount"].iloc[0] == 41.48      # 작은 쪽이 남는다


def test_keep_first_였다면_정상운행이_지워진다():
    """naive 비교 지표가 'keep=first는 위험'을 실제로 보여줘야 한다 (문서 §3.5)."""
    rows = [
        _row("2026-05-01 00:02", "2026-05-01 00:12", fare=-12.1, total=-17.85),
        _row("2026-05-01 00:02", "2026-05-01 00:12", fare=12.1, total=17.85),
    ]
    res = deduplicate(_frame(rows), CFG)
    naive = res.metrics["naive_keep_first"]
    # 음수가 먼저 오므로 keep='first'는 양수(정상 운행)를 지운다
    assert naive["positive_rows_dropped"] == 1
    assert naive["negative_rows_left"] == 1


def test_서로_다른_운행은_유지된다():
    """상쇄쌍도 2배도 아니면 건드리지 않는다 (문서 §3.6 ③)."""
    rows = [
        _row("2026-05-01 00:02", "2026-05-01 00:12", total=16.02),
        _row("2026-05-01 00:02", "2026-05-01 00:12", total=20.16),
    ]
    res = deduplicate(_frame(rows), CFG)
    assert res.metrics["other_groups"] == 1
    assert res.metrics["rows_dropped"] == 0
    assert len(res.df) == 2


def test_중복이_없으면_아무것도_안_한다():
    rows = [
        _row("2026-05-01 00:02", "2026-05-01 00:12"),
        _row("2026-05-01 01:02", "2026-05-01 01:12"),
    ]
    res = deduplicate(_frame(rows), CFG)
    assert res.metrics["duplicate_groups"] == 0
    assert len(res.df) == 2


def test_분류_함수는_빈_입력을_견딘다():
    """중복이 하나도 없을 때 _classify가 죽지 않아야 한다."""
    df = _frame([_row("2026-05-01 00:02", "2026-05-01 00:12")])
    lo, hi, void, double = _classify(df, CFG)
    assert len(lo) == 0 and len(hi) == 0
    assert void.size == 0 and double.size == 0


# ---------------------------------------------------------------- 이상치
def test_기간_이탈은_승차시각으로만_판정한다():
    """하차가 다음 달로 넘어간 심야운행은 살아야 한다 (문서 §4)."""
    rows = [
        _row("2026-05-31 23:50", "2026-06-01 00:10"),   # 정상 심야운행
        _row("2026-04-30 23:50", "2026-04-30 23:59"),   # 기간 이탈
    ]
    res = filter_outliers(_frame(rows), CFG)
    assert len(res.df) == 1
    assert res.df["tpep_pickup_datetime"].iloc[0].day == 31


def test_소요시간_0이하는_제거된다():
    rows = [
        _row("2026-05-01 00:00", "2026-05-01 00:10"),
        _row("2026-05-01 01:00", "2026-05-01 01:00"),   # 0초
        _row("2026-05-01 02:00", "2026-05-01 01:50"),   # 역전
    ]
    res = filter_outliers(_frame(rows), CFG)
    assert len(res.df) == 1


def test_flag_정책이면_소요시간_이상행이_남는다():
    """duration_policy='flag'는 삭제 대신 표시만 해야 한다 (문서 §6-5)."""
    from dataclasses import replace

    cfg = replace(CFG, outliers=replace(CFG.outliers, duration_policy="flag"))
    rows = [
        _row("2026-05-01 00:00", "2026-05-01 00:10"),
        _row("2026-05-01 01:00", "2026-05-01 01:00"),   # 0초
    ]
    res = filter_outliers(_frame(rows), cfg)
    assert len(res.df) == 2                              # 지우지 않았다
    assert res.df["duration_valid"].tolist() == [True, False]


def test_사업자_전멸을_경고한다():
    """특정 VendorID가 필터로 전량 사라지면 지표에 남아야 한다 (문서 §4 보완)."""
    rows = [
        _row("2026-05-01 00:00", "2026-05-01 00:10", vendor=2),
        _row("2026-05-01 01:00", "2026-05-01 01:00", vendor=7),   # 0초, vendor 7뿐
    ]
    res = filter_outliers(_frame(rows), CFG)
    wiped = res.metrics["vendors_wiped_out"]
    assert len(wiped) == 1
    assert wiped[0]["vendor"] == 7
    assert wiped[0]["nonpositive_duration_ratio"] == 1.0
    assert any("경고" in n for n in res.notes)


def test_거리와_금액_이상치가_제거된다():
    rows = [
        _row("2026-05-01 00:00", "2026-05-01 00:10"),
        _row("2026-05-01 01:00", "2026-05-01 01:10", dist=0.0),      # 거리 0
        _row("2026-05-01 02:00", "2026-05-01 02:10", dist=500.0),    # 거리 과다
        _row("2026-05-01 03:00", "2026-05-01 03:10", total=0.0),     # 총액 0
        _row("2026-05-01 04:00", "2026-05-01 04:10", fare=-5.0),     # 요금 음수
    ]
    res = filter_outliers(_frame(rows), CFG)
    assert len(res.df) == 1
    assert res.metrics["rows_dropped"] == 4


# ---------------------------------------------------------------- 파생변수
# 🆕🆕🆕 [신규 2026-08-06 / 유길선] engineer_features 단계 테스트 🆕🆕🆕
def test_러시아워는_승차시각_기준으로_판정된다():
    """아침 8시·저녁 17시는 True, 낮 13시는 False여야 한다."""
    rows = [
        _row("2026-05-01 08:00", "2026-05-01 08:10"),   # 아침 러시아워
        _row("2026-05-01 17:00", "2026-05-01 17:10"),   # 저녁 러시아워
        _row("2026-05-01 13:00", "2026-05-01 13:10"),   # 러시아워 아님
    ]
    res = engineer_features(_frame(rows), CFG)
    assert res.df["is_rush_hour"].tolist() == [True, True, False]


def test_공항트립은_요율_또는_요금으로_판정되고_partial행은_결측이다():
    """RatecodeID 공항코드나 Airport_fee>0이면 True, partial 행(§2.7)은 NA여야 한다."""
    rows = [
        _row("2026-05-01 00:00", "2026-05-01 00:10", rate=2),      # JFK 요율
        _row("2026-05-01 01:00", "2026-05-01 01:10", air=1.75),    # 공항 부가금
        _row("2026-05-01 02:00", "2026-05-01 02:10"),              # 일반 트립
    ]
    partial = _row("2026-05-01 03:00", "2026-05-01 03:10")
    for col in CFG.columns.missing_group:
        partial[col] = None
    rows.append(partial)

    df = prepare_missing(_frame(rows), CFG).df
    res = engineer_features(df, CFG)
    assert res.df["is_airport_trip"].tolist()[:3] == [True, True, False]
    assert pd.isna(res.df["is_airport_trip"].iloc[3])   # False가 아니라 결측이어야 함


# ---------------------------------------------------------------- 통계분석
def test_효과크기_구간이_경계에서_바뀐다():
    """Cohen's d 구간 경계는 해석 문장을 좌우하므로 고정해 둔다."""
    from taxi_pipeline.steps.statistics import _effect_label

    assert _effect_label(0.19) == "무시 가능"
    assert _effect_label(0.2) == "작음"
    assert _effect_label(0.5) == "중간"
    assert _effect_label(-0.9) == "큼"          # 부호와 무관하게 크기로 판정


def test_p값_언더플로는_하한으로_표기된다():
    """scipy가 0.0을 돌려줄 때 'p = 0'으로 쓰면 사실과 다르다."""
    from taxi_pipeline.steps.statistics import P_MIN, fmt_p

    assert fmt_p(0.0) == f"p < {P_MIN:.3g}"
    assert fmt_p(0.03) == "p = 0.03"


def test_큰_표본에서_작은_차이는_유의하지만_무시_가능으로_해석된다():
    """p-value만 보면 '유의'인데 효과크기가 작으면 해석 문장이 달라져야 한다."""
    from taxi_pipeline.steps.statistics import _interpret

    r = {"group_a": {"label": "A", "n": 500_000, "mean": 0.2447, "std": 0.13},
         "group_b": {"label": "B", "n": 2_000_000, "mean": 0.2426, "std": 0.13},
         "mean_diff": 0.0021, "t_statistic": 9.47, "p_value": 2.8e-21,
         "significant": True, "cohens_d": 0.016, "effect_size": "무시 가능"}
    text = _interpret(r, 0.05)
    assert "실질적 의미는 없다" in text
    assert "2,500,000건으로 커서" in text        # 표본 크기를 원인으로 지목


def test_효과크기가_중간이면_의미_있는_차이로_해석된다():
    from taxi_pipeline.steps.statistics import _interpret

    r = {"group_a": {"label": "장거리", "n": 477_000, "mean": 0.1716, "std": 0.12},
         "group_b": {"label": "단거리", "n": 2_182_779, "mean": 0.2585, "std": 0.12},
         "mean_diff": -0.0869, "t_statistic": -450.96, "p_value": 0.0,
         "significant": True, "cohens_d": -0.703, "effect_size": "중간"}
    text = _interpret(r, 0.05)
    assert "실질적으로 의미 있는 차이" in text
    assert "p < " in text                        # 언더플로 하한 표기


def test_유의하지_않으면_차이_근거_없음으로_해석된다():
    from taxi_pipeline.steps.statistics import _interpret

    r = {"group_a": {"label": "A", "n": 100, "mean": 0.10, "std": 0.1},
         "group_b": {"label": "B", "n": 100, "mean": 0.11, "std": 0.1},
         "mean_diff": -0.01, "t_statistic": -0.7, "p_value": 0.48,
         "significant": False, "cohens_d": -0.1, "effect_size": "무시 가능"}
    assert "근거가 없다" in _interpret(r, 0.05)


def test_ttest는_ttest_ind를_쓴다():
    """검정 구현이 조용히 다른 것으로 바뀌지 않게 고정한다.

    직접 구현한 t 통계량은 자유도 계산을 틀리기 쉽다.
    """
    import inspect

    from taxi_pipeline.steps import statistics as st

    assert "stats.ttest_ind" in inspect.getsource(st.statistics_step)


# ---------------------------------------------------------------- ML Pipeline
def test_누수_컬럼이_피처에_없다():
    """total_amount는 팁을 포함한 합계라 넣으면 정답을 입력하는 것과 같다.

    이 목록이 흔들리면 모델 성능이 비현실적으로 높아지므로 고정한다.
    """
    from taxi_pipeline.steps.model import CATEGORICAL, LEAKAGE, NUMERIC

    assert set(LEAKAGE) == {"tip_amount", "total_amount", "payment_type"}
    for col in LEAKAGE:
        assert col not in NUMERIC and col not in CATEGORICAL


def test_전처리가_Pipeline_안에_있다():
    """전처리를 Pipeline 밖에서 하면 테스트셋 정보가 학습에 새어 들어간다."""
    from sklearn.pipeline import Pipeline

    from taxi_pipeline.steps.model import CATEGORICAL, NUMERIC, _build_pipeline

    pipe = _build_pipeline(CFG, NUMERIC, CATEGORICAL)
    assert isinstance(pipe, Pipeline)
    assert [n for n, _ in pipe.steps] == ["preprocess", "classifier"]

    # 전처리 안에 결측 대체·스케일링·원-핫이 모두 들어 있어야 한다
    pre = pipe.named_steps["preprocess"]
    inner = {n for name, trans, _ in pre.transformers for n, _ in trans.steps}
    assert {"imputer", "scaler", "onehot"} <= inner


def test_표본이_적으면_학습을_건너뛴다():
    """데이터가 말라붙었을 때 예외로 죽지 않고 건너뛰어야 한다."""
    from taxi_pipeline.steps.model import train_model

    rows = [_row("2026-05-01 00:00", "2026-05-01 00:10") for _ in range(5)]
    res = train_model(_frame(rows), CFG)
    assert res.metrics.get("skipped") is True
    assert res.artifacts == []


def test_모델이_joblib으로_저장되고_다시_불러진다(tmp_path=None):
    """저장한 Pipeline이 그대로 다시 예측 가능해야 한다."""
    import tempfile

    import joblib
    from sklearn.pipeline import Pipeline

    from taxi_pipeline.steps.model import CATEGORICAL, NUMERIC, _build_pipeline

    pipe = _build_pipeline(CFG, NUMERIC, CATEGORICAL)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.joblib"
        joblib.dump(pipe, p)
        assert p.stat().st_size > 0
        assert isinstance(joblib.load(p), Pipeline)


# ---------------------------------------------------------------- 직렬화
def test_dataclass가_객체로_직렬화된다():
    """jsonable이 dataclass를 str()로 뭉개면 manifest.json이 문자열 한 줄이 된다.

    예외가 안 나고 파일도 정상적으로 생겨서 한참 뒤에야 발견되는 종류의 버그라
    테스트로 고정해 둔다.
    """
    import json

    from taxi_pipeline.observability import jsonable
    from taxi_pipeline.storage import RunManifest

    m = RunManifest(
        run_id="x", started_at="t", config_file="c", config_digest="d",
        input_file="i", input_digest="h", input_bytes=1,
    )
    out = json.loads(json.dumps(jsonable(m)))
    assert isinstance(out, dict), f"dict가 아니라 {type(out).__name__}로 직렬화됐다"
    assert out["run_id"] == "x"
    assert out["steps"] == [] and out["outputs"] == {}


def test_numpy_타입이_직렬화된다():
    """metrics에 numpy 값이 섞여도 JSON 덤프가 죽지 않아야 한다."""
    import json

    import numpy as np

    from taxi_pipeline.observability import jsonable

    payload = {"a": np.int64(3), "b": np.float64(1.5), "c": np.bool_(True),
               "d": np.float64("nan"), "e": pd.Timestamp("2026-05-01")}
    out = json.loads(json.dumps(jsonable(payload)))
    assert out["a"] == 3 and out["b"] == 1.5 and out["c"] is True
    assert out["d"] is None                       # NaN은 JSON에 없으므로 null
    assert out["e"].startswith("2026-05-01")


# ---------------------------------------------------------------- 입력 확보
def test_URL에_월이_치환된다():
    """month를 바꾸면 다운로드 URL이 따라가야 한다 (경로·URL 이중 관리 방지)."""
    from dataclasses import replace

    from taxi_pipeline.fetch import build_url

    assert build_url(CFG).endswith("yellow_tripdata_2026-05.parquet")
    assert build_url(replace(CFG, month="2026-06")).endswith("2026-06.parquet")


def test_파일이_있으면_다운로드하지_않는다():
    """ensure_input은 기존 파일을 건드리지 않아야 한다.

    네트워크를 타면 테스트가 느리고 불안정해지므로, download를 호출하면
    바로 실패하도록 바꿔치기해 '호출되지 않음'을 검증한다.
    """
    from taxi_pipeline import fetch

    if not CFG.paths.raw.is_file():
        return                                    # 원본이 없는 환경에서는 건너뛴다

    original = fetch.download
    fetch.download = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("파일이 있는데 다운로드를 시도했다"))
    try:
        assert fetch.ensure_input(CFG) == CFG.paths.raw
    finally:
        fetch.download = original


def test_auto_download_꺼져있고_파일_없으면_안내와_함께_실패한다():
    """네트워크가 막힌 환경에서 조용히 멈추지 않고 명확히 실패해야 한다."""
    from dataclasses import replace

    from taxi_pipeline import fetch

    cfg = replace(
        CFG,
        source=replace(CFG.source, auto_download=False),
        paths=replace(CFG.paths, raw=ROOT / "data" / "raw" / "__없는파일__.parquet"),
    )
    try:
        fetch.ensure_input(cfg)
    except FileNotFoundError as e:
        assert "auto_download" in str(e) and "fetch" in str(e)
        return
    raise AssertionError("파일이 없는데 예외가 나지 않았다")


# ---------------------------------------------------------------- 설정
def test_설정_해시는_내용이_같으면_같다():
    a = load_config(ROOT / "config" / "pipeline.toml", root=ROOT)
    b = load_config(ROOT / "config" / "pipeline.toml", root=ROOT)
    assert a.digest == b.digest and len(a.digest) == 12


def test_잘못된_소요시간_정책은_거부된다():
    from dataclasses import replace

    try:
        replace(CFG.outliers, duration_policy="ignore")
    except ValueError:
        return
    raise AssertionError("잘못된 duration_policy가 통과했다")


if __name__ == "__main__":
    # pytest 없이도 돌아가게 해 둔다. 자동화 환경에 pytest가 없을 수 있다.
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
