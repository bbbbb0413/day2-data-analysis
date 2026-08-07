"""설정 로딩·입력 확보·JSON 직렬화 테스트 — 파이프라인 바깥 껍데기를 고정한다."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from helpers import CFG, ROOT

from taxi_pipeline import fetch
from taxi_pipeline.config import load_config
from taxi_pipeline.fetch import build_url
from taxi_pipeline.observability import jsonable
from taxi_pipeline.storage import RunManifest


# ---------------------------------------------------------------- 직렬화
def test_dataclass가_객체로_직렬화된다():
    """jsonable이 dataclass를 str()로 뭉개면 manifest.json이 문자열 한 줄이 된다.

    예외가 안 나고 파일도 정상적으로 생겨서 한참 뒤에야 발견되는 종류의 버그라
    테스트로 고정해 둔다.
    """
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
    payload = {"a": np.int64(3), "b": np.float64(1.5), "c": np.bool_(True),
               "d": np.float64("nan"), "e": pd.Timestamp("2026-05-01")}
    out = json.loads(json.dumps(jsonable(payload)))
    assert out["a"] == 3 and out["b"] == 1.5 and out["c"] is True
    assert out["d"] is None                       # NaN은 JSON에 없으므로 null
    assert out["e"].startswith("2026-05-01")


# ---------------------------------------------------------------- 입력 확보
def test_URL에_월이_치환된다():
    """month를 바꾸면 다운로드 URL이 따라가야 한다 (경로·URL 이중 관리 방지)."""
    assert build_url(CFG).endswith("yellow_tripdata_2026-05.parquet")
    assert build_url(replace(CFG, month="2026-06")).endswith("2026-06.parquet")


def test_원본_경로도_월을_따라간다():
    """paths.raw의 {month}가 치환되지 않으면 다른 달을 돌릴 때 예전 파일을 재사용한다."""
    cfg = load_config(ROOT / "config" / "pipeline.toml", root=ROOT)
    assert cfg.paths.raw.name == f"yellow_tripdata_{cfg.month}.parquet"


def test_파일이_있으면_다운로드하지_않는다():
    """ensure_input은 기존 파일을 건드리지 않아야 한다.

    네트워크를 타면 테스트가 느리고 불안정해지므로, download를 호출하면
    바로 실패하도록 바꿔치기해 '호출되지 않음'을 검증한다.
    """
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
    try:
        replace(CFG.outliers, duration_policy="ignore")
    except ValueError:
        return
    raise AssertionError("잘못된 duration_policy가 통과했다")
