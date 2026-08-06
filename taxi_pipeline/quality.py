"""단계별 지표가 설정한 품질 기준을 충족하는지 확인한다."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    """품질 게이트 한 항목의 검사 결과를 저장한다."""

    name: str
    passed: bool
    expected: Any
    actual: Any
    kind: str            # "exact" 또는 "range"이다.

    def describe(self) -> str:
        """검사 결과를 한 줄 문자열로 반환한다."""
        mark = "PASS" if self.passed else "FAIL"
        # 일부 단계만 실행하면 실측값이 없을 수 있다.
        actual = "없음" if self.actual is None else (
            f"{self.actual:,}" if isinstance(self.actual, int)
            else (f"{self.actual:.6f}" if isinstance(self.actual, float)
                  else str(self.actual)))
        if self.kind == "exact":
            return f"[{mark}] {self.name}: 기대 {self.expected:,} / 실측 {actual}"
        lo, hi = self.expected
        return f"[{mark}] {self.name}: 허용 {lo}~{hi} / 실측 {actual}"


def collect_gate_values(metrics: dict[str, dict]) -> dict[str, Any]:
    """단계별 지표에서 품질 검사에 사용할 값을 모은다."""
    m = metrics
    out: dict[str, Any] = {}

    if am := m.get("analyze_missing"):
        out["total_rows"] = am.get("rows")
        out["missing_rows"] = am.get("missing_rows")
        out["missing_ratio"] = am.get("missing_ratio")
    if dd := m.get("deduplicate"):
        out["exact_duplicates"] = dd.get("exact_duplicates")
        out["duplicate_groups"] = dd.get("duplicate_groups")
        out["void_pairs"] = dd.get("void_pairs")
        out["double_rows"] = dd.get("double_rows")
    if stt := m.get("statistics"):
        # 통계 지표는 다른 월에도 적용할 수 있도록 범위로 검사한다.
        out["cohens_d"] = stt.get("cohens_d")
    if ml := m.get("model"):
        # F1이 지나치게 높으면 누수를 의심할 수 있다.
        out["f1"] = ml.get("f1")
        out["train_rows"] = ml.get("train_rows")
    if vz := m.get("visualize"):
        # 차트 생성 개수로 누락 여부를 확인한다.
        out["figure_count"] = vz.get("figure_count")
    if fo := m.get("filter_outliers"):
        out["final_rows"] = fo.get("rows_out")
        out["negative_total"] = fo.get("negative_total_after")

    # 보존율은 원본 행 수를 기준으로 계산한다.
    if out.get("total_rows") and out.get("final_rows") is not None:
        out["retention_ratio"] = out["final_rows"] / out["total_rows"]
    return out


def evaluate(metrics: dict[str, dict], cfg: Config) -> list[GateResult]:
    """설정의 기대값과 실측값을 비교한다."""
    values = collect_gate_values(metrics)
    exp = cfg.expectations
    results: list[GateResult] = []

    if exp.check_exact:
        for name, expected in exp.exact.items():
            actual = values.get(name)
            results.append(GateResult(
                name=name, passed=actual == expected,
                expected=expected, actual=actual, kind="exact"))
    else:
        log.info("check_exact=false — 정확값 검사를 건너뛴다(기준 데이터가 아닌 경우)")

    for name, bounds in exp.range.items():
        # range 항목이 min과 max를 가진 dict인지 확인한다.
        if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
            raise ValueError(
                f"[expectations.range] {name} 은 {{min=..., max=...}} 형태여야 합니다"
                f" (현재: {bounds!r}). TOML에서 스칼라 키가 하위 테이블 뒤에 오면"
                " 그 테이블에 속하게 됩니다.")
        actual = values.get(name)
        lo, hi = bounds["min"], bounds["max"]
        ok = actual is not None and lo <= actual <= hi
        results.append(GateResult(
            name=name, passed=bool(ok), expected=(lo, hi), actual=actual, kind="range"))

    for r in results:
        (log.info if r.passed else log.error)(r.describe())
    return results


def failures(results: list[GateResult]) -> list[str]:
    """실패한 품질 게이트의 설명을 반환한다."""
    return [r.describe() for r in results if not r.passed]
