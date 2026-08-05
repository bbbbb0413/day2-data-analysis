"""품질 게이트 — 자동화의 안전장치.

스케줄러는 로그를 읽지 않는다. 사람이 매일 아침 출력을 확인해 줄 거라고
가정하면 안 된다. 기대와 다르면 **프로세스가 0이 아닌 코드로 죽어야** 알림이 울린다.

두 종류를 검사한다.
  exact : 기준 데이터(2026-05)에서 재현되어야 하는 정확한 값.
          로직 회귀(리팩터링하다 결과가 바뀌는 사고)를 잡는다.
  range : 다른 월에도 통하는 일반 규칙(보존율·결측률 등).
          데이터 자체의 이상(공급처 포맷 변경 등)을 잡는다.

다른 월을 돌릴 때는 설정에서 check_exact=false 로 두고 range만 검사한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    """게이트 검사 하나의 결과.

    kind가 "exact"면 expected는 정확값, "range"면 (하한, 상한) 튜플이다.
    actual이 None일 수 있다 — --steps로 일부 단계만 돌리면 지표가 없다.
    """

    name: str
    passed: bool
    expected: Any
    actual: Any
    kind: str            # "exact" | "range"

    def describe(self) -> str:
        """로그와 리포트에 쓰는 한 줄 설명."""
        mark = "PASS" if self.passed else "FAIL"
        # 지표가 아예 없을 수 있다(--steps로 일부 단계만 돌린 경우).
        # None을 숫자 포맷에 넣으면 TypeError가 나므로 별도로 처리한다.
        actual = "없음" if self.actual is None else (
            f"{self.actual:,}" if isinstance(self.actual, int)
            else (f"{self.actual:.6f}" if isinstance(self.actual, float)
                  else str(self.actual)))
        if self.kind == "exact":
            return f"[{mark}] {self.name}: 기대 {self.expected:,} / 실측 {actual}"
        lo, hi = self.expected
        return f"[{mark}] {self.name}: 허용 {lo}~{hi} / 실측 {actual}"


def collect_gate_values(metrics: dict[str, dict]) -> dict[str, Any]:
    """단계별 지표에서 게이트가 검사할 값만 뽑아 평평하게 만든다.

    게이트 설정이 단계 이름을 몰라도 되게 하는 층이다. 단계를 리팩터링해
    지표 위치가 바뀌어도 이 함수만 고치면 설정 파일은 그대로 쓸 수 있다.
    """
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
        # 검정 결과가 무너지면 데이터 이상 신호다. 정확한 값을 고정하면
        # 다른 달 데이터에서 바로 실패하므로 범위로 검사한다.
        out["cohens_d"] = stt.get("cohens_d")
    if ml := m.get("model"):
        # F1 상한을 두는 이유: 지나치게 높으면 성능이 아니라 누수 신호다.
        out["f1"] = ml.get("f1")
        out["train_rows"] = ml.get("train_rows")
    if vz := m.get("visualize"):
        # 차트 생성 실패를 조용히 넘기지 않는다
        out["figure_count"] = vz.get("figure_count")
    if fo := m.get("filter_outliers"):
        out["final_rows"] = fo.get("rows_out")
        out["negative_total"] = fo.get("negative_total_after")

    # 보존율은 원본 대비로 계산한다(단계별 보존율이 아니라 전체 파이프라인 기준)
    if out.get("total_rows") and out.get("final_rows") is not None:
        out["retention_ratio"] = out["final_rows"] / out["total_rows"]
    return out


def evaluate(metrics: dict[str, dict], cfg: Config) -> list[GateResult]:
    """설정의 기대값과 실측을 대조한다."""
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
        # 설정 오타를 여기서 명확히 잡는다. TOML에서 하위 테이블 순서를 잘못 두면
        # 스칼라 값이 range 안으로 딸려 들어와 AttributeError로 터진다.
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
    """실패한 게이트 설명만 뽑는다. 매니페스트와 종료 코드 결정에 쓴다."""
    return [r.describe() for r in results if not r.passed]
