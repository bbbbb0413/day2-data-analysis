"""파생변수 추가 단계"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from .base import StepResult

log = logging.getLogger(__name__)

# ################################################################################
# 🆕🆕🆕 [신규 2026-08-06 / 유길선] 파생변수 상수 🆕🆕🆕
# 다른 임계값(예: outliers.speed_max_kmh)은 config/pipeline.toml에서 관리하지만,
# 이 두 상수는 일부러 여기 로컬 상수로 뒀다 — config.py의 Config dataclass를
# 새로 늘리는 건 파이프라인 구조 담당(윤서준) 영역이라, 최소 수정 원칙에 따라
# 일단 이렇게 두고 필요해지면 TOML로 옮기기로 함.
# ################################################################################
RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}     # 출퇴근 시간대: 아침 7~9시, 저녁 16~19시
AIRPORT_RATECODES = {2, 3}                 # TLC 코드북: 2=JFK, 3=Newark


def engineer_features(df: pd.DataFrame, cfg: Config) -> StepResult:
    """is_rush_hour · is_airport_trip 파생변수를 추가한다."""
    pu = cfg.columns.pickup
    notes: list[str] = []

    # ---- 🆕 is_rush_hour ---------------------------------------------------
    # 승차시각(pu)은 전 컬럼 중 결측 0건(§2.1)이라 record_source와 무관하게
    # 모든 행에서 계산 가능하다.
    df["is_rush_hour"] = df[pu].dt.hour.isin(RUSH_HOURS)

    # ---- 🆕 is_airport_trip -------------------------------------------------
    # RatecodeID·Airport_fee는 §2의 5개 구조적 결측 컬럼에 속한다.
    # record_source="partial" 행은 이 두 컬럼 자체가 없으므로, False로 채우면
    # "공항 아님"과 "모름"을 섞어 fillna(0)이 congestion_surcharge를 왜곡시켰던
    # 것(§2.5)과 같은 실수가 된다. 그래서 partial 행은 True/False 대신 명시적
    # 결측(pd.NA)으로 남긴다 — §2.7 기준 ①·⑤와 같은 원칙을 그대로 적용한 것.
    is_airport = df["RatecodeID"].isin(AIRPORT_RATECODES) | (df["Airport_fee"] > 0)
    is_airport = is_airport.astype("boolean")           # nullable boolean (pd.NA 허용)
    if "record_source" in df.columns:
        is_airport = is_airport.mask(df["record_source"] == "partial", pd.NA)
        notes.append(
            "[파생변수] is_airport_trip은 record_source='partial' 행(§2.7)에서 "
            "RatecodeID·Airport_fee 자체가 없어 False가 아닌 결측(NA)으로 남겼다.")
    df["is_airport_trip"] = is_airport

    rush_ratio = float(df["is_rush_hour"].mean()) if len(df) else float("nan")
    airport_known = df["is_airport_trip"].dropna()
    airport_ratio = float(airport_known.mean()) if len(airport_known) else float("nan")
    notes.append(
        f"[파생변수] is_rush_hour=True 비율 {rush_ratio:.1%} · "
        f"is_airport_trip=True 비율 {airport_ratio:.1%} "
        f"(결측 {len(df) - len(airport_known):,}행 제외)")

    metrics = {
        "rows": len(df),
        "rush_hour_ratio": rush_ratio,
        "airport_trip_ratio": airport_ratio,
        "airport_trip_unknown": int(len(df) - len(airport_known)),
    }
    log.info("파생변수 추가: is_rush_hour=%.1f%% is_airport_trip=%.1f%%(결측 제외)",
             rush_ratio * 100, airport_ratio * 100)
    return StepResult(df=df, metrics=metrics, notes=notes)
