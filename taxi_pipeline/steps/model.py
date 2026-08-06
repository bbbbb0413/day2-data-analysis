"""[단계 8] ML Pipeline — 문서 04_ML파이프라인_근거.md

고액팁(팁 비율 25% 이상) 여부를 예측하는 이진 분류 모델을 학습·평가하고 저장한다.

문제 정의는 docs/01_분석설계_근거.md 에서 데이터로 확정했다.
  타깃   : 팁 비율 ≥ 25% (고액팁 여부) — 양성 56.13%로 클래스 균형이 좋다
  모집단 : payment_type == 1 (카드결제) — 현금은 팁이 100% 0으로 기록된다
  표본   : 50만 층화 추출 — 전체 266만 대비 ROC-AUC 차이 +0.0005, 시간은 5.7배

★ 전처리를 Pipeline 안에 넣는다.
  대체값·스케일링 기준을 전체 데이터로 미리 계산하면 테스트셋 정보가 학습에
  새어 들어간다(전처리 누수). Pipeline은 fit에서 학습셋 기준만 학습하고
  transform에서 적용하므로 이 실수를 구조적으로 막는다.

★ 타깃과 산술적으로 연결된 컬럼은 절대 피처에 넣지 않는다.
  total_amount는 팁을 포함한 합계라 역산이 가능하다. 넣으면 정확도가 0.99를
  넘지만 아무것도 예측한 게 아니며, 운행 시작 시점에는 알 수 없는 값이다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import Config
from .base import Artifact, StepResult

log = logging.getLogger(__name__)

# 피처 목록 (STEP 1 §B-4에서 확정)
# 모두 '운행이 끝난 시점에 확정되는 관측값'이다.
# 🆕🆕🆕 [2026-08-06 / 유길선] speed_kmh·is_rush_hour·is_airport_trip 추가 🆕🆕🆕
# 원래 이 파일은 내 담당이 아니지만, "전처리·파생변수 전/후로 모델 성능이
# 바뀌는가"를 비교하려면 파생변수가 실제로 피처에 들어가 있어야 해서 최소
# 범위로 건드림. X = card[[... if c in card.columns]] 라 이 컬럼들이 없는
# 데이터(예: raw 비교용 실행)로 돌려도 에러 없이 자동으로 빠진다.
# LEAKAGE 세트는 절대 건드리지 않았다(테스트로 고정돼 있음).
NUMERIC = ["trip_distance", "fare_amount", "duration_min", "passenger_count",
           "hour", "dayofweek", "extra", "tolls_amount", "congestion_surcharge",
           "speed_kmh"]
CATEGORICAL = ["VendorID", "RatecodeID", "PULocationID", "DOLocationID",
               "is_rush_hour", "is_airport_trip"]

# 절대 피처로 쓰지 않는 컬럼. 타깃과 산술적으로 연결돼 있다.
LEAKAGE = ["tip_amount", "total_amount", "payment_type"]


def _prepare(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, dict]:
    """모집단을 좁히고 타깃과 피처를 만든다."""
    m = cfg.model

    # 카드결제 한정. 현금·무료·분쟁은 팁이 예외 없이 0으로 기록되므로,
    # 섞으면 모델이 '결제수단 맞히기'를 하게 된다.
    card = df[df["payment_type"] == m.payment_type].copy()

    # 타깃: 팁 비율. 절대금액이 아니라 비율을 쓰는 이유는 100달러 운행의 10달러와
    # 10달러 운행의 10달러가 전혀 다른 행동인데 금액으로는 같기 때문이다.
    rate = card["tip_amount"] / card["fare_amount"].replace(0, np.nan)
    ok = rate.notna() & (rate < 2)          # 요금 0(계산 불가)과 극단값 제외
    card, rate = card[ok], rate[ok]

    # 시각 컬럼은 그대로 모델에 넣을 수 없어 파생 피처로 바꾼다
    card["duration_min"] = ((card[cfg.columns.dropoff] - card[cfg.columns.pickup])
                            .dt.total_seconds() / 60)
    card["hour"] = card[cfg.columns.pickup].dt.hour
    card["dayofweek"] = card[cfg.columns.pickup].dt.dayofweek

    y = (rate >= m.target_tip_rate).astype(int)
    X = card[[c for c in NUMERIC + CATEGORICAL if c in card.columns]]

    info = {
        "population_rows": len(card),
        "positive_ratio": float(y.mean()),
        "target_threshold": m.target_tip_rate,
        "features_numeric": [c for c in NUMERIC if c in card.columns],
        "features_categorical": [c for c in CATEGORICAL if c in card.columns],
        "excluded_leakage": LEAKAGE,
    }
    return X, y, info


def _build_pipeline(cfg: Config, num: list[str], cat: list[str]) -> Pipeline:
    """전처리 + 모델을 하나의 Pipeline으로 묶는다.

    전처리를 Pipeline 밖에서 미리 하면 대체값·스케일 기준이 테스트셋까지 보고
    계산되어 평가 점수가 부풀려진다. Pipeline이 그 실수를 막는다.
    """
    m = cfg.model
    pre = ColumnTransformer([
        # 수치형: 결측 대체 → 스케일링
        #   스케일링은 트리 모델 성능에 영향이 없지만, 모델을 선형 계열로
        #   바꿀 때 전처리를 다시 짜지 않아도 되게 넣어 둔다.
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        # 범주형: 결측 대체 → 원-핫
        #   min_frequency로 희귀 범주를 하나로 묶는다. 존이 255개나 되어
        #   제한 없이 원-핫하면 509컬럼(밀집 10GB)이 되어 학습이 불가능하다.
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   min_frequency=m.min_frequency))]),
         cat),
    ], sparse_threshold=0.0)   # HistGradientBoosting은 희소 행렬을 받지 못한다

    return Pipeline([
        ("preprocess", pre),
        ("classifier", HistGradientBoostingClassifier(random_state=m.random_state)),
    ])


def train_model(df: pd.DataFrame, cfg: Config) -> StepResult:
    """모델을 학습·평가하고 joblib으로 저장한다. DataFrame은 바꾸지 않는다."""
    import joblib
    import sklearn

    m = cfg.model
    X, y, info = _prepare(df, cfg)
    notes: list[str] = []

    if len(X) < 1000:
        log.warning("모델 학습 생략 — 표본이 %d행뿐입니다", len(X))
        return StepResult(df=df, metrics={"skipped": True, "rows": len(X)},
                          notes=["[모델] 표본이 부족해 학습을 건너뛰었다."])

    # 층화 추출: 타깃 비율을 보존해야 실행 간 성능 비교가 공정하다.
    # 전체 266만으로 학습해도 ROC-AUC는 +0.0005 오르는 데 그치고 시간은 5.7배 든다.
    if m.train_sample and len(X) > m.train_sample:
        idx, _ = train_test_split(np.arange(len(X)), train_size=m.train_sample,
                                  stratify=y, random_state=m.random_state)
        X, y = X.iloc[idx], y.iloc[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=m.test_size, stratify=y, random_state=m.random_state)

    pipe = _build_pipeline(cfg, info["features_numeric"], info["features_categorical"])
    log.info("모델 학습 시작 (학습 %s행, 피처 %d개)",
             f"{len(X_train):,}", X.shape[1])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    scores = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    # 기준선: 다수 클래스만 예측하는 모델의 정확도.
    # 이 값과 비교해야 실제 모델이 얼마나 나은지 판단할 수 있다.
    majority = float(max(y_test.mean(), 1 - y_test.mean()))

    notes.append(
        f"[모델] 대상은 카드결제 {info['population_rows']:,}건이다. 현금·무료·분쟁 "
        f"결제는 팁이 100% 0으로 기록돼 섞으면 모델이 '결제수단 맞히기'를 하게 되므로 "
        f"제외했다. 타깃은 팁 비율 {m.target_tip_rate:.0%} 이상(양성 "
        f"{info['positive_ratio']:.1%})으로, 클래스가 균형에 가까워 정확도가 의미를 갖는다.")
    notes.append(
        f"[모델] 누수 방지를 위해 {', '.join(LEAKAGE)}를 피처에서 제외했다. "
        f"특히 total_amount는 팁을 포함한 합계라 역산이 가능하고, 운행 시작 시점에는 "
        f"알 수 없는 값이다. 넣으면 정확도가 0.99를 넘지만 예측한 것이 아니다.")
    notes.append(
        f"[모델] 전처리(결측 대체·스케일링·원-핫)를 Pipeline 안에 넣었다. "
        f"밖에서 미리 처리하면 대체값·스케일 기준이 테스트셋까지 보고 계산되어 "
        f"평가 점수가 부풀려진다.")
    notes.append(
        f"[모델] 정확도 {scores['accuracy']:.4f}, F1 {scores['f1']:.4f}, "
        f"ROC-AUC {scores['roc_auc']:.4f}. 다수 클래스만 예측하는 기준선의 정확도가 "
        f"{majority:.4f}이므로 {scores['accuracy'] - majority:+.4f} 개선했다.")
    notes.append(
        "[한계] 학습/평가를 무작위로 분할했다. '미래 예측'이 목적이라면 시간 순 "
        "분할이 옳다. 한 달치 데이터라 추세가 크지 않아 무작위 분할을 썼다. "
        "하이퍼파라미터는 튜닝하지 않은 기본값이다.")

    # ---- 모델 저장 ----------------------------------------------------------
    # 전처리까지 포함한 Pipeline 객체 전체를 저장한다. 불러온 뒤 원본 형태의
    # DataFrame을 그대로 넣어 예측할 수 있다.
    def save_model(path: Path) -> None:
        """Pipeline 전체를 저장한다. 전처리가 함께 들어가 그대로 재사용된다."""
        joblib.dump(pipe, path)

    artifact = Artifact(
        name="model.joblib", save=save_model, kind="model",
        caption=(f"전처리 + HistGradientBoostingClassifier Pipeline. "
                 f"joblib.load() 후 원본 형태의 DataFrame을 넣어 예측할 수 있다. "
                 f"학습 scikit-learn {sklearn.__version__}."))

    metrics = {
        **info,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
        "scores": scores,
        "baseline_accuracy": majority,
        "improvement_over_baseline": scores["accuracy"] - majority,
        "model": type(pipe.named_steps["classifier"]).__name__,
        "sklearn_version": sklearn.__version__,
        "random_state": m.random_state,
        **scores,          # 게이트가 바로 읽을 수 있도록 평탄화해 둔다
    }
    log.info("모델 완료: 정확도 %.4f F1 %.4f ROC-AUC %.4f (기준선 %.4f)",
             scores["accuracy"], scores["f1"], scores["roc_auc"], majority)
    return StepResult(df=df, metrics=metrics, notes=notes, artifacts=[artifact])
