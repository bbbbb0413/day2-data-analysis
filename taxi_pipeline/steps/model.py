"""고액 팁 여부를 예측하는 분류 모델을 학습하고 저장한다."""

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

# 모델 입력에 사용할 피처를 정의한다.
NUMERIC = ["trip_distance", "fare_amount", "duration_min", "passenger_count",
           "hour", "dayofweek", "extra", "tolls_amount", "congestion_surcharge"]
CATEGORICAL = ["VendorID", "RatecodeID", "PULocationID", "DOLocationID"]

# 타깃 누수를 방지하기 위해 제외할 컬럼을 정의한다.
LEAKAGE = ["tip_amount", "total_amount", "payment_type"]


def _prepare(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, dict]:
    """모집단을 좁히고 타깃과 피처를 만든다."""
    m = cfg.model

    # 카드결제 데이터만 사용한다.
    card = df[df["payment_type"] == m.payment_type].copy()

    # 팁 비율을 기준으로 타깃을 생성한다.
    rate = card["tip_amount"] / card["fare_amount"].replace(0, np.nan)
    ok = rate.notna() & (rate < 2)          # 계산할 수 없는 값과 극단값을 제외
    card, rate = card[ok], rate[ok]

    # 시간 관련 파생 피처를 생성한다.
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
    """전처리와 분류 모델을 하나의 Pipeline으로 구성한다."""
    m = cfg.model
    pre = ColumnTransformer([
        # 수치형 변수는 중앙값으로 대체한 후 스케일링한다.
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        # 범주형 변수는 최빈값으로 대체한 후 원-핫 인코딩한다.
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   min_frequency=m.min_frequency))]),
         cat),
    ], sparse_threshold=0.0)   # HistGradientBoosting 사용을 위해 dense 형태로 출력한다.

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

    # 타깃 비율을 유지하면서 표본을 추출한다.
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

    # 다수 클래스 정확도를 기준선으로 사용한다.
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

    # 전처리를 포함한 Pipeline 전체를 저장한다.
    def save_model(path: Path) -> None:
        """학습된 Pipeline을 저장한다."""
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
        **scores,          # 품질 게이트에서 바로 사용하도록 저장한다.
    }
    log.info("모델 완료: 정확도 %.4f F1 %.4f ROC-AUC %.4f (기준선 %.4f)",
             scores["accuracy"], scores["f1"], scores["roc_auc"], majority)
    return StepResult(df=df, metrics=metrics, notes=notes, artifacts=[artifact])
