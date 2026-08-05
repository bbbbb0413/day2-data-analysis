# NYC Yellow Taxi Question-Driven EDA + ML Pipeline

NYC Yellow Taxi 2026년 5월 Parquet 데이터를 대상으로 **질문 중심 EDA**, 통계 검정, 머신러닝 Pipeline, 자동 보고서 생성을 한 번에 수행하는 프로젝트입니다.

기존의 단순한 `head()`, `describe()`, 결측치 개수 확인에서 끝나지 않고 다음 흐름으로 구성했습니다.

```text
전체 데이터 로딩·검증
    ↓
컬럼별 머신러닝 역할 진단
    ↓
결측·중복·논리 오류·극단값 후보 분석
    ↓
단변량·이변량·다변량 Question-Driven EDA
    ↓
Seaborn 정적 차트 + Plotly 인터랙티브 차트
    ↓
기술통계·상관계수·Welch t-test
    ↓
sklearn Pipeline 학습·평가·joblib 저장
    ↓
report.md + presentation_5min.md 자동 생성
```

## 1. 분석 주제

### 통계 분석

평일과 주말의 **카드 결제 팁 비율 평균**에 차이가 있는지 Welch 독립표본 t-test로 확인합니다.

### 머신러닝

결제 전 금액 대비 카드 팁 비율이 20% 이상인지 예측합니다.

```text
high_tip = 1: tip_amount / (total_amount - tip_amount) >= 0.20
high_tip = 0: 위 조건 미만
```

20%는 프로젝트에서 정한 분류 기준이며 코드 상단의 `HIGH_TIP_THRESHOLD`로 변경할 수 있습니다.

## 2. Question-Driven EDA 구성

코드는 다음 질문에 실제 수치와 차트로 답하도록 구성되어 있습니다.

1. 데이터의 행·열·자료형은 어떠하며 각 컬럼은 ML에서 어떤 역할인가?
2. 평균·중앙값·분위수·왜도·첨도는 어떤 분포 문제를 보여주는가?
3. 결측치는 어느 컬럼과 Vendor에 집중되며, 결측 자체가 Target 신호인가?
4. 카드 팁 비율과 high_tip 클래스는 얼마나 치우쳐 있는가?
5. IQR 극단값 후보가 많은 변수는 무엇이며 실제 오류와 어떻게 구분할 것인가?
6. 날짜·거리·시간·요금·공식 코드의 논리 검증 결과는 어떠한가?
7. 범주형 변수의 고유값 수, 편중도, 희소 범주는 어떤 인코딩 문제를 만드는가?
8. 수치형 변수와 high_tip의 관계는 어떠한가?
9. 시간·요일·Vendor·Ratecode·지역에 따라 고팁률이 달라지는가?
10. 시간대×요일, 거리×요금×Target, 승차×하차 경로를 함께 볼 때 어떤 패턴이 나타나는가?

각 질문은 자동 보고서에서 다음 형식으로 정리됩니다.

```text
Finding
Evidence
Possible Action
```

## 3. 극단값 처리 원칙

큰 거리·긴 시간·큰 요금이라는 이유만으로 행을 삭제하지 않습니다.

```text
논리적으로 불가능한 값
- 하차일시가 승차일시보다 빠름
- 음수 거리
- 음수 요금
→ 품질 플래그 생성, 정상 운행 모델 코호트에서는 제외 가능

드물지만 실제일 수 있는 값
- 장거리 운행
- 장시간 운행
- 고액 요금
- 100% 초과 팁
→ 원본 유지, IQR·상위 분위수·속도·요금 구성 차이로 검토
```

IQR은 오류 삭제 규칙이 아니라 **검토 후보를 찾는 통계 기준**으로만 사용합니다.

## 4. 결측치 처리 원칙

### EDA

결측치를 임의로 모두 채우지 않고 다음을 먼저 확인합니다.

- 컬럼별 결측 건수와 비율
- 상위 동시 결측 패턴
- Vendor별 결측률
- 결측 여부별 high_tip 비율

### ML Pipeline

```text
수치형
SimpleImputer(strategy="median", add_indicator=True)
→ StandardScaler

범주형
SimpleImputer(strategy="constant", fill_value="Unknown")
→ OneHotEncoder(handle_unknown="ignore", min_frequency=20)
```

목표값을 만드는 `tip_amount`, `total_amount`가 결측이면 임의 대체하지 않습니다. 가짜 정답을 만들 수 있기 때문입니다.

## 5. 표본 설정

코드 상단에서 단계별 표본 크기를 변경할 수 있습니다.

```python
ANALYSIS_SAMPLE_SIZE: int | None = 500_000
PLOT_SAMPLE_SIZE: int | None = 80_000
MODEL_SAMPLE_SIZE: int | None = 200_000
TTEST_MAX_PER_GROUP: int | None = 150_000
```

전체 데이터를 사용하려면 해당 값을 따옴표 없이 `None`으로 변경합니다.

```python
ANALYSIS_SAMPLE_SIZE: int | None = None
```

`ANALYSIS_SAMPLE_SIZE=None`은 후속 Pandas EDA에 전체 데이터를 사용한다는 뜻입니다. 시각화, ML, t-test에는 각각 별도 설정이 있으므로 모든 단계에서 전체 데이터를 쓰려면 네 값을 모두 `None`으로 바꿔야 합니다.

일반 PC에서는 다음 설정을 권장합니다.

```python
ANALYSIS_SAMPLE_SIZE: int | None = 500_000
PLOT_SAMPLE_SIZE: int | None = 80_000
MODEL_SAMPLE_SIZE: int | None = 200_000
TTEST_MAX_PER_GROUP: int | None = 150_000
```

Pandas와 Polars의 행·열·결측치·중복 비교는 표본 설정과 관계없이 전체 Parquet 파일을 기준으로 수행합니다.

## 6. 실행 방법

```bash
python -m venv .venv
```

macOS 또는 Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

패키지 설치와 실행:

```bash
pip install -r requirements.txt
python main.py
```

데이터가 없으면 코드가 다음 파일을 자동으로 내려받습니다.

```text
data/yellow_tripdata_2026-05.parquet
data/taxi_zone_lookup.csv
```

인터넷 연결이 제한된 환경에서는 해당 파일을 `data` 폴더에 직접 넣습니다.

## 7. 주요 자동 생성 결과

```text
report.md
presentation_5min.md
outputs/
├── figures/
│   ├── 01_missing_rate_by_column.png
│   ├── 02_quality_flag_rates.png
│   ├── 03_tip_rate_distribution.png
│   ├── 04_numeric_feature_distributions.png
│   ├── 05_iqr_candidate_boxplots.png
│   ├── 06_model_safe_correlation_heatmap.png
│   ├── 07_daily_trip_count.png
│   ├── 08_hourly_trip_count.png
│   ├── 09_tip_rate_weekday_weekend.png
│   ├── 10_distance_fare_high_tip.png
│   ├── 11_high_tip_rate_by_hour.png
│   ├── 12_top_pickup_zones.png
│   ├── 13_model_confusion_matrix.png
│   ├── 14_model_roc_curve.png
│   └── 15_model_precision_recall_curve.png
├── interactive/
│   ├── 01_hour_day_trip_heatmap.html
│   ├── 02_distance_fare_high_tip_scatter.html
│   └── 03_top_pickup_zones.html
├── models/
│   └── yellow_taxi_high_tip_pipeline.joblib
├── tables/
│   ├── column_ml_profile.csv
│   ├── missing_summary_analysis_sample.csv
│   ├── top_missing_patterns.csv
│   ├── missing_by_vendor.csv
│   ├── quality_flag_summary.csv
│   ├── iqr_outlier_candidate_summary.csv
│   ├── numeric_distribution_summary.csv
│   ├── categorical_cardinality_summary.csv
│   ├── numeric_target_relationships.csv
│   ├── categorical_target_relationships.csv
│   ├── ttest_weekday_weekend_tip_rate.json
│   ├── model_metrics.json
│   └── model_logistic_coefficients.csv
└── run_manifest.json
```

## 8. 모델링 핵심 기준

- 카드 결제 운행만 팁 분석에 사용합니다.
- `tip_amount`, `total_amount`, `tip_rate`, `high_tip`은 모델 입력에서 제외해 Target 누수를 방지합니다.
- 카드 전용 코호트에서 `payment_type`은 모두 같은 값이므로 입력에서 제외합니다.
- 시간 순서 분할을 우선 적용해 과거 데이터로 학습하고 이후 데이터로 평가합니다.
- 클래스 불균형을 고려해 Accuracy뿐 아니라 Balanced Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision을 출력합니다.
- 전처리기와 모델 전체를 하나의 sklearn Pipeline으로 저장합니다.

## 9. 검증 범위

- `python -m py_compile main.py` 구문 검증 완료
- 합성 Yellow Taxi 스키마로 파생변수, 결측 분석, 극단값 플래그, 정적·인터랙티브 차트, Welch t-test, ML Pipeline, joblib 저장, 자동 보고서 생성까지 Smoke Test 완료
- 공식 대용량 Parquet의 실제 지표는 사용자의 실행 환경에서 `python main.py`를 실행하면 자동 산출됩니다.
