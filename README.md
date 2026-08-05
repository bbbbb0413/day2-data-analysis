# NYC Yellow Taxi End-to-End 데이터 분석

## 1. 프로젝트 목적

NYC TLC의 `yellow_tripdata_2026-05.parquet`를 사용하여 다음 평가 항목을 한 번에 수행합니다.

- Pandas와 Polars 전체 로딩 결과 비교
- 결측치·완전 중복·논리적으로 비정상적인 운행값 처리
- 기본 EDA와 기술통계·상관계수
- Seaborn 정적 차트와 Plotly 인터랙티브 차트
- `scipy.stats.ttest_ind`를 이용한 Welch t-test와 p-value 해석
- 시간 파생변수 생성 후 `sklearn.pipeline.Pipeline` 기반 전처리 + 분류 모델
- Accuracy, Precision, Recall, F1-score, ROC-AUC 출력
- `joblib` 모델 저장
- 실행 결과를 `report.md`로 자동 생성

## 2. 분석 주제

### 통계 분석

평일과 주말의 **카드 결제 팁 비율 평균**이 다른지 검정합니다.

```text
팁 비율 = tip_amount / (total_amount - tip_amount)
```

NYC TLC 데이터 사전상 현금 팁은 `tip_amount`에 포함되지 않으므로 팁 분석은
`payment_type == 1`인 신용카드 결제만 사용합니다.

### 머신러닝

결제 전 금액 대비 팁 비율이 20% 이상인지 분류합니다.

```text
팁 비율 20% 미만  -> high_tip = 0
팁 비율 20% 이상  -> high_tip = 1
```

`tip_amount`와 `total_amount`는 정답 계산에 직접 사용되므로 모델 입력에서 제외합니다.

## 3. 실행 방법

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

데이터 파일이 없으면 공식 NYC TLC 주소에서 자동으로 다운로드합니다.
인터넷 연결이 제한된 환경에서는 다음 파일을 직접 내려받아 `data` 폴더에 넣습니다.

```text
data/yellow_tripdata_2026-05.parquet
```

## 4. 대용량 데이터 처리 기준

원본 Parquet는 Pandas와 Polars에서 각각 전체 로딩하여 행 수, 열 수, 컬럼명,
결측치 수, 메모리 사용량, 로딩 시간을 비교합니다.

후속 시각화와 머신러닝은 일반 노트북에서도 실행 가능하도록 기본적으로
고정 난수 표본을 사용합니다.

```python
ANALYSIS_SAMPLE_SIZE = 500_000
ML_SAMPLE_SIZE = 150_000
RANDOM_STATE = 42
```

전체 데이터로 후속 분석하려면 `main.py`에서 다음과 같이 변경합니다.

```python
ANALYSIS_SAMPLE_SIZE = None
```

## 5. 주요 정제 기준

다음은 공식 오류 판정 규칙이 아니라 실습을 위해 설정한 분석용 품질 규칙입니다.

- 완전히 동일한 행 제거
- 승차일시가 2026년 5월인 기록
- 하차일시가 승차일시보다 늦은 기록
- 운행시간 1~180분
- 운행거리 0.1~100마일
- 미터요금 0초과~500달러
- 총 결제금액 0초과~1000달러
- 승차·하차 위치 ID 1~265
- 승객 수가 결측이거나 0~8명

EDA용 데이터는 수치형 중앙값과 범주형 `Unknown`으로 결측치를 처리합니다.
ML에서는 Pipeline 내부의 `SimpleImputer`가 학습 데이터 기준으로 처리합니다.

## 6. 자동 생성 파일

실행 후 다음 파일이 생성됩니다.

```text
report.md
outputs/
├── loader_comparison.csv
├── loader_consistency_check.json
├── cleaning_summary.csv
├── descriptive_statistics.csv
├── correlation_matrix.csv
├── payment_type_summary.csv
├── high_tip_target_summary.csv
├── ttest_result.json
├── model_metrics.json
├── yellow_taxi_high_tip_pipeline.joblib
├── seaborn_tip_rate_weekday_weekend.png
└── plotly_hourly_trip_share.html
```

## 7. 제출 화면 캡처 권장 순서

1. Pandas·Polars 로딩 비교와 일관성 확인
2. 정제 단계별 제거 행 수와 결측치 처리 결과
3. 기술통계와 상관계수
4. Seaborn Boxplot
5. Plotly 인터랙티브 차트
6. t-statistic, p-value, 해석 문장
7. Accuracy와 F1-score를 포함한 모델 평가 결과
8. `report.md`와 `.joblib` 파일 생성 결과

## 8. 저장 모델 재사용 시 주의

저장된 `.joblib` 파일에는 결측치 대체, 스케일링, 원-핫 인코딩,
로지스틱 회귀 모델이 함께 들어 있습니다. 새 원본 운행 데이터를 예측할 때는
`main.py`의 `prepare_model_features()`로 운행시간·시간대·요일 변수를 먼저 만든 뒤
모델 메타데이터에 저장된 `input_features` 순서로 전달해야 합니다.
