# NYC Yellow Taxi 분석 5분 발표안

## 1분: 데이터와 목표

- 2026년 5월 Yellow Taxi 전체 데이터: **4,090,836행**
- Pandas와 Polars 전체 로딩 결과를 비교하여 행·열·결측치 일관성을 확인했습니다.
- 통계 주제: 평일과 주말의 카드 팁 비율 평균 비교
- ML 주제: 카드 팁 비율 20% 이상 여부 예측
- 카드 결제 분석 행: **333,464개**

## 1분: Question-Driven EDA

- 컬럼을 날짜·연속형·코드형 범주·목표 생성 변수로 구분했습니다.
- 결측치 비율뿐 아니라 Vendor별 결측과 동시 결측 패턴도 확인했습니다.
- 큰 거리·요금은 삭제하지 않고 IQR·속도·요금합 불일치 플래그로 관리했습니다.
- high_tip과 수치형 관계 상위: trip_distance(-0.2035), improvement_surcharge(0.2018), fare_amount(-0.1935)

## 1분: 통계 검정

- 평일 평균 팁 비율: **16.8727%**
- 주말 평균 팁 비율: **17.2170%**
- p-value: **9.269935e-13**
- Cohen's d: **-0.0301 (매우 작음)**
- 큰 표본에서는 작은 차이도 유의할 수 있어 효과크기와 민감도 분석을 함께 봤습니다.

## 1분: ML Pipeline

```text
수치형: 중앙값 대체 + 결측 표시 + 표준화
범주형: Unknown 대체 + One-Hot Encoding
모델: class_weight='balanced' LogisticRegression
```

- `tip_amount`, `total_amount`, `tip_rate`는 Target 누수 방지를 위해 제외했습니다.
- 시간 순서 분할을 우선 적용해 미래 예측 상황에 가깝게 평가했습니다.

## 1분: 결과와 개선

- Accuracy: **0.5664**
- F1-score: **0.6233**
- Balanced Accuracy: **0.6035**
- ROC-AUC: **0.6507**
- 개선 방향: 여러 달 데이터, 날씨·공휴일 결합, 비선형 모델 비교, 확률 임계값 최적화
