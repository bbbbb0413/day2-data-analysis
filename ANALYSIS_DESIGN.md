# Question-Driven EDA 설계서

## 분석 목표

단순히 차트를 많이 생성하는 것이 아니라, 각 EDA 결과를 머신러닝의 전처리·변수 선택·평가 방식으로 연결합니다.

## 질문과 산출물 매핑

| 질문 | 확인 내용 | 주요 산출물 | ML 판단 |
| --- | --- | --- | --- |
| Q1 데이터 구조 | dtype, 고유값, 결측, 상수·편중 | `column_ml_profile.csv` | 수치형·범주형·날짜·누수 변수 구분 |
| Q2 기술통계 | 평균, 중앙값, 표준편차, 분위수, 왜도, 첨도 | `numeric_distribution_summary.csv` | 로그 변환·Robust 처리 후보 |
| Q3 결측 구조 | 결측률, 동시 결측, Vendor별 차이 | `missing_*` 표 | 대체 방법과 Missing Indicator 결정 |
| Q4 Target | 팁 비율·high_tip 분포 | Target 표·Histogram | 클래스 불균형 평가 지표 결정 |
| Q5 극단값 | IQR, P95, P99, P99.9, 최댓값 | `iqr_outlier_candidate_summary.csv` | 삭제가 아닌 플래그·변환 여부 판단 |
| Q6 품질 규칙 | 시간 순서, 음수값, 속도, 요금합, 코드 | `quality_flag_summary.csv` | 목적별 모델 코호트 결정 |
| Q7 범주형 | Cardinality, 최빈값 편중, 희소 범주 | 범주 분포·Cardinality 표 | One-Hot·희소 통합·고유 ID 제외 |
| Q8 수치–Target | 상관계수, 그룹 평균·중앙값, Cohen's d | `numeric_target_relationships.csv` | 선형 관계와 비선형 모델 필요성 판단 |
| Q9 범주–Target | 시간·요일·Vendor·지역별 고팁률 | `categorical_target_relationships.csv` | 표본 지지도를 고려한 범주 해석 |
| Q10 교차 탐색 | 요일×시간, 거리×요금×Target, 경로 | Plotly HTML 3개 | 상호작용 변수·시간 분할 필요성 판단 |

## 극단값 정책

```text
원본 데이터
└─ 삭제하지 않음

품질 플래그
├─ 논리 오류: invalid_time_order, negative_distance 등
├─ 검토 후보: implausible_speed, component_mismatch
└─ 통계 후보: flag_iqr_outlier__*

모델 코호트
├─ 시간 계산 가능
├─ 운행시간 > 0
├─ 거리·요금·총액 >= 0
└─ 큰 양수값은 유지
```

## 결측치 정책

```text
Target 생성에 필요한 결측
→ 대체하지 않고 Target 코호트에서 제외

수치형 설명변수 결측
→ 학습 데이터 중앙값 대체 + 결측 Indicator

범주형 설명변수 결측
→ Unknown 범주

날짜 결측
→ 시간 파생변수 계산 불가; 정상 운행 모델 코호트에서 제외
```

## 데이터 누수 점검

다음 값은 high_tip 계산에 직접 연결되므로 입력에서 제외합니다.

```text
tip_amount
total_amount
pre_tip_amount
tip_rate
tip_rate_pct
high_tip
```

`payment_type`은 카드 결제 코호트 안에서 상수이므로 정보력이 없어 제외합니다.

## 평가 방식

```text
Baseline Accuracy
Accuracy
Balanced Accuracy
Precision
Recall
F1-score
ROC-AUC
Average Precision
Confusion Matrix
Classification Report
```

시간 순서 Holdout을 우선 사용하며, 한 클래스가 사라지는 경우 층화 무작위 Holdout으로 자동 전환합니다.
