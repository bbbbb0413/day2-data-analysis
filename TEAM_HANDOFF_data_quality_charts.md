# [연동 요청] 자동화 파이프라인 report.md에 결측치/이상치 시각화 붙이기

## 배경
시각화 파트에서 만든 EDA 노트북(`visualization_original.ipynb`) 중 **Part 5(결측치 & 데이터 품질 이슈)**, **Part 12(수치형 이상치 IQR 박스플롯)** 두 섹션을 자동화 파이프라인의 `report.md` 생성 결과에 포함시키고 싶습니다. 매달(4월/5월/6월...) 원본 데이터가 다르므로 결측 건수·이상치 비율·차트가 **실행할 때마다 그 달 데이터 기준으로 다시 계산/생성**돼야 합니다. 지금은 6월 데이터만 우선 처리하면 되고, 여러 달을 동시에 비교 보관하는 기능은 이번 스코프에 없습니다.

## 넘기는 것
`data_quality_charts.py` 1개 파일(현재 `taxi_pipeline/viz/quality_charts.py`). 노트북 Part 5 + Part 12 로직을 함수 하나로 뺀 것이고, 실제 5월치 원본 데이터로 정상 동작(이미지 생성, 통계 계산)까지 확인했습니다.

```python
from taxi_pipeline.viz.quality_charts import generate_data_quality_charts

result = generate_data_quality_charts(
    df,                          # 그 달 원본(정제 전) DataFrame — pd.read_parquet(...) 결과 그대로
    output_dir="output",         # PNG 저장 폴더 (원하는 경로로 바꿔도 됨)
    month_label="2026-06",       # 리포트 문구에 들어갈 라벨
)
```

## `result`에 들어있는 것 (3가지, 필요한 것만 골라 쓰면 됨)

1. **`result["images"]`** — 생성된 PNG 5개의 경로 dict
   `missing_matrix`, `missing_bar`, `missing_heatmap`, `quality_issues`, `iqr_boxplot`
2. **`result["stats"]`** — 관찰 텍스트에 넣을 실제 수치 (컬럼별 결측 건수/비율, 품질 이슈별 건수/비율, 컬럼별 IQR 이상치 건수/비율 등). report.md 쪽 템플릿에 값만 채워 넣고 싶으면 이걸 쓰면 됩니다.
3. **`result["markdown"]`** — 위 두 개를 이미 표+이미지 마크다운으로 조합해놓은 완성된 블록(`str`). report.md 생성 로직이 "섹션 문자열을 이어붙이는 방식"이면 이 값을 그 자리에 그대로 삽입하면 끝입니다.

셋 중 어떤 걸 쓸지는 report.md를 만드는 방식(문자열 조합인지 Jinja2 템플릿인지)에 맞춰서 고르면 됩니다 — `markdown`을 통째로 넣는 게 제일 빠르고, 기존 템플릿 스타일에 맞추고 싶으면 `stats`만 뽑아 쓰면 됩니다.

## 연동 방법 (예시 — 기존 report.md 생성 로직 안에서)

```python
from taxi_pipeline.viz.quality_charts import generate_data_quality_charts

# 파이프라인이 이미 그 달 원본 df를 들고 있는 지점에서 호출
dq = generate_data_quality_charts(df, output_dir="output", month_label=month_label)

# report.md를 문자열로 조합하고 있다면:
report_sections.append(dq["markdown"])

# 혹은 기존 템플릿에 숫자만 채워 넣고 싶다면 예:
missing_pct = dq["stats"]["missing"]["passenger_count"]["pct"]
outlier_pct = dq["stats"]["iqr_outliers"]["total_amount"]["pct"]
```

## 전제 조건 / 확인해주셔야 할 것

- **입력 데이터는 정제 전 원본**이어야 함 (결측치/이상치를 있는 그대로 보여주는 게 목적이라, 이미 정제된 df를 넣으면 결측 관련 차트가 전부 0으로 나와서 의미 없어짐). 파이프라인에서 정제 단계 **이전에** 이 함수를 호출해주세요.
- 컬럼 스키마는 표준 NYC Yellow Taxi 스키마를 가정합니다: `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, `Airport_fee`(결측 대상), `tpep_pickup_datetime`/`tpep_dropoff_datetime`, `total_amount`, `fare_amount`, `trip_distance`. 혹시 컬럼명이 파이프라인 쪽 스키마와 다르면 얘기해주세요 — `nullable_columns`/`numeric_cols` 파라미터로 오버라이드는 가능하지만 5-B 품질 이슈 계산 부분은 컬럼명이 하드코딩돼 있어서 다르면 수정이 필요합니다.
- 신규 의존성 1개 추가 필요: `pip install missingno` (kaleido처럼 가볍습니다)
- `output_dir`은 파라미터로 열어뒀습니다 — 지금 당장은 그냥 `"output"`처럼 고정 경로로 써도 되지만, 나중에 여러 달을 구분 저장하고 싶어지면 호출부에서 `f"output/{month_label}"`처럼 바꾸기만 하면 됩니다 (모듈 코드 수정 불필요).

## 확인해주시면 좋은 것
- 결측 5만행 샘플링(`missing_sample_n=50_000`, matrix 차트 전용 — 4백만 행을 그대로 그리면 매우 느려서 샘플링)이 파이프라인 실행 시간 예산에 문제없는지
- report.md 생성 로직이 문자열 조합인지 템플릿 엔진인지 알려주시면 `markdown`/`stats` 중 뭘 쓸지 같이 정하겠습니다
