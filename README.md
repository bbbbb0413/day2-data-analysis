# NYC Yellow Taxi 정제·분석 파이프라인

2026-05 Yellow Taxi 운행 기록(409만 행)을 정제하고 EDA·시각화·통계 검정·
분류 모델까지 자동으로 수행한다.

**모든 데이터 처리에는 근거가 있다.** 관례로 정하지 않고 이 데이터에서
실제로 관측된 값으로 결정했으며, 그 기록은
[`결측치_중복_처리기준.md`](결측치_중복_처리기준.md)와 [`docs/`](docs/)에 있다.

---

## 빠른 시작

**요구 사항**: Python **3.11 이상** (설정 파일을 읽는 `tomllib`이 3.11부터 표준 라이브러리)

```bash
pip install -r requirements.txt

python run_pipeline.py --dry-run      # 무엇을 할지 먼저 확인
python run_pipeline.py                # 전체 실행 (원본 자동 다운로드 포함, 약 22초)
python tests/test_steps.py            # 단위 테스트 32개 (원본 없이 0.5초)
```

원본 데이터를 미리 받아둘 필요가 없다. `data/raw/`에 파일이 없으면
NYC TLC 공개 엔드포인트에서 자동으로 내려받는다(66.5MB, 약 1초).
이미 있으면 건드리지 않는다.

정상 실행 시 터미널 마지막에 이렇게 나온다.

```
품질 게이트: 14개 전부 통과
최종 3,884,622행 (원본 대비 보존율 94.96%)
리포트: outputs/runs/<run_id>/report.md
```

---

## 파이프라인 8단계

```
1. compare_loaders   Pandas·Polars 로딩 결과 비교
2. analyze_missing   결측 구조 진단 (삭제·대체가 왜 안 되는지 근거 수집)
3. prepare_missing   위장 결측 변환 + record_source 플래그 (행 삭제 없음)
4. deduplicate       중복 유형 판정 후 선택 제거
5. filter_outliers   기간·소요시간·거리·금액 이상치 제거
6. visualize         Seaborn 정적 7개 + Plotly 인터랙티브 3개 차트
7. statistics        기술통계·상관계수·t-test와 p-value 해석
8. model             Pipeline으로 전처리+모델 학습, 평가 지표, joblib 저장
```

`python run_pipeline.py list-steps`로 확인할 수 있다.
1·2·6·7·8은 DataFrame을 바꾸지 않는 분석 단계이고, 3·4·5만 행을 변형한다.

---

## 산출물

### 1. 정제 데이터 — `data/processed/yellow_2026-05_clean.parquet`

**76MB, 3,884,622행 × 21열** (원본 대비 94.96% 보존).

```python
import pandas as pd
df = pd.read_parquet("data/processed/yellow_2026-05_clean.parquet")
```

**제거된 것 — 총 206,214행**

| 구분 | 행 수 | 내용 |
|---|---:|---|
| 중복 | 42,688 | 상쇄쌍 13,301쌍(26,602행) 양쪽 + 이중계상 16,086행 큰 쪽 |
| 거리 이상 | 108,631 | 0마일 또는 100마일 초과 |
| 소요시간 이상 | 53,006 | 0초 이하 또는 6시간 초과 |
| 금액 이상 | 1,875 | 총액 0 이하 또는 요금 음수 |
| 기간 이탈 | 14 | 2026-05 밖 승차 |

**남아 있는 것 — 오해하기 쉬운 부분**

- **결측행 863,163행(22.2%)은 지우지 않았다.** `dropna()`를 하지 않는 것이
  이 파이프라인의 핵심 결정이다. 대신 21번째 컬럼 `record_source`가
  `full`(3,021,459) / `partial`(863,163)로 구분한다.
- **결측은 오히려 늘었다.** 위장 결측 1,137,013건(`payment_type=0`,
  `RatecodeID=99`, `PULocationID=264/265` 등)을 NaN으로 바꿨기 때문이다.
  원본을 그대로 집계하면 이 값들이 정상값으로 섞여 평균을 오염시킨다.
- **중복키가 12건 남아 있다.** 상쇄쌍도 2배도 아닌 "서로 다른 운행"으로
  판정한 그룹이라 의도적으로 유지했다.

**쓸 때 주의**

```python
# 승객수·요율 분석은 full 한정 — partial 소스에는 그 컬럼 자체가 없다
full = df[df.record_source == "full"]

# 요금·거리·시각은 결측 0건이라 전체를 그대로 쓴다
df.groupby(df.tpep_pickup_datetime.dt.hour)["total_amount"].mean()
```

### 2. 실행 기록 — `outputs/runs/<run_id>/`

`run_id`는 `시각_설정해시_입력해시` 형식이라, 디렉터리 정렬만으로 시간순이 되고
같은 입력·같은 기준의 재실행인지 눈으로 구분된다.

| 파일 | 용도 | 언제 보는가 |
|---|---|---|
| `report.md` | 사람이 읽는 리포트 (약 330줄) | **여기부터 본다** |
| `figures/` | 차트 PNG 18개 + 인터랙티브 HTML 3개 | 원본 EDA 차트 8개는 리포트에 자동 임베드 |
| `model.joblib` | 학습된 Pipeline (약 390KB) | 예측에 재사용할 때 |
| `metrics.json` | 전 단계 지표 | 이전 실행과 비교할 때 |
| `manifest.json` | 입력·설정 해시, 단계별 소요, 게이트 결과 | "이 결과가 뭘로 만들어졌나" 추적 |

```bash
# 최신 실행 리포트 열기
open "$(ls -td outputs/runs/*/ | head -1)/report.md"
```

**`report.md` 구성** — 분석을 진행한 순서 그대로, 읽는 사람의 질문 순서로 배치했다.

| 섹션 | 답하는 질문 |
|---|---|
| 1. 개요 | 무엇을 한 분석인가? |
| 2. 데이터 이해 | 이 데이터의 컬럼은 무엇을 뜻하나? |
| 3. 주요 시각화 요약 (정제 전) | 원본에 어떤 문제가 있었나? |
| 4. 처리 기준 설계 | 그 문제를 어떤 근거로 어떻게 다뤘나? |
| 5. 핵심 결과 | **한 표로 요약하면?** |
| 6. 통계 분석 | 눈으로 본 차이가 통계적으로도 성립하나? |
| 7. ML Pipeline | 그래서 예측할 수 있나? |
| 8. 자동화·품질 검증 및 한계 | 이 결과를 믿어도 되나? 어디까지 믿으면 되나? |

3장에는 정제 전 원본 EDA 차트 8개가 임베드된다. 정제 후 차트 10개와 인터랙티브
HTML 3개는 `figures/`에 생성되며, 별도 보고서에서 다룬다.

각 단계가 낸 근거 문장이 그대로 실린다.

```
- drop_duplicates(keep='first')를 썼다면 양수 총액 29,370건(정상 운행)이
  지워지고 음수 14,847건(취소 기록)이 남는다. 순서 기반 제거는 사용 금지.
- [경고] VendorID=7가 51,750건 전량 삭제됐다 (소요시간<=0 비율 100.0%)...
```

**저장된 모델 사용법**

```python
import joblib
model = joblib.load("outputs/runs/<run_id>/model.joblib")
model.predict(df[FEATURES])   # 전처리가 Pipeline 안에 들어 있다
```

**`metrics.json` 활용** — 다음 달 실행과 비교해 데이터 사고를 잡는 용도다.

```python
import json
m = json.load(open("outputs/runs/<run_id>/metrics.json"))
m["deduplicate"]["naive_keep_first"]
# {'positive_rows_dropped': 29370, 'negative_rows_dropped': 30, 'negative_rows_left': 14847}
m["filter_outliers"]["vendors_wiped_out"]
# [{'vendor': 7, 'rows': 51750, 'nonpositive_duration_ratio': 1.0}]
m["model"]["scores"]["f1"]        # 0.8217
```

### 3. 중간 산출물 — `data/interim/` (선택)

`--checkpoint`를 줄 때만 생긴다. 단계마다 parquet을 써서(각 70MB 내외)
"이 단계 직후 데이터가 어땠는지"를 사후에 열어볼 수 있게 한다. 기본은 꺼져 있다.

> **아직 없는 기능**: 저장한 중간 산출물을 **다시 읽어 이어서 실행하는
> `--from <step>`은 구현돼 있지 않다.** 지금은 쓰기 전용 스냅샷이다.
> 전체 실행이 22초라 재시작의 실익이 없어 미뤄뒀다.

---

## 구조

```
day2-data-analysis/
├── config/pipeline.toml        ★ 모든 임계값이 여기 있다 (코드에 상수 없음)
├── run_pipeline.py             CLI 진입점 — 인자 파싱, 종료 코드
├── taxi_pipeline/
│   ├── config.py               TOML → dataclass, 설정 해시
│   ├── fetch.py                원본 확보 (없을 때만 다운로드)
│   ├── observability.py        로깅(text/json), JSON 직렬화
│   ├── storage.py              입출력, 실행 기록(lineage)
│   ├── quality.py              품질 게이트 — 실패 시 종료 코드 1
│   ├── report.py               지표 → 마크다운 리포트
│   ├── runner.py               오케스트레이션 — 순서·시간·기록
│   └── steps/
│       ├── base.py             단계 계약 (StepResult, Artifact)
│       ├── __init__.py         ★ 단계 목록 = 파이프라인의 목차
│       ├── loaders.py          Pandas·Polars 로딩 비교
│       ├── missing.py          결측
│       ├── duplicates.py       중복
│       ├── outliers.py         이상치
│       ├── visualize.py        차트 10개
│       ├── statistics.py       기술통계·상관계수·t-test
│       └── model.py            ML Pipeline
├── docs/                       단계별 근거 문서 7개
├── tests/test_steps.py         단위 테스트 32개
├── data/
│   ├── raw/                    원본 (자동 다운로드, git 제외)
│   ├── interim/                --checkpoint 시 중간 산출물
│   └── processed/              ★ 정제 데이터
└── outputs/runs/<run_id>/      실행 기록
```

### 설계 원칙 세 가지

**1. 분석 로직은 파일을 읽지도 출력하지도 않는다**

모든 단계가 같은 모양이다.

```python
def 단계(df: DataFrame, cfg: Config) -> StepResult:
    return StepResult(df=처리결과, metrics={...}, notes=["근거 문장"])
```

파일 읽기·쓰기는 `storage.py`, 출력은 `report.py`가 맡는다. 이 분리 덕분에
409만 행을 읽지 않고 3행짜리 DataFrame으로 로직을 테스트할 수 있다
(테스트 32개가 0.5초에 끝난다).

차트·모델처럼 파일이 되는 산출물은 `Artifact`로 **저장하는 방법만** 넘기고,
경로는 runner가 정한다. 단계는 여전히 순수 함수로 남고 산출물은
실행마다 `run_id` 디렉터리에 격리된다.

**2. 임계값은 코드가 아니라 설정에 있다**

소요시간 상한을 6시간에서 2시간으로 바꾸려면 `config/pipeline.toml`만 고친다.
코드 diff에 로직 변경과 값 조정이 섞이지 않고, 스케줄러가 월별로 다른 설정을
넘기는 것도 가능하다. 설정 해시가 매니페스트에 남아 재현도 된다.

**3. 기대와 다르면 프로세스가 실패로 끝난다**

스케줄러는 로그를 읽지 않는다. 품질 게이트가 종료 코드로 말한다.

| 코드 | 의미 |
|---|---|
| 0 | 성공 — 모든 게이트 통과 |
| 1 | 게이트 실패 — 산출물은 생겼지만 신뢰할 수 없다 |
| 2 | 실행 오류 — 파일 없음·설정 오류·예외 |

게이트는 두 종류다.

- **exact (7개)** — 2026-05에서 재현돼야 하는 정확한 값. 리팩터링하다 결과가
  바뀌는 로직 회귀를 잡는다.
- **range (7개)** — 보존율·결측률·음수 총액·차트 수·효과크기·F1·학습 표본 수.
  월이 달라도 통하는 규칙이라 데이터 자체의 이상을 잡는다.

> F1 게이트에 **상한 0.95**를 둔 것이 중요하다. 하한만 두면 누수를 못 잡는다.
> F1이 0.95를 넘으면 성능이 좋은 게 아니라 타깃 정보가 피처에 새어 들어갔다는
> 신호다.

---

## 주요 처리 기준

자세한 근거는 [`결측치_중복_처리기준.md`](결측치_중복_처리기준.md)와
[`docs/`](docs/)에 있다.

### 결측·중복

| 항목 | 통상적인 처리 | 이 데이터에서의 결론 |
|---|---|---|
| 결측치 | `dropna()` | **삭제 금지** — 5개 컬럼이 100% 같은 행에서 결측(23.35%). 지우면 심야·주말·특정 사업자가 편향되어 사라진다 |
| 결측 대체 | `fillna(0)` | **금지** — 결측행의 61.5%는 총액 역산상 실제로 2.50이 부과됐다 |
| 중복 | `drop_duplicates()` | 완전중복 **0건**. 그냥 쓰면 아무것도 안 지워지고 "중복 없음"으로 오판 |
| 중복 제거 | `keep='first'` | **금지** — 정상 운행 29,370건을 지우고 취소 기록을 남긴다 |

대신 이렇게 한다.

- 결측행은 `record_source` 컬럼으로 **표시만** 하고 남긴다
- 위장 결측(`payment_type=0`, `RatecodeID=99` 등 114만 건)을 NaN으로 변환
- 중복은 그룹의 금액 관계로 유형을 판정해 선택 제거
  - 상쇄쌍(취소·환불, 13,301쌍) → 양쪽 제거
  - 이중계상(총액 정확히 2배, 16,086건) → 큰 쪽만 제거

### 통계 검정

t-검정 결과에 **p-value만 쓰지 않는다.** n이 266만이면 아주 작은 차이도
p < 0.001이 되어 "유의하다"가 변별력을 잃는다. 효과크기를 함께 낸다.

```
장거리(≥5mi)(477,000건, 평균 17.16%) vs 단거리(<5mi)(2,182,779건, 평균 25.85%)
— 차이 -8.69%p, t = -450.96, p < 2.23e-308, Cohen's d = -0.703(중간).
유의하며 효과크기도 중간 이상이라 실질적으로 의미 있는 차이다.
```

### ML 모델

고액팁(팁 비율 25% 이상) 여부를 예측하는 이진 분류.

| 항목 | 값 | 이유 |
|---|---|---|
| 모집단 | 카드결제 2,659,779건 | 현금·무료·분쟁 결제는 팁이 **100% 0**으로 기록된다. 섞으면 "결제수단 맞히기"가 된다 |
| 타깃 임계값 | 25% | 양성 56.13%로 균형. 15%면 "전부 양성" 모델이 정확도 0.79로 실제 모델(0.77)을 이긴다 |
| 누수 제외 | `tip_amount`, `total_amount`, `payment_type` | `total_amount`는 팁을 포함한 합계라 역산된다 |
| 학습 표본 | 50만 층화 추출 | 전체 266만 대비 ROC-AUC +0.0005인데 시간은 5.7배 |

정확도 0.7703 / F1 0.8217 / ROC-AUC 0.7890
(다수 클래스만 예측하는 기준선 0.5614 대비 **+0.2089**)

---

## 자주 쓰는 명령

```bash
python run_pipeline.py fetch                 # 원본만 내려받기
python run_pipeline.py --force-download      # 원본을 새로 받아 실행
python run_pipeline.py list-steps            # 등록된 단계 확인
python run_pipeline.py --dry-run             # 실행 계획만 출력
python run_pipeline.py --steps deduplicate   # 특정 단계만
python run_pipeline.py --checkpoint          # 단계별 중간 산출물 저장
python run_pipeline.py --no-save             # 검증만, parquet 저장 안 함
python run_pipeline.py --log-format json     # 로그 수집기용
python run_pipeline.py --input <다른.parquet> # 입력 직접 지정
python run_pipeline.py compare-loaders       # 로딩 비교 상세 출력
```

`--steps`로 일부만 돌리면 뒤 단계의 지표가 없어 **게이트가 실패하고 종료 코드 1이
나온다**. 디버깅 목적이라면 정상 동작이다.

### 다른 월 데이터 처리

```bash
cp config/pipeline.toml config/pipeline-2026-06.toml
# month 수정 + check_exact = false (정확값은 2026-05 전용이므로)
# url_template과 paths.raw의 {month}는 자동 치환된다
python run_pipeline.py --config config/pipeline-2026-06.toml
```

`check_exact = false`로 두면 범위 게이트만 검사한다. 이 규칙들은 월이 달라도
통하는 일반 규칙이라 데이터 자체의 이상은 계속 잡힌다.

> TOML 주의: `check_exact`는 반드시 `[expectations.exact]`보다 **위에** 둬야 한다.
> 하위 테이블이 열린 뒤의 키는 그 테이블에 속하게 된다.

### 스케줄 등록

```cron
# 매월 5일 새벽 3시. 종료 코드가 0이 아니면 스케줄러가 실패로 처리한다.
0 3 5 * * cd /path/to/day2-data-analysis && mkdir -p logs && \
    .venv/bin/python run_pipeline.py --log-format json >> logs/cron.log 2>&1
```

Airflow에서는 `BashOperator`로 같은 명령을 걸면 되고, 단계별로 나누고 싶으면
`--steps`로 태스크를 쪼갤 수 있다. 다만 **`deduplicate` → `filter_outliers`
순서는 지켜야 한다** — 상쇄쌍을 먼저 지워야 음수 금액이 89% 줄어들고,
순서를 바꾸면 이상치 필터가 취소 쌍의 한쪽만 지워 짝이 깨진 기록이 남는다.

---

## 알려진 판단 지점

`duration_policy` 설정 하나는 분석 목적에 따라 바꿔야 한다.

소요시간 ≤ 0 필터가 `VendorID=7`을 **51,750건 전량** 제거한다. 이 사업자는
하차시각을 승차시각과 같게 기록하기 때문이다. 값이 이상한 게 아니라 소스가 다르다.

| 설정 | 최종 행 수 | VendorID=7 | 적합한 분석 |
|---|---:|---|---|
| `duration_policy = "drop"` (기본) | 3,884,622 (94.96%) | 전량 삭제 | 소요시간·속도 |
| `duration_policy = "flag"` | 3,936,456 (96.23%) | **보존** | 운행량·요금·존 |

`flag`는 삭제 대신 `duration_valid` 컬럼을 달아 행을 남긴다. 소요시간을 쓰는
집계에서만 `df[df.duration_valid]`로 걸러 쓰면 된다.

바꿔 실행할 때는 `check_exact = false`도 함께 둬야 한다. `exact` 기대값 7개는
`drop` 기준으로 기록된 값이라 그대로 두면 게이트가 실패한다.

파이프라인은 이런 "사업자 전멸"을 자동 감지해 경고 로그와 리포트에 남긴다.

### 그 밖의 한계

실행할 때마다 리포트의 「한계」 섹션에 자동으로 실린다.

- **partial 소스 95만 행**은 행은 남겼지만 승객수·요율·결제수단 컬럼 자체가 없다.
  해당 컬럼을 쓰는 분석에서는 자동 제외되므로 **그 결과는 부분집합에 대한 것**이다.
- **관측이 완전히 독립이 아니다.** 같은 기사·차량이 하루에 여러 번 운행하는데
  식별자가 없어 보정할 수 없다. p-value가 실제보다 작게 나온다.
- **모델은 무작위 분할**이라 "미래 예측" 성능은 과대평가될 수 있다.
  하이퍼파라미터도 튜닝하지 않았다.

---

## 문서

| 문서 | 내용 |
|---|---|
| [`결측치_중복_처리기준.md`](결측치_중복_처리기준.md) | 결측·중복·이상치 기준과 실측 근거 |
| [`docs/00_과제_수행계획.md`](docs/00_과제_수행계획.md) | 전체 작업 계획 |
| [`docs/01_분석설계_근거.md`](docs/01_분석설계_근거.md) | t-test 가설·ML 문제 정의 |
| [`docs/02_시각화_근거.md`](docs/02_시각화_근거.md) | 차트 선택과 표현 결정 |
| [`docs/03_통계분석_근거.md`](docs/03_통계분석_근거.md) | 검정 방법·전제·해석 규칙 |
| [`docs/04_ML파이프라인_근거.md`](docs/04_ML파이프라인_근거.md) | 피처·전처리·모델·평가 결정 |
| [`docs/05_리포트_근거.md`](docs/05_리포트_근거.md) | 리포트 구성 결정 |
| [`docs/06_최종점검.md`](docs/06_최종점검.md) | 최종 점검 결과 |

모든 결정은 `결정 / 근거 / 반례 / 한계` 형식으로 기록했다.
**반례**는 "그렇게 하지 않았다면 무슨 일이 생기는가"를 실측으로 보인 것이다.

---

## 저장소에 포함되지 않는 것

`.gitignore`로 제외한다. 코드가 아니거나 언제든 재생성되는 것들이다.

| 경로 | 이유 |
|---|---|
| `data/` | 원본 66.5MB + 정제 결과 76MB. 입력물이지 코드가 아니다 |
| `outputs/` | `run_pipeline.py`로 재생성된다 |
| `logs/` | cron 로그 |
| `.venv/`, `__pycache__/` | 환경·캐시 |

---

## 함께 있는 다른 파일

| 파일 | 성격 |
|---|---|
| `visualization_original.ipynb` | 시각화 파트의 원본 EDA 노트북. Part 5·12는 `taxi_pipeline/viz/quality_charts.py`로 옮겨 파이프라인이 매 실행 다시 계산한다 |
| `NYC_Yellow_Taxi_제출용_요약.md` | 제출용 요약. 리포트 구성의 기준이 된 문서다 |
| `NYC_Yellow_Taxi_원본_데이터_시각화_분석_보고서.pdf` | 원본 데이터 시각화 분석 보고서 |

초기 탐색용 단일 스크립트(`taxi_data_overview.py`, `taxi_eda_pipeline.py`)는
`taxi_pipeline/` 패키지로 재구성이 끝나 삭제했다. 필요하면 git 이력에서 찾을 수 있다.
