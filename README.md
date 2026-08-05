# NYC Yellow Taxi 정제·EDA 파이프라인

2026-05 Yellow Taxi 운행 기록(409만 행)을 정제하고 기본 EDA를 수행한다.
**어떤 기준으로 결측치와 중복을 처리할지**는 실제 데이터에서 근거를 찾아 정했고,
그 근거는 [`결측치_중복_처리기준.md`](결측치_중복_처리기준.md)에 있다.
이 저장소의 코드는 그 문서의 기준을 실행 가능한 형태로 옮긴 것이다.

---

## 빠른 시작

**요구 사항**: Python **3.11 이상** (설정 파일을 읽는 `tomllib`이 3.11부터 표준 라이브러리)

```bash
pip install -r requirements.txt

python run_pipeline.py --dry-run      # 무엇을 할지 먼저 확인
python run_pipeline.py                # 전체 실행 (원본 자동 다운로드 포함, 약 8초)
python tests/test_steps.py            # 단위 테스트 22개 (원본 없이 0.5초)
```

원본 데이터를 미리 받아둘 필요가 없다. `data/raw/`에 파일이 없으면
NYC TLC 공개 엔드포인트에서 자동으로 내려받는다(66.5MB, 약 1초).
이미 있으면 건드리지 않는다.

```bash
python run_pipeline.py fetch            # 원본만 받아두기
python run_pipeline.py --force-download # 원본이 갱신됐을 때 새로 받기
```

정상 실행 시 터미널 마지막에 이렇게 나온다.

```
품질 게이트: 10개 전부 통과
최종 3,884,622행 (원본 대비 보존율 94.96%)
리포트: outputs/runs/20260804T232745Z_001fa159e74d_9aa5a1609e2b/report.md
```

---

## 입력 데이터

`config/pipeline.toml`의 `[source]`가 출처를 정한다.

```toml
[source]
url_template = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
auto_download = true      # false면 없을 때 받지 않고 안내와 함께 실패
timeout_sec = 120
```

`{month}`는 `project.month`로 치환되므로, **다른 달을 처리할 때 `month` 하나만
바꾸면 저장 경로와 다운로드 URL이 함께 따라간다.**

다운로드는 세 가지를 지킨다.

- **원자적 쓰기** — `.part`로 받아 검증에 통과해야 최종 이름으로 바꾼다.
  중간에 끊긴 파일이 `data/raw/`에 남으면 다음 실행이 그걸 "파일 있음"으로
  판단해 깨진 데이터로 돌아버린다.
- **받은 뒤 검증** — 크기와 parquet 메타데이터(행 수)를 확인한다.
  HTTP 200인데 본문이 에러 페이지인 경우를 여기서 잡는다.
- **없을 때만** — 이미 있으면 다시 받지 않는다. 원본이 바뀌면 입력 해시가
  달라져 매니페스트에 드러난다.

네트워크가 막힌 환경이라면 `auto_download = false`로 두면 된다.
조용히 멈추지 않고 받을 URL을 알려주며 종료 코드 2로 실패한다.

---

## 산출물

실행하면 두 곳에 결과가 생긴다.

### 1. 정제 데이터 — `data/processed/yellow_2026-05_clean.parquet`

파이프라인의 최종 결과물. **76MB, 3,884,622행 × 21열** (원본 대비 94.96% 보존).

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
  판정한 그룹이라 의도적으로 유지했다(문서 §3.6 ③).

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
| `report.md` | 사람이 읽는 근거 리포트 | **여기부터 본다** |
| `metrics.json` | 지표 48개 (기계가 읽음) | 이전 실행과 비교할 때 |
| `manifest.json` | 입력·설정 해시, 단계별 소요, 게이트 결과 | "이 결과가 뭘로 만들어졌나" 추적할 때 |

```bash
# 최신 실행 리포트 열기
open "$(ls -td outputs/runs/*/ | head -1)/report.md"
```

**`report.md` 구성** — 품질 게이트 → 단계별 처리 → **처리 근거** → 기본 EDA

근거 섹션이 핵심이다. 각 단계가 왜 그렇게 처리했는지를 실측 숫자와 함께 남긴다.

```
### deduplicate
- 20개 컬럼 완전중복 0건 — drop_duplicates()는 아무 행도 지우지 않는다.
- drop_duplicates(keep='first')를 썼다면 양수 총액 29,370건(정상 운행)이
  지워지고 음수 14,847건(취소 기록)이 남는다. 순서 기반 제거는 사용 금지.

### filter_outliers
- [경고] VendorID=7가 51,750건 전량 삭제됐다 (소요시간<=0 비율 100.0%)...
```

**`metrics.json` 활용** — 다음 달 실행과 비교해 데이터 사고를 잡는 용도다.

```python
import json
m = json.load(open("outputs/runs/<run_id>/metrics.json"))
m["deduplicate"]["naive_keep_first"]
# {'positive_rows_dropped': 29370, 'negative_rows_dropped': 30, 'negative_rows_left': 14847}
m["filter_outliers"]["vendors_wiped_out"]
# [{'vendor': 7, 'rows': 51750, 'nonpositive_duration_ratio': 1.0}]
m["profile"]["describe"]["fare_amount"]["mean"]   # 21.45
```

### 3. 중간 산출물 — `data/interim/` (선택)

`--checkpoint`를 줄 때만 생긴다. 단계마다 parquet을 써서(각 70MB 내외)
"이 단계 직후 데이터가 어땠는지"를 사후에 열어볼 수 있게 한다. 기본은 꺼져 있다.

> **아직 없는 기능**: 저장한 중간 산출물을 **다시 읽어 이어서 실행하는
> `--from <step>`은 구현돼 있지 않다.** 지금은 쓰기 전용 스냅샷이다.
> 전체 실행이 8초라 재시작의 실익이 없어 미뤄뒀다. 단계가 늘어 실행이
> 길어지면 그때 붙이면 된다.

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
│       ├── base.py             단계 계약 (StepResult)
│       ├── __init__.py         ★ 단계 목록 = 파이프라인의 목차
│       ├── missing.py          결측 (문서 §2)
│       ├── duplicates.py       중복 (문서 §3)
│       ├── outliers.py         이상치 (문서 §4)
│       └── profile.py          기본 EDA
├── tests/test_steps.py         단위 테스트 22개
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
(테스트 22개가 0.5초에 끝난다).

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
- **range (3개)** — 보존율·결측률·음수 총액. 월이 달라도 통하는 규칙이라
  데이터 자체의 이상(공급처 포맷 변경 등)을 잡는다.

---

## 처리 기준 요약

자세한 근거는 [`결측치_중복_처리기준.md`](결측치_중복_처리기준.md)에 있다.

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
python run_pipeline.py compare-loaders       # pandas vs polars 로딩 비교
```

`--steps`로 일부만 돌리면 뒤 단계의 지표가 없어 **게이트가 실패하고 종료 코드 1이
나온다**. 디버깅 목적이라면 정상 동작이다.

### 다른 월 데이터 처리

```bash
cp config/pipeline.toml config/pipeline-2026-06.toml
# month, paths.raw 수정 + check_exact = false (정확값은 2026-05 전용이므로)
# url_template은 {month}가 자동 치환되므로 손댈 필요 없다
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
나머지 판단 지점은 문서 §6에 정리돼 있다.

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

## 함께 있는 다른 스크립트

| 파일 | 성격 |
|---|---|
| `taxi_data_overview.py` | 데이터 첫 탐색용. 기본정보를 7단계로 출력만 한다(읽기 전용) |
| `taxi_eda_pipeline.py` | 같은 기준의 **단일 스크립트 버전**. 읽기 쉽지만 print 기반이라 자동화에 맞지 않아 `taxi_pipeline/` 패키지로 재구성했다. 결과(7개 기대값)는 양쪽 동일 |

두 파일 모두 파이프라인 동작에는 관여하지 않는다. 정리해도 무방하다.
