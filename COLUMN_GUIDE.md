# Yellow Taxi 주요 컬럼 설명

| 컬럼 | 분석에서의 의미 |
|---|---|
| `VendorID` | 운행 기록을 제공한 기술 사업자 코드 |
| `tpep_pickup_datetime` | 미터기가 켜진 승차 일시 |
| `tpep_dropoff_datetime` | 미터기가 꺼진 하차 일시 |
| `passenger_count` | 운전자가 입력한 승객 수 |
| `trip_distance` | 미터기에 기록된 운행거리(마일) |
| `RatecodeID` | 운행 종료 시 적용된 요금 코드 |
| `PULocationID` | 승차 택시 존 ID |
| `DOLocationID` | 하차 택시 존 ID |
| `payment_type` | 결제 유형 코드 |
| `fare_amount` | 시간·거리 기준 미터요금 |
| `tip_amount` | 카드 팁 금액이며 현금 팁은 포함되지 않음 |
| `tolls_amount` | 통행료 총액 |
| `total_amount` | 승객에게 청구된 총액이며 현금 팁은 포함되지 않음 |
| `congestion_surcharge` | 혼잡 통행 관련 추가 요금 |
| `airport_fee` | JFK·LaGuardia 공항 승차 관련 요금 |
| `cbd_congestion_fee` | 2025년부터 추가된 혼잡 완화 구역 운행 요금 |

## 결제 유형 코드

| 코드 | 의미 |
|---:|---|
| 0 | Flex Fare |
| 1 | Credit card |
| 2 | Cash |
| 3 | No charge |
| 4 | Dispute |
| 5 | Unknown |
| 6 | Voided trip |
