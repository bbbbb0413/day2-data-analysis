"""NYC Yellow Taxi 정제·EDA 파이프라인.

계층 구조 (위가 아래에 의존한다):

    cli / run_pipeline.py     진입점 — 인자 파싱, 종료 코드
      └ runner.py             오케스트레이션 — 순서·시간·기록
          ├ steps/            분석 로직 (DataFrame, Config) -> StepResult
          ├ quality.py        기대값 검증, 실패 시 종료 코드
          └ report.py         지표 → 사람이 읽는 리포트
              └ storage.py    입출력, 실행 기록(lineage)
                  └ config.py 설정 (config/pipeline.toml)
                      observability.py  로깅·직렬화

핵심 원칙:
  - 분석 로직은 파일을 읽거나 쓰지 않고, 출력도 하지 않는다.
    입력은 DataFrame과 Config, 출력은 DataFrame과 지표뿐이다.
  - 임계값은 코드가 아니라 TOML에 있다.
  - 기대와 다르면 프로세스가 실패로 끝난다.

기준의 근거는 결측치_중복_처리기준.md 에 있다.
"""

from .config import Config, load_config
from .runner import RunResult, run_pipeline

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "run_pipeline", "RunResult", "__version__"]
