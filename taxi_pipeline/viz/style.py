"""차트의 색상·라벨·폰트 설정을 한곳에서 관리한다.

여기 있는 값은 전부 "그림에 보이는 것"이다. 두 시각화 단계(`steps/visualize.py`,
`steps/visualize_raw.py`)가 같은 값을 각자 하드코딩하면 한쪽만 바뀌어 같은 개념이
다른 색·다른 이름으로 나온다. 실제로 결제수단 컬럼이 한쪽은 "결제수단",
다른 쪽은 "결제방식"으로 갈려 있었다.
"""

from __future__ import annotations

import logging

import matplotlib
# 화면이 없는 환경에서도 저장할 수 있도록 Agg 백엔드를 사용한다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import seaborn as sns                     # noqa: E402

log = logging.getLogger(__name__)

# Seaborn deep 팔레트에서 고른 색이다. 의미로 이름을 붙여 두면
# "파란 막대"가 아니라 "기본 계열"로 읽힌다.
PRIMARY = "#4C72B0"      # 기본 계열 — 건수·분포 등 주 지표
ACCENT = "#C44E52"       # 강조 계열 — 평균선·주말·비교 대상
POSITIVE = "#55A868"     # 보조 계열 — 중앙값·필터링 후 등 대비군
SECONDARY = "#DD8452"    # 두 번째 그룹 — 소스별 비교의 나머지 한쪽

DOW = ["월", "화", "수", "목", "금", "토", "일"]

# 결제수단 코드를 화면에 표시할 이름으로 변환한다.
PAYMENT = {1: "카드", 2: "현금", 3: "무료", 4: "분쟁"}
# 결제수단을 담는 임시 컬럼의 이름. 그대로 축 레이블·범례로 쓰인다.
PAYMENT_LABEL = "결제수단"


def setup_style(cfg) -> str:
    """한글 폰트와 공통 차트 설정을 적용하고 선택된 폰트 이름을 반환한다."""
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in cfg.visualize.font_candidates if f in available), None)
    if not chosen:
        log.warning("한글 폰트를 찾지 못했습니다. 축 레이블이 깨질 수 있습니다.")
        chosen = "DejaVu Sans"

    plt.rcParams["font.family"] = chosen
    # 한글 폰트에서 음수 기호가 깨지지 않도록 설정한다.
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=chosen, rc={"axes.unicode_minus": False})
    return chosen
