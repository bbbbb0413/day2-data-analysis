"""yellow_sample_1000.csv 간단 시각화."""
import matplotlib.pyplot as plt
import pandas as pd

# macOS 기본 폰트는 한글 지원 안 해서 라벨이 깨짐 -> 한글 폰트로 교체
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("yellow_sample_1000.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 이동거리 분포
axes[0].hist(df["trip_distance"], bins=30, color="steelblue")
axes[0].set_title("이동거리 분포")
axes[0].set_xlabel("trip_distance (마일)")
axes[0].set_ylabel("빈도")

# 이동거리 vs 요금 산점도
axes[1].scatter(df["trip_distance"], df["fare_amount"], alpha=0.5, color="teal")
axes[1].set_title("이동거리 vs 요금")
axes[1].set_xlabel("trip_distance (마일)")
axes[1].set_ylabel("fare_amount ($)")

plt.tight_layout()
plt.savefig("yellow_sample_1000_plot.png", dpi=150)
print("저장 완료: yellow_sample_1000_plot.png")
plt.show()
