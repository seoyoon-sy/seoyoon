"""그림의 y축(온도)을 0.2 ℃ 간격 눈금으로 맞추는 도우미. 패널마다 데이터 범위에 맞춰 y축 범위를 줄인다(범위는 0.2의 배수).
눈금이 많아지면(범위가 2.2 ℃ 넘음) 눈금 선은 0.2 ℃마다 두고 숫자 표시만 건너뛴다(max_labels개 이하)."""
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

def fit_y(ax, lo, hi, step=0.2, pad=0.1, max_labels=12):
    y0 = np.floor((lo - pad) / step + 1e-9) * step; y1 = np.ceil((hi + pad) / step - 1e-9) * step
    ax.set_ylim(y0, y1); ax.yaxis.set_major_locator(MultipleLocator(step))
    n = int(round((y1 - y0) / step)) + 1; k = next(c for c in (1, 2, 5, 10, 25, 50) if n / c <= max_labels)   # 숫자는 0.2, 0.4, 1.0, 2.0, 5.0 ℃ 단위로만 건너뜀
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}" if int(round(y / step)) % k == 0 else ""))
    return y1 - y0, k
