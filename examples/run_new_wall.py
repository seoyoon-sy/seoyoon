"""VOLTRA 파일 없이, 코드만으로 새 벽체의 실내 표면온도를 계산하는 예.
벽 구조(층, 물성), 날씨(배열)를 직접 넣는다. TRY3 와 다른 구조도 가능."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hbcalc import Boundary, Solver, Layer, TJunction, build_tjunction, Material

# 1) 재료와 벽 구조 (실외 -> 실내). 두께 mm, 물성 SI.
materials = {
    "brick":    Material("brick", 0.72, 1800, 840),
    "XPS":      Material("XPS", 0.034, 30, 1450),
    "concrete": Material("concrete", 1.6, 2200, 795),
    "plaster":  Material("plaster", 0.7, 1400, 840),
    "breaker":  Material("breaker", 0.2, 800, 1000),      # 열교 차단 띠
}
spec = TJunction(
    layers=[Layer("brick", 90), Layer("XPS", 100), Layer("concrete", 180), Layer("plaster", 15)],
    materials=materials, partition_material="concrete", partition_mm=150,
    strip_material="breaker", strip_mm=10, strip_from_layer=1, strip_extend_mm=100,
    offsets_mm=(0, 10, 20, 50, 100, 200, 500),
)
model = build_tjunction(spec)

# 2) 경계조건: 한 달(744시간), 실외온도 일변화, 실내 20 도 일정, 일사 없음 (G 를 주면 일사 계산됨)
n = 744
h = np.arange(n)
T_out = -5 + 5 * np.sin(2 * np.pi * (h % 24 - 9) / 24)
bc = Boundary(T_out=T_out, T_in=20.0, h_out=17.0, h_in=1.5, dt=3600.0)

# 3) 계산
s = Solver(model)
s.run(bc)
night = np.zeros(n, bool)
for d in range(30):
    night[18 + 24 * d: 18 + 24 * d + 14] = True
print("격자", model.shape, "노드", s.N)
print("야간 평균 실내 표면온도 (접합부면에서 떨어진 거리 mm : ℃)")
for o in spec.offsets_mm:
    print(f"  {o:4d} mm : {s.series(f'surf_{o}')[night].mean():.2f}")
print("  기준(먼 곳)   :", round(s.series("ref_surf")[night].mean(), 2))
