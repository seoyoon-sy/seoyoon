"""TRISCO 결과(T1~T4, 대칭 격자)와 hbcalc 비교: 열류(들어옴, 나감, 순)와 영역 표면 최저, 최고 온도."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.compat_trisco import model_from_trc

# TRISCO 결과 (사용자 제공 리포트): 색 1 오른쪽 실내, 색 8 왼쪽 실내, 색 2 실외
TRISCO = {
 "T1_fine_sym_strong":    dict(in1=10.8599, out1=0.0000, in8=10.8655, out8=0.0000, out2=21.7254, t1=(14.6037, 19.8816), t8=(14.6047, 19.8817)),
 "T2_fine_sym_weak":      dict(in1=8.9275,  out1=0.4410, in8=8.9300,  out8=0.4408, out2=16.9756, t1=(16.4424, 19.8346), t8=(16.4455, 19.8347)),
 "T3_default_sym_strong": dict(in1=10.8883, out1=0.0000, in8=10.8927, out8=0.0000, out2=21.7812, t1=(14.6255, 19.8630), t8=(14.6265, 19.8631)),
 "T4_default_sym_weak":   dict(in1=9.0228,  out1=0.4964, in8=9.0249,  out8=0.4961, out2=17.0552, t1=(16.4550, 19.8181), t8=(16.4589, 19.8182)),
}

for name, t in TRISCO.items():
    m, zv = model_from_trc(os.path.join(ROOT, "trisco_cases", name + ".trc"))
    b = Boundary(np.array([20.0]), np.array([20.0]), np.array([1.0]), np.array([1.0]),
                 zone_values={c: (np.array([zv[c][0]]), np.array([zv[c][1]])) for c in zv})
    print(f"\n== {name}   TRISCO: 순 Φ(우) {t['in1']-t['out1']:.4f}, Φ2(좌) {t['in8']-t['out8']:.4f}, 실외 유출 {t['out2']:.4f}")
    for cl in ("half", "low"):
        s = Solver(m, conv_lump=cl); s.run(b, nsteps=1)
        F = s.zone_flows_into_out()
        T0 = s.T_all[0]
        r = {}
        for z, key in ((1, "1"), (8, "8")):
            i, o = F[z][0][0], F[z][1][0]
            tz = T0[s.zone_nodes[z]]
            r[z] = (i, o, i - o, tz.min(), tz.max())
        o2 = F[2][1][0] - F[2][0][0]
        print(f" {cl:4s}: 우 in {r[1][0]:.4f} out {r[1][1]:.4f} 순 {r[1][2]:.4f} ({(r[1][2]/(t['in1']-t['out1'])-1)*100:+.2f} %) | "
              f"좌 순 {r[8][2]:.4f} ({(r[8][2]/(t['in8']-t['out8'])-1)*100:+.2f} %) | 좌우비 {r[8][2]/r[1][2]:.4f} (TRISCO {(t['in8']-t['out8'])/(t['in1']-t['out1']):.4f}) | 실외 {o2:.4f}")
        print(f"       표면 최저온도 우 {r[1][3]:.4f} (TRISCO {t['t1'][0]}), 좌 {r[8][3]:.4f} (TRISCO {t['t8'][0]}); 최고 우 {r[1][4]:.4f} ({t['t1'][1]})")
