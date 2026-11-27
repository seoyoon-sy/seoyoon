"""격자 시험(대칭 격자 L0 ~ L3, 균일 U10, U5) VOLTRA 정상상태 결과를 3D 솔버(작은 형상계수 버리기 포함)로 비교 (CLAUDE.md 8.79).
사용: python examples/compare_mesh_study_solver3d.py L0,L1,L2"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.solver3d import Solver3D
from hbcalc.presets import try3_model, try3_excel_strip_props
from hbcalc.voltra_report import read_report
LV = {"L0": (10, 100, 40), "L1": (5, 50, 20), "L2": (2.5, 25, 10), "L3": (1.25, 12.5, 5)}
P = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), diffuse_obstruction=True)
for lv in sys.argv[1].split(","):
    for x in (1.6, 0.0001):
        a, b, c = LV[lv]; V = np.array(read_report(f"{ROOT}/voltra_results/mesh_study/MS_X{x:g}_{lv}.txt")[1][0][7:31][0::2])
        rho, cp = try3_excel_strip_props(x)
        m = try3_model({"EPS": 0.032, "XPS": x}, h_min_mm=a, h_max_mm=b, h_layer_mm=c, symmetric_mesh=True).with_properties(rho={"XPS": rho}, cp={"XPS": cp})
        s2 = Solver(m); s2.run(bc, nsteps=1); h2 = np.array([s2.series(f"surf_{p}")[0] for p in P])
        row = [f"X{x:g} {lv} 노드 {s2.N}", f"2D {abs(h2 - V).mean():.3f}/{abs(h2 - V).max():.3f} 0mm {h2[0] - V[0]:+.3f} 500mm {h2[-1] - V[-1]:+.3f}"]
        for vf in (0.0, 1e-4):
            t = time.time(); s3 = Solver3D(m, nz=20, vf_min=vf); s3.run(bc, nsteps=1); h3 = np.array([s3.series(f"surf_{p}")[0] for p in P])
            row.append(f"3D cut{vf:g} {abs(h3 - V).mean():.3f}/{abs(h3 - V).max():.3f} 0mm {h3[0] - V[0]:+.3f} 500mm {h3[-1] - V[-1]:+.3f} ({time.time() - t:.0f}s, 500mm {h3[-1]:.3f} V {V[-1]:.2f})")
        print(" | ".join(row), flush=True)
