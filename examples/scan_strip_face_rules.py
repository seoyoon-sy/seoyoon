"""결로 방지 단열재(XPS 판) 노출 면의 대류 면적 배분 규칙 탐색 (8.34 후속).
벽체 8개(N1 ~ N6, N5b, N6b)의 VOLTRA 정상상태(t=0) 접합부 표면온도 12위치와 비교.
XPS 판에 접한 면은 (sx, sy), 그 밖의 면은 (wx, wy) 가중치(1 = 좌표가 작은 쪽 끝 노드에 전부, 0 = 큰 쪽, 0.5 = 절반씩).
가로면(ax = x)과 세로면(ax = y)을 따로 둔다. 사용: python examples/scan_strip_face_rules.py"""
import os, sys, pickle, itertools, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.voltra_report import read_report
import test_walls_def as T

POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=0.0)
STRIP = {"N5": (0.5, 1200, 1000), "N6": (0.04, 40, 1450)}

def model_with_strip(key, strip_from):
    text, th, part, strip, ext, props = T.WALLS[key]
    props = dict(props); props["XPS"] = STRIP[strip_from]
    mats = {T.NAMES[k]: T.M(T.NAMES[k], *v) for k, v in props.items()}
    layers = [Layer(T.NAMES[k], t) for k, t in zip(("concrete", "EPS", "block", "gypsum"), th) if t > 0]
    return build_tjunction(replace(try3_spec(snap_mm=1), layers=layers, partition_mm=part, strip_mm=strip, strip_extend_mm=ext, materials=mats))

res = pickle.load(open(os.path.join(ROOT, "voltra_results/test_walls/_cmp.pkl"), "rb"))
W = {}
for k in res:
    W[k[:2]] = (T.build(k)["mj"], res[k]["junction"]["Vs"][0])
for nm, key, sf in (("N6b", "N6_noblock_insstrip", "N5"), ("N5b", "N5_noblock", "N6")):
    _, a = read_report(os.path.join(ROOT, "voltra_results/strip_split", {"N6b": "N6b_strip0.5", "N5b": "N5b_strip0.04"}[nm] + ".txt"))
    W[nm] = (model_with_strip(key, sf), a[0, 7:][0::2])

class Case:
    def __init__(self, m, V):
        self.s = Solver(m, conv_lump="half"); self.V = V
        xid = [k for k, v in self.s.m.materials.items() if v.name == "XPS"][0]; cid = self.s.m.cid
        self.rows = []
        for f, lo, hi in zip(self.s.faces, self.s._face_low_node, self.s._face_high_node):
            i, j, ax, zn, eps, refl, area, side = f
            mat = (cid[i, j - 1] if side == "up" else cid[i, j]) if ax == "x" else (cid[i - 1, j] if side == "right" else cid[i, j])
            self.rows.append((zn, lo, hi, area, ax, mat == xid))
    def dev(self, sx, sy, wx, wy):
        s = self.s; cA = {z: np.zeros(s.N) for z in s.zone_ids}
        for zn, lo, hi, area, ax, isx in self.rows:
            w = (sx if ax == "x" else sy) if isx else (wx if ax == "x" else wy)
            cA[zn][lo] += w * area; cA[zn][hi] += (1 - w) * area
        s.conv_A = cA; s.run(bc, nsteps=1)
        M = np.array([s.series(f"surf_{p}")[0] for p in POS]); return M - self.V

if __name__ == "__main__":
    C = {k: Case(*v) for k, v in W.items()}; keys = list(C)
    ws = (0.0, 0.25, 0.5, 0.75, 1.0)
    t0 = time.time(); out = []
    for sx, sy, wx, wy in itertools.product(ws, repeat=4):
        d = [C[k].dev(sx, sy, wx, wy) for k in keys]
        rm = [np.sqrt((x ** 2).mean()) for x in d]; z = [x[0] for x in d]
        out.append((max(rm), float(np.mean(rm)), (sx, sy, wx, wy), rm, z))
    print("조합 %d개, %.0f초" % (len(out), time.time() - t0)); out.sort(key=lambda r: r[0])
    print("벽체:", keys)
    for mx, mean, p, rm, z in out[:10]:
        print(f"(sx,sy,wx,wy)={p} 최대RMSE {mx:.3f} 평균RMSE {mean:.3f} | RMSE " + " ".join(f"{r:.3f}" for r in rm) + " | 0mm " + " ".join(f"{v:+.3f}" for v in z))
    base = [r for r in out if r[2] == (1.0, 1.0, 1.0, 1.0)][0]
    print("현재 low(1,1,1,1): 최대RMSE %.3f 평균 %.3f | RMSE " % (base[0], base[1]) + " ".join(f"{r:.3f}" for r in base[3]) + " | 0mm " + " ".join(f"{v:+.3f}" for v in base[4]))
    pickle.dump(out, open(os.path.join(ROOT, "results", "scan_strip_face_rules.pkl"), "wb"))
