"""격자 독립성 시험 (CLAUDE.md 8.68). TRY3 구성(콘크리트 200, EPS 180, 블록 90, 석고 10, 내벽 180, 결로 방지 판 10 mm)에서
격자를 5단계로 촘촘히 하며(좌우 대칭, 접합부 최소 칸 10 -> 0.625 mm) 현재 기본 솔버(voltra 방식, 2D, 일사 보정 0)의 결과 변화를 본다.
지표: (1) TRISCO 조건 정상상태 Ψ (실내 20, 실외 -3.7 ℃, 실내 hc 1.415/1.42, 실외 16.03, hrb 5.1)
      (2) 744시간(난방 20 ℃) 야간 평균 표면온도 12위치와 ΔTs(500 mm - 0 mm)
      (3) 가장 촘촘한 격자 대비 야간 시간별 표면온도 차이(12위치 최대)
      (4) 노드 수와 계산 시간
참고 행: 현재 데이터 생성에 쓰는 자동 격자(비대칭, 정수 mm), 사용자 VOLTRA 손격자(TRY3_Plus_QTest2).
VOLTRA 쪽 격자 시험용 .vtr(정상상태 calc 3600, 난방)도 만든다: voltra_inputs/mesh_study/
사용: python examples/mesh_independence.py   (결과 results/mesh_independence/)"""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from hbcalc import Boundary, Solver
from hbcalc.presets import try3_model, try3_voltra_mesh_model, try3_excel_strip_props
from hbcalc.validate import night_mask
from hbcalc.write_vtr import write_vtr
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500); N = 744; mk = night_mask(N)
OUT = os.path.join(ROOT, "results", "mesh_independence"); os.makedirs(OUT, exist_ok=True)
VOUT = os.path.join(ROOT, "voltra_inputs", "mesh_study"); os.makedirs(VOUT, exist_ok=True)
CASES = {"강함(결로 방지 1.6)": 1.6, "중간(결로 방지 0.04)": 0.04, "약함(결로 방지 0.0001)": 0.0001}
LEVELS = {"L0": (10, 100, 40), "L1": (5, 50, 20), "L2": (2.5, 25, 10), "L3": (1.25, 12.5, 5), "L4": (0.625, 6.25, 2.5)}
LAYERS = [("concrete", 200), ("EPS", 180), ("0.5B cement block", 90), ("gypsum", 10)]

def model(x, lv):
    rho, cp = try3_excel_strip_props(x); lam = {"EPS": 0.032, "XPS": x}
    if lv == "자동(현재)": m = try3_model(lam, snap_mm=1)
    elif lv == "VOLTRA 손격자": m = try3_voltra_mesh_model(lam)
    else:
        a, b, c = LEVELS[lv]; m = try3_model(lam, h_min_mm=a, h_max_mm=b, h_layer_mm=c, symmetric_mesh=True)
    return m.with_properties(rho={"XPS": rho}, cp={"XPS": cp})

def psi_trisco(m):
    outd = [z for z in m.zones if m.zones[z].kind == "outdoor"][0]; ind = sorted(z for z in m.zones if m.zones[z].kind == "indoor")
    Ti, Te, ho, hrb0 = 20.0, -3.7, 16.03, m.hrb; m.hrb = 5.1
    bz = Boundary(np.array([Ti]), np.array([Ti]), np.array([1.0]), np.array([1.0]),
                  zone_values={outd: (np.array([Te]), np.array([ho])), ind[0]: (np.array([Ti]), np.array([1.42])), ind[-1]: (np.array([Ti]), np.array([1.415]))})
    s = Solver(m); s.run(bz, nsteps=1); F = s.zone_flows_into_out(); m.hrb = hrb0
    phis = [F[z][0][0] - F[z][1][0] for z in ind]
    R = 1 / (1.415 + 0.9 * 5.1) + 1 / (ho + 0.9 * 5.1) + sum(t / 1000 / next(v.lam for v in m.materials.values() if v.name == n) for n, t in LAYERS)
    phi2d = (Ti - Te) * m.xe[-1] * m.zt / R
    return (sum(phis) - phi2d) / (Ti - Te), phis, np.array([s.series(f"surf_{p}")[0] for p in POS])

def job(a):
    cname, lv = a; x = CASES[cname]; m = model(x, lv)
    t0 = time.time(); psi, phis, Tst = psi_trisco(m); t_st = time.time() - t0
    bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
    t0 = time.time(); s = Solver(m); s.run(bc, nsteps=N); t_run = time.time() - t0
    M = np.array([s.series(f"surf_{p}")[:N] for p in POS]).astype(np.float32)
    return a, dict(nodes=s.N, cells=f"{m.shape[0]}x{m.shape[1]}", hmin=float(np.diff(m.xe).min() * 1000), psi=psi, phis=phis, Tst=Tst, M=M, t_run=t_run, t_st=t_st)

if __name__ == "__main__":
    # VOLTRA 격자 시험용 .vtr (정상상태 calc 3600, 난방 20 ℃): 강함, 약함 x L0 ~ L3
    rows = []
    for cname, x in CASES.items():
        if x == 0.04: continue
        for lv in ("L0", "L1", "L2", "L3"):
            m = model(x, lv); nm = f"MS_X{x:g}_{lv}"
            write_vtr(m, os.path.join(ROOT, "TRY3_Plus_QTest2.vtr"), os.path.join(VOUT, nm + ".vtr"), calc_s=3600, indoor_bc="heating")
            bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
            s = Solver(m); s.run(bc, nsteps=2)
            r = {"파일": nm, "격자": f"{m.shape[0]}x{m.shape[1]}", "노드": s.N}
            for p in POS: r[f"t0_표면_{p}mm"] = s.series(f"surf_{p}")[0]
            rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(VOUT, "expected_hbcalc_steady.csv"), index=False, encoding="utf-8-sig")
    tasks = [(c, lv) for c in CASES for lv in ("L4", "L3", "L2", "L1", "L0", "자동(현재)", "VOLTRA 손격자")]
    res = {}
    with ProcessPoolExecutor(3) as ex:
        for a, r in ex.map(job, tasks): res[a] = r; print(a, r["nodes"], f"{r['t_run']:.0f}s", flush=True)
    pickle.dump(res, open(os.path.join(OUT, "raw.pkl"), "wb"))
    rows = []
    for c in CASES:
        ref = res[(c, "L4")]; Vn = ref["M"][:, mk]
        for lv in ("L0", "L1", "L2", "L3", "L4", "자동(현재)", "VOLTRA 손격자"):
            r = res[(c, lv)]; Mn = r["M"][:, mk]; nmean = Mn.mean(1); rmean = Vn.mean(1)
            d = Mn - Vn
            rows.append({"케이스": c, "격자": lv, "칸": r["cells"], "노드": r["nodes"], "최소칸_mm": round(r["hmin"], 3), "744h_계산초": round(r["t_run"], 1),
                         "정상Ψ": r["psi"], "Ψ_차이_L4대비": r["psi"] - ref["psi"],
                         "야간평균_0mm": nmean[0], "0mm_차이_L4대비": nmean[0] - rmean[0], "야간평균_500mm": nmean[11],
                         "ΔTs(500-0)": nmean[11] - nmean[0], "ΔTs_차이_L4대비": (nmean[11] - nmean[0]) - (rmean[11] - rmean[0]),
                         "시간별_최대차이_12위치_L4대비": float(np.abs(d).max()), "시간별_평균절대차이_L4대비": float(np.abs(d).mean()),
                         "정상_0mm": r["Tst"][0], "정상_0mm_차이_L4대비": r["Tst"][0] - ref["Tst"][0]})
    df = pd.DataFrame(rows); df.round(5).to_csv(os.path.join(OUT, "summary.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(df.drop(columns=["정상_0mm", "야간평균_500mm"]).round(4).to_string(index=False))
