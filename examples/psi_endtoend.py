"""끝에서 끝까지 시험: VOLTRA 표면온도 대신 hbcalc 표면온도를 사용자의 학습 코드(training/psi_model_v6_3.py)에 넣었을 때
정답 Ψ(정상상태_정답.xlsx) 예측이 얼마나 달라지는가.
사용자 코드는 수정하지 않고, 경로만 바꿔서 불러온다. 모사 결과는 VOLTRA 엑셀과 같은 형식의 엑셀로 만들어 같은 전처리를 거친다.
전제: results/_abs_cache.pkl (python examples/compare_surface_temperature.py 로 생성). 독립 테스트는 Model A 를 제외한 9개.
사용: python examples/psi_endtoend.py [출력폴더]"""
import os, sys, re, pickle, types, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.planar import planar_model
from hbcalc.presets import TRY3_MATERIALS
from hbcalc.validate import parse_sheet

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "psi_endtoend")
os.makedirs(OUT, exist_ok=True)


def load_user_module():
    src = open(os.path.join(ROOT, "training", "psi_model_v6_3.py"), encoding="utf-8").read()
    src = src.replace("C:/Users/nolyb/Desktop/vol/", ROOT.replace("\\", "/") + "/")
    mod = types.ModuleType("psi_model_v6_3"); mod.__file__ = "psi_model_v6_3.py"
    exec(compile(src, "psi_model_v6_3.py", "exec"), mod.__dict__)
    return mod


M = load_user_module()
TEST9 = [c for c in M.TEST_CASES if c[0] != "Model A"]
DEPTHS = M.DEPTHS
N = 744
BCF = {"난방": "Wall_TI.FTE", "비난방": "Wall_TI_2.FTE"}


def planar_series(eps, ti, shift):
    mats = dict(TRY3_MATERIALS); mats["EPS"] = replace(mats["EPS"], lam=eps)
    pm = planar_model([("concrete", 200), ("EPS", 180), ("0.5B cement block", 90), ("gypsum", 10)], mats, hrb=5.25)
    bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=shift)
    s = Solver(pm); s.run(bc, nsteps=N)
    return s.series("surf")[:N], bc.T_out[:N], bc.T_in[:N]


def make_emulator_excels(cache, shift):
    """모사 결과를 VOLTRA 엑셀과 같은 열 이름으로 저장. 반환: 4개 파일 경로."""
    paths = {k: os.path.join(OUT, f"emu_{k}_shift{shift}.xlsx") for k in ("multi", "planar", "nh_tsi", "nh_q2d")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    names_h = [nm for (k, nm) in cache if k == "난방"]
    names_n = [nm for (k, nm) in cache if k == "비난방"]
    key = f"M_surf_{float(shift)}"
    with pd.ExcelWriter(paths["multi"]) as wm, pd.ExcelWriter(paths["planar"]) as wp:
        for nm in names_h:
            d = cache[("난방", nm)]
            df = pd.DataFrame({f"{p} 표면온도(℃)": d[key][i] for i, p in enumerate(DEPTHS)})
            df["실외 온도(℃)"], df["실내 온도(℃)"] = d["T_out"], d["T_in"]
            df.to_excel(wm, sheet_name=nm, index=False)
            e, _ = parse_sheet(nm); e = e if e is not None else 0.032
            P, To, Ti = planar_series(e, BCF["난방"], shift)
            dfp = pd.DataFrame({"실외 온도(℃)": To, "실내 온도(℃)": Ti, "비열교부 표면온도(℃)": P, "비열교부 석고보드 국부 열류량(W/m2)": 0.0,
                                 "실내로 들어오는 열류량1(W)": 0.0, "실내에서 나가는 열류량1(W)": 0.0, "실내로 들어오는 열류량2(W)": 0.0, "실내에서 나가는 열류량2(W)": 0.0})
            dfp.to_excel(wp, sheet_name=M.NAME_MAP_TO_PLANAR.get(nm, nm)[:31], index=False)
    with pd.ExcelWriter(paths["nh_tsi"]) as wt, pd.ExcelWriter(paths["nh_q2d"]) as wq:
        for nm in names_n:
            d = cache[("비난방", nm)]
            df = pd.DataFrame({f"{p} 표면온도(℃)": d[key][i] for i, p in enumerate(DEPTHS)})
            df["실외 온도(℃)"], df["실내 온도(℃)"] = d["T_out"], d["T_in"]
            df.to_excel(wt, sheet_name=nm, index=False)
            e, _ = parse_sheet(nm); e = e if e is not None else 0.032
            P, To, Ti = planar_series(e, BCF["비난방"], shift)
            pd.DataFrame({"비열교부 표면온도(℃)": P}).to_excel(wq, sheet_name=M.NAME_MAP_TO_NH_Q2D.get(nm, nm)[:31], index=False)
    return paths


def build_all(ef_multi, ef_planar, answer):
    tr, tg, te, ta = {}, {}, {}, {}
    for s in M.TRAIN_SHEETS:
        tr[s] = M.build_case(ef_multi, ef_planar, s, M.NAME_MAP_TO_PLANAR.get(s, s), False); tg[s] = M.get_answer(answer, s)
    for s2, s0 in TEST9:
        te[s2] = M.build_case(ef_multi, ef_planar, s2, s0, False); ta[s2] = M.get_answer(answer, s2)
    return dict(tr=tr, tg=tg, te=te, ta=ta)


def nh_seqs(tsi_path, q2d_path):
    M.FILE_NH_TSI, M.FILE_NH_Q2D = tsi_path, q2d_path
    return {c: M.build_case_2var_nonheating(c) for c in M.TRAIN_SHEETS}


def run_scenarios(Vd, Ed, label, rows_loco, rows_test):
    """Vd: VOLTRA 데이터, Ed: 모사 데이터. 시나리오 A(V->V), B(V->E), C(E->V), D(E->E)."""
    sc = {"A: 학습 VOLTRA, 시험 VOLTRA": (Vd, Vd), "B: 학습 VOLTRA, 시험 모사": (Vd, Ed),
          "C: 학습 모사, 시험 VOLTRA": (Ed, Vd), "D: 학습 모사, 시험 모사": (Ed, Ed)}
    for sname, (Tr, Te) in sc.items():
        if label != "shift0" and sname.startswith("A"):
            continue                       # A 는 모사와 무관하므로 한 번만
        names = M.TRAIN_SHEETS
        # LOCO: 학습 29개(Tr), 시험 = 빠진 벽체(Te) 야간별
        for held in names:
            fold = [c for c in names if c != held]
            Xtr = np.concatenate([Tr["tr"][c].mean(axis=1) for c in fold]); ytr = np.concatenate([np.full(M.N_NIGHTS, Tr["tg"][c]) for c in fold])
            Xte = Te["tr"][held].mean(axis=1); act = Te["tg"][held]
            pr = M.fit_predict_all(Xtr, ytr, Xte, 2)
            for m in M.MODEL_NAMES:
                for n in range(M.N_NIGHTS):
                    rows_loco.append(dict(shift=label, scen=sname, case=held, night=n + 1, model=m, actual=act, pred=float(pr[m][n])))
        # 독립 테스트: 학습 30개 전체(Tr), 시험 9개(Te)
        Xtr = np.concatenate([Tr["tr"][c].mean(axis=1) for c in names]); ytr = np.concatenate([np.full(M.N_NIGHTS, Tr["tg"][c]) for c in names])
        for s2, _ in TEST9:
            Xte = Te["te"][s2].mean(axis=1); act = Te["ta"][s2]
            pr = M.fit_predict_all(Xtr, ytr, Xte, 2)
            for m in M.MODEL_NAMES:
                rows_test.append(dict(shift=label, scen=sname, case=s2, model=m, actual=act, pred=float(np.mean(pr[m])), pred_nights=[float(x) for x in pr[m]]))
        print(label, sname, "완료", flush=True)


def main():
    warnings.filterwarnings("ignore")
    cache = pickle.load(open(os.path.join(ROOT, "results", "_abs_cache.pkl"), "rb"))
    answer = M.load_answer_table()
    ef_m = pd.ExcelFile(os.path.join(ROOT, "voltra_30cases_night420_통합_v2.xlsx"))
    ef_p = pd.ExcelFile(os.path.join(ROOT, "voltra_plus_result_30케이스_00mm.xlsx"))
    Vd = build_all(ef_m, ef_p, answer)
    rows_loco, rows_test = [], []
    for sh in (0.0, -0.25):
        paths = make_emulator_excels(cache, sh)
        Ed = build_all(pd.ExcelFile(paths["multi"]), pd.ExcelFile(paths["planar"]), answer)
        run_scenarios(Vd, Ed, "shift0" if sh == 0.0 else "shift-0.25", rows_loco, rows_test)
    pd.DataFrame(rows_loco).to_pickle(os.path.join(OUT, "loco.pkl")); pd.DataFrame(rows_test).to_pickle(os.path.join(OUT, "test.pkl"))
    print("저장:", OUT)


if __name__ == "__main__":
    main()
