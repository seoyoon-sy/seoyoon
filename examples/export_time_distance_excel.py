"""시간대별(하루 24시) x 거리별(접합부 면에서 12개 위치) 표면온도와 오차 분포 엑셀: results/accuracy/time_distance_detail.xlsx
대상: TRY3 난방 39 + 비난방 30케이스(VOLTRA 엑셀), 추가 벽체 N1 ~ N6(난방 20도), N1 ~ N6와 A ~ D(비난방). 744시간 전체(낮 일사 포함).
시트: 읽는 법, 케이스목록, 시간대별_거리별(모든 케이스 긴 형식), 강도별_시간대_편향, 강도별_시간대_MAE, 시간별_요약, 상세_케이스(선택 케이스 744시간 전체).
사용: python examples/export_time_distance_excel.py   (약 8분, 중간 결과 캐시 results/_time_distance_cache.pkl 은 gitignore)"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter
from hbcalc import Boundary, Solver
from hbcalc.validate import EXCEL_LABELS, SIM_POS_VOLTRA_ORDER, load_cases, run_case, parse_sheet
from hbcalc.voltra_report import read_report
import test_walls_def as T

R = ROOT + "/"; OUT = R + "results/accuracy/time_distance_detail.xlsx"; CACHE = R + "results/_time_distance_cache.pkl"
POSX = list(EXCEL_LABELS); SIMP = list(SIM_POS_VOLTRA_ORDER); POSM = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
H = 744
def bc_of(ti): return Boundary.from_voltra_files(R + "wall_TO.FTE", R + ti, R + "Wall_HO.FHT", R + "Wall_HI.FHT", R + "Wall_IG2.FSG", R + "Wall_ID2.FSD", solar_shift_h=0.0)
bcH, bcN = bc_of("Wall_TI.FTE"), bc_of("Wall_TI_2.FTE")

def collect():
    data = {}   # (조건, 케이스) -> dict(V, M, 열위치, 노드위치, 출처)
    for cond, bc, xlsx in (("난방", bcH, "voltra_30cases_night420_통합_v2.xlsx"), ("비난방", bcN, "Non_heating_Tsi.xlsx")):
        xl, names = load_cases(R + xlsx)
        for nm in names:
            s, m = run_case(nm, bc, mesh="voltra", positions=sorted(set(POSX) | set(SIMP)))
            df = xl.parse(nm)
            V = np.array([df[f"{p}mm 표면온도(℃)"].values[:H] for p in POSX]).T
            M = np.array([s.series(f"surf_{p}")[:H] for p in SIMP]).T
            data[(cond, nm)] = dict(V=V, M=M, col=POSX, node=SIMP, src="TRY3 손격자(VOLTRA 격자), 엑셀")
            print(cond, nm, flush=True)
    tw = pickle.load(open(R + "voltra_results/test_walls/_cmp.pkl", "rb")); vr = pickle.load(open(R + "voltra_results/_cmp.pkl", "rb"))
    for k in tw:
        j = tw[k]["junction"]; data[("난방", k)] = dict(V=j["Vs"][:H], M=j["Ms"][:H], col=POSM, node=POSM, src="추가 벽체, 자동 격자, VOLTRA 리포트")
    for k in ("A_EPS100", "B_EPS250", "C_partition250", "D_strip20"):
        d = vr[(k, 0.0)]; data[("비난방", k)] = dict(V=d["Vs"][:H], M=d["Ms"][:H], col=POSM, node=POSM, src="변형 벽체, 자동 격자, VOLTRA 리포트")
    for key in T.WALLS:
        f = R + f"voltra_results/test_walls_nonheating/{key}_junction.txt"
        if not os.path.exists(f): continue
        _, a = read_report(f); V = a[:H, 7:][:, 0::2]
        s = Solver(T.build(key)["mj"]); s.run(bcN, nsteps=H); M = np.array([s.series(f"surf_{p}")[:H] for p in POSM]).T
        data[("비난방", key)] = dict(V=V, M=M, col=POSM, node=POSM, src="추가 벽체, 자동 격자, VOLTRA 리포트")
    return data
if os.path.exists(CACHE): data = pickle.load(open(CACHE, "rb"))
else: data = collect(); pickle.dump(data, open(CACHE, "wb"))

per = pd.read_csv(R + "results/accuracy/per_case.csv", encoding="utf-8-sig").set_index(["조건", "케이스"])["열교"].to_dict()
def strength(cond, nm, V):
    if (cond, nm) in per: return per[(cond, nm)]
    return "-"
t = np.arange(H); hod = t % 24; day = t // 24 + 1
rows, summ = [], []
for (cond, nm), d in data.items():
    V, M = d["V"], d["M"]; E = M - V; g = strength(cond, nm, V)
    for h in range(24):
        sel = hod == h
        ev, mv, vv = E[sel].mean(0), np.abs(E[sel]).mean(0), V[sel].mean(0); mm = M[sel].mean(0)
        for i in range(12):
            rows.append((cond, nm, g, h, d["col"][i], d["node"][i], vv[i], mm[i], ev[i], mv[i]))
L = pd.DataFrame(rows, columns=["조건", "케이스", "열교강도", "시각", "열 이름 위치(mm)", "모사 노드 위치(mm)", "VOLTRA 평균", "모사 평균", "오차 평균(모사-VOLTRA)", "평균절대오차"])
# 강도별 시간대 x 거리 (TRY3 69케이스만: 열교강도 분류가 있는 케이스)
T3 = L[L.열교강도 != "-"]
def piv(col):
    out = []
    for (cond, g), q in T3.groupby(["조건", "열교강도"]):
        p = q.groupby(["시각", "열 이름 위치(mm)"])[col].mean().unstack()
        p.columns = [f"{c} mm" for c in p.columns]; p.insert(0, "열교강도", g); p.insert(0, "조건", cond); out.append(p.reset_index())
    return pd.concat(out)
PB, PM = piv("오차 평균(모사-VOLTRA)"), piv("평균절대오차")
# 시간별 요약: 시각별 전 위치 평균 MAE (조건 x 강도), 야간(18~07) 표시
S = L.groupby(["조건", "열교강도", "시각"]).agg(MAE=("평균절대오차", "mean"), 편향=("오차 평균(모사-VOLTRA)", "mean")).reset_index()
S["야간(18~07시)"] = np.where((S.시각 >= 18) | (S.시각 <= 7), "야간", "주간")
cases = pd.DataFrame([dict(조건=c, 케이스=n, 열교강도=strength(c, n, d["V"]), 출처=d["src"], 열이름위치=",".join(str(x) for x in d["col"]), 노드위치=",".join(str(x) for x in d["node"])) for (c, n), d in data.items()])
# 상세 시트: 선택 케이스
SEL = [("난방", "결로 방지 단열재 0.0001"), ("난방", "결로 방지 단열재 0.014"), ("난방", "단열재 조정 0.045"), ("난방", "결로 방지 단열재 10"), ("난방", "N6_noblock_insstrip"),
       ("비난방", "결로 방지 단열재 0.0001"), ("비난방", "결로 방지 단열재 10"), ("비난방", "N1_light")]
SEL = [s for s in SEL if s in data]
def detail(cond, nm):
    d = data[(cond, nm)]; bc = bcH if cond == "난방" else bcN
    base = pd.DataFrame({"시간(h)": t, "일": day, "시": hod, "실외온도": bc.T_out[:H], "실내온도": bc.T_in[:H]})
    cols = {}
    for i in range(12): cols[f"VOLTRA {d['col'][i]}mm"] = d["V"][:, i]
    for i in range(12): cols[f"모사 {d['col'][i]}mm"] = d["M"][:, i]
    for i in range(12): cols[f"오차 {d['col'][i]}mm"] = d["M"][:, i] - d["V"][:, i]
    return pd.concat([base, pd.DataFrame(cols)], axis=1)
README = pd.DataFrame({"항목": ["내용", "대상", "시각", "야간", "오차", "열 이름 위치와 노드 위치", "시트", "한계"], "설명": [
    "접합부 표면온도(12개 위치)가 하루 중 시각에 따라 어떻게 달라지는지, VOLTRA와 모사(hbcalc 현재 기본 설정 conv_lump='auto', 일사 보정 0)를 비교한 자료",
    f"{len(data)}케이스: TRY3 난방 39 + 비난방 30(손격자), 추가 벽체 난방 N1 ~ N6, 비난방 N1 ~ N6와 A ~ D(자동 격자). 비난방 N5 접합부는 VOLTRA 리포트가 아직 없어 제외",
    "시각 0 ~ 23은 하루 중 시(0 = 0시). 31일(744시간)의 같은 시각을 평균. 일사가 있는 낮(대략 8 ~ 17시)도 포함",
    "18 ~ 07시. 정확도 집계(accuracy_detail.xlsx)는 야간만 쓴다",
    "모사 - VOLTRA (도). 양수면 모사가 더 따뜻함. 평균절대오차는 시각별로 평균한 |오차|가 아니라 31일 각 시각의 |오차|를 평균한 값",
    "TRY3 엑셀의 열 이름은 0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500 mm이고 모사 노드 위치는 0, 5, 10, 20, 30, 50 ... (10 ~ 40 mm 열 = 노드 5 ~ 30 mm, 8.4). 추가 벽체는 열 이름과 노드 위치가 같다(0, 5, 10, 20, 30, 50 ...)",
    "읽는 법, 케이스목록, 시간대별_거리별(모든 케이스, 긴 형식: 필터와 피벗으로 사용), 강도별_시간대_편향, 강도별_시간대_MAE(열교 강도별 시각 x 거리 행렬), 시간별_요약(시각별 전 위치 평균), 상세 시트 8개(선택 케이스의 744시간 전체 VOLTRA, 모사, 오차)",
    "열교 강도 분류는 TRY3 69케이스만 있음(추가 벽체는 '-'). 모든 케이스를 개별 시트로 만들면 85개가 되어 긴 형식 한 시트로 합침"]})
with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    README.to_excel(w, sheet_name="읽는 법", index=False); cases.to_excel(w, sheet_name="케이스목록", index=False)
    L.to_excel(w, sheet_name="시간대별_거리별", index=False); PB.to_excel(w, sheet_name="강도별_시간대_편향", index=False); PM.to_excel(w, sheet_name="강도별_시간대_MAE", index=False)
    S.to_excel(w, sheet_name="시간별_요약", index=False)
    for c, n in SEL:
        nm = ("상세_" + ("난" if c == "난방" else "비") + "_" + n.replace("결로 방지 단열재 ", "결로").replace("단열재 조정 ", "조정").replace("_noblock_insstrip", "").replace("_light", ""))[:31]
        detail(c, n).to_excel(w, sheet_name=nm, index=False)
wb = load_workbook(OUT); HF = PatternFill("solid", fgColor="DCE6F5"); RED = PatternFill("solid", fgColor="F8D7DA")
for ws in wb.worksheets:
    for c in ws[1]: c.font = Font(bold=True); c.fill = HF; c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "A2" if ws.title in ("읽는 법", "케이스목록") else ("E2" if ws.title.startswith("상세") else "D2")
    for i, col in enumerate(ws.columns, 1):
        mx = max((len(str(c.value)) if c.value is not None else 0) for c in col[:80]); ws.column_dimensions[get_column_letter(i)].width = min(max(9, mx * 1.1), 110 if ws.title == "읽는 법" else 40)
        if ws.title != "읽는 법":
            for c in col[1:]:
                if isinstance(c.value, float): c.number_format = "0.000"
    if ws.title == "읽는 법":
        for row in ws.iter_rows(min_row=2):
            for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
    if ws.title in ("강도별_시간대_편향",):
        rng = f"D2:{get_column_letter(ws.max_column)}{ws.max_row}"; ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0.05"], fill=RED)); ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["-0.05"], fill=RED))
    if ws.title in ("강도별_시간대_MAE",):
        rng = f"D2:{get_column_letter(ws.max_column)}{ws.max_row}"; ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0.05"], fill=RED))
    if ws.title == "시간대별_거리별":
        ws.auto_filter.ref = ws.dimensions
wb.save(OUT)
print("saved", OUT, [ws.title for ws in wb.worksheets]); print("행", len(L))
# 요약 출력: 시각별 평균 MAE (전체), 주간 대 야간
q = S.groupby("야간(18~07시)").MAE.mean(); print("TRY3+추가 전체 평균 MAE: ", q.round(4).to_dict())
for cond in ("난방", "비난방"):
    x = S[S.조건 == cond].groupby("시각").MAE.mean().round(3); print(cond, "시각별 MAE:", x.to_dict())
