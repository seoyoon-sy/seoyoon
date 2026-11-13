"""상세 정확도 엑셀: results/accuracy/accuracy_detail.xlsx (현재 기본 설정 conv_lump='auto', 일사 보정 0, VOLTRA 격자).
입력: results/accuracy/raw.pkl(accuracy_report.py), per_case.csv, per_case_before_auto.csv(auto 도입 전 low 기준),
voltra_results/test_walls/_cmp.pkl, voltra_results/_cmp.pkl(추가 벽체). 사용: python examples/export_accuracy_excel.py"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter
from hbcalc.validate import EXCEL_LABELS, night_mask

R = ROOT + "/"; OUT = R + "results/accuracy/accuracy_detail.xlsx"; mk = night_mask(744)
POS = list(EXCEL_LABELS); PN = [f"{p} mm" for p in POS]
raw = pickle.load(open(R + "results/accuracy/raw.pkl", "rb"))
new = pd.read_csv(R + "results/accuracy/per_case.csv", encoding="utf-8-sig"); old = pd.read_csv(R + "results/accuracy/per_case_before_auto.csv", encoding="utf-8-sig")
j = new.merge(old[["조건", "케이스", "MAE", "RMSE", "MAE_0mm", "편향_0mm", "최대절대", "psi_환산오차율%"]], on=["조건", "케이스"], suffixes=("", "_이전"))
j["MAE_변화"] = j.MAE - j.MAE_이전
# ---- 케이스 x 위치 행렬
rows_b, rows_m, rows_v, rows_h = [], [], [], []
for (cond, nm), v in raw.items():
    d, S = v["d"], v["S"]; g = new[(new.조건 == cond) & (new.케이스 == nm)].열교.iloc[0]
    base = dict(조건=cond, 케이스=nm, 열교=g)
    rows_b.append({**base, **dict(zip(PN, d.mean(0)))}); rows_m.append({**base, **dict(zip(PN, np.abs(d).mean(0)))})
    rows_v.append({**base, **{f"VOLTRA {p}": x for p, x in zip(PN, S.mean(0))}, **{f"모사 {p}": x for p, x in zip(PN, (S + d).mean(0))}})
B = pd.DataFrame(rows_b); M = pd.DataFrame(rows_m); T = pd.DataFrame(rows_v)
def grp(df): return df.groupby(["조건", "열교"])[PN].mean().reset_index()
GB, GM = grp(B), grp(M)
# ---- 요약
def summ(a, b):
    f = lambda x: x.mean()
    return dict(케이스수=len(a), 평균절대오차_이전=b.MAE_이전.mean(), 평균절대오차=a.MAE.mean(), RMSE_이전=b.RMSE_이전.mean(), RMSE=a.RMSE.mean(),
                MAE_0mm_이전=b.MAE_0mm_이전.mean(), MAE_0mm=a.MAE_0mm.mean(), 편향_0mm_이전=b.편향_0mm_이전.mean(), 편향_0mm=a.편향_0mm.mean(),
                최대절대오차_최댓값=a.최대절대.max(), MAE_0_05_이하_개수=int((a.MAE <= 0.05).sum()), dTs오차_평균절대=a.dTs_오차.abs().mean(),
                Psi환산오차율_평균=a["psi_환산오차율%"].abs().mean(), Psi환산오차율_최대=a["psi_환산오차율%"].abs().max())
S_rows = []
for cond in ("난방", "비난방", "전체"):
    q = j if cond == "전체" else j[j.조건 == cond]
    for g in ("전체", "약함(ΔTs<1)", "중간(1~3)", "강함(≥3)"):
        qq = q if g == "전체" else q[q.열교 == g]
        if len(qq): S_rows.append({"조건": cond, "열교 강도": g, **summ(qq, qq)})
SUM = pd.DataFrame(S_rows)
# ---- 추가 벽체
tw = pickle.load(open(R + "voltra_results/test_walls/_cmp.pkl", "rb")); vr = pickle.load(open(R + "voltra_results/_cmp.pkl", "rb"))
DESC = {"N1_light": "N1 경량, EPS 120", "N2_heavy": "N2 중량, EPS 220", "N3_thin": "N3 얇은 단열 EPS 80", "N4_weak": "N4 약한 열교 (판 0.002)", "N5_noblock": "N5 블록 없음 (판 0.5)",
        "N6_noblock_insstrip": "N6 블록 없음 (판 0.04)", "A_EPS100": "A EPS 100", "B_EPS250": "B EPS 250", "C_partition250": "C 내벽 250", "D_strip20": "D 판 20, 돌출 120"}
W = []
for k in tw: W.append((k, "난방 20 ℃", tw[k]["junction"]["Vs"], tw[k]["junction"]["Ms"], tw[k]["junction"]["Vb"], tw[k]["junction"]["Mb"], tw[k]["planar"]))
for k in ("A_EPS100", "B_EPS250", "C_partition250", "D_strip20"): W.append((k, "비난방", vr[(k, 0.0)]["Vs"], vr[(k, 0.0)]["Ms"], vr[(k, 0.0)]["Vb"], vr[(k, 0.0)]["Mb"], None))
EW, EB, EP = [], [], []
for k, cond, Vs, Ms, Vb, Mb, pl in W:
    e = (Ms - Vs)[mk]; ew = (Mb - Vb)[mk]
    r = {"벽체": k, "설명": DESC[k], "조건": cond, "접합부 표면 평균절대오차": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "0 mm 평균절대오차": np.abs(e[:, 0]).mean(), "0 mm 편향": e[:, 0].mean(),
         "벽체쪽 평균절대오차": np.abs(ew).mean(), "최대절대오차": np.abs(e).max()}
    if pl is not None:
        r["평면 표면 평균절대오차"] = np.abs(pl["Ms"] - pl["Vs"])[mk].mean()
        dv = pl["Vs"][mk].mean() - Vs[mk, 0].mean(); dm = pl["Ms"][mk].mean() - Ms[mk, 0].mean(); r.update({"ΔTs VOLTRA": dv, "ΔTs 모사": dm, "ΔTs 오차": dm - dv})
    EW.append(r)
    EB.append({"벽체": k, "조건": cond, **dict(zip(PN, e.mean(0)))}); EP.append({"벽체": k, "조건": cond, **dict(zip(PN, np.abs(e).mean(0)))})
EW, EB, EP = pd.DataFrame(EW), pd.DataFrame(EB), pd.DataFrame(EP)
# ---- N6 진단: 정상상태 대류 배분별 편차
import scan_strip_face_rules as SS
diag = []
LAM = {"N1": 0.3, "N2": 0.8, "N3": 1.0, "N4": 0.002, "N5": 0.5, "N6": 0.04, "N6b": 0.5, "N5b": 0.04}
for k, (m, V) in SS.W.items():
    c = SS.Case(m, V); ws = np.round(np.arange(0.3, 1.001, 0.05), 2)
    rm = {w: np.sqrt((c.dev(w, w, w, w) ** 2).mean()) for w in ws}; wb = min(rm, key=rm.get)
    dh, dl = c.dev(0.5, 0.5, 0.5, 0.5), c.dev(1, 1, 1, 1)
    from hbcalc.lump_rule import strip_weight
    wa = strip_weight(LAM[k]); da = c.dev(wa, wa, wa, wa)
    diag.append({"벽체": k, "XPS 판 열전도율": LAM[k], "블록 있음": "아니오" if k in ("N5", "N6", "N5b", "N6b") else "예", "auto가 고른 w": wa,
                 "half 0 mm 편차": dh[0], "low 0 mm 편차": dl[0], "auto 0 mm 편차": da[0], "half RMSE": np.sqrt((dh ** 2).mean()), "low RMSE": np.sqrt((dl ** 2).mean()),
                 "auto RMSE": np.sqrt((da ** 2).mean()), "최적 w": wb, "최적 w RMSE": rm[wb]})
DIAG = pd.DataFrame(diag)
# ---- 쓰기
README = pd.DataFrame({"항목": [
    "설정", "비교 대상", "야간", "오차 정의", "평균절대오차(MAE)", "0 mm 편향", "열교 강도", "ΔTs", "Ψ 환산", "2D/3D", "이전/이후", "한계", "시트 안내"],
    "내용": [
    "hbcalc 기본값: conv_lump='auto'(결로 방지 단열재 열전도율에 따라 대류 배분 선택), 일사 시각 보정 0, VOLTRA와 같은 격자",
    "난방(실내 20 ℃ 고정) 39케이스 voltra_30cases_night420_통합_v2.xlsx, 비난방(실내온도 변동 Wall_TI_2.FTE) 30케이스 Non_heating_Tsi.xlsx, 추가 벽체 10종",
    "매일 18시 ~ 다음날 07시(14시간 x 30일 = 420시간), 744시간 중 야간만 사용",
    "모사 - VOLTRA (℃). 양수면 모사가 더 따뜻하다",
    "시간별 오차의 절대값을 12개 위치와 420시간에 대해 평균",
    "0 mm(접합부 면) 위치의 야간 시간별 평균 오차(부호 있음)",
    "VOLTRA ΔTs 기준: 약함(1 미만), 중간(1 ~ 3), 강함(3 이상)",
    "비열교부(평면 벽체 모델 야간 평균) - 접합부 0 mm 야간 평균 표면온도",
    "정답 Ψ(정상상태_정답.xlsx, TRISCO)와 VOLTRA ΔTs의 3차 다항 관계로 모사 ΔTs 오차를 Ψ 오차로 환산한 대리 계산. 학습모델 입력이 ΔTs만이라는 가정",
    "hbcalc는 2D 전도(x, y). z(깊이) 방향 전도는 계산하지 않고, 복사 형상계수에만 깊이(1000 mm)와 중간 높이를 반영. 3D 시험(steady3d_test.py)에서 z 전도의 영향은 작았음(CLAUDE.md 8.21)",
    "'이전'은 대류 배분 low 고정(2026-11-13 이전), '이후'는 auto. 규칙은 TRY3 난방 케이스에서 맞춘 것이라 난방 개선분은 표본 내 값이다",
    "블록이 없고 판 열전도율 0.04인 N6, N5b는 규칙이 맞지 않음(최적 w 0.8 ~ 0.85, auto는 1.0). 판이 단열재급인 약한 열교의 0 mm는 아직 +0.07 ~ +0.09",
    "요약, 케이스별, 위치별_편향, 위치별_MAE, 강도별_위치편향, 강도별_위치MAE, 야간평균표면온도, 추가벽체, 추가벽체_위치편향, 추가벽체_위치MAE, N6진단"]})
with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    README.to_excel(w, sheet_name="읽는 법", index=False); SUM.to_excel(w, sheet_name="요약", index=False)
    j[["조건", "케이스", "열교", "VOLTRA_dTs", "MAE_이전", "MAE", "MAE_변화", "RMSE_이전", "RMSE", "MAE_0mm_이전", "MAE_0mm", "편향_0mm_이전", "편향_0mm", "최대절대_이전", "최대절대", "dTs_오차",
       "psi_정답", "psi_환산오차", "psi_환산오차율%_이전", "psi_환산오차율%"]].to_excel(w, sheet_name="케이스별", index=False)
    B.to_excel(w, sheet_name="위치별_편향", index=False); M.to_excel(w, sheet_name="위치별_MAE", index=False); GB.to_excel(w, sheet_name="강도별_위치편향", index=False); GM.to_excel(w, sheet_name="강도별_위치MAE", index=False)
    T.to_excel(w, sheet_name="야간평균표면온도", index=False); EW.to_excel(w, sheet_name="추가벽체", index=False); EB.to_excel(w, sheet_name="추가벽체_위치편향", index=False); EP.to_excel(w, sheet_name="추가벽체_위치MAE", index=False); DIAG.to_excel(w, sheet_name="N6진단", index=False)
wb = load_workbook(OUT); HF = PatternFill("solid", fgColor="DCE6F5"); RED = PatternFill("solid", fgColor="F8D7DA"); GRN = PatternFill("solid", fgColor="D7EFDC")
for ws in wb.worksheets:
    for c in ws[1]: c.font = Font(bold=True); c.fill = HF; c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "C2" if ws.title not in ("읽는 법",) else "A2"
    for i, col in enumerate(ws.columns, 1):
        mx = max((len(str(c.value)) if c.value is not None else 0) for c in col[:60]); ws.column_dimensions[get_column_letter(i)].width = min(max(10, mx * (1.15 if ws.title != "읽는 법" else 1.0)), 60 if ws.title != "읽는 법" else 130)
        if ws.title != "읽는 법":
            for c in col[1:]:
                if isinstance(c.value, float): c.number_format = "0.000"
    if ws.title == "읽는 법":
        for row in ws.iter_rows(min_row=2):
            for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
    hdr = [c.value for c in ws[1]]
    if ws.title in ("위치별_편향", "추가벽체_위치편향", "강도별_위치편향"):
        rng = f"{get_column_letter(4 if ws.title != '추가벽체_위치편향' else 3)}2:{get_column_letter(ws.max_column)}{ws.max_row}"
        ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0.05"], fill=RED)); ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["-0.05"], fill=RED))
    if ws.title in ("위치별_MAE", "추가벽체_위치MAE", "강도별_위치MAE"):
        rng = f"{get_column_letter(4 if ws.title != '추가벽체_위치MAE' else 3)}2:{get_column_letter(ws.max_column)}{ws.max_row}"
        ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0.05"], fill=RED))
    if ws.title == "케이스별":
        for name in ("MAE", "RMSE"):
            ci = hdr.index(name) + 1; L = get_column_letter(ci); ws.conditional_formatting.add(f"{L}2:{L}{ws.max_row}", CellIsRule(operator="greaterThan", formula=["0.05"], fill=RED))
        ci = hdr.index("MAE_변화") + 1; L = get_column_letter(ci); ws.conditional_formatting.add(f"{L}2:{L}{ws.max_row}", CellIsRule(operator="lessThan", formula=["-0.005"], fill=GRN))
wb.save(OUT); print("saved", OUT, [ws.title for ws in wb.worksheets])
