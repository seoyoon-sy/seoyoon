"""변형 벽체(.vtr) 생성 + 같은 조건의 hbcalc 야간 평균 표면온도(기대값) 표.
사용: python examples/make_variant_vtr.py [출력폴더]
VOLTRA 에서 생성된 .vtr 를 열어 돌린 결과 엑셀을 expected_hbcalc.csv 와 비교하면 된다.
(.vtr 안의 경계조건 파일 경로는 템플릿 그대로이므로 사용자 PC 경로에 맞춰야 한다.)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Layer, Boundary, Solver, build_tjunction
from hbcalc.presets import try3_spec
from hbcalc.validate import night_mask
from hbcalc.write_vtr import write_vtr

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "vtr_cases")
os.makedirs(OUT, exist_ok=True)
TPL = os.path.join(ROOT, "TRY3_Plus_QTest2.vtr")
L = lambda eps=180, con=200, cb=90: [Layer("concrete", con), Layer("EPS", eps), Layer("0.5B cement block", cb), Layer("gypsum", 10)]
CASES = {                                    # 이름: (설명, spec 변경)
    "A_EPS100": ("EPS 두께 100 mm", dict(layers=L(100))),
    "B_EPS250": ("EPS 두께 250 mm", dict(layers=L(250))),
    "C_partition250": ("내벽 두께 250 mm", dict(partition_mm=250)),
    "D_strip20": ("열교 차단 띠 두께 20 mm, 돌출 120 mm", dict(strip_mm=20, strip_extend_mm=120)),
}
POS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
BCS = {"비난방(Wall_TI_2.FTE, 템플릿 기본)": "Wall_TI_2.FTE", "난방 20도(Wall_TI.FTE)": "Wall_TI.FTE"}
rows = []
for key, (desc, kw) in CASES.items():
    spec = replace(try3_spec(snap_mm=1), **kw)
    model = build_tjunction(spec)                            # 재료 물성은 템플릿 기본값 (EPS 0.032, XPS 1.6)
    write_vtr(model, TPL, os.path.join(OUT, f"{key}.vtr"))
    s = Solver(model)
    for bname, ti in BCS.items():
        for sh in (0.0, -0.25):
            bc = Boundary.from_voltra_files(*(os.path.join(ROOT, f) for f in ("wall_TO.FTE", ti, "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")), solar_shift_h=sh)
            s.run(bc); mk = night_mask(len(bc.T_out))
            r = {"케이스": key, "설명": desc, "경계조건": bname, "일사보정(h)": sh, "격자": f"{model.shape[0]}x{model.shape[1]}"}
            for p in POS: r[f"표면_{p}mm"] = s.series(f"surf_{p}")[mk].mean()
            for p in POS: r[f"벽체쪽_{p}mm"] = s.series(f"back_{p}")[mk].mean()
            rows.append(r)
    print(key, model.shape, flush=True)
pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, "expected_hbcalc.csv"), index=False, encoding="utf-8-sig")
print("완료:", OUT)
