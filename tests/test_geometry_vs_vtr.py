"""코드로 만든 형상(VOLTRA 격자 지정)이 .vtr 를 읽은 모델과 같은 셀 배치, 같은 계산 결과를 주는지 확인."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Solver, Boundary
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.presets import try3_voltra_mesh_model

LAM = {"EPS": 0.045, "XPS": 0.3}


def test_cells_identical():
    a = model_from_vtr(f"{ROOT}/TRY3_Plus_QTest2.vtr")
    b = try3_voltra_mesh_model()
    assert a.shape == b.shape
    assert np.abs(a.xe - b.xe).max() < 1e-6 and np.abs(a.ye - b.ye).max() < 1e-6
    # 재료 번호 체계가 다르므로 재료 이름과 경계종류로 비교
    name = lambda m: np.vectorize(lambda c: m.materials[c].name if c > 0 else m.zones[-c].kind)(m.cid)
    na, nb = name(a), name(b)
    # 경계 셀: 왼쪽/오른쪽 방은 둘 다 indoor 이므로 종류만 비교
    diff = (na != nb)
    print("다른 셀 수:", diff.sum())
    assert diff.sum() == 0
    assert a.outputs  # 출력 위치: VOLTRA onod 의 (ix,iy) 와 builder 의 surf_/back_ 이 같은 격자점인지
    on = {v for v in a.outputs.values()}
    mine = {b.outputs[f"surf_{o}"] for o in (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)} | \
           {b.outputs[f"back_{o}"] for o in (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)}
    assert on == mine, (sorted(on ^ mine))


def test_results_identical(n=100):
    bc = Boundary.from_voltra_files(f"{ROOT}/wall_TO.FTE", f"{ROOT}/Wall_TI.FTE", f"{ROOT}/Wall_HO.FHT",
                                    f"{ROOT}/Wall_HI.FHT", f"{ROOT}/Wall_IG2.FSG", f"{ROOT}/Wall_ID2.FSD")
    a = model_from_vtr(f"{ROOT}/TRY3_Plus_QTest2.vtr").with_properties(lam=LAM)
    b = try3_voltra_mesh_model(LAM)
    sa, sb = Solver(a), Solver(b)
    sa.run(bc, nsteps=n); sb.run(bc, nsteps=n)
    err = 0
    for o in (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500):
        err = max(err, np.abs(sa.node_series(*a.outputs[[k for k, v in a.outputs.items() if v == b.outputs[f"surf_{o}"]][0]]) - sb.series(f"surf_{o}")).max())
    print("최대 차이 ℃:", err)
    assert err < 1e-3


if __name__ == "__main__":
    test_cells_identical(); test_results_identical()
