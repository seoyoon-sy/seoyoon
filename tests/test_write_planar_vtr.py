"""평면 벽체 .vtr 생성기 시험: (1) 템플릿을 같은 칸 수로 재생성하면 파일이 같다 (2) 새 벽체를 읽어 들인 모델이 hbcalc 평면 모델과 같은 결과
(3) 시험 벽체 접합부 .vtr 가 물성(열전도율, 밀도, 비열)과 난방 실내온도 경계를 정확히 담는다."""
import sys, os, tempfile, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Boundary, Solver
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.model import Material
from hbcalc.planar import planar_model
from hbcalc.presets import TRY3_MATERIALS
from hbcalc.write_vtr import write_planar_vtr

TPL = f"{ROOT}/voltra_inputs/TRY3_Plus_2D.vtr"
L = [("concrete", 200), ("EPS", 180), ("0.5B cement block", 90), ("gypsum", 10)]


def _text(p):
    b = open(p, "rb").read()
    try: return b.decode("utf-8")
    except UnicodeDecodeError: return b.decode("cp949")


def test_regenerate_template_identical():
    mats = {n: TRY3_MATERIALS[n] for n, _ in L}
    with tempfile.TemporaryDirectory() as d:
        write_planar_vtr(L, mats, TPL, f"{d}/p.vtr", cells=[10, 9, 2, 1])
        assert _text(f"{d}/p.vtr") == _text(TPL)


def test_new_wall_matches_hbcalc_planar():
    L2 = [("concrete", 250), ("EPS", 220), ("0.5B cement block", 100), ("gypsum", 15)]
    mats = {"concrete": Material("concrete", 2.3, 2400, 880), "EPS": Material("EPS", 0.03, 30, 1500),
            "0.5B cement block": Material("0.5B cement block", 0.9, 1900, 840), "gypsum": Material("gypsum", 0.21, 700, 1100)}
    bc = Boundary.from_voltra_files(*(f"{ROOT}/{f}" for f in ("wall_TO.FTE", "Wall_TI.FTE", "Wall_HO.FHT", "Wall_HI.FHT", "Wall_IG2.FSG", "Wall_ID2.FSD")))
    with tempfile.TemporaryDirectory() as d:
        _, cells = write_planar_vtr(L2, mats, TPL, f"{d}/p.vtr", indoor_bc="heating")
        back = model_from_vtr(f"{d}/p.vtr")
        txt = _text(f"{d}/p.vtr")
    for m in back.materials.values():                       # 물성이 그대로 들어갔다
        w = mats[m.name]; assert (m.lam, m.rho, m.cp) == (w.lam, w.rho, w.cp)
    assert "Wall_TI.FTE" in txt and "Wall_TI_2.FTE" not in txt and "funp= 20 2 1440 360 0 0" in txt
    ref = planar_model(L2, mats, cells=cells, width=2.2, nx=44)
    from dataclasses import replace as _rep                  # 템플릿 x 격자의 첫 칸은 폭 0 이고 어떤 블록에도 속하지 않음: 떼고 푼다
    back = _rep(back, xe=back.xe[1:], cid=back.cid[1:], outputs={k: (i - 1, j) for k, (i, j) in back.outputs.items()})
    a = Solver(back, conv_lump="half"); a.run(bc, nsteps=48)
    b = Solver(ref, conv_lump="half"); b.run(bc, nsteps=48)
    for va, vb in ((a.series("onod0"), b.series("surf")), (a.series("onod1"), b.series("back"))):
        assert np.abs(va - vb).max() < 1e-6, np.abs(va - vb).max()


def test_test_wall_junction_files():
    files = sorted(glob.glob(f"{ROOT}/voltra_inputs/test_walls/*_junction.vtr"))
    assert len(files) >= 4
    for f in files:
        txt = _text(f); m = model_from_vtr(f)
        assert "Wall_TI.FTE" in txt and "Wall_TI_2.FTE" not in txt
        assert len(m.outputs) == 24 and len(m.materials) == 5


if __name__ == "__main__":
    test_regenerate_template_identical(); test_new_wall_matches_hbcalc_planar(); test_test_wall_junction_files(); print("통과")
