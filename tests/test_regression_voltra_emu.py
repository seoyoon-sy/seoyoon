"""새 솔버가 기존 voltra_emu.py 와 같은 격자에서 같은 결과를 내는지 확인 (솔버 이식 검증)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Solver, Boundary
from hbcalc.compat_voltra import model_from_vtr
from voltra_emu import VoltraEmu

BC = dict(T01=f"{ROOT}/wall_TO.FTE", T02=f"{ROOT}/Wall_TI.FTE", G01=f"{ROOT}/Wall_IG2.FSG",
          D01=f"{ROOT}/Wall_ID2.FSD", H01=f"{ROOT}/Wall_HO.FHT", H02=f"{ROOT}/Wall_HI.FHT")


def test_same_as_old(n=200, shift=0.0):
    vtr = f"{ROOT}/TRY3_Plus_QTest2.vtr"
    old = VoltraEmu(vtr, lamb={"EPS": 0.045, "XPS": 0.3})
    Told = old.run(BC, solar_shift=shift, nsteps=n)
    m = model_from_vtr(vtr).with_properties(lam={"EPS": 0.045, "XPS": 0.3})
    bc = Boundary.from_voltra_files(BC["T01"], BC["T02"], BC["H01"], BC["H02"], BC["G01"], BC["D01"],
                                    solar_shift_h=shift)
    s = Solver(m, conv_lump="half")
    Tnew = s.run(bc, nsteps=n)
    assert Tnew.shape == Told.shape
    err = np.abs(Tnew - Told).max()
    print("max |new-old| =", err)
    assert err < 1e-6


if __name__ == "__main__":
    test_same_as_old(); test_same_as_old(shift=-0.25)
