"""3D 솔버 단위 시험 (CLAUDE.md 8.77): 형상계수 조각 합, 대칭 접기, 에너지 수지, 균일 온도."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hbcalc import Boundary
from hbcalc.solver3d import Solver3D, strip_pair_z
from hbcalc.radiation import exchange_area
from hbcalc.presets import try3_voltra_mesh_model


def test_view_factor_slices_sum_to_depth_average():
    for args in [(0, 0.05, 0, 0.01, 0.2, 0), (0.1, 0.3, 0, 0.02, 0.5, 0)]:
        tot = strip_pair_z(*args, np.linspace(0, 1, 7)).sum()
        assert abs(tot / exchange_area(*args, 1.0, "avg") - 1) < 1e-6


def _bc(n=12):
    t = np.arange(n)
    return Boundary(T_out=-5 + 3 * np.sin(t / 3), T_in=np.full(n, 20.0), h_out=15 + 5 * np.cos(t / 2), h_in=np.full(n, 1.4),
                    G=np.where((t % 24 > 8) & (t % 24 < 16), 300.0, 0.0), D=np.where((t % 24 > 8) & (t % 24 < 16), 100.0, 0.0),
                    diffuse_obstruction=True)


def test_fold_equals_full_and_energy_balance():
    m = try3_voltra_mesh_model({"EPS": 0.032, "XPS": 1.6})
    bc = _bc()
    a = Solver3D(m, nz=4); b = Solver3D(m, nz=4, fold=False)
    a.run(bc); b.run(bc)
    for n in m.outputs:
        assert abs(a.series(n) - b.series(n)).max() < 1e-9
    T = b.T_all; E = (b.cap * T).sum(1)
    q = np.array([(b.Krad @ T[k] + b._Hk(b.hT, k) @ T[k] - b._rhs(b.hT, b.Isol, k)).sum() for k in range(T.shape[0])])
    lhs = np.diff(E) / bc.dt; rhs = -0.5 * (q[1:] + q[:-1])
    assert abs(lhs - rhs).max() <= 1e-6 * abs(rhs).max()


def test_uniform_temperature_stays_uniform():
    m = try3_voltra_mesh_model({"EPS": 0.032, "XPS": 1.6})
    n = 3
    bc = Boundary(T_out=np.full(n, 7.0), T_in=np.full(n, 7.0), h_out=np.full(n, 15.0), h_in=np.full(n, 1.4))
    s = Solver3D(m, nz=4); s.run(bc)
    assert abs(s.T_all - 7.0).max() < 1e-9
