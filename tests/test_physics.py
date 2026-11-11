"""코드 자체의 물리 검증 (VOLTRA와 비교하기 전에 먼저 통과해야 함)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy import integrate
from hbcalc import Solver, Boundary
from hbcalc.model import Material, Zone, Model
from hbcalc.radiation import zint, exchange_area


def slab_model(layers, nsub=4, width=0.2, hrb=5.25, zt=1.0):
    """1D 평판: 아래(y 작음)=실외 경계, 위=실내 경계. layers=[(Material, 두께 m)]. x 방향 2칸(옆면 단열)."""
    ye = [0.0, 1.0]
    cid_rows = [-1]
    mats = {}
    for k, (m, d) in enumerate(layers, start=1):
        mats[k] = m
        for _ in range(nsub):
            ye.append(ye[-1] + d / nsub); cid_rows.append(k)
    ye.append(ye[-1] + 1.0); cid_rows.append(-2)
    xe = np.array([0.0, width / 2, width])
    cid = np.tile(np.array(cid_rows)[None, :], (2, 1))
    return Model(xe=xe, ye=np.array(ye), zt=zt, cid=cid, materials=mats,
                 zones={1: Zone("outdoor"), 2: Zone("indoor")}, hrb=hrb)


def test_1d_steady_state_U():
    L = [(Material("a", 1.6, 2200, 800, eps=0.9), 0.2), (Material("b", 0.032, 25, 1500, eps=0.9), 0.18),
         (Material("c", 0.6, 1400, 1000, eps=0.9), 0.09), (Material("d", 0.16, 900, 840, eps=0.8), 0.01)]
    m = slab_model(L)
    ho, hi, To, Ti = 17.0, 1.5, -8.0, 20.0
    s = Solver(m)
    s.run(Boundary.constant(3, To, Ti, ho, hi), nsteps=3)
    e_o, e_i = L[0][0].eps, L[-1][0].eps
    R = 1 / (ho + e_o * m.hrb) + sum(d / mt.lam for mt, d in L) + 1 / (hi + e_i * m.hrb)
    q = (Ti - To) / R                                       # W/m2
    Ts_in = Ti - q / (hi + e_i * m.hrb)
    Ts_out = To + q / (ho + e_o * m.hrb)
    iy_in, iy_out = m.shape[1] - 1, 1
    Tin_num = s.node_series(1, iy_in)[2]; Tout_num = s.node_series(1, iy_out)[2]
    print("U =", 1 / R, "Ts_in", Ts_in, Tin_num, "Ts_out", Ts_out, Tout_num)
    assert abs(Tin_num - Ts_in) < 1e-9 and abs(Tout_num - Ts_out) < 1e-9
    # 열류량: 실내 대류 + 복사 = q  (대류만 보면 hi*(Ts-Ti))
    Qc = s.zone_convective_flow("indoor")[2]
    assert abs(-Qc / (m.xe[-1] * m.zt) - hi * q / (hi + e_i * m.hrb)) < 1e-9


def test_lumped_capacity_decay():
    mt = Material("m", 200.0, 2.0e6 / 1000, 1000.0, eps=0.9)       # rho*c = 2e6
    L, hc, hrb = 0.02, 10.0, 5.25
    m = slab_model([(mt, L)], nsub=2, hrb=hrb)
    n, dt = 400, 30.0
    Ta = np.where(np.arange(n) >= 1, 10.0, 0.0)
    bc = Boundary(Ta, Ta, np.full(n, hc), np.full(n, hc), dt=dt)
    s = Solver(m); s.run(bc)
    h = hc + 0.9 * hrb
    tau = mt.rho * mt.cp * L / (2 * h)
    # Crank-Nicolson 은 시각별 입력을 선형으로 잇는다: Ta 는 t=0 에서 0, t=dt 에서 10 으로 선형 증가 후 일정.
    t = np.arange(n) * dt
    x = dt / tau
    ramp = 10 * (t / dt - (1 - np.exp(-t / tau)) / x)                       # 0 <= t <= dt (정확해)
    after = 10 * (1 - (np.exp(x) - 1) / x * np.exp(-t / tau))               # t >= dt
    exact = np.where(t <= dt, ramp, after)
    num = s.node_series(1, 2)                                      # 평판 중간(층 경계) 노드
    err = np.abs(num - exact).max()
    print("tau =", tau, "max err =", err)
    assert err < 1e-3 * 10


def test_energy_balance_tjunction():
    from hbcalc.compat_voltra import model_from_vtr
    m = model_from_vtr(f"{ROOT}/TRY3_Plus_QTest2.vtr")
    bc = Boundary.from_voltra_files(f"{ROOT}/wall_TO.FTE", f"{ROOT}/Wall_TI.FTE", f"{ROOT}/Wall_HO.FHT",
                                    f"{ROOT}/Wall_HI.FHT", f"{ROOT}/Wall_IG2.FSG", f"{ROOT}/Wall_ID2.FSD")
    s = Solver(m); s.run(bc, nsteps=120)
    Q = s.boundary_heat_flow()
    dE = (s.cap * (s.T_all[-1] - s.T_all[0])).sum()
    out = (0.5 * (Q[:-1] + Q[1:]) * bc.dt).sum()
    print("저장에너지 변화 J", dE, " 경계 유출 적분 J", out)
    assert abs(dE + out) < 1e-6 * max(abs(dE), abs(out), 1.0)


def test_z_integral_matches_quadrature():
    Lz, a = 0.7, 0.09
    f = lambda z1, z2: 1.0 / (a + (z1 - z2) ** 2) ** 2
    avg = integrate.dblquad(f, 0, Lz, 0, Lz)[0]
    mid = Lz * integrate.quad(lambda z2: f(Lz / 2, z2), 0, Lz)[0]
    assert abs(zint(Lz, a, "avg") - avg) < 1e-6 * avg
    assert abs(zint(Lz, a, "mid") - mid) < 1e-6 * mid


def test_view_factor_perpendicular_strips_long():
    # 모서리를 공유하는 직각 띠 (z 방향 아주 긴 경우): A_a F_ab = Lz (a + b - sqrt(a^2 + b^2)) / 2
    a, b, Lz = 0.5, 0.3, 2000.0
    ex = (a + b - np.hypot(a, b)) / 2 * Lz
    for zm in ("avg", "mid", "inf"):
        v = exchange_area(0, a, 0, 0, b, 0, Lz, zm, ngl=64)
        print(zm, v, ex, v / ex - 1)
        assert abs(v / ex - 1) < (1e-12 if zm == 'inf' else 2e-3)


def test_view_sum_le_one_tjunction():
    from hbcalc.compat_voltra import model_from_vtr
    s = Solver(model_from_vtr(f"{ROOT}/TRY3_Plus_QTest2.vtr"))
    assert s.view_sum.max() <= 1 + 1e-9


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("통과:", k)


def test_zone_flows_sum_equals_boundary_flow():
    """영역별 노드 열류의 합이 경계 전체 순열류와 같아야 한다 (열류 출력의 일관성)."""
    from hbcalc.compat_voltra import model_from_vtr
    m = model_from_vtr(f"{ROOT}/TRY3_Plus_QTest2.vtr")
    bc = Boundary.from_voltra_files(f"{ROOT}/wall_TO.FTE", f"{ROOT}/Wall_TI.FTE", f"{ROOT}/Wall_HO.FHT",
                                    f"{ROOT}/Wall_HI.FHT", f"{ROOT}/Wall_IG2.FSG", f"{ROOT}/Wall_ID2.FSD")
    s = Solver(m); s.run(bc, nsteps=60)
    tot = sum(q.sum(1) for q in s.zone_node_flows().values())
    assert np.abs(tot - s.boundary_heat_flow()).max() < 1e-6
    # 정상상태(첫 시각): 전 영역 순열류의 합 = 0 (실외로 나가는 열 = 실내에서 들어오는 열)
    F = s.zone_flows_into_out()
    net0 = sum(F[z][1][0] - F[z][0][0] for z in F)
    scale = max(abs(F[z][1][0] - F[z][0][0]) for z in F)
    assert abs(net0) < 1e-9 * max(scale, 1.0)


def test_planar_model_steady_U():
    from hbcalc.planar import planar_model
    from hbcalc.presets import TRY3_MATERIALS
    layers = [("concrete", 200), ("EPS", 180), ("0.5B cement block", 90), ("gypsum", 10)]
    m = planar_model(layers, TRY3_MATERIALS, hrb=5.1)
    ho, hi, To, Ti = 16.03, 1.415, -3.7, 20.0
    s = Solver(m); s.run(Boundary.constant(2, To, Ti, ho, hi), nsteps=1)
    R = 1 / (hi + 0.9 * 5.1) + 1 / (ho + 0.9 * 5.1) + 0.2 / 1.6 + 0.18 / 0.032 + 0.09 / 0.6 + 0.01 / 0.16
    q = (Ti - To) / R
    Ts = Ti - q / (hi + 0.9 * 5.1)
    assert abs(s.series("surf")[0] - Ts) < 1e-9
