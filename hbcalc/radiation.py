"""표면 복사: 평행하지 않은(직각) 표면 요소 사이 형상계수와 회색면 복사망."""
from __future__ import annotations
import numpy as np


def _I0(L, a):
    return L / (2 * a * (a + L ** 2)) + np.arctan(L / np.sqrt(a)) / (2 * a ** 1.5)


def _I1(L, a):
    return 1 / (2 * a) - 1 / (2 * (a + L ** 2))


def zint(Lz, a, zmode="mid"):
    """Z(a) = ∬ dz1 dz2 / (a + (z1-z2)^2)^2.
    'avg': z1, z2 모두 0..Lz 전체. 'mid': z1 은 Lz/2 한 점에서 본 값에 Lz 를 곱함."""
    if zmode == "avg":
        return 2 * (Lz * _I0(Lz, a) - _I1(Lz, a))
    if zmode == "mid":
        return 2 * _I0(Lz / 2, a) * Lz
    raise ValueError(zmode)


def exchange_area(lo_a, hi_a, plane_a, lo_b, hi_b, plane_b, Lz, zmode="mid", ngl=24):
    """직각인 두 띠 사이 A_a F_ab (m2).
    띠 a: y=plane_a 평면, x in [lo_a, hi_a].  띠 b: x=plane_b 평면, y in [lo_b, hi_b].
    두 띠는 서로 마주 보는 쪽에 있어야 한다(호출하는 쪽에서 검사)."""
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    u1, u2 = sorted([abs(lo_a - plane_b), abs(hi_a - plane_b)])
    v1, v2 = sorted([abs(lo_b - plane_a), abs(hi_b - plane_a)])
    u = (u2 - u1) / 2 * xg + (u2 + u1) / 2
    wu = (u2 - u1) / 2 * wg
    v = (v2 - v1) / 2 * xg + (v2 + v1) / 2
    wv = (v2 - v1) / 2 * wg
    U, V = np.meshgrid(u, v, indexing="ij")
    return (U * V * zint(Lz, U ** 2 + V ** 2, zmode) * np.outer(wu, wv)).sum() / np.pi


def grey_network(E, W, A, hrb):
    """E: 요소 목록(dict, 'eps' 포함), W: A_i F_ij, A: 면적.
    반환: Gt(요소 간 유효 전도), Gs(요소 -> 먼 곳(기준온도) 유효 전도), F행합.
    복사 열전달계수 hrb 로 선형화한 흑체 복사망에 방사율 저항을 추가한 회색면 해석."""
    eps = np.array([e["eps"] for e in E])
    F = W / A[:, None]
    g = eps * A * hrb / (1 - eps)
    R = hrb * W
    Rsky = hrb * A * np.clip(1 - F.sum(1), 0, None)
    Binv = np.linalg.inv(np.diag(g + R.sum(1) + Rsky) - R)
    Gt = np.diag(g) - g[:, None] * Binv * g[None, :]
    Gs = g * (Binv @ Rsky)
    return Gt, Gs, F.sum(1)
