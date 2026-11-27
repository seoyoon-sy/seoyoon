"""3D(z 조각) 비정상 열전도 + 표면 대류 + 조각별 형상계수 복사 + 외벽 일사 흡수 솔버 (CLAUDE.md 8.77).

VOLTRA 는 깊이 방향(z)까지 계산하는 3D 다(8.66). 2D 솔버(hbcalc.solver.Solver)는 중간 높이에서 본 형상계수만 쓰고
z 전도가 없어서 접합부가 0.01 ~ 0.03 도 차갑게 나온다(8.65, 8.75). 이 솔버는 같은 2D 격자를 z 평면마다 복제한다.

이산화
  - 노드: 2D 노드(격자선 교점) x z 평면(z_edges). 평면 노드의 z 두께는 이웃 조각의 절반씩(끝 평면은 절반).
  - 평면 안 전도와 열용량: 2D 값(깊이 zt 기준)을 평면 두께 비율로 나눔. z 전도: 노드 단면(인접 재료 셀 면적의 1/4 합) x λ / 조각 두께.
  - 표면: 2D 면(선분) x z 조각 = 사각형 면. 대류와 복사의 표면 처리는 2D voltra 방식(8.52)을 z 방향에도 같은 규칙으로 확장했다:
      각 모서리 노드는 자기 쪽 1/4 사각형(면적 A/4)의 중심 온도로 열교환한다. 1/4 사각형 중심 온도는 양선형 보간
      (9 T_자기 + 3 T_x이웃 + 3 T_z이웃 + T_대각)/16 이고, 행렬로는 A M, M = M1 (x) M1, M1 = [[3/8, 1/8], [1/8, 3/8]].
      z 방향 확장은 2D 규칙에서 유추한 것이고 VOLTRA 에서 따로 확인하지 않았다(가정).
  - 복사: 사각형 면마다 흑체 노드(방사율 저항)로 회색면 망. 면 온도 = 네 모서리 평균, Krad = B^T Gt B + sum g (M - J/16).
    같은 zone 의 직각 면끼리 해석 형상계수(z 조각 쌍마다 정확한 적분), 나머지 시야(열린 끝면 포함)는 먼 곳 = 공기온도.
  - 대칭: z 조각이 깊이 중간면에 대해 대칭이고 조각 수가 짝수면 절반만 푼다(정확히 같은 해, 열류는 전체 깊이 값).
  - 시간: 첫 스텝 정상상태, 이후 Crank-Nicolson. 매 스텝 행렬(대류계수)이 바뀌어 전처리 켤레기울기법으로 푼다.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from .model import Model
from .boundary import Boundary
from .radiation import grey_network
from .solver import Solver

M1 = np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]])
MQ = np.kron(M1, M1)                     # 순서: (z 아래, x 낮음), (z 아래, x 높음), (z 위, x 낮음), (z 위, x 높음)


def _K2(t, a):
    """d2/dt2 K2(t) = 1/(a + t^2)^2, K2(0) = 0, 짝함수"""
    t = np.abs(t)
    return t * np.arctan(t / np.sqrt(a)) / (2 * a ** 1.5)


def strip_pair_z(lo_a, hi_a, plane_a, lo_b, hi_b, plane_b, z_edges, ngl=12):
    """직각인 두 띠(a: y=plane_a 평면 x in [lo_a,hi_a], b: x=plane_b 평면 y in [lo_b,hi_b])를 z 조각으로 나눈 사각형 면들 사이의
    교환면적 A F (m2), (nz, nz). W[p, q] = 조각 p 의 a 면과 조각 q 의 b 면."""
    u1, u2 = sorted([abs(lo_a - plane_b), abs(hi_a - plane_b)])
    v1, v2 = sorted([abs(lo_b - plane_a), abs(hi_b - plane_a)])
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    u = (u2 - u1) / 2 * xg + (u2 + u1) / 2; wu = (u2 - u1) / 2 * wg
    v = (v2 - v1) / 2 * xg + (v2 + v1) / 2; wv = (v2 - v1) / 2 * wg
    U, V = np.meshgrid(u, v, indexing="ij")
    w = (U * V * np.outer(wu, wv)).ravel(); a = (U ** 2 + V ** 2).ravel()
    z = np.asarray(z_edges, float)
    D = z[None, :] - z[:, None]                                  # D[i, j] = z_j - z_i
    Kt = (_K2(D[..., None], a[None, None, :]) * w).sum(-1) / np.pi
    # W[p,q] = K(z_{q+1} - z_p) - K(z_{q+1} - z_{p+1}) - K(z_q - z_p) + K(z_q - z_{p+1})
    return Kt[:-1, 1:] - Kt[1:, 1:] - Kt[:-1, :-1] + Kt[1:, :-1]


class Solver3D:
    def __init__(self, model: Model, nz: int = 6, z_edges=None, fold: bool = True):
        """nz: 깊이 방향 균일 조각 수(z_edges 를 주지 않을 때). z_edges: 0 ~ zt 의 조각 경계 좌표(m).
        fold: 깊이 중간면 대칭이면 절반만 푼다."""
        self.m = m = model
        zt = m.zt
        ze = np.linspace(0, zt, nz + 1) if z_edges is None else np.asarray(z_edges, float)
        if abs(ze[0]) > 1e-12 or abs(ze[-1] - zt) > 1e-9:
            raise ValueError("z_edges 는 0 에서 zt 까지여야 한다")
        self.ze, self.nz = ze, len(ze) - 1
        s2 = Solver(model, conv_lump="half")                # 2D 격자, 전도, 열용량, 표면 면 목록 재사용
        self.s2, self.idx, self.N, self.zone_ids = s2, s2.idx, s2.N, s2.zone_ids
        N, nzs, P = s2.N, self.nz, self.nz + 1
        dz = np.diff(ze)
        hk = np.zeros(P); hk[:-1] += dz / 2; hk[1:] += dz / 2      # 평면 노드의 z 두께
        self.hk = hk
        # ---- 전도와 열용량
        Kin = sp.kron(sp.diags(hk), s2.Kc / zt)
        cap = np.kron(hk, s2.cap / zt)
        dx, dy = np.diff(m.xe), np.diff(m.ye)
        lam_area = np.zeros(N)
        ii, jj = np.nonzero(s2.ismat)
        q = s2.lam[ii, jj] * dx[ii] * dy[jj] / 4
        for a in (0, 1):
            for b in (0, 1):
                np.add.at(lam_area, s2.idx[ii + a, jj + b], q)
        r, c, v = [], [], []
        for k in range(nzs):
            g = lam_area / dz[k]; A_ = np.arange(N) + k * N; B_ = A_ + N
            r += [A_, B_, A_, B_]; c += [A_, B_, B_, A_]; v += [g, g, -g, -g]
        Kz = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(N * P, N * P))
        # ---- 표면 사각형 면
        F = s2.faces; nf = len(F)
        nodes2 = np.array([((s2.idx[f[0], f[1]], s2.idx[f[0] + 1, f[1]]) if f[2] == "x" else (s2.idx[f[0], f[1]], s2.idx[f[0], f[1] + 1])) for f in F])
        L = np.array([f[6] for f in F]) / zt                      # 면 길이
        eps2 = np.array([f[4] for f in F]); refl2 = np.array([f[5] for f in F]); zone2 = np.array([f[3] for f in F])
        nq = nf * nzs
        fq = np.repeat(np.arange(nf), nzs); sq = np.tile(np.arange(nzs), nf)     # 사각형 q = f * nzs + s
        Aq = L[fq] * dz[sq]
        corners = np.stack([nodes2[fq, 0] + sq * N, nodes2[fq, 1] + sq * N,
                            nodes2[fq, 0] + (sq + 1) * N, nodes2[fq, 1] + (sq + 1) * N], 1)
        Bq = sp.csr_matrix((np.full(4 * nq, 0.25), (np.repeat(np.arange(nq), 4), corners.ravel())), shape=(nq, N * P))

        def local(w, Mloc):            # sum_q w_q Mloc (4x4) 를 노드 행렬로
            rr = np.repeat(corners, 4, 1).ravel(); cc = np.tile(corners, (1, 4)).ravel()
            vv = (w[:, None] * Mloc.ravel()[None, :]).ravel()
            return sp.csr_matrix((vv, (rr, cc)), shape=(N * P, N * P))
        # ---- 복사: 사각형 면 사이 교환면적
        xe, ye = m.xe, m.ye
        ext = [((xe[f[0]], xe[f[0] + 1], ye[f[1]]) if f[2] == "x" else (ye[f[1]], ye[f[1] + 1], xe[f[0]])) for f in F]
        W = np.zeros((nq, nq))
        for ia in range(nf):
            if F[ia][2] != "x":
                continue
            la, ha, pa = ext[ia]
            for ib in range(nf):
                if F[ib][2] != "y" or F[ib][3] != F[ia][3]:
                    continue
                lb, hb, pb = ext[ib]
                nx2 = 1 if F[ib][7] == "right" else -1; ny1 = 1 if F[ia][7] == "up" else -1
                if ((la + ha) / 2 - pb) * nx2 <= 0 or ((lb + hb) / 2 - pa) * ny1 <= 0:
                    continue
                blk = strip_pair_z(la, ha, pa, lb, hb, pb, ze)
                W[ia * nzs:(ia + 1) * nzs, ib * nzs:(ib + 1) * nzs] = blk
                W[ib * nzs:(ib + 1) * nzs, ia * nzs:(ia + 1) * nzs] = blk.T
        epsq = eps2[fq]
        Gt, Gs, self.view_sum = grey_network([dict(eps=e) for e in epsq], W, Aq, m.hrb)
        gq = epsq * Aq * m.hrb / (1 - epsq)
        Krad = (Bq.T @ sp.csr_matrix(Gt) @ Bq + local(gq, MQ - 1 / 16)).tocsr()
        zq = zone2[fq]
        rad_zone = {z: Bq.T @ (Gs * (zq == z)) for z in self.zone_ids}
        conv_A = {z: Bq.T @ (Aq * (zq == z)) for z in self.zone_ids}             # 모서리마다 A/4
        conv_C = {z: local(Aq * (zq == z), MQ - np.eye(4) / 4) for z in self.zone_ids}
        sol_A = {z: Bq.T @ (Aq * (zq == z) * (1 - refl2[fq])) for z in self.zone_ids}
        self.quad = dict(A=Aq, zone=zq, W=W, corners=corners)
        # ---- 대칭 접기
        sym = np.allclose(ze + ze[::-1], zt) and nzs % 2 == 0
        self.fold = fold and sym
        if self.fold:
            Pk = nzs // 2 + 1
            keep = np.minimum(np.arange(P), nzs - np.arange(P))       # 전체 평면 -> 남긴 평면
            rows = np.arange(N * P); cols = (keep[:, None] * N + np.arange(N)[None, :]).ravel()
            Pm = sp.csr_matrix((np.ones(N * P), (rows, cols)), shape=(N * P, N * Pk))
        else:
            Pk = P; Pm = sp.identity(N * P, format="csr")
        self.Pk, self.Pm = Pk, Pm
        fK = lambda K: (Pm.T @ K @ Pm).tocsr()
        fv = lambda x: Pm.T @ x
        self.Kst = fK(Kin + Kz + Krad)
        self.Krad = fK(Krad)
        self.cap = fv(cap)
        self.H = {z: fK(sp.diags(conv_A[z]) + conv_C[z]) for z in self.zone_ids}   # 대류 (h 를 곱함)
        self.conv_A = {z: fv(conv_A[z]) for z in self.zone_ids}
        self.rad_zone = {z: fv(rad_zone[z]) for z in self.zone_ids}
        self.sol_A = {z: fv(sol_A[z]) for z in self.zone_ids}
        self.zone_nodes = {z: self.conv_A[z] > 0 for z in self.zone_ids}
        self.n_nodes = N * Pk
        mid = nzs / 2
        if abs(mid - round(mid)) > 1e-9:
            raise ValueError("조각 수가 짝수여야 중간면 평면이 있다")
        self.mid_plane = int(round(mid)) if not self.fold else Pk - 1

    # ---------------------------------------------------------------- 시간 적분
    def _zone_hT(self, bc: Boundary):
        n = bc.n; out = {}
        for z in self.zone_ids:
            kind = self.m.zones[z].kind
            if bc.zone_values and z in bc.zone_values:
                Ta, h = (np.broadcast_to(np.asarray(v, float), (n,)) for v in bc.zone_values[z])
            elif kind == "outdoor":
                h, Ta = bc.h_out, bc.T_out
            else:
                h, Ta = bc.h_in, bc.T_in
            out[z] = (np.asarray(h, float), np.asarray(Ta, float), kind)
        return out

    def _rhs(self, hT, Isol, k):
        b = np.zeros(self.n_nodes)
        for z, (h, Ta, kind) in hT.items():
            b += h[k] * Ta[k] * self.conv_A[z] + Ta[k] * self.rad_zone[z]
            if kind == "outdoor":
                b += Isol[k] * self.sol_A[z]
        return b

    def _Hk(self, hT, k):
        H = None
        for z, (h, _, _) in hT.items():
            M = h[k] * self.H[z]; H = M if H is None else H + M
        return H

    def run(self, bc: Boundary, nsteps: int | None = None, store: str = "all", rtol: float = 1e-10):
        """store: 'all'(모든 노드 시계열) 또는 'outputs'(모델 출력 노드의 모든 평면만)."""
        nT = nsteps or bc.n
        Isol = bc.irradiance(); self.solar_irradiance = Isol
        hT = self._zone_hT(bc); self.bc, self.hT, self.Isol = bc, hT, Isol
        dt = bc.dt; Cd = sp.diags(self.cap / dt)
        H0 = self._Hk(hT, 0)
        T = spl.spsolve((self.Kst + H0).tocsc(), self._rhs(hT, Isol, 0))
        href = {z: h.mean() for z, (h, _, _) in hT.items()}
        Aref = (Cd + 0.5 * (self.Kst + sum(href[z] * self.H[z] for z in self.zone_ids))).tocsc()
        lu = spl.splu(Aref)
        Mp = spl.LinearOperator(Aref.shape, matvec=lu.solve)
        sel = None
        if store == "outputs":
            sel = np.unique([p * self.N + self.idx[o[0], o[1]] for o in self.m.outputs.values() for p in range(self.Pk)])
        out = np.zeros((nT, self.n_nodes if sel is None else len(sel)))
        out[0] = T if sel is None else T[sel]
        Hprev, bprev, self.iters = H0, self._rhs(hT, Isol, 0), []
        for k in range(1, nT):
            Hk = self._Hk(hT, k); bk = self._rhs(hT, Isol, k)
            A = Cd + 0.5 * (self.Kst + Hk)
            rhs = Cd @ T - 0.5 * ((self.Kst + Hprev) @ T) + 0.5 * (bprev + bk)
            it = [0]
            T, info = spl.cg(A, rhs, x0=T, M=Mp, rtol=rtol, atol=0, maxiter=200, callback=lambda x: it.__setitem__(0, it[0] + 1))
            if info != 0:
                T = spl.spsolve(A.tocsc(), rhs)
            self.iters.append(it[0])
            out[k] = T if sel is None else T[sel]
            Hprev, bprev = Hk, bk
        self.T_all, self._sel = out, sel
        return out

    # ---------------------------------------------------------------- 후처리
    def node_series(self, ix, iy, plane=None):
        p = self.mid_plane if plane is None else plane
        g = p * self.N + self.idx[ix, iy]
        if self._sel is None:
            return self.T_all[:, g]
        return self.T_all[:, np.searchsorted(self._sel, g)]

    def series(self, name, plane=None):
        """출력 노드 시계열. 기본은 깊이 중간면(VOLTRA 출력 노드 z = zt/2 와 같은 위치)."""
        return self.node_series(*self.m.outputs[name][:2], plane=plane)

    def zone_node_flows(self):
        """영역별 표면 노드의 구조체 -> 영역 순열류 (n, 노드) W, 전체 깊이 값(접은 경우 대칭 쌍 합). store='all' 필요."""
        if self._sel is not None:
            raise ValueError("store='all' 로 돌려야 한다")
        T = self.T_all; n = T.shape[0]
        Kr = self.Krad
        out = np.zeros_like(T)
        for k in range(n):
            out[k] = Kr @ T[k] + self._Hk(self.hT, k) @ T[k] - self._rhs(self.hT, self.Isol, k)
        return {z: out[:, self.zone_nodes[z]] for z in self.zone_ids}

    def zone_flows_into_out(self):
        res = {}
        for z, q in self.zone_node_flows().items():
            res[z] = (np.where(q < 0, -q, 0.0).sum(1), np.where(q > 0, q, 0.0).sum(1))
        return res
