"""2D 비정상 열전도 + 표면 대류 + 형상계수 복사 + 외벽 일사 흡수 솔버.

이산화
  - 노드: 격자선 교점 중 재료 셀에 닿는 것. 재료는 셀 단위(셀 중심 값).
  - 전도: 인접 노드 사이 유한체적 전도도 (셀 반폭 x 깊이 x λ / 노드 간격).
  - 열용량: 셀 용량의 1/4 을 네 모서리 노드에 배분.
  - 경계: 재료 셀과 경계 셀(zone) 사이 면. 대류 hc*(T - Ta), 복사는 hrb 로 선형화한 회색면 복사망
    (같은 zone 의 직각 표면끼리 형상계수, 나머지 시야는 먼 곳 = 기준온도).
  - 시간: 첫 스텝은 정상상태, 이후 Crank-Nicolson.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from .model import Model
from .boundary import Boundary
from .radiation import exchange_area, grey_network


class Solver:
    def __init__(self, model: Model, zmode: str = "mid", vf_cutoff: float = 0.0, vf_rebalance: str = "sky",
                 conv_lump="voltra"):
        """vf_cutoff: 이 값보다 작은 형상계수(행 기준)는 0으로 버린다 (VOLTRA 의 'smallest accepted view factor' 시험용).
        vf_rebalance: 버린 만큼을 'sky'(먼 곳)로 보낼지, 'scale'(남은 형상계수를 비례 확대해 행 합을 유지)할지.
        conv_lump: 표면 칸(면)의 대류와 복사 열교환을 노드에 어떻게 줄지. 기본값은 'voltra'(VOLTRA 와 같은 처리, 경험 보정 없음).
          아래 'auto', 'low', 'half', (wx, wy)는 이전 경험 보정이다(8.19 ~ 8.46). 비교, 회귀 시험용으로 남겨 둔다.
          'auto' : (기본값) 모델에 결로 방지 단열재(XPS 판, meta['strip_material'])가 있으면 그 열전도율에 따라 hbcalc.lump_rule.strip_weight 로
                   (w, w)를 정하고, 없으면 'low'.
          'low'  : 좌표가 작은 쪽 끝 노드에 전부 (VOLTRA 결과와 가장 잘 맞음: 정상상태 표면온도 RMSE
                   자동 격자 변형 4종 0.087 -> 0.016, TRY3 손격자 0.170 -> 0.060 ℃)
          'half' : 양 끝 노드에 절반씩 (대칭적인 일반 유한체적 방식)
          (wx, wy) : 가로 면(x축 선분), 세로 면(y축 선분)별로 낮은 좌표 쪽 끝 노드에 가는 비율 (시험용)
          'voltra' : VOLTRA 의 표면 처리(CLAUDE.md 8.52). 면(선분)의 반쪽마다 그 반쪽 중점(끝 노드에서 L/4) 온도 (3 Ta + Tb)/4 로
                     대류와 복사 열교환을 계산해 가까운 노드에 준다. 복사는 면마다 흑체 노드(두 끝 노드 평균 온도와 방사율 저항으로 연결).
                     VOLTRA 온도장에서 표면 노드 열 평형 잔차 0.001 W/m (경험 보정 없음)"""
        if conv_lump == "auto":
            conv_lump = "low"
            sm = model.meta.get("strip_material") if model.meta else None
            lam_s = [mat.lam for mat in model.materials.values() if mat.name == sm]
            if lam_s:
                from .lump_rule import strip_weight
                w = strip_weight(lam_s[0], "hand" if model.meta.get("hand_mesh") else "auto"); conv_lump = "low" if w >= 1.0 else (w, w)
        self.voltra = conv_lump == "voltra"
        if self.voltra:
            conv_lump = "half"
        if conv_lump not in ("low", "half") and not (isinstance(conv_lump, (tuple, list)) and len(conv_lump) == 2):
            raise ValueError(conv_lump)
        self.conv_lump = conv_lump
        self.m = model
        self.zmode = zmode
        self.vf_cutoff, self.vf_rebalance = vf_cutoff, vf_rebalance
        m = model
        nx, ny = m.shape
        self.nx, self.ny = nx, ny
        cid = m.cid
        self.ismat = cid > 0
        lam = np.zeros((nx, ny)); rc = np.zeros((nx, ny)); ep = np.zeros((nx, ny)); rf = np.zeros((nx, ny))
        for k, mat in m.materials.items():
            s = cid == k
            lam[s], rc[s], ep[s], rf[s] = mat.lam, mat.rho * mat.cp, mat.eps, mat.refl
        self.lam, self.rc, self.epsm, self.refl = lam, rc, ep, rf
        ex = np.zeros((nx + 2, ny + 2), bool)
        ex[1:-1, 1:-1] = self.ismat
        node = np.zeros((nx + 1, ny + 1), bool)
        for a in (0, 1):
            for b in (0, 1):
                node |= ex[a:a + nx + 1, b:b + ny + 1]
        self.idx = -np.ones((nx + 1, ny + 1), int)
        self.idx[node] = np.arange(node.sum())
        self.N = int(node.sum())
        self._assemble_conduction()
        self._find_faces()
        self._setup_radiation()
        if self.voltra:
            self._setup_voltra_surface()

    # ---------------------------------------------------------------- 전도
    def _assemble_conduction(self):
        m, nx, ny, idx = self.m, self.nx, self.ny, self.idx
        dx, dy, zt = np.diff(m.xe), np.diff(m.ye), m.zt
        mat = self.ismat
        cap = np.zeros(self.N)
        q = self.rc * dx[:, None] * dy[None, :] * zt / 4 * mat
        for a in (0, 1):
            for b in (0, 1):
                ii, jj = np.nonzero(mat)
                np.add.at(cap, idx[ii + a, jj + b], q[ii, jj])
        self.cap = cap
        # x 방향 링크: 노드 (i,j)-(i+1,j), 위아래 셀 j-1, j 가 기여
        wx = self.lam * mat * (dy[None, :] / 2) * zt / dx[:, None]       # (nx, ny)
        gx = np.zeros((nx, ny + 1)); gx[:, :-1] += wx; gx[:, 1:] += wx
        wy = self.lam * mat * (dx[:, None] / 2) * zt / dy[None, :]
        gy = np.zeros((nx + 1, ny)); gy[:-1, :] += wy; gy[1:, :] += wy
        r, c, v = [], [], []
        ii, jj = np.nonzero(gx > 0)
        a, b, g = idx[ii, jj], idx[ii + 1, jj], gx[ii, jj]
        r += [a, b, a, b]; c += [a, b, b, a]; v += [g, g, -g, -g]
        ii, jj = np.nonzero(gy > 0)
        a, b, g = idx[ii, jj], idx[ii, jj + 1], gy[ii, jj]
        r += [a, b, a, b]; c += [a, b, b, a]; v += [g, g, -g, -g]
        self.Kc = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                                shape=(self.N, self.N))

    # ---------------------------------------------------------------- 경계면
    def _find_faces(self):
        nx, ny, cid = self.nx, self.ny, self.m.cid
        dx, dy, zt = np.diff(self.m.xe), np.diff(self.m.ye), self.m.zt
        ism = lambda i, j: 0 <= i < nx and 0 <= j < ny and self.ismat[i, j]
        isbc = lambda i, j: 0 <= i < nx and 0 <= j < ny and not self.ismat[i, j]
        faces = []      # (i, j, 축, zone, 방사율, 일사반사율, 면적, 방향)
        for j in range(ny + 1):
            for i in range(nx):
                if ism(i, j - 1) and isbc(i, j):
                    faces.append((i, j, "x", -cid[i, j], self.epsm[i, j - 1], self.refl[i, j - 1], dx[i] * zt, "up"))
                if ism(i, j) and isbc(i, j - 1):
                    faces.append((i, j, "x", -cid[i, j - 1], self.epsm[i, j], self.refl[i, j], dx[i] * zt, "down"))
        for i in range(nx + 1):
            for j in range(ny):
                if ism(i - 1, j) and isbc(i, j):
                    faces.append((i, j, "y", -cid[i, j], self.epsm[i - 1, j], self.refl[i - 1, j], dy[j] * zt, "right"))
                if ism(i, j) and isbc(i - 1, j):
                    faces.append((i, j, "y", -cid[i - 1, j], self.epsm[i, j], self.refl[i, j], dy[j] * zt, "left"))
        self.faces = faces
        self.zone_ids = sorted({f[3] for f in faces})
        # 면 -> 낮은 좌표 쪽 끝 노드 (가로 면: (i,j), 세로 면: (i,j))
        self._face_low_node = [self.idx[f[0], f[1]] for f in faces]
        self._face_high_node = [self.idx[f[0] + 1, f[1]] if f[2] == "x" else self.idx[f[0], f[1] + 1] for f in faces]

    # ---------------------------------------------------------------- 복사
    def _setup_radiation(self):
        m, N, zt, xe, ye = self.m, self.N, self.m.zt, self.m.xe, self.m.ye
        el = {}
        for (i, j, ax, zn, eps, refl, area, side) in self.faces:
            if ax == "x":
                nodes, coord, plane = [(i, j), (i + 1, j)], [xe[i], xe[i + 1]], ye[j]
            else:
                nodes, coord, plane = [(i, j), (i, j + 1)], [ye[j], ye[j + 1]], xe[i]
            for k in (0, 1):
                key = (self.idx[nodes[k]], zn, ax, side)
                e = el.setdefault(key, dict(node=self.idx[nodes[k]], zone=zn, ax=ax, side=side, plane=plane,
                                            lo=coord[k], hi=coord[k], eps=eps, refl=refl))
                mid = (coord[0] + coord[1]) / 2
                e["lo"] = min(e["lo"], mid, coord[k]); e["hi"] = max(e["hi"], mid, coord[k])
        E = list(el.values()); n = len(E)
        A = np.array([(e["hi"] - e["lo"]) * zt for e in E])
        W = np.zeros((n, n))
        for ia, ea in enumerate(E):
            if ea["ax"] != "x":
                continue
            for ib, eb in enumerate(E):
                if eb["ax"] != "y" or eb["zone"] != ea["zone"]:
                    continue
                nx2 = 1 if eb["side"] == "right" else -1
                ny1 = 1 if ea["side"] == "up" else -1
                if ((ea["lo"] + ea["hi"]) / 2 - eb["plane"]) * nx2 <= 0: continue
                if ((eb["lo"] + eb["hi"]) / 2 - ea["plane"]) * ny1 <= 0: continue
                W[ia, ib] = W[ib, ia] = exchange_area(ea["lo"], ea["hi"], ea["plane"],
                                                      eb["lo"], eb["hi"], eb["plane"], zt, self.zmode)
        if self.vf_cutoff > 0:
            F0 = W / A[:, None]
            keep = (F0 >= self.vf_cutoff) & (F0.T >= self.vf_cutoff)
            if self.vf_rebalance == "scale":
                Fk = np.where(keep, F0, 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = np.where(Fk.sum(1) > 0, F0.sum(1) / Fk.sum(1), 1.0)
                Wn = Fk * f[:, None] * A[:, None]
                W = 0.5 * (Wn + Wn.T)
            else:
                W = np.where(keep, W, 0.0)
        self.rad_elements, self.rad_area, self.rad_W = E, A, W      # 진단용
        self._apply_network(W)

    def _face_exchange_areas(self):
        """면(선분) 요소 간 교환면적 A_i F_ij (m2). 같은 zone 의 직각 면만(해석 형상계수, 중간 높이 등)."""
        m, zt, xe, ye = self.m, self.m.zt, self.m.xe, self.m.ye
        F = self.faces; nf = len(F)
        ext = [((xe[f[0]], xe[f[0] + 1], ye[f[1]]) if f[2] == "x" else (ye[f[1]], ye[f[1] + 1], xe[f[0]])) for f in F]
        W = np.zeros((nf, nf))
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
                W[ia, ib] = W[ib, ia] = exchange_area(la, ha, pa, lb, hb, pb, zt, self.zmode)
        return W

    def _setup_voltra_surface(self, W=None):
        """VOLTRA 식 표면 교환(8.52). 면 요소 f: 끝 노드 (a, b), 면적 A_f, 방사율 e. 각 노드는 자기 쪽 반쪽 면(면적 A_f/2)의 중점 온도
        T_q = (3 T_self + T_other)/4 로 대류와 복사 열교환을 한다.
          복사: 면 흑체 노드 B_f 가 두 반쪽에 같은 전도 g_f/2 (g_f = e A_f hrb/(1-e))로 연결 -> 면 요소 온도 (Ta+Tb)/2 의 회색면 망(Gt, Gs)을 풀고
                Krad = B^T Gt B (열량을 두 노드에 절반씩) + sum (g_f/8) [[1,-1],[-1,1]] (끝 노드 온도차에 비례하는 표면 연결)
          대류: 노드 대각(면적 절반, 기존 half)에 시간 변하는 연결 -(h A_f / 8) [[1,-1],[-1,1]] 를 더한다."""
        F = self.faces; nf = len(F); N = self.N
        nodes = np.array([((self.idx[f[0], f[1]], self.idx[f[0] + 1, f[1]]) if f[2] == "x" else (self.idx[f[0], f[1]], self.idx[f[0], f[1] + 1])) for f in F])
        A = np.array([f[6] for f in F]); eps = np.array([f[4] for f in F]); zone = np.array([f[3] for f in F])
        W = self._face_exchange_areas() if W is None else np.asarray(W, float)
        Bm = sp.csr_matrix((np.full(2 * nf, 0.5), (np.repeat(np.arange(nf), 2), nodes.ravel())), shape=(nf, N))
        E = [dict(eps=e) for e in eps]
        Gt, Gs, self.view_sum = grey_network(E, W, A, self.m.hrb)
        g = eps * A * self.m.hrb / (1 - eps)
        def lap(w):                                           # sum_f w_f [[1,-1],[-1,1]] (노드 N x N)
            r = np.concatenate([nodes[:, 0], nodes[:, 1], nodes[:, 0], nodes[:, 1]]); c = np.concatenate([nodes[:, 0], nodes[:, 1], nodes[:, 1], nodes[:, 0]])
            v = np.concatenate([w, w, -w, -w]); return sp.csr_matrix((v, (r, c)), shape=(N, N))
        self.Krad = (Bm.T @ sp.csr_matrix(Gt) @ Bm + lap(g / 8)).tocsr()
        self.rad_zone = {z: Bm.T @ (Gs * (zone == z)) for z in self.zone_ids}
        self.vf_faces = dict(nodes=nodes, A=A, eps=eps, zone=zone, W=W, g=g)
        self.conv_coupling = {z: lap(A * (zone == z) / 8) for z in self.zone_ids}      # 대류 연결 행렬(h 를 곱해 뺀다)
        self.conv_A = {z: Bm.T @ (A * (zone == z)) * 1.0 for z in self.zone_ids}       # 대류 대각 면적 = 각 노드에 면적 절반
        self.rad_W = W

    def _conv_coupling_k(self, k):
        """시각 k 의 대류 연결 행렬 -(h A_f / 8) L 의 영역 합 (voltra 방식), 없으면 None"""
        if not self.voltra:
            return None
        C = None
        for z in self.zone_ids:
            h = float(self._h_zone[z][k]); M = (-h) * self.conv_coupling[z]
            C = M if C is None else C + M
        return C

    def set_exchange_areas(self, W):
        """요소 간 교환면적 A_i F_ij (m2, 대칭) 를 외부 값(예: VOLTRA .vwf)으로 바꾸고 복사망을 다시 만든다."""
        self.rad_W = np.asarray(W, float)
        self._apply_network(self.rad_W)

    def _apply_network(self, W):
        m, N = self.m, self.N
        E, A = self.rad_elements, self.rad_area
        n = len(E)
        Gt, Gs, self.view_sum = grey_network(E, W, A, m.hrb)
        S = sp.csr_matrix((np.ones(n), ([e["node"] for e in E], np.arange(n))), shape=(N, n))
        self.Krad = (S @ sp.csr_matrix(Gt) @ S.T).tocsr()
        zones = np.array([e["zone"] for e in E])
        refl = np.array([e["refl"] for e in E])
        self.rad_zone = {z: S @ (Gs * (zones == z)) for z in self.zone_ids}
        self.conv_A = {z: S @ (A * (zones == z)) for z in self.zone_ids}
        # 복사 요소가 속한 노드 (영역별). 대류 면적 배분 방식과 무관하게 영역 소속을 정한다.
        self.zone_nodes = {z: (S @ (zones == z).astype(float)) > 0 for z in self.zone_ids}
        if self.conv_lump != "half":
            wx, wy = (1.0, 1.0) if self.conv_lump == "low" else self.conv_lump
            conv_w = {z: np.zeros(N) for z in self.zone_ids}
            for f, lo, hi in zip(self.faces, self._face_low_node, self._face_high_node):
                w = wx if f[2] == "x" else wy
                conv_w[f[3]][lo] += w * f[6]; conv_w[f[3]][hi] += (1 - w) * f[6]
            self.conv_A = conv_w
        self.sol_A = {z: S @ (A * (zones == z) * (1 - refl)) for z in self.zone_ids}

    # ---------------------------------------------------------------- 시간 적분
    def _bc_terms(self, bc: Boundary, Isol):
        """각 시각의 대각 추가항 G(k) (n,N)과 우변 b(k) (n,N)"""
        n = bc.n
        G = np.zeros((n, self.N)); b = np.zeros((n, self.N)); self._h_zone = {}
        for z in self.zone_ids:
            kind = self.m.zones[z].kind
            if bc.zone_values and z in bc.zone_values:
                Ta, h = (np.broadcast_to(np.asarray(v, float), (n,)) for v in bc.zone_values[z])
            elif kind == "outdoor":
                h, Ta = bc.h_out, bc.T_out
            elif kind == "indoor":
                h, Ta = bc.h_in, bc.T_in
            else:
                raise ValueError(kind)
            A = self.conv_A[z]
            self._h_zone[z] = np.asarray(h, float)
            G += h[:, None] * A[None, :]
            b += (h * Ta)[:, None] * A[None, :] + Ta[:, None] * self.rad_zone[z][None, :]
            if kind == "outdoor":
                b += Isol[:, None] * self.sol_A[z][None, :]
        return G, b

    def run(self, bc: Boundary, nsteps: int | None = None, store: bool = True):
        nT = nsteps or bc.n
        Isol = bc.irradiance()
        self.solar_irradiance = Isol
        G, b = self._bc_terms(bc, Isol)
        Kst = (self.Kc + self.Krad).tocsr()
        dt = bc.dt
        if self.voltra:
            return self._run_voltra(bc, nT, G, b, Kst, dt)
        T = spl.spsolve((Kst + sp.diags(G[0])).tocsc(), b[0])      # 첫 스텝: 정상상태
        out = np.zeros((nT, self.N)); out[0] = T
        Cd = self.cap / dt
        base = (sp.diags(Cd) + 0.5 * Kst).tocsc()
        base.sort_indices()
        # 대각 원소 위치 (열마다)
        pos = np.empty(self.N, int)
        for j in range(self.N):
            s, e = base.indptr[j], base.indptr[j + 1]
            pos[j] = s + np.searchsorted(base.indices[s:e], j)
        d0 = base.data[pos].copy()
        for k in range(1, nT):
            rhs = Cd * T - 0.5 * (Kst @ T + G[k - 1] * T) + 0.5 * (b[k - 1] + b[k])
            base.data[pos] = d0 + 0.5 * G[k]
            T = spl.splu(base).solve(rhs)
            out[k] = T
        self.T_all = out
        self.bc = bc
        self.G_all, self.b_all = G[:nT], b[:nT]
        return out

    def _run_voltra(self, bc, nT, G, b, Kst, dt):
        """voltra 방식: 시간에 따라 변하는 대류 연결 행렬이 있어 매 스텝 행렬을 다시 만들어 푼다."""
        C0 = self._conv_coupling_k(0)
        T = spl.spsolve((Kst + C0 + sp.diags(G[0])).tocsc(), b[0])
        out = np.zeros((nT, self.N)); out[0] = T; Cd = self.cap / dt
        Cprev = C0
        for k in range(1, nT):
            Ck = self._conv_coupling_k(k)
            rhs = Cd * T - 0.5 * ((Kst + Cprev) @ T + G[k - 1] * T) + 0.5 * (b[k - 1] + b[k])
            base = (sp.diags(Cd + 0.5 * G[k]) + 0.5 * (Kst + Ck)).tocsc()
            T = spl.splu(base).solve(rhs); out[k] = T; Cprev = Ck
        self.T_all = out; self.bc = bc; self.G_all, self.b_all = G[:nT], b[:nT]
        return out

    # ---------------------------------------------------------------- 후처리
    def node_series(self, ix, iy):
        return self.T_all[:, self.idx[ix, iy]]

    def series(self, name):
        return self.node_series(*self.m.outputs[name])

    def boundary_heat_flow(self):
        """구조체에서 모든 경계로 나가는 순열류량 (n,) W = Σ (K T + G T - b). 에너지 수지 검사용."""
        Kst = self.Kc + self.Krad
        return np.array([(Kst @ self.T_all[k]).sum() + (self.G_all[k] * self.T_all[k]).sum() - self.b_all[k].sum()
                         for k in range(self.T_all.shape[0])])

    def zone_convective_flow(self, zone_kind):
        """zone 종류(indoor/outdoor) 표면에서 공기로 나가는 대류 열류량 W (n,) (복사 제외)"""
        out = np.zeros(self.T_all.shape[0])
        for z in self.zone_ids:
            if self.m.zones[z].kind != zone_kind:
                continue
            h = self.bc.h_in if zone_kind == "indoor" else self.bc.h_out
            Ta = self.bc.T_in if zone_kind == "indoor" else self.bc.T_out
            out += h * (self.T_all @ self.conv_A[z] - Ta * self.conv_A[z].sum())
        return out

    def zone_node_flows(self):
        """경계 영역(zone)별 표면 노드의 구조체 -> 영역 순열류 (n, 노드) W. 양수 = 구조체에서 나감.
        (K_rad T + G T - b) 를 해당 영역 노드에서만 취한 값. VOLTRA 리포트의 'into/out of object' 와 같은 방식으로
        노드별 부호를 나눠 합하려면 이 값을 쓴다."""
        T = self.T_all
        out = (self.Krad @ T.T).T + self.G_all * T - self.b_all
        if self.voltra:
            for k in range(T.shape[0]):
                out[k] += self._conv_coupling_k(k) @ T[k]
        return {z: out[:, self.zone_nodes[z]] for z in self.zone_ids}

    def zone_flows_into_out(self):
        """VOLTRA 리포트 형식: {zone: (into_object[W], out_of_object[W])}, 노드별 부호를 나눠 합산."""
        res = {}
        for z, q in self.zone_node_flows().items():
            res[z] = (np.where(q < 0, -q, 0.0).sum(1), np.where(q > 0, q, 0.0).sum(1))
        return res
