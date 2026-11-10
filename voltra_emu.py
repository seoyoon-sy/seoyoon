#!/usr/bin/env python3
"""
VOLTRA 모사기 (표면온도만, 열류량 없음)

VOLTRA .vtr 파일(격자, 블록, 재료, 경계조건, 출력 노드 onod)과 경계조건 파일(FTE/FHT/FSG/FSD)을 읽어서
2D 비정상 열전도 + 실내외 표면 열전달 + 형상계수 복사 + 외벽 일사 흡수를 풀고,
.vtr 의 출력 노드(onod) 위치의 표면온도 시계열을 돌려준다.

VOLTRA 매뉴얼 v9.0 근거
  - 노드는 격자선 교점, 재료는 격자 셀 단위 (매뉴얼 B.6, B.21, D.6)
  - 첫 스텝은 정상상태, 이후 Crank-Nicolson (D.6)
  - BC_SKY: 대류계수는 사용자 함수(H##), 복사는 형상계수와 흑체 복사계수 hrb (D.1.4, D.2.3)
  - 회색면은 방사율로 보정 (D.2.3)
  - 일사: X축 북, Y축 서, 지면반사율, 태양 장애물 (B.18.1, D.9)

사용 예 (파이썬)
    from voltra_emu import simulate
    df = simulate("template.vtr",
                  lamb={"EPS": 0.032, "XPS": 1.6},
                  bc_files={"T01": "wall_TO.FTE", "T02": "Wall_TI.FTE", "G01": "Wall_IG2.FSG",
                            "D01": "Wall_ID2.FSD", "H01": "Wall_HO.FHT", "H02": "Wall_HI.FHT"})
    df.to_excel("result.xlsx")

사용 예 (명령줄)
    python voltra_emu.py template.vtr result.xlsx --lamb EPS=0.032 XPS=1.6 \
        --bc T01=wall_TO.FTE T02=Wall_TI.FTE G01=Wall_IG2.FSG D01=Wall_ID2.FSD H01=Wall_HO.FHT H02=Wall_HI.FHT

한계 (반드시 읽을 것)
  - 이 코드는 VOLTRA의 계산 방식을 모사한 것으로 VOLTRA 자체가 아니며 결과가 완전히 같지는 않다.
    엑셀의 VOLTRA 결과 39개 케이스와 비교한 오차는 voltra_emu_validate.py 결과를 볼 것.
  - 일사 시각 보정(solar_shift=-0.25 시간)은 엑셀 4개 케이스에 맞춰 정한 값이다.
  - 3D 중 z 방향 열전도는 무시하고 출력 노드 높이(z 중간)의 형상계수만 사용한다.
  - 외벽 방위(동향), 흡수율(1-refs), 지면반사율은 .vtr 와 매뉴얼에서 읽은 값을 쓴다.
"""
import argparse
import re
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


# ----------------------------------------------------------------- 파일 읽기
def _text(path):
    raw = open(path, 'rb').read()
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('cp949', errors='replace')


def read_vtr(path):
    txt = _text(path)
    L = txt.split('\r\n') if '\r\n' in txt else txt.split('\n')
    d = dict(xw=[], yw=[], zw=[], blok=[], colr={}, onod=[], funt=[], par={})
    cur, region = None, 'hdr'
    for l in L:
        if '=' not in l:
            continue
        k, v = l.split('=', 1)
        k = k.strip()
        if k == 'xwid': d['xw'].append(float(v))
        elif k == 'ywid': d['yw'].append(float(v))
        elif k == 'zwid': d['zw'].append(float(v))
        elif k == 'blok': d['blok'].append([int(t) for t in v.split()])
        elif k == 'onod': d['onod'].append([int(t) for t in v.split()])
        elif k == 'sobs': d.setdefault('sobs', []).append([float(t) for t in v.split()])
        elif k == 'rptf': region, cur = 'mat', None
        elif k == 'nnbc': region = 'post'
        elif k == 'colr':
            cur = int(v)
            d['colr'].setdefault(cur, {})
        elif region == 'hdr' and k == 'type' and cur is not None:
            d['colr'][cur]['type'] = int(v)
        elif region == 'mat' and cur is not None:
            d['colr'][cur][k] = v.strip()
        elif k == 'funr': d['funt'].append({'ref': v.strip()})
        elif k == 'funn': d['funt'][-1]['file'] = v.strip()
        elif k == 'funp': d['funt'][-1]['par'] = v
        elif region in ('hdr', 'post'):
            d['par'][k] = v.strip()
    return d


def read_series(path, tag):
    out = []
    for l in _text(path).splitlines():
        if l.startswith(tag):
            out.append(float(l.replace(',', ' ').split()[1]))
    return np.array(out)


def read_site(path):
    """FSG 파일 DAT 줄에서 위도, 경도, 시간대"""
    for l in _text(path).splitlines():
        if l.startswith('DAT'):
            p = l.replace(',', ' ').split()
            return float(p[2]), float(p[3]), float(p[4]) if len(p) > 4 else 9.0
    return 37.5, 127.0, 9.0


# ----------------------------------------------------------------- 일사
def sun_position(n, lat, lon, tz, shift, day0=1):
    """시간 인덱스 k 의 태양 고도, 방위(남 0, 서 +). shift: 시간 보정(시간)"""
    k = np.arange(n) + shift
    dayno = day0 + np.floor((k + 24) / 24) - 1
    clock = np.mod(k, 24)
    B = np.radians(360 / 365 * (dayno - 81))
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    ts = clock + (lon - 15 * tz) / 15 + eot / 60
    dec = np.radians(23.45 * np.sin(np.radians(360 / 365 * (284 + dayno))))
    ha = np.radians(15 * (ts - 12))
    ph = np.radians(lat)
    alt = np.arcsin(np.clip(np.sin(ph) * np.sin(dec) + np.cos(ph) * np.cos(dec) * np.cos(ha), -1, 1))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(ph) - np.tan(dec) * np.cos(ph))
    return alt, az


def solar_on_surface(G, D, alt, az, obst, surf_az_deg, tilt_deg, rho_g):
    a, z = np.degrees(alt), np.degrees(az)
    Gh = np.maximum(G, 0)
    Dh = np.minimum(np.maximum(D, 0), Gh)
    blocked = np.zeros(len(a), bool)
    for o in obst:                         # [번호, 방위min, 방위max, 고도min, 고도max]
        blocked |= (z >= o[1]) & (z <= o[2]) & (a >= o[3]) & (a <= o[4])
    blocked |= a <= 0
    Ib = np.where(blocked, 0.0, (Gh - Dh) / np.maximum(np.sin(alt), 1e-3))
    g, t = np.radians(surf_az_deg), np.radians(tilt_deg)
    cos_i = np.sin(alt) * np.cos(t) + np.cos(alt) * np.sin(t) * np.cos(az - g)
    return Ib * np.maximum(cos_i, 0) + Dh * (1 + np.cos(t)) / 2 + rho_g * Gh * (1 - np.cos(t)) / 2


# ----------------------------------------------------------------- 형상계수
def _I0(Lz, a):
    return Lz / (2 * a * (a + Lz ** 2)) + np.arctan(Lz / np.sqrt(a)) / (2 * a ** 1.5)


def _I1(Lz, a):
    return 1 / (2 * a) - 1 / (2 * (a + Lz ** 2))


def _Z(Lz, a, zmode):
    if zmode == 'avg':                      # 두 요소 모두 z 전체 평균
        return 2 * (Lz * _I0(Lz, a) - _I1(Lz, a))
    return 2 * _I0(Lz / 2, a) * Lz          # 출력 노드 높이(z 중간)에서 본 형상계수


# ----------------------------------------------------------------- 모사기
class VoltraEmu:
    def __init__(self, vtr, lamb=None, rho=None, cp=None, eps=None):
        """lamb, rho, cp, eps: {재료이름: 값} 으로 .vtr 값을 덮어쓴다."""
        d = read_vtr(vtr)
        self.d = d
        xe = np.r_[0, np.cumsum(d['xw'])] / 1000.0
        ye = np.r_[0, np.cumsum(d['yw'])] / 1000.0
        self.xe, self.ye = xe, ye
        self.nx, self.ny = len(d['xw']), len(d['yw'])
        self.zt = np.sum(d['zw']) / 1000.0
        nx, ny = self.nx, self.ny
        cid = np.zeros((nx, ny), int)
        for b in d['blok']:                  # 나중 블록이 앞 블록을 덮어쓴다
            cid[b[2]:b[3], b[4]:b[5]] = b[1]
        self.cid = cid
        mats = {}
        for c, p in d['colr'].items():
            if 'lamb' in p:
                nm = p.get('name', str(c))
                m = dict(name=nm, lamb=float(p['lamb']), rho=float(p['ro']), cp=float(p['c']),
                         eps=float(p['epsi']), refs=float(p.get('refs', 0.1)))
                for dct, key in ((lamb, 'lamb'), (rho, 'rho'), (cp, 'cp'), (eps, 'eps')):
                    if dct and nm in dct:
                        m[key] = dct[nm]
                mats[c] = m
        if lamb:
            unknown = set(lamb) - {m['name'] for m in mats.values()}
            if unknown:
                raise KeyError(f"없는 재료 이름: {unknown}. 있는 재료: {[m['name'] for m in mats.values()]}")
        self.mats = mats
        self.ismat = np.isin(cid, list(mats))
        lam = np.zeros((nx, ny)); rc = np.zeros((nx, ny)); ep = np.zeros((nx, ny)); rf = np.zeros((nx, ny))
        for c, m in mats.items():
            s = cid == c
            lam[s], rc[s], ep[s], rf[s] = m['lamb'], m['rho'] * m['cp'], m['eps'], m['refs']
        self.lam, self.rc, self.epsm, self.refs = lam, rc, ep, rf
        ex = np.zeros((nx + 2, ny + 2), bool)
        ex[1:-1, 1:-1] = self.ismat
        node = np.zeros((nx + 1, ny + 1), bool)
        for a in (0, 1):
            for b in (0, 1):
                node |= ex[a:a + nx + 1, b:b + ny + 1]
        self.idx = -np.ones((nx + 1, ny + 1), int)
        self.idx[node] = np.arange(node.sum())
        self.N = int(node.sum())
        self._assemble()
        self._rad_key = None

    # --- 전도, 용량, 경계면
    def _assemble(self):
        nx, ny, xe, ye, zt = self.nx, self.ny, self.xe, self.ye, self.zt
        dxs, dys = np.diff(xe), np.diff(ye)
        cap = np.zeros(self.N)
        for i in range(nx):
            for j in range(ny):
                if self.ismat[i, j]:
                    q = self.rc[i, j] * dxs[i] * dys[j] * zt / 4
                    for a in (0, 1):
                        for b in (0, 1):
                            cap[self.idx[i + a, j + b]] += q
        self.cap = cap
        K = {}
        for j in range(ny + 1):
            for i in range(nx):
                g = sum(self.lam[i, jc] * (dys[jc] / 2) * zt / dxs[i]
                        for jc in (j - 1, j) if 0 <= jc < ny and self.ismat[i, jc])
                if g > 0:
                    K[(self.idx[i, j], self.idx[i + 1, j])] = K.get((self.idx[i, j], self.idx[i + 1, j]), 0) + g
        for i in range(nx + 1):
            for j in range(ny):
                g = sum(self.lam[ic, j] * (dxs[ic] / 2) * zt / dys[j]
                        for ic in (i - 1, i) if 0 <= ic < nx and self.ismat[ic, j])
                if g > 0:
                    K[(self.idx[i, j], self.idx[i, j + 1])] = K.get((self.idx[i, j], self.idx[i, j + 1]), 0) + g
        r, c, v = [], [], []
        for (a, b), g in K.items():
            r += [a, b, a, b]; c += [a, b, b, a]; v += [g, g, -g, -g]
        self.Kc = sp.csr_matrix((v, (r, c)), shape=(self.N, self.N))
        # 재료와 BC 셀이 만나는 경계면
        faces = []
        isbc = lambda i, j: 0 <= i < nx and 0 <= j < ny and not self.ismat[i, j]
        ism = lambda i, j: 0 <= i < nx and 0 <= j < ny and self.ismat[i, j]
        for j in range(ny + 1):
            for i in range(nx):
                if ism(i, j - 1) and isbc(i, j):
                    faces.append((i, j, 'x', self.cid[i, j], self.epsm[i, j - 1], self.refs[i, j - 1], dxs[i] * zt, 'up'))
                if ism(i, j) and isbc(i, j - 1):
                    faces.append((i, j, 'x', self.cid[i, j - 1], self.epsm[i, j], self.refs[i, j], dxs[i] * zt, 'down'))
        for i in range(nx + 1):
            for j in range(ny):
                if ism(i - 1, j) and isbc(i, j):
                    faces.append((i, j, 'y', self.cid[i, j], self.epsm[i - 1, j], self.refs[i - 1, j], dys[j] * zt, 'right'))
                if ism(i, j) and isbc(i - 1, j):
                    faces.append((i, j, 'y', self.cid[i - 1, j], self.epsm[i, j], self.refs[i, j], dys[j] * zt, 'left'))
        self.faces = faces
        self.zones = sorted({f[3] for f in faces})

    # --- 복사망 (흑체 노드 + 회색면 추가 저항)
    def _setup_radiation(self, hrb, zmode, ngl=24):
        xe, ye, Lz, N = self.xe, self.ye, self.zt, self.N
        el = {}
        for (i, j, ax, col, eps, refs, area, side) in self.faces:
            if ax == 'x':
                nodes, coord, plane = [(i, j), (i + 1, j)], [xe[i], xe[i + 1]], ye[j]
            else:
                nodes, coord, plane = [(i, j), (i, j + 1)], [ye[j], ye[j + 1]], xe[i]
            for k in (0, 1):
                key = (self.idx[nodes[k]], col, ax, side)
                e = el.setdefault(key, dict(node=self.idx[nodes[k]], col=col, ax=ax, side=side, plane=plane,
                                           lo=coord[k], hi=coord[k], eps=eps, refs=refs))
                mid = (coord[0] + coord[1]) / 2
                e['lo'] = min(e['lo'], mid, coord[k]); e['hi'] = max(e['hi'], mid, coord[k])
        E = list(el.values()); n = len(E)
        A = np.array([(e['hi'] - e['lo']) * Lz for e in E])
        eps = np.array([e['eps'] for e in E])
        W = np.zeros((n, n))
        xg, wg = np.polynomial.legendre.leggauss(ngl)
        for ia, ea in enumerate(E):
            if ea['ax'] != 'x':
                continue
            for ib, eb in enumerate(E):
                if eb['ax'] != 'y' or eb['col'] != ea['col']:
                    continue
                nx2 = 1 if eb['side'] == 'right' else -1
                ny1 = 1 if ea['side'] == 'up' else -1
                if ((ea['lo'] + ea['hi']) / 2 - eb['plane']) * nx2 <= 0: continue
                if ((eb['lo'] + eb['hi']) / 2 - ea['plane']) * ny1 <= 0: continue
                u1, u2 = sorted([abs(ea['lo'] - eb['plane']), abs(ea['hi'] - eb['plane'])])
                v1, v2 = sorted([abs(eb['lo'] - ea['plane']), abs(eb['hi'] - ea['plane'])])
                u = (u2 - u1) / 2 * xg + (u2 + u1) / 2; wu = (u2 - u1) / 2 * wg
                v = (v2 - v1) / 2 * xg + (v2 + v1) / 2; wv = (v2 - v1) / 2 * wg
                U, V = np.meshgrid(u, v, indexing='ij')
                val = (U * V * _Z(Lz, U ** 2 + V ** 2, zmode) * np.outer(wu, wv)).sum() / np.pi
                W[ia, ib] = W[ib, ia] = val
        F = W / A[:, None]
        g = eps * A * hrb / (1 - eps)
        R = hrb * W
        Rsky = hrb * A * np.clip(1 - F.sum(1), 0, None)
        Binv = np.linalg.inv(np.diag(g + R.sum(1) + Rsky) - R)
        Gt = np.diag(g) - g[:, None] * Binv * g[None, :]
        Gs = g * (Binv @ Rsky)
        S = sp.csr_matrix((np.ones(n), ([e['node'] for e in E], np.arange(n))), shape=(N, n))
        self.Krad = (S @ sp.csr_matrix(Gt) @ S.T).tocsr()
        cols = np.array([e['col'] for e in E])
        refs = np.array([e['refs'] for e in E])
        self.rad_zone = {c: S @ (Gs * (cols == c)) for c in self.zones}
        self.conv_A = {c: S @ (A * (cols == c)) for c in self.zones}
        self.sol_A = {c: S @ (A * (cols == c) * (1 - refs)) for c in self.zones}
        self.view_sum = F.sum(1)

    # --- 계산
    def run(self, bc_files=None, hrb=None, zmode='mid', solar=True, solar_shift=-0.25, nsteps=None,
            surf_az_deg=-90.0, tilt_deg=90.0):
        """모든 노드의 온도 (nsteps, N). surf_az_deg: 외벽 방위(남 0, 서 +90, 동 -90)"""
        d = self.d
        F = {}
        for f in d['funt']:
            ref = f['ref']
            path = (bc_files or {}).get(ref, f.get('file'))
            F[ref] = read_series(path, {'T': 'FTE', 'H': 'FHT', 'G': 'FSG', 'D': 'FSD'}[ref[0]])
        hrb = hrb if hrb is not None else float(d['par'].get('hrbl', 5.1))
        if self._rad_key != (hrb, zmode):
            self._setup_radiation(hrb, zmode)
            self._rad_key = (hrb, zmode)
        dt = float(d['par']['step'])
        nT = nsteps or int(round(float(d['par']['calc']) / dt))
        Isol = np.zeros(nT)
        if solar and 'G01' in F and 'D01' in F:
            gfile = (bc_files or {}).get('G01') or next(f['file'] for f in d['funt'] if f['ref'] == 'G01')
            lat, lon, tz = read_site(gfile)
            alt, az = sun_position(nT, lat, lon, tz, solar_shift, int(d['par'].get('day0', 1)))
            rho_g = float(d['par'].get('refg', 0.2))
            Isol = solar_on_surface(F['G01'][:nT], F['D01'][:nT], alt, az, d.get('sobs', []),
                                    surf_az_deg, tilt_deg, rho_g)
        self.solar_irradiance = Isol
        prm = {c: d['colr'][c] for c in self.zones}

        def bc(k):
            G = np.zeros(self.N); b = np.zeros(self.N)
            for c in self.zones:
                p = prm[c]
                hc, ta, tr = F[p['hcfu']][k], F[p['tafu']][k], F[p['trfu']][k]
                G += self.conv_A[c] * hc
                b += self.conv_A[c] * hc * ta + self.rad_zone[c] * tr
                if p.get('solz', '0') == '1':
                    b += self.sol_A[c] * Isol[k]
            return sp.diags(G), b

        Gm, b0 = bc(0)
        Kp = self.Kc + self.Krad + Gm
        T = spl.spsolve(Kp.tocsc(), b0)         # 첫 스텝: 정상상태
        out = np.zeros((nT, self.N)); out[0] = T
        C = sp.diags(self.cap / dt)
        bp = b0
        for k in range(1, nT):
            Gm, b1 = bc(k)
            K1 = self.Kc + self.Krad + Gm
            T = spl.spsolve((C + 0.5 * K1).tocsc(), (C - 0.5 * Kp) @ T + 0.5 * (bp + b1))  # Crank-Nicolson
            out[k] = T
            Kp, bp = K1, b1
        self.T_all = out
        self.fn = F
        return out

    def node_series(self, ix, iy):
        return self.T_all[:, self.idx[ix, iy]]

    def planar_reference(self):
        """비열교부 참조점: 모델 오른쪽 끝(격자 끝)의 실내 표면 노드 (열교 중심에서 가장 먼 곳)"""
        iy = max(o[2] for o in self.d['onod'])
        return self.node_series(self.nx, iy)


def simulate(vtr, lamb=None, rho=None, cp=None, eps=None, bc_files=None, zmode='mid', solar_shift=-0.25,
             hrb=None, start='2026-01-01 00:00', include_planar=True):
    """출력 노드(onod)의 표면온도를 pandas DataFrame 으로 돌려준다."""
    import pandas as pd
    e = VoltraEmu(vtr, lamb=lamb, rho=rho, cp=cp, eps=eps)
    e.run(bc_files, hrb=hrb, zmode=zmode, solar_shift=solar_shift)
    n = e.T_all.shape[0]
    cols = {}
    onod = e.d['onod']
    x0 = e.xe[min(o[1] for o in onod)]
    for o in onod:
        _, ix, iy, _ = o
        cols[f"x={1000 * (e.xe[ix] - x0):g}mm, y={1000 * e.ye[iy]:g}mm (노드 {ix},{iy})"] = e.node_series(ix, iy)
    if include_planar:
        cols["비열교부 참조 (모델 끝)"] = e.planar_reference()
    idx = pd.date_range(start, periods=n, freq='h')
    df = pd.DataFrame(cols, index=idx)
    df.insert(0, '실내 온도(℃)', e.fn['T02'][:n] if 'T02' in e.fn else np.nan)
    df.insert(0, '실외 온도(℃)', e.fn['T01'][:n] if 'T01' in e.fn else np.nan)
    return df


def _kv(items):
    out = {}
    for s in items or []:
        k, v = s.split('=', 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description='VOLTRA 모사기: 출력 노드 표면온도 시계열')
    ap.add_argument('vtr'); ap.add_argument('out', help='.xlsx 또는 .csv')
    ap.add_argument('--lamb', nargs='*', help='EPS=0.032 XPS=1.6')
    ap.add_argument('--rho', nargs='*'); ap.add_argument('--cp', nargs='*')
    ap.add_argument('--bc', nargs='*', help='T01=경로 T02=경로 G01=경로 D01=경로 H01=경로 H02=경로')
    ap.add_argument('--zmode', default='mid', choices=['mid', 'avg'])
    ap.add_argument('--solar-shift', type=float, default=-0.25)
    a = ap.parse_args()
    f = lambda kv: {k: float(v) for k, v in _kv(kv).items()} or None
    df = simulate(a.vtr, lamb=f(a.lamb), rho=f(a.rho), cp=f(a.cp), bc_files=_kv(a.bc) or None,
                  zmode=a.zmode, solar_shift=a.solar_shift)
    if a.out.lower().endswith('.csv'):
        df.to_csv(a.out, encoding='utf-8-sig')
    else:
        df.round(3).to_excel(a.out)
    print('저장:', a.out, df.shape)


if __name__ == '__main__':
    main()
