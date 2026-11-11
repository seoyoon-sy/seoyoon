"""벽 구조 입력 -> 격자와 재료 배치. .vtr 파일이 필요 없다.

현재 지원하는 형상: T자 접합부 (외벽 + 직각으로 붙은 내벽, 접합부에 단열 띠(열교 차단재) 선택).

          실외(y=0)
   ──────────────────────────  외벽 층: layers (실외 -> 실내 순서)
   ... 층 | 내벽 ........ | 층 ...   
   ──────────────────────────  실내 마감면 (y = y_in)
        실내A   | 내벽 |   실내B
                | 내벽 |           내벽은 실내로 room_depth 만큼 돌출

x 방향: [왼쪽 외벽 wall_half] [띠 s][내벽 코어 P][띠 s] [오른쪽 외벽 wall_half]
출력 위치는 오른쪽 내벽 면(x_R)에서 오른쪽으로 offsets 만큼 떨어진 실내 표면 점이다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .model import Material, Zone, Model

MM = 1e-3


@dataclass
class Layer:
    material: str
    thickness_mm: float


@dataclass
class TJunction:
    layers: list                         # 실외 -> 실내 순서 Layer 목록
    materials: dict                      # {이름: Material}
    partition_material: str = "concrete"
    partition_mm: float = 180.0          # 내벽 코어 두께
    strip_material: str | None = "XPS"   # 열교 차단 띠 (None 이면 없음)
    strip_mm: float = 10.0
    strip_from_layer: int = 1            # 띠가 시작하는 층 번호 (그 층의 실외쪽 면에서 시작)
    strip_extend_mm: float = 170.0       # 실내 마감면에서 실내쪽으로 더 뻗는 길이 (TRY3: 1480 -> 1650 mm)
    wall_half_mm: float = 1000.0         # 내벽 양쪽 외벽 길이 (좌우 같을 때)
    wall_left_mm: float | None = None    # 값을 주면 왼쪽 외벽 길이를 따로 지정
    wall_right_mm: float | None = None   # 값을 주면 오른쪽 외벽 길이를 따로 지정 (출력 위치는 오른쪽)
    room_depth_mm: float = 900.0         # 내벽이 실내로 돌출한 길이
    outdoor_depth_mm: float = 1000.0     # 실외 경계 영역(재료 없음) 두께. 계산에는 영향 없음
    z_mm: float = 1000.0
    offsets_mm: tuple = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
    hrb: float = 5.25
    # 격자 조절
    h_min_mm: float = 5.0                # 접합부 근처 최소 셀 크기
    h_max_mm: float = 50.0               # 최대 셀 크기
    h_layer_mm: float = 20.0             # 외벽 층 두께 방향 최대 셀 크기
    growth: float = 0.25                 # 접합부에서 멀어질 때 셀 크기 증가율
    symmetric_mesh: bool = False         # True 면 왼쪽 접합부도 오른쪽처럼 촘촘히
    mesh_mm: float | None = None         # 값을 주면 균일 격자(셀 크기 mesh_mm). 구조 경계와 출력 위치는 mesh_mm 의 배수여야 함
    snap_mm: float | None = None         # 값을 주면 자동 격자의 모든 격자선을 snap_mm 의 배수로 맞춤 (예: 1)
    x_edges_mm: tuple | None = None      # 격자선을 직접 지정하면 위 격자 조절 값은 무시
    y_edges_mm: tuple | None = None


def _march(a, b, cap, focus, min_cells=1, single=False):
    """구간 [a,b]를 나눈 격자선 좌표(양 끝 포함). cap(d): focus 와의 거리 d 에서 허용 셀 크기."""
    L = b - a
    if single or L <= 1e-12:
        return [a, b]
    focus = list(focus)
    dist = lambda s: min((abs(s - f) for f in focus), default=1e9)
    # 구간에서 focus 와 가장 가까운 끝에서 시작
    from_a = dist(a) <= dist(b)
    pts, pos, rem = [], 0.0, L
    sizes = []
    while rem > 1e-9:
        s = (a + pos) if from_a else (b - pos)
        d = dist(s)
        h = cap(d)
        if rem <= 1.5 * h:
            sizes.append(rem); break
        sizes.append(h); pos += h; rem -= h
    if len(sizes) < min_cells:                      # 최소 셀 수 보장 (균등 분할)
        sizes = [L / min_cells] * min_cells
    if not from_a:
        sizes = sizes[::-1]
    out = [a]
    for h in sizes:
        out.append(out[-1] + h)
    out[-1] = b
    return out


def _axis(breaks, cap, focus, caps_interval=None, min_cells=None, single=None):
    breaks = sorted(set(round(b, 9) for b in breaks))
    caps_interval = caps_interval or {}
    min_cells = min_cells or {}
    single = single or set()
    edges = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        key = (round(a, 9), round(b, 9))
        ci = caps_interval.get(key, np.inf)
        c = (lambda d, ci=ci: min(cap(d), ci))
        seg = _march(a, b, c, focus, min_cells.get(key, 1), key in single)
        edges += seg[1:]
    return np.array(edges)


def _uniform(a, b, h, nm):
    n = (b - a) / h
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"{nm} 방향 길이 {b - a} mm 가 mesh_mm={h} 의 배수가 아님")
    return a + h * np.arange(int(round(n)) + 1)


def build_tjunction(spec: TJunction) -> Model:
    s = spec
    names = {m.name for m in s.materials.values()} if isinstance(next(iter(s.materials.values())), Material) else set()
    mats = {}
    ids = {}
    for i, (nm, m) in enumerate(s.materials.items(), start=1):
        mats[i] = m
        ids[nm] = i

    def need(nm):
        if nm not in ids:
            raise KeyError(f"재료 '{nm}' 이(가) materials 에 없음. 있는 재료: {sorted(ids)}")
        return ids[nm]

    for l in s.layers:
        need(l.material)
    need(s.partition_material)
    if s.strip_material:
        need(s.strip_material)

    # --- 좌표 (mm)
    P, st = s.partition_mm, (s.strip_mm if s.strip_material else 0.0)
    xL = s.wall_left_mm if s.wall_left_mm is not None else s.wall_half_mm
    wR = s.wall_right_mm if s.wall_right_mm is not None else s.wall_half_mm
    xR = xL + 2 * st + P
    W = xR + wR
    Yo = s.outdoor_depth_mm
    ycum = [Yo]
    for l in s.layers:
        ycum.append(ycum[-1] + l.thickness_mm)
    y_in = ycum[-1]
    y_top = y_in + s.room_depth_mm
    y_strip0 = ycum[s.strip_from_layer] if s.strip_material else None
    y_strip1 = y_in + s.strip_extend_mm if s.strip_material else None

    # --- 격자선
    h_min, h_max, g = s.h_min_mm, s.h_max_mm, s.growth
    cap = lambda d: min(h_max, h_min + g * d)
    xb = [0, xL, xL + st, xR - st, xR, W] + [xR + o for o in s.offsets_mm]
    xfocus = [xR] + ([xL] if s.symmetric_mesh else [])
    xedges = np.array(s.x_edges_mm, float) if s.x_edges_mm is not None else _axis(xb, cap, xfocus, caps_interval={})
    if s.mesh_mm and s.x_edges_mm is None:
        xedges = _uniform(0.0, W, s.mesh_mm, "x")
    yb = [0, Yo] + ycum[1:] + [y_top]
    if s.strip_material:
        yb += [y_strip0, y_strip1]
    capi = {}
    mc = {}
    for a, b in zip(ycum[:-1], ycum[1:]):
        capi[(round(a, 9), round(b, 9))] = s.h_layer_mm
        mc[(round(a, 9), round(b, 9))] = 2
    yfocus = [y_in]
    yedges = (np.array(s.y_edges_mm, float) if s.y_edges_mm is not None else
              _axis(yb, cap, yfocus, caps_interval=capi, min_cells=mc, single={(round(0, 9), round(Yo, 9))}))
    if s.mesh_mm and s.y_edges_mm is None:
        yedges = np.r_[0.0, _uniform(Yo, y_top, s.mesh_mm, "y")]
    # 층 경계 사이 구간이 strip 끝점 등으로 쪼개진 경우도 층 규칙을 따르게 하려면 위 breaks 로 충분하다

    if s.snap_mm and not s.mesh_mm:
        snap = lambda e: np.unique(np.round(e / s.snap_mm) * s.snap_mm)
        if s.x_edges_mm is None: xedges = snap(xedges)
        if s.y_edges_mm is None: yedges = snap(yedges)
    # 격자선을 직접 지정한 경우, 층 경계와 띠 위치가 격자선 위에 있어야 한다 (두께를 바꾸고 옛 격자를 쓰는 실수 방지)
    for nm, br, ed, given in (("x", xb, xedges, s.x_edges_mm), ("y", yb, yedges, s.y_edges_mm)):
        if given is not None:
            bad = [b for b in br if np.abs(ed - b).min() > 1e-3]
            if bad:
                raise ValueError(f"지정한 {nm} 격자선이 구조 경계 {bad} mm 와 맞지 않음. 두께를 바꿨다면 x_edges_mm/y_edges_mm 를 지우고 자동 격자를 쓰세요.")
    nx, ny = len(xedges) - 1, len(yedges) - 1
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    cid = np.zeros((nx, ny), int)

    zones = {1: Zone("outdoor", "실외"), 2: Zone("indoor", "실내 왼쪽"), 3: Zone("indoor", "실내 오른쪽")}
    cid[Y < Yo] = -1
    cid[(Y >= y_in) & (X < xL)] = -2
    cid[(Y >= y_in) & (X > xR)] = -3
    # 외벽 층
    for k, l in enumerate(s.layers):
        sel = (Y >= ycum[k]) & (Y < ycum[k + 1]) & ((X < xL) | (X > xR))
        cid[sel] = need(l.material)
    # 내벽 (외벽 바깥면에서 실내 끝까지) + 열교 차단 띠
    sel = (X >= xL) & (X <= xR) & (Y >= Yo) & (Y < y_top)
    cid[sel] = need(s.partition_material)
    if s.strip_material:
        for x0, x1 in ((xL, xL + st), (xR - st, xR)):
            sel = (X >= x0) & (X <= x1) & (Y >= y_strip0) & (Y < y_strip1)
            cid[sel] = need(s.strip_material)
    assert (cid != 0).all(), "채워지지 않은 셀이 있음"

    # --- 출력 위치
    def find(arr, v):
        i = int(np.argmin(np.abs(arr - v)))
        assert abs(arr[i] - v) < 1e-3, f"격자선이 {v} mm 에 없음"
        return i

    outputs = {}
    iy_surf = find(yedges, y_in)
    iy_back = find(yedges, ycum[-2]) if len(s.layers) >= 2 else None
    # 마지막 층(마감재) 뒷면 = 마지막 층의 실외쪽 면
    for o in s.offsets_mm:
        ix = find(xedges, xR + o)
        outputs[f"surf_{o:g}"] = (ix, iy_surf)
        if iy_back is not None:
            outputs[f"back_{o:g}"] = (ix, iy_back)
    outputs["ref_surf"] = (nx, iy_surf)         # 접합부에서 가장 먼 실내 표면 (비열교부 기준점)
    if iy_back is not None:
        outputs["ref_back"] = (nx, iy_back)

    meta = dict(xL=xL, xR=xR, W=W, st=st, Yo=Yo, ycum=list(ycum), y_in=y_in, y_top=y_top,
                y_strip0=y_strip0, y_strip1=y_strip1, layers=[l.material for l in s.layers],
                partition_material=s.partition_material, strip_material=s.strip_material,
                offsets_mm=tuple(s.offsets_mm), z_mm=s.z_mm)
    return Model(xe=xedges * MM, ye=yedges * MM, zt=s.z_mm * MM, cid=cid, materials=mats,
                 zones=zones, outputs=outputs, hrb=s.hrb, meta=meta)
