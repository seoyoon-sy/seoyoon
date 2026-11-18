"""모델이 어떤 RC 회로인지 보이는 그림: 왼쪽은 실제 격자(TRY3 접합부 확대, 재료별 색과 노드), 오른쪽은 노드 연결 회로도.
출력: results/fig_rc_network.png
사용: python examples/plot_rc_diagram.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from hbcalc import build_tjunction
from hbcalc.presets import try3_spec
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
INK, MUTED, GRID, SURF = "#1f2328", "#59636e", "#d0d4d8", "#fcfcfb"
BLUE, ORANGE, GREEN, RED = "#2f6fd0", "#e8590c", "#2b9348", "#c2185b"

fig = plt.figure(figsize=(18, 9.6), facecolor=SURF)
axL = fig.add_axes([0.05, 0.19, 0.41, 0.66], facecolor=SURF); axR = fig.add_axes([0.495, 0.14, 0.49, 0.71], facecolor=SURF)
fig.text(0.03, 0.955, "이 모델은 어떤 RC 회로인가: 노드 약 2000개가 저항(전도)과 용량(열용량)으로 이어진 회로", fontsize=20, weight="bold", color=INK)
fig.text(0.03, 0.915, "왼쪽: 실제 격자(TRY3 접합부 확대, 재료별 색, 점이 노드). 오른쪽: 노드 몇 개를 떼어 낸 회로도. 모든 노드에 같은 식이 적용된다.", fontsize=12.5, color=MUTED)

# ---------------- 왼쪽: 실제 격자 확대 ----------------
m = build_tjunction(try3_spec(snap_mm=1)); xe = np.array(m.xe) * 1000; ye = np.array(m.ye) * 1000; cid = np.array(m.cid)
X0, X1, Y0, Y1 = 930, 1330, 1180, 1700
COL = {1: "#b9bec4", 2: "#f3e08a", 3: "#d9b48a", 4: "#f2f2ee", 5: "#f0a9c4"}; NAME = {1: "콘크리트", 2: "EPS", 3: "시멘트 블록", 4: "석고", 5: "XPS 판"}
nx, ny = cid.shape; mat = cid > 0
for i in range(nx):
    for j in range(ny):
        x0, x1, y0, y1 = xe[i], xe[i + 1], ye[j], ye[j + 1]
        if x1 < X0 or x0 > X1 or y1 < Y0 or y0 > Y1: continue
        c = cid[i, j]; axL.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=COL.get(c, "#eaf2fb"), ec="#8a9096", lw=0.5, zorder=1))
touch = np.zeros((nx + 1, ny + 1), bool)
for di in (0, 1):
    for dj in (0, 1): touch[di:di + nx, dj:dj + ny] |= mat
gx, gy = np.meshgrid(xe, ye, indexing="ij"); sel = touch & (gx >= X0) & (gx <= X1) & (gy >= Y0) & (gy <= Y1)
axL.plot(gx[sel], gy[sel], "o", ms=3.4, color=INK, zorder=3)
# 대표 노드: 내부 노드 하나와 표면 노드 하나에 연결선을 표시
def nearest(px, py):
    k = np.argmin((gx - px) ** 2 + (gy - py) ** 2 + 1e9 * (~touch)); i, j = np.unravel_index(k, gx.shape); return i, j
for (px, py, col, lab) in ((1100, 1600, BLUE, "내부 노드"), (1100, 1480, RED, "표면 노드")):
    i, j = nearest(px, py); axL.plot(gx[i, j], gy[i, j], "o", ms=9, mfc="none", mec=col, mew=2.2, zorder=5)
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        a, b = i + di, j + dj
        if 0 <= a <= nx and 0 <= b <= ny and touch[a, b]: axL.plot([gx[i, j], gx[a, b]], [gy[i, j], gy[a, b]], "-", color=col, lw=2.2, zorder=4)
axL.set_xlim(X0, X1); axL.set_ylim(Y0, Y1); axL.set_aspect("equal"); axL.set_xlabel("벽 길이 방향 x (mm)", color=MUTED); axL.set_ylabel("벽 두께 방향 y (mm), 아래가 실외, 위가 실내", color=MUTED)
for sp in ("top", "right"): axL.spines[sp].set_visible(False)
axL.set_title("실제 격자: 칸마다 재료가 정해지고, 칸 모서리가 노드", loc="left", fontsize=13.5, color=INK)
h = [Rectangle((0, 0), 1, 1, fc=COL[k], ec="#8a9096") for k in (1, 2, 3, 4, 5)] + [Rectangle((0, 0), 1, 1, fc="#eaf2fb", ec="#8a9096")]
axL.legend(h, [NAME[k] for k in (1, 2, 3, 4, 5)] + ["공기(경계 칸, 노드 없음)"], loc="upper center", bbox_to_anchor=(0.5, -0.085), fontsize=9, frameon=False, ncol=6, columnspacing=1.0, handletextpad=0.4)
fig.text(0.05, 0.075, "파랑 = 내부 노드가 이웃 4개와 잇는 전도 저항.  빨강 = 표면 노드의 전도 저항(공기 쪽 연결은 오른쪽 회로도).", fontsize=10, color=INK)

# ---------------- 오른쪽: 회로도 ----------------
ax = axR; ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
def res(p, q, col=INK, label=None, off=(0, 0), fs=9, w=0.26, n=6):
    p, q = np.array(p, float), np.array(q, float); d = q - p; L = np.hypot(*d); u = d / L; v = np.array([-u[1], u[0]])
    a, b = 0.32 * L, 0.68 * L; pts = [p, p + u * a]
    for k in range(n): pts.append(p + u * (a + (b - a) * (k + 0.5) / n) + v * w * (1 if k % 2 == 0 else -1))
    pts += [p + u * b, q]; pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.6, zorder=2)
    if label: c = (p + q) / 2 + np.array(off); ax.text(c[0], c[1], label, fontsize=fs, color=col, ha="center", va="center", zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none"))
def cap(p, direction=(0.7, 0.7), col=GREEN, label=None):
    p = np.array(p, float); u = np.array(direction, float); u /= np.hypot(*u); v = np.array([-u[1], u[0]]); a = p + u * 0.55; b = p + u * 0.72
    ax.plot([p[0], a[0]], [p[1], a[1]], "-", color=col, lw=1.4, zorder=2); ax.plot([a[0] - v[0] * 0.26, a[0] + v[0] * 0.26], [a[1] - v[1] * 0.26, a[1] + v[1] * 0.26], "-", color=col, lw=2.2, zorder=2)
    ax.plot([b[0] - v[0] * 0.26, b[0] + v[0] * 0.26], [b[1] - v[1] * 0.26, b[1] + v[1] * 0.26], "-", color=col, lw=2.2, zorder=2)
    e = b + u * 0.22; ax.plot([e[0] - v[0] * 0.3, e[0] + v[0] * 0.3], [e[1] - v[1] * 0.3, e[1] + v[1] * 0.3], "-", color=col, lw=1.4, zorder=2)
    if label: ax.text(e[0] + 0.1, e[1] + 0.2, label, fontsize=8.5, color=col, ha="left", va="bottom")
def node(p, lab, col=INK, r=0.26): ax.add_patch(Circle(p, r, fc="white", ec=col, lw=2, zorder=5)); ax.text(p[0], p[1], lab, fontsize=8, ha="center", va="center", color=col, zorder=6)
xs, ys = [1.6, 4.6, 7.6], [8.6, 6.4, 4.6]
P = {(r, c): (xs[c], ys[r]) for r in range(3) for c in range(3)}
ax.add_patch(Rectangle((0.2, 3.7), 9.0, 6.1, fc="#f1f2f4", ec="none", zorder=0)); pass
for r in range(3):
    for c in range(2): res(P[r, c], P[r, c + 1], label="K" if (r, c) == (1, 0) else None, off=(0, 0.42), fs=10)
for c in range(3):
    for r in range(2): res(P[r, c], P[r + 1, c], label="K" if (r, c) == (0, 1) else None, off=(0.42, 0), fs=10)
for (r, c), p in P.items(): cap(p, (0.75, 0.55)); node(p, "T")
# 표면(실내) 연결
ax.add_patch(Rectangle((0.2, 0.2), 11.6, 1.5, fc="#eaf2fb", ec="none", zorder=0)); ax.add_patch(Rectangle((0.2, 1.7), 11.6, 2.0, fc="white", ec="none", zorder=0))
ax.plot([0.5, 11.5], [0.75, 0.75], "-", color=BLUE, lw=3, zorder=1); ax.text(11.5, 0.45, "실내 공기온도 T_air(t)", fontsize=10.5, color=BLUE, ha="right", va="top")
ax.plot([0.5, 11.5], [2.35, 2.35], "-", color=ORANGE, lw=3, zorder=1); ax.text(11.7, 2.62, "복사 흑체 노드 J\n(표면마다 하나, 서로 별 모양 연결)", fontsize=9.5, color=ORANGE, ha="right", va="bottom")
res(P[2, 0], (xs[0], 0.75), col=BLUE, label="1/(hA)\n대류", off=(0.95, 0), fs=9.5)
ax.plot(xs[0], 2.35, "s", ms=0)  # 대류선은 J 선을 건너뜀(연결 아님)
res(P[2, 1], (xs[1], 2.35), col=ORANGE, label="복사\n(방사율 저항)", off=(1.1, 0), fs=9.5)
ax.plot(xs[1], 2.35, "o", ms=7, color=ORANGE, zorder=6)
res(P[2, 2], (xs[2], 0.75), col=BLUE, label="1/(hA)", off=(0.8, 0), fs=9.5)
res((10.4, 2.35), (10.4, 0.75), col=ORANGE, label="나머지\n(먼 곳)", off=(0.9, 0), fs=9); ax.plot(10.4, 2.35, "o", ms=7, color=ORANGE, zorder=6); ax.plot(10.4, 0.75, "o", ms=7, color=BLUE, zorder=6)
for xx in xs[0::2]: ax.plot(xx, 0.75, "o", ms=7, color=BLUE, zorder=6)
ax.text(5.9, 3.78, "표면 노드", fontsize=10, color=RED, ha="left", va="bottom"); ax.add_patch(Rectangle((0.7, 4.0), 7.9, 1.2, fc="none", ec=RED, lw=1.2, ls="--", zorder=1))
# 일사
ax.add_patch(FancyArrowPatch((1.6, 10.0), (1.6, 8.95), arrowstyle="-|>", mutation_scale=16, color=ORANGE, lw=2.2, zorder=4)); ax.text(1.85, 9.78, "일사 q_sol(t)와 외기 T_out(t)\n(바깥 표면 노드에 들어감)", fontsize=9.5, color=ORANGE, va="center")
# 범례와 식
ax.text(9.5, 9.35, "정리", fontsize=11, weight="bold", color=INK, va="top")
ax.text(9.5, 8.95, "K 전도 저항\n   λA/L, 이웃 노드 사이\nC 열용량\n   ρ c V, 노드마다\nh 대류계수 (시간별 입력)\n복사는 hrb 5.25, 방사율 0.9", fontsize=9, color=MUTED, va="top", linespacing=1.4)
ax.add_patch(Rectangle((9.45, 6.4), 0.25, 0.25, fc=GREEN, ec="none")); ax.text(9.8, 6.52, "녹색 = 열용량 C", fontsize=9, color=GREEN, va="center")
ax.text(9.45, 5.9, "※ 표면 노드는 모두 대류와\n   복사를 둘 다 가진다.\n   그림은 한 노드씩만 그렸다.", fontsize=8.5, color=MUTED, va="top")
ax.set_title("회로도 (회색 = 벽체 안쪽, 아래 = 실내 공기)", loc="left", fontsize=13.5, color=INK)
fig.text(0.5, 0.03, "노드마다 이 식 하나:   C dT/dt  =  Σ K (T이웃 - T)  +  h A (T_air - T)  +  복사  +  일사        시간 적분: 1시간 간격 Crank-Nicolson, 첫 시각은 dT/dt = 0 (정상상태)",
         ha="center", fontsize=13, color=INK, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID))
fig.savefig(os.path.join(ROOT, "results", "fig_rc_network.png"), dpi=130, facecolor=SURF); plt.close(fig); print("ok")
