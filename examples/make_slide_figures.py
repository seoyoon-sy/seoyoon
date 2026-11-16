"""발표용 설명 그림 3장: (1) 경량 RC 대 이 코드의 노드 구성, (2) 노드 하나 확대, (3) 계산 흐름.
출력: results/slides/slide1_rc_vs_grid.png, slide2_one_node.png, slide3_flow.png
사용: python examples/make_slide_figures.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Polygon
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})
INK, MUTED, GRID, SURF = "#1f2328", "#59636e", "#e4e6e8", "#fcfcfb"
BLUE, RED, GREEN, ORANGE = "#2f6fd0", "#d1453b", "#2b9348", "#e8590c"
COL = dict(concrete="#c2c5c9", eps="#f6e7a1", block="#dcb892", gypsum="#f4f4f1", xps="#7cc6c3")
OUT = os.path.join(ROOT, "results", "slides"); os.makedirs(OUT, exist_ok=True)

def zigzag(ax, x0, y0, x1, y1, n=6, amp=0.12, color=INK, lw=1.8):
    """저항 기호: (x0,y0) -> (x1,y1) 지그재그"""
    t = np.linspace(0, 1, 2 * n + 1); d = np.array([x1 - x0, y1 - y0]); L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]])
    pts = [np.array([x0, y0]) + d * ti + nrm * (amp * (1 if i % 2 else -1) if 0 < i < 2 * n else 0) * 1 for i, ti in enumerate(t)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, solid_joinstyle="miter", zorder=3)

def cap(ax, x, y, h=0.9, w=0.45, color=INK):
    """커패시터 기호: 노드 (x,y)에서 아래로 내려가는 C"""
    ax.plot([x, x], [y, y - h * 0.45], color=color, lw=1.8, zorder=3)
    ax.plot([x - w / 2, x + w / 2], [y - h * 0.45] * 2, color=color, lw=2.4, zorder=3)
    ax.plot([x - w / 2, x + w / 2], [y - h * 0.62] * 2, color=color, lw=2.4, zorder=3)
    ax.plot([x, x], [y - h * 0.62, y - h], color=color, lw=1.8, zorder=3)
    ax.plot([x - w / 2, x + w / 2], [y - h] * 2, color=MUTED, lw=1.6, zorder=3)

def node(ax, x, y, r=0.17, fc=BLUE, ec="white", z=5):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.5, zorder=z))

def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=12, lw=1.5, color=INK, weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight, zorder=3, linespacing=1.35)

def arrow(ax, p, q, color=INK, lw=1.8, style="-|>", ms=14, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=4, ls=ls))

# ------------------------------------------------------------------ 그림 1
def slide1():
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.945, "같은 RC인데 노드를 잡는 방식이 다르다", fontsize=25, weight="bold", color=INK)
    fig.text(0.03, 0.905, "경량 RC는 벽을 한 줄로 보고 노드를 몇 개만 둔다. 이 코드는 접합부 단면을 격자로 쪼개 노드 약 2000개를 둔다.", fontsize=13.5, color=MUTED)
    # ---------- 왼쪽: 경량 RC
    axL = fig.add_axes([0.03, 0.30, 0.43, 0.56]); axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
    axL.add_patch(FancyBboxPatch((0, 0), 10, 10, boxstyle="round,pad=0,rounding_size=0.2", fc="white", ec=GRID, lw=1.5))
    axL.text(0.3, 9.45, "경량 RC (보통 1차원)", fontsize=17, weight="bold", color=INK)
    # 벽 단면(가로 띠): 실외 -> 실내
    xs = [(1.0, 3.2, "concrete", "콘크리트"), (3.2, 6.4, "eps", "단열재"), (6.4, 7.6, "block", "블록"), (7.6, 8.0, "gypsum", "")]
    for a, b, k, lab in xs:
        axL.add_patch(Rectangle((a, 6.6), b - a, 1.5, fc=COL[k], ec=MUTED, lw=1)); axL.text((a + b) / 2, 7.35, lab, ha="center", va="center", fontsize=11, color=INK)
    axL.text(0.35, 7.35, "실외", ha="center", va="center", fontsize=11, color=MUTED); axL.text(8.55, 7.35, "실내", ha="center", va="center", fontsize=11, color=MUTED)
    axL.annotate("", xy=(9.5, 5.95), xytext=(0.5, 5.95), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
    axL.text(5.0, 6.15, "열은 두께 방향 한 줄로만 흐른다", ha="center", va="bottom", fontsize=12, color=RED, weight="bold")
    # 회로
    y = 3.9
    axL.plot([0.6, 9.4], [y, y], color=INK, lw=1.8, zorder=2)
    for x in (2.1, 5.0, 7.9): pass
    xs_r = [(0.6, 2.0), (2.0, 3.6), (3.6, 6.4), (6.4, 8.0), (8.0, 9.4)]
    axL.add_patch(Rectangle((0.6, y - 0.4), 8.8, 0.8, fc="white", ec="white", zorder=2))
    zigzag(axL, 0.9, y, 2.1, y, n=4, amp=0.2); node(axL, 2.5, y, 0.22, BLUE); cap(axL, 2.5, y - 0.22, h=1.1)
    zigzag(axL, 2.9, y, 4.6, y, n=5, amp=0.2); node(axL, 5.0, y, 0.22, BLUE); cap(axL, 5.0, y - 0.22, h=1.1)
    zigzag(axL, 5.4, y, 7.1, y, n=5, amp=0.2); node(axL, 7.5, y, 0.22, BLUE); cap(axL, 7.5, y - 0.22, h=1.1)
    zigzag(axL, 7.9, y, 9.1, y, n=4, amp=0.2)
    axL.plot([0.6, 0.9], [y, y], color=INK, lw=1.8); axL.plot([9.1, 9.4], [y, y], color=INK, lw=1.8)
    axL.text(0.6, y + 0.55, "T실외", ha="center", fontsize=11, color=INK); axL.text(9.4, y + 0.55, "T실내", ha="center", fontsize=11, color=INK)
    axL.text(2.5, y + 0.55, "T1", ha="center", fontsize=11, color=BLUE, weight="bold"); axL.text(5.0, y + 0.55, "T2", ha="center", fontsize=11, color=BLUE, weight="bold"); axL.text(7.5, y + 0.55, "T3", ha="center", fontsize=11, color=BLUE, weight="bold")
    axL.text(5.0, 1.15, "노드 = 층 하나(또는 둘)를 한 덩어리로 본 온도 1개\n저항 R, 용량 C는 층 두께와 물성으로 계산", ha="center", fontsize=12, color=INK, linespacing=1.5)
    axL.text(5.0, 0.35, "열교(접합부)는 따로 계산한 보정값(Ψ)으로 더한다", ha="center", fontsize=12, color=RED, weight="bold")
    # ---------- 오른쪽: 이 코드
    axR = fig.add_axes([0.50, 0.30, 0.47, 0.56]); axR.axis("off")
    axR.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec=GRID, lw=1.5, transform=axR.transAxes))
    axR.text(0.02, 0.945, "이 코드 (접합부 단면, 2차원)", fontsize=17, weight="bold", color=INK, transform=axR.transAxes)
    axD = fig.add_axes([0.512, 0.315, 0.30, 0.50]); axD.set_aspect("equal"); axD.axis("off")
    # 접합부 단면: x(벽 길이 방향) -520..520, y(실외 0 -> 실내 480, 내벽은 위로 650만 표시)
    th = [("concrete", 0, 200), ("eps", 200, 380), ("block", 380, 470), ("gypsum", 470, 480)]
    X0, X1 = -450, 600; P = 90; top = 480 + 650
    for k, a_, b_ in th: axD.add_patch(Rectangle((X0, a_), X1 - X0, b_ - a_, fc=COL[k], ec="none"))
    axD.add_patch(Rectangle((-P, 480), 2 * P, top - 480, fc=COL["concrete"], ec="none"))
    for x0 in (-P - 10, P): axD.add_patch(Rectangle((x0, 480), 10, 170, fc=COL["xps"], ec="#168a85", lw=0.6))
    def edges(a_, b_, focus, h0=10, g=0.25, hmax=50):
        pts = [a_]; x = a_
        while x < b_ - 1e-6:
            d = abs(x - focus); h = min(hmax, h0 + g * d); x = min(b_, x + h); pts.append(x)
        return np.array(pts)
    right = P + np.cumsum(np.r_[0, np.diff(edges(P, X1, P))]); right = np.r_[P, edges(P, X1, P)[1:]]
    ex = np.unique(np.r_[-right, np.linspace(-P, P, 10), right])
    ey = np.unique(np.r_[np.arange(0, 480 + 1, 40), edges(480, top, 480, h0=8, g=0.3, hmax=50)])
    for x in ex: axD.plot([x, x], [0, 480], color="white", lw=0.35, alpha=0.9, zorder=2)
    for y in ey: axD.plot([X0, X1] if y <= 480 else [-P, P], [y, y], color="white", lw=0.35, alpha=0.9, zorder=2)
    xx, yy = np.meshgrid(ex, ey[ey <= 480]); axD.plot(xx.ravel(), yy.ravel(), ".", ms=1.8, color=BLUE, zorder=3)
    xx2, yy2 = np.meshgrid(np.linspace(-P, P, 10), ey[ey > 480]); axD.plot(xx2.ravel(), yy2.ravel(), ".", ms=1.8, color=BLUE, zorder=3)
    # 출력 노드: 실내 마감면(y=480)에서 내벽 면으로부터 0 ~ 500 mm (그림은 520 mm까지만 그려서 500 mm 점은 오른쪽 끝에 가깝다)
    xo = [P + d for d in (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)]
    axD.plot(xo, [480] * len(xo), "o", ms=6, mfc=RED, mec="white", mew=0.8, zorder=6)
    axD.annotate("", xy=(P + 500, 535), xytext=(P + 25, 535), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4))
    axD.text(P + 60, 560, "표면온도를 뽑는 위치\n내벽 면에서 0 ~ 500 mm", ha="left", va="bottom", fontsize=10.5, color=RED, weight="bold", linespacing=1.3)
    axD.set_xlim(X0 - 15, X1 + 15); axD.set_ylim(-70, top + 30)
    # 층 이름은 띠 안에 직접 쓴다
    for x_, y_, t_ in ((-330, 100, "콘크리트"), (-330, 290, "단열재 (EPS)"), (-330, 425, "블록")):
        axD.text(x_, y_, t_, ha="center", va="center", fontsize=10.5, color=INK, zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    axD.annotate("석고 10 mm", xy=(-380, 475), xytext=(-330, 560), fontsize=10.5, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    axD.annotate("내벽 (콘크리트)", xy=(-P, 900), xytext=(-330, 900), fontsize=10.5, color=INK, ha="center", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    axD.annotate("XPS 판", xy=(-P - 5, 600), xytext=(-330, 690), fontsize=10.5, color="#168a85", ha="center", va="center", weight="bold", arrowprops=dict(arrowstyle="-", color="#168a85", lw=0.9))
    axD.text(X0, -45, "실외 (아래)", fontsize=10.5, color=MUTED, ha="left", va="center"); axD.text(X1, -45, "실내는 위", fontsize=10.5, color=MUTED, ha="right", va="center")
    # 오른쪽 설명
    axT = fig.add_axes([0.822, 0.325, 0.15, 0.48]); axT.axis("off"); axT.set_xlim(0, 1); axT.set_ylim(0, 1)
    items = [(0.89, BLUE, "파란 점 = 노드", "약 1700 ~ 2000개.\n점마다 온도 1개와\n열용량 C 1개"), (0.63, INK, "이웃과 저항으로 연결", "위, 아래, 좌, 우로\n열이 흐른다 (2차원)"),
             (0.40, RED, "빨간 점 = 표면온도", "열교 바로 옆(0 mm)부터\n500 mm까지 12곳"), (0.17, "#168a85", "접합부 근처는 촘촘", "5 mm 간격에서 시작해\n멀어지면 50 mm까지")]
    for y, c, h, s_ in items:
        axT.add_patch(Circle((0.04, y + 0.045), 0.035, fc=c, ec="none", transform=axT.transAxes))
        axT.text(0.12, y + 0.055, h, fontsize=11, weight="bold", color=INK, va="center"); axT.text(0.12, y - 0.005, s_, fontsize=10, color=MUTED, va="top", linespacing=1.35)
    # ---------- 아래 비교 표
    axB = fig.add_axes([0.03, 0.035, 0.94, 0.235]); axB.axis("off"); axB.set_xlim(0, 100); axB.set_ylim(0, 10)
    cols = [0, 17, 58]; hdr = ["", "경량 RC", "이 코드"]; rows = [("노드 수", "몇 개 (왼쪽 그림은 3개)", "약 1700 ~ 2000개 (접합부 단면 전체)"), ("열이 흐르는 방향", "벽 두께 한 방향", "두께와 벽 길이 방향 둘 다 (접합부 모서리 효과가 자동으로 들어감)"),
                                                          ("열교(접합부)", "보정값을 따로 더함", "형상 자체를 풀어서 표면온도가 위치마다 나옴"), ("출력", "벽 표면온도 1개", "접합부 0 ~ 500 mm 12곳의 표면온도 시계열"),
                                                          ("저항 R, 용량 C", "층별로 계산 (보통 일부 보정)", "치수와 물성으로 전부 자동 계산, 맞춘 계수 없음")]
    for c, h in zip(cols, hdr): axB.text(c + 1, 9.2, h, fontsize=13, weight="bold", color=BLUE if h == "이 코드" else INK)
    axB.plot([0, 100], [8.4, 8.4], color=INK, lw=1.2)
    for i, (a, b, c) in enumerate(rows):
        y = 7.3 - i * 1.55
        axB.text(1, y, a, fontsize=11.5, weight="bold", color=INK, va="center"); axB.text(18, y, b, fontsize=11.5, color=INK, va="center"); axB.text(59, y, c, fontsize=11.5, color=INK, va="center")
        axB.plot([0, 100], [y - 0.77, y - 0.77], color=GRID, lw=0.8)
    fig.savefig(os.path.join(OUT, "slide1_rc_vs_grid.png"), dpi=130, facecolor=SURF); plt.close(fig)


# ------------------------------------------------------------------ 그림 2
def slide2():
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.945, "노드 하나를 확대하면: 작은 RC 한 조각", fontsize=25, weight="bold", color=INK)
    fig.text(0.03, 0.905, "격자의 교점마다 같은 조각이 하나씩 있고, 이웃 조각끼리 저항으로 이어진다. 표면 노드는 공기와 다른 표면에도 연결된다.", fontsize=13.5, color=MUTED)
    ax = fig.add_axes([0.03, 0.04, 0.60, 0.84]); ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 12, 10, boxstyle="round,pad=0,rounding_size=0.2", fc="white", ec=GRID, lw=1.5))
    # 고체(아래)와 공기(위)
    ax.add_patch(Rectangle((0.3, 0.3), 11.4, 4.9, fc="#dfe2e6", ec="none")); ax.text(11.5, 0.6, "벽 (고체)", ha="right", fontsize=12, color=MUTED)
    ax.add_patch(Rectangle((0.3, 5.2), 11.4, 4.5, fc="#f2f8ff", ec="none")); ax.text(11.5, 9.35, "실내 공기", ha="right", fontsize=12, color=MUTED)
    ax.plot([0.3, 11.7], [5.2, 5.2], color=INK, lw=2.2)
    cx, cy = 5.2, 5.2
    # 이웃 노드와 전도 저항 (왼쪽, 오른쪽, 아래)
    for (nx_, ny_, lab) in ((2.2, 5.2, "왼쪽 노드"), (8.2, 5.2, "오른쪽 노드"), (5.2, 2.2, "아래 노드")):
        node(ax, nx_, ny_, 0.2, "#8fb3e8")
        ax.text(nx_, ny_ + (-0.55 if ny_ == 5.2 else -0.55), lab, ha="center", fontsize=11, color=MUTED)
    zigzag(ax, 2.45, 5.2, 4.95, 5.2, n=5, amp=0.18); zigzag(ax, 5.45, 5.2, 7.95, 5.2, n=5, amp=0.18); zigzag(ax, 5.2, 2.45, 5.2, 4.95, n=5, amp=0.18)
    ax.text(3.7, 5.55, "전도 R", ha="center", fontsize=12, color=INK, weight="bold"); ax.text(6.7, 5.55, "전도 R", ha="center", fontsize=12, color=INK, weight="bold"); ax.text(4.85, 2.95, "전도 R", ha="right", fontsize=12, color=INK, weight="bold")
    node(ax, cx, cy, 0.3, BLUE)
    ax.text(cx + 0.4, 6.3, "표면 노드 (온도 T)", ha="center", fontsize=12.5, color=BLUE, weight="bold")
    # 제어체적(이 노드가 맡는 영역)
    ax.add_patch(Rectangle((3.7, 3.7), 3.0, 1.5, fc="none", ec=BLUE, lw=1.4, ls="--")); ax.text(3.55, 4.0, "이 노드가 맡는 영역\n(표면이라 아래 반쪽)", ha="right", fontsize=10.5, color=BLUE, linespacing=1.3)
    # 열용량: 노드에서 오른쪽 아래로 빼서 C 기호
    ax.plot([cx + 0.2, 6.05], [cy - 0.2, 4.55], color=INK, lw=1.6); ax.plot([6.05, 6.05], [4.55, 4.35], color=INK, lw=1.6)
    ax.plot([5.75, 6.35], [4.35, 4.35], color=INK, lw=2.6); ax.plot([5.75, 6.35], [4.1, 4.1], color=INK, lw=2.6); ax.plot([6.05, 6.05], [4.1, 3.95], color=INK, lw=1.6); ax.plot([5.75, 6.35], [3.95, 3.95], color=MUTED, lw=1.6)
    ax.text(6.9, 4.22, "열용량 C", ha="left", fontsize=12, color=INK, weight="bold")
    # 공기 쪽: 대류 저항과 공기 노드
    node(ax, 3.4, 8.6, 0.22, "#6bb6ff"); ax.text(3.4, 9.15, "실내 공기 T", ha="center", fontsize=11.5, color=INK, weight="bold")
    ax.plot([3.4, 3.4], [8.38, 7.9], color=INK, lw=1.6); zigzag(ax, 3.4, 7.9, 3.4, 6.5, n=4, amp=0.18); ax.plot([3.4, cx - 0.2], [6.5, cy + 0.2], color=INK, lw=1.6)
    ax.text(3.0, 7.2, "대류\n1/h", ha="right", va="center", fontsize=12, color=INK, weight="bold", linespacing=1.3)
    # 복사: 다른 표면 (예: 내벽 면)
    node(ax, 8.2, 8.6, 0.22, "#f0a5a0"); ax.text(8.2, 9.15, "마주 보는 다른 표면 T", ha="center", fontsize=11.5, color=INK, weight="bold")
    ax.plot([8.2, 8.2], [8.38, 7.9], color=RED, lw=1.6); zigzag(ax, 8.2, 7.9, 8.2, 6.5, n=4, amp=0.18, color=RED); ax.plot([8.2, cx + 0.2], [6.5, cy + 0.2], color=RED, lw=1.6)
    ax.text(8.6, 7.2, "복사\n(형상계수로 계산)", ha="left", va="center", fontsize=12, color=RED, weight="bold", linespacing=1.3)
    # 일사 (실외 쪽만 해당)
    # ---------- 오른쪽: 식과 표면 규칙
    axE = fig.add_axes([0.65, 0.04, 0.32, 0.84]); axE.set_xlim(0, 10); axE.set_ylim(0, 10); axE.axis("off")
    axE.add_patch(FancyBboxPatch((0, 0), 10, 10, boxstyle="round,pad=0,rounding_size=0.2", fc="white", ec=GRID, lw=1.5))
    axE.text(0.4, 9.35, "노드 하나의 식", fontsize=16, weight="bold", color=INK)
    axE.text(0.4, 8.15, r"$C_i\,\frac{dT_i}{dt}=\sum_j G_{ij}(T_j-T_i)+hA\,(T_{air}-T_i)+Q_{rad}+Q_{sun}$", fontsize=14.5, color=INK)
    axE.text(0.4, 7.15, "용량 x 온도 변화 = 전도 + 대류 + 복사(rad) + 일사(sun)", fontsize=11.5, color=MUTED)
    axE.text(0.4, 6.75, "일사 Q_sun은 실외 쪽 표면 노드에만 들어간다", fontsize=11, color=ORANGE)
    axE.text(0.4, 6.25, "이런 식이 노드마다 하나씩, 약 2000개", fontsize=12.5, color=INK, weight="bold")
    axE.text(0.4, 5.75, "묶어서 한 줄로 쓰면  C dT/dt = -(K + G) T + b", fontsize=12.5, color=BLUE, weight="bold")
    axE.plot([0.4, 9.6], [5.2, 5.2], color=GRID, lw=1.2)
    axE.text(0.4, 4.65, "표면의 대류와 복사를 노드에 주는 규칙", fontsize=15, weight="bold", color=INK)
    axE.text(0.4, 4.1, "(VOLTRA 해 파일을 분석해서 찾았다)", fontsize=11, color=MUTED)
    # 선분 그림
    axE.plot([1.4, 8.6], [2.35, 2.35], color=INK, lw=3.2); node(axE, 1.4, 2.35, 0.22, BLUE); node(axE, 8.6, 2.35, 0.22, BLUE)
    axE.text(1.4, 1.75, "노드 a", ha="center", fontsize=11.5, color=BLUE, weight="bold"); axE.text(8.6, 1.75, "노드 b", ha="center", fontsize=11.5, color=BLUE, weight="bold")
    axE.plot([3.2, 3.2], [2.1, 2.6], color=RED, lw=2.2); axE.plot([6.8, 6.8], [2.1, 2.6], color=RED, lw=2.2)
    axE.annotate("", xy=(3.2, 3.15), xytext=(3.2, 2.7), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8)); axE.annotate("", xy=(6.8, 3.15), xytext=(6.8, 2.7), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    axE.text(3.2, 3.45, "a쪽 반쪽의\n중점 온도", ha="center", fontsize=10.5, color=RED, linespacing=1.25); axE.text(6.8, 3.45, "b쪽 반쪽의\n중점 온도", ha="center", fontsize=10.5, color=RED, linespacing=1.25)
    axE.text(5.0, 0.95, r"$T_q=(3T_a+T_b)/4$   (끝 노드에서 L/4 지점)", ha="center", fontsize=12.5, color=RED)
    axE.text(5.0, 0.35, "그 온도로 대류, 복사를 계산해 가까운 노드에 준다", ha="center", fontsize=11.5, color=INK)
    fig.savefig(os.path.join(OUT, "slide2_one_node.png"), dpi=130, facecolor=SURF); plt.close(fig)

# ------------------------------------------------------------------ 그림 3
def slide3():
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.945, "계산은 이렇게 돌아간다", fontsize=25, weight="bold", color=INK)
    fig.text(0.03, 0.905, "준비는 한 번, 시간 계산은 1시간씩 744번 반복한다.", fontsize=13.5, color=MUTED)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.85]); ax.set_xlim(0, 16); ax.set_ylim(0, 9.5); ax.axis("off")
    # 1 입력
    box(ax, 0.2, 6.55, 3.6, 2.4, "", fc="white", ec=BLUE, lw=2); ax.text(2.0, 8.6, "1. 입력", ha="center", fontsize=15, weight="bold", color=BLUE)
    ax.text(0.45, 8.3, "벽 구조: 층 두께, 내벽,\nXPS 판 두께와 돌출", fontsize=10.5, color=INK, va="top", linespacing=1.3)
    ax.text(0.45, 7.45, "재료: 열전도율, 밀도, 비열\n날씨: 실외, 실내온도, 일사,\n대류계수 (시간별 744개)", fontsize=10.5, color=INK, va="top", linespacing=1.3)
    arrow(ax, (3.9, 7.75), (4.7, 7.75), lw=2.2)
    # 2 격자
    box(ax, 4.8, 6.55, 3.6, 2.4, "", fc="white", ec=BLUE, lw=2); ax.text(6.6, 8.6, "2. 격자와 재료 배치", ha="center", fontsize=15, weight="bold", color=BLUE)
    ax.text(5.05, 7.95, "접합부 단면을 칸으로 나눔\n(5 mm에서 시작, 최대 50 mm)\n칸마다 재료를 지정", fontsize=11, color=INK, va="top", linespacing=1.4)
    ax.text(5.05, 6.85, "노드 약 1700 ~ 2000개", fontsize=11.5, color=BLUE, weight="bold", va="center")
    arrow(ax, (8.5, 7.75), (9.3, 7.75), lw=2.2)
    # 3 행렬
    box(ax, 9.4, 6.55, 6.4, 2.4, "", fc="white", ec=BLUE, lw=2); ax.text(12.6, 8.6, "3. 행렬 만들기 (한 번)", ha="center", fontsize=15, weight="bold", color=BLUE)
    ax.text(9.65, 8.0, "C  노드 열용량 (용량 C)", fontsize=11.5, color=INK, va="top"); ax.text(9.65, 7.5, "K  전도 + 표면 복사 연결 (저항 R의 역수)", fontsize=11.5, color=INK, va="top")
    ax.text(9.65, 7.0, "G(t), b(t)  대류 연결과 입력 (시간마다 값이 바뀜)", fontsize=11.5, color=INK, va="top")
    arrow(ax, (12.6, 6.45), (12.6, 5.95), lw=2.2)
    # 4 정상상태
    box(ax, 9.4, 4.65, 6.4, 1.2, "", fc="white", ec=INK, lw=2); ax.text(9.65, 5.55, "4. 시작: 0시는 정상상태로 푼다", fontsize=13.5, weight="bold", color=INK, va="center")
    ax.text(9.65, 5.0, r"$(K+G_0)\,T_0=b_0$   (시간 항 없음)", fontsize=13, color=INK, va="center")
    arrow(ax, (12.6, 4.55), (12.6, 4.05), lw=2.2)
    # 5 시간 루프
    box(ax, 0.2, 0.3, 15.6, 3.7, "", fc="#f7faff", ec=BLUE, lw=2.4, ls="--")
    ax.text(0.5, 3.65, "5. 시간 루프: k = 1, 2, ... 743  (Δt = 1시간)", fontsize=15, weight="bold", color=BLUE)
    ax.text(0.5, 3.0, "앞 시각의 온도 T(k-1)에서 다음 시각의 온도 T(k)를 구한다 (Crank-Nicolson)", fontsize=12, color=INK)
    ax.text(0.5, 2.15, r"$\left(\frac{C}{\Delta t}+\frac{1}{2}(K+G_k)\right)T_k=\left(\frac{C}{\Delta t}-\frac{1}{2}(K+G_{k-1})\right)T_{k-1}+\frac{1}{2}(b_{k-1}+b_k)$", fontsize=17, color=INK, va="center")
    ax.text(0.5, 1.15, "왼쪽: 새 시각의 미지수   |   오른쪽: 이미 아는 값(앞 시각 온도, 앞뒤 시각 입력의 평균)", fontsize=11, color=MUTED)
    ax.text(0.5, 0.65, "행렬을 희소 LU로 풀어 노드 약 2000개의 온도를 한 번에 구한다. 대류 연결은 시간마다 바뀌어서 매 시각 행렬을 다시 만든다.", fontsize=11, color=MUTED)
    # 시간축 그림 (오른쪽 아래)
    for i in range(6):
        x = 9.6 + i * 1.05; ax.plot([x, x], [2.85, 3.15], color=INK, lw=1.6); ax.text(x, 2.55, f"{i}h" if i < 4 else ("..." if i == 4 else "743h"), ha="center", fontsize=10, color=INK)
    ax.plot([9.5, 15.2], [3.0, 3.0], color=INK, lw=1.2)
    for i in range(3): arrow(ax, (9.6 + i * 1.05 + 0.1, 3.4), (9.6 + (i + 1) * 1.05 - 0.1, 3.4), color=BLUE, lw=1.8, ms=11, rad=-0.4)
    # 6 출력과 활용
    box(ax, 0.2, 4.6, 8.6, 1.5, "", fc="white", ec=GREEN, lw=2); ax.text(0.45, 5.75, "6. 출력과 활용", fontsize=15, weight="bold", color=GREEN, va="center")
    ax.text(0.45, 5.15, "접합부 표면 12곳의 온도 시계열 (744시간)  →  야간(18 ~ 07시) 표면온도차 ΔTs", fontsize=11.5, color=INK, va="center")
    ax.text(0.45, 4.8, "→  선형열관류율(Ψ) 예측 학습모델의 입력 데이터", fontsize=11.5, color=INK, va="center")
    arrow(ax, (4.5, 4.05), (4.5, 4.55), lw=2.2, color=GREEN)
    fig.savefig(os.path.join(OUT, "slide3_flow.png"), dpi=130, facecolor=SURF); plt.close(fig)

if __name__ == "__main__":
    slide1(); slide2(); slide3(); print("저장 완료")
