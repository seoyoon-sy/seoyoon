"""접합부 없는 평면 벽체(비열교부 기준) 모델. 폭 2 m 20칸, 양옆 단열, 아래 = 실외 경계, 위 = 실내 경계.
출력은 폭 가운데 노드(가장자리 노드의 대류 면적 배분 영향이 닿지 않음)."""
import numpy as np
from .model import Zone, Model


def planar_cells(layers, h_max_mm=20.0, min_cells=1):
    """층마다 같은 두께로 쪼갠 칸 수: 칸 두께가 h_max_mm 이하가 되는 최소 개수 (VOLTRA .vtr 평면 벽체 생성기와 같은 규칙)."""
    return [max(min_cells, int(np.ceil(t / h_max_mm - 1e-9))) for _, t in layers]


def planar_model(layers, materials, hrb=5.25, h_cell_mm=5.0, zt=1.0, width=2.0, nx=20, cells=None):
    """layers: [(재료이름, 두께 mm)] 실외 -> 실내. materials: {이름: Material}. 반환 Model (outputs: 'surf', 'back').
    cells: 층별 칸 수 목록(없으면 h_cell_mm 이하로 쪼갬, 층마다 최소 2칸)."""
    ids = {nm: i for i, nm in enumerate(materials, start=1)}
    ye, rows = [0.0, 1.0], [-1]
    for k, (nm, t) in enumerate(layers):
        n = cells[k] if cells is not None else max(2, int(np.ceil(t / h_cell_mm)))
        for _ in range(n):
            ye.append(ye[-1] + t / 1000.0 / n); rows.append(ids[nm])
    iy_surf = len(ye) - 1
    ye.append(ye[-1] + 1.0); rows.append(-2)
    cid = np.tile(np.array(rows)[None, :], (nx, 1))
    m = Model(xe=np.linspace(0.0, width, nx + 1), ye=np.array(ye), zt=zt, cid=cid,
              materials={i: materials[nm] for nm, i in ids.items()},
              zones={1: Zone("outdoor"), 2: Zone("indoor")}, hrb=hrb)
    # 마감재(마지막 층) 뒷면 = 마지막 층 두께만큼 아래
    last_n = cells[-1] if cells is not None else max(2, int(np.ceil(layers[-1][1] / h_cell_mm)))
    m.outputs = {"surf": (nx // 2, iy_surf), "back": (nx // 2, iy_surf - last_n)}
    return m
