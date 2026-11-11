"""접합부 없는 평면 벽체(비열교부 기준) 모델. 폭 0.2 m 2칸, 양옆 단열, 아래 = 실외 경계, 위 = 실내 경계."""
import numpy as np
from .model import Zone, Model


def planar_model(layers, materials, hrb=5.25, h_cell_mm=5.0, zt=1.0):
    """layers: [(재료이름, 두께 mm)] 실외 -> 실내. materials: {이름: Material}. 반환 Model (outputs: 'surf', 'back')."""
    ids = {nm: i for i, nm in enumerate(materials, start=1)}
    ye, rows = [0.0, 1.0], [-1]
    for nm, t in layers:
        n = max(2, int(np.ceil(t / h_cell_mm)))
        for _ in range(n):
            ye.append(ye[-1] + t / 1000.0 / n); rows.append(ids[nm])
    iy_surf = len(ye) - 1
    ye.append(ye[-1] + 1.0); rows.append(-2)
    cid = np.tile(np.array(rows)[None, :], (2, 1))
    m = Model(xe=np.array([0.0, 0.1, 0.2]), ye=np.array(ye), zt=zt, cid=cid,
              materials={i: materials[nm] for nm, i in ids.items()},
              zones={1: Zone("outdoor"), 2: Zone("indoor")}, hrb=hrb)
    # 마감재(마지막 층) 뒷면 = 마지막 층 두께만큼 아래
    last_n = max(2, int(np.ceil(layers[-1][1] / h_cell_mm)))
    m.outputs = {"surf": (1, iy_surf), "back": (1, iy_surf - last_n)}
    return m
