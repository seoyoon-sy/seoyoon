"""VOLTRA 형상계수 파일(.vwf) 읽기. 면(face)은 격자 칸 x z 조각(3D)이고 vwfc 줄이 i->j 형상계수.
반환: dict(zones={zone:면수}, faces=[(id, 필드3개, 면적, 플래그, (x0,x1,y0,y1,z0,z1)mm)], F=scipy.sparse (nfac x nfac))"""
import numpy as np
import scipy.sparse as sp


def read_vwf(path):
    zones, faces, rows, cols, vals, cur = {}, [], [], [], [], None
    for l in open(path, "rb").read().decode("ascii", "replace").replace("\r", "").split("\n"):
        if l.startswith("zone="):
            a = l[5:].split(); zones[int(a[0])] = int(a[1])
        elif l.startswith("face="):
            a = l[5:].split(); cur = (int(a[0]), tuple(int(x) for x in a[1:4]), float(a[4]), float(a[5]))
        elif l.startswith("box "):
            b = tuple(float(x) for x in l[l.index("=") + 1:].split())
            faces.append((*cur, b))
        elif l.startswith("vwfc="):
            a = l[5:].split(); rows.append(int(a[0]) - 1); cols.append(int(a[1]) - 1); vals.append(float(a[2]))
    n = len(faces)
    F = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return dict(zones=zones, faces=faces, F=F)
