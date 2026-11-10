"""호환용: VOLTRA .vtr 파일을 hbcalc Model 로 바꾼다 (회귀 시험과 기존 케이스 재현용).
핵심 계산 코드는 이 모듈을 쓰지 않는다."""
import numpy as np
from .model import Material, Zone, Model


def _text(path):
    raw = open(path, "rb").read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp949", errors="replace")


def model_from_vtr(path, hrb=None):
    xw, yw, zw, blok, onod = [], [], [], [], []
    colr, cur, region, par = {}, None, "hdr", {}
    for l in _text(path).replace("\r", "").split("\n"):
        if "=" not in l:
            continue
        k, v = l.split("=", 1); k = k.strip()
        if k == "xwid": xw.append(float(v))
        elif k == "ywid": yw.append(float(v))
        elif k == "zwid": zw.append(float(v))
        elif k == "blok": blok.append([int(t) for t in v.split()])
        elif k == "onod": onod.append([int(t) for t in v.split()])
        elif k == "rptf": region, cur = "mat", None
        elif k == "nnbc": region = "post"
        elif k == "colr": cur = int(v); colr.setdefault(cur, {})
        elif region == "hdr" and k == "type" and cur is not None: colr[cur]["type"] = int(v)
        elif region == "mat" and cur is not None: colr[cur][k] = v.strip()
        elif region in ("hdr", "post"): par[k] = v.strip()
    xe = np.r_[0, np.cumsum(xw)] / 1000.0
    ye = np.r_[0, np.cumsum(yw)] / 1000.0
    nx, ny = len(xw), len(yw)
    raw = np.zeros((nx, ny), int)
    for b in blok:
        raw[b[2]:b[3], b[4]:b[5]] = b[1]
    mats, zones = {}, {}
    cid = np.zeros((nx, ny), int)
    for c, p in colr.items():
        s = raw == c
        if "lamb" in p:
            mats[c] = Material(p.get("name", str(c)), float(p["lamb"]), float(p["ro"]), float(p["c"]),
                               float(p["epsi"]), float(p.get("refs", 0.1)))
            cid[s] = c
        else:
            kind = "outdoor" if p.get("solz", "0") == "1" else "indoor"
            zones[c] = Zone(kind, f"colr{c}")
            cid[s] = -c
    outputs = {}
    for o in onod:
        outputs[f"onod{o[0]}"] = (o[1], o[2])
    h = float(par.get("hrbl", 5.1)) if hrb is None else hrb
    return Model(xe=xe, ye=ye, zt=sum(zw) / 1000.0, cid=cid, materials=mats, zones=zones, outputs=outputs, hrb=h)
