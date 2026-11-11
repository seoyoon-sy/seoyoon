"""TRISCO 입력(.trc) 읽기 (정상상태 비교용). 형식은 .vtr 와 거의 같고 BC 색은 tair, hcon, trad 를 직접 가진다.
반환: (Model, {zone id: (공기온도, 대류계수)}, hrb).  재료의 밀도와 비열은 정상상태에서 쓰이지 않으므로 1 로 둔다."""
import numpy as np
from .model import Material, Zone, Model


def read_trc(path):
    raw = open(path, "rb").read()
    try: txt = raw.decode("utf-8")
    except UnicodeDecodeError: txt = raw.decode("cp949", errors="replace")
    xw, yw, zw, blok, col, par = [], [], [], [], {}, {}
    cur, region = None, "hdr"
    for l in txt.replace("\r", "").split("\n"):
        if "=" not in l: continue
        k, v = l.split("=", 1); k = k.strip(); v = v.strip()
        if k == "xwid": xw.append(float(v))
        elif k == "ywid": yw.append(float(v))
        elif k == "zwid": zw.append(float(v))
        elif k == "blok": blok.append([int(t) for t in v.split()])
        elif k == "vwpt": region = "mat"       # 여기서부터 색별 물성과 경계조건
        elif k == "colr": cur = int(v); col.setdefault(cur, {})
        elif region == "mat" and cur is not None and k in ("lamb", "epsi", "tair", "hcon", "trad"): col[cur][k] = float(v)
        elif k == "hrbl": par["hrbl"] = float(v)
    return xw, yw, zw, blok, col, par


def model_from_trc(path):
    xw, yw, zw, blok, col, par = read_trc(path)
    xe = np.r_[0, np.cumsum(xw)] / 1000.0; ye = np.r_[0, np.cumsum(yw)] / 1000.0
    nx, ny = len(xw), len(yw)
    raw = np.zeros((nx, ny), int)
    for b in blok: raw[b[2]:b[3], b[4]:b[5]] = b[1]
    cid = np.zeros((nx, ny), int); mats, zones, zv = {}, {}, {}
    for c, p in col.items():
        s = raw == c
        if "lamb" in p:
            mats[c] = Material(f"colr{c}", p["lamb"], 1.0, 1.0, p.get("epsi", 0.9)); cid[s] = c
        elif "tair" in p:
            zones[c] = Zone("outdoor" if p["tair"] < 10 else "indoor", f"colr{c}"); cid[s] = -c
            zv[c] = (p["tair"], p["hcon"])
    m = Model(xe=xe, ye=ye, zt=sum(zw) / 1000.0, cid=cid, materials=mats, zones=zones, hrb=par.get("hrbl", 5.1))
    return m, zv
