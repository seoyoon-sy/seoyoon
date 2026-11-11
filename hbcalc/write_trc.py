"""hbcalc 모델(T자 접합부) -> TRISCO .trc 입력. 템플릿(.trc, 예: trisco_inputs/TRY3_Plus.trc)의 격자, 블록, 재료 열전도율만 바꾸고
나머지(경계조건 tair, hcon, trad, 복사 설정)는 템플릿 그대로 둔다.
TRISCO 템플릿의 색 번호: 2 실외, 8 왼쪽 실내, 1 오른쪽 실내, 3 콘크리트, 4 EPS, 5 0.5B 시멘트 블록, 6 석고, 7 XPS 띠.
주의: 이 환경에는 TRISCO가 없어서 생성한 파일이 TRISCO에서 열리는지는 시험하지 못했다(템플릿 구조를 그대로 따름)."""
import re
import numpy as np

COLOR = {"outdoor": 2, "left": 8, "right": 1, "concrete": 3, "EPS": 4, "0.5B cement block": 5, "gypsum": 6, "XPS": 7}


def _read(path):
    raw = open(path, "rb").read()
    try: return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError: return raw.decode("cp949"), "cp949"


def write_trc(model, template_path, out_path):
    m = model.meta
    if not m: raise ValueError("model.meta 가 비어 있음: build_tjunction 으로 만든 모델만 지원")
    text, enc = _read(template_path)
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    xe, ye = model.xe * 1000.0, model.ye * 1000.0
    nx, ny = len(xe) - 1, len(ye) - 1
    nz = int(round(m["z_mm"] / 50.0))

    def ix(v):
        i = int(np.argmin(np.abs(xe - v))); assert abs(xe[i] - v) < 1e-3, ("x", v); return i

    def iy(v):
        i = int(np.argmin(np.abs(ye - v))); assert abs(ye[i] - v) < 1e-3, ("y", v); return i

    xL, xR, st = m["xL"], m["xR"], m["st"]
    c = COLOR
    blocks = [(c[m["partition_material"]], 0, nx, iy(m["Yo"]), ny)]
    if m["strip_material"]:
        for x0, x1 in ((xL, xL + st), (xR - st, xR)):
            blocks.append((c[m["strip_material"]], ix(x0), ix(x1), iy(m["y_strip0"]), iy(m["y_strip1"])))
    blocks += [(c["outdoor"], 0, nx, 0, iy(m["Yo"])), (c["left"], 0, ix(xL), iy(m["y_in"]), ny), (c["right"], ix(xR), nx, iy(m["y_in"]), ny)]
    L = m["layers"]
    for k in range(len(L) - 1, 0, -1):
        y0, y1 = iy(m["ycum"][k]), iy(m["ycum"][k + 1])
        blocks += [(c[L[k]], 0, ix(xL), y0, y1), (c[L[k]], ix(xR), nx, y0, y1)]
    lam = {c["EPS"]: None, c["XPS"]: None}
    for mt in model.materials.values():
        if mt.name in ("EPS", "XPS"): lam[c[mt.name]] = mt.lam
    out, i, cur, region = [], 0, None, False
    while i < len(lines):
        l = lines[i]
        if l.startswith("nxwi="):
            out += [f"nxwi={nx}"] + [f"xwid={np.format_float_positional(v, trim='-', precision=4)}" for v in np.diff(xe)]
            i += 1
            while lines[i].startswith("xwid="): i += 1
            continue
        if l.startswith("nywi="):
            out += [f"nywi={ny}"] + [f"ywid={np.format_float_positional(v, trim='-', precision=4)}" for v in np.diff(ye)]
            i += 1
            while lines[i].startswith("ywid="): i += 1
            continue
        if l.startswith("nblk="):
            out += [f"nblk={len(blocks)}"] + [f"blok={k} {cc} {a} {b} {d} {e} 0 {nz}" for k, (cc, a, b, d, e) in enumerate(blocks)]
            i += 1
            while lines[i].startswith("blok="): i += 1
            continue
        if l.startswith("qin2="): region = True
        if region and l.startswith("colr="): cur = int(l.split("=")[1])
        if region and cur in lam and lam[cur] is not None and l.startswith("lamb="):
            l = f"lamb={lam[cur]:g}"
        out.append(l); i += 1
    open(out_path, "wb").write(nl.join(out).encode(enc))
    return out_path
