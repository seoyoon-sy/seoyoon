"""hbcalc 모델(T자 접합부) -> VOLTRA .vtr 파일. 템플릿(.vtr)의 격자, 블록, 재료 물성, 출력 노드만 바꾸고
나머지(계산 설정, 경계조건 파일 경로 등)는 템플릿 그대로 둔다.

제한: 템플릿과 같은 재료 구성(경계 영역 3개 + 재료 5개)과 같은 층 개수/순서만 지원한다.
      출력 위치는 12개(노드 24개)여야 한다 (템플릿의 rept 줄이 그 개수를 전제로 함).
주의: 이 환경에는 VOLTRA 가 없어서 생성한 파일이 VOLTRA 에서 열리는지는 시험하지 못했다.
      원본 템플릿 재생성이 원본과 의미상 같은지(tests/test_write_vtr.py)만 확인했다.
"""
import re
import numpy as np


def _read(path):
    raw = open(path, "rb").read()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp949"), "cp949"


def _fmt(v):
    return f"{v:.4f}".rstrip("0").rstrip(".")


def write_vtr(model, template_path, out_path, calc_s=None):
    m = model.meta
    if not m:
        raise ValueError("model.meta 가 비어 있음: build_tjunction 으로 만든 모델만 지원")
    if len(m["offsets_mm"]) != 12:
        raise ValueError("출력 위치는 12개여야 함 (템플릿의 rept 줄 때문)")
    text, enc = _read(template_path)
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    xe, ye = model.xe * 1000.0, model.ye * 1000.0
    nx, ny = len(xe) - 1, len(ye) - 1
    nz = int(round(m["z_mm"] / 50.0))

    # 템플릿에서 색(colr) 번호 찾기: 재료 이름 -> 번호, 경계 영역 번호
    names, zones, cur, region = {}, {}, None, False
    for l in lines:
        if l.startswith("rptf="):
            region = True
        if region and l.startswith("colr="):
            cur = int(l.split("=")[1])
        elif region and l.startswith("name=") and cur is not None:
            names[l.split("=", 1)[1].strip()] = cur
        elif region and l.startswith("solz=") and cur is not None:
            zones.setdefault(cur, l.split("=")[1].strip())
    outdoor = [c for c, v in zones.items() if v == "1"]
    indoor = sorted(c for c, v in zones.items() if v == "0")
    if len(outdoor) != 1 or len(indoor) != 2:
        raise ValueError(f"템플릿의 경계 영역 구성이 예상과 다름: {zones}")
    col = lambda nm: names[nm]

    def ix(v):
        i = int(np.argmin(np.abs(xe - v)));  assert abs(xe[i] - v) < 1e-3, ("x", v); return i

    def iy(v):
        i = int(np.argmin(np.abs(ye - v)));  assert abs(ye[i] - v) < 1e-3, ("y", v); return i

    xL, xR, st = m["xL"], m["xR"], m["st"]
    blocks = [(col(m["partition_material"]), 0, nx, iy(m["Yo"]), ny)]
    if m["strip_material"]:
        c = col(m["strip_material"])
        blocks += [(c, ix(xL), ix(xL + st), iy(m["y_strip0"]), iy(m["y_strip1"])),
                   (c, ix(xR - st), ix(xR), iy(m["y_strip0"]), iy(m["y_strip1"]))]
    blocks += [(outdoor[0], 0, nx, 0, iy(m["Yo"])),
               (indoor[0], 0, ix(xL), iy(m["y_in"]), ny), (indoor[1], ix(xR), nx, iy(m["y_in"]), ny)]
    L = m["layers"]
    for k in range(len(L) - 1, 0, -1):          # 첫 층(구조층)은 내벽과 같은 재료라 템플릿처럼 생략
        y0, y1 = iy(m["ycum"][k]), iy(m["ycum"][k + 1])
        blocks += [(col(L[k]), 0, ix(xL), y0, y1), (col(L[k]), ix(xR), nx, y0, y1)]
    if L[0] != m["partition_material"]:
        y0, y1 = iy(m["ycum"][0]), iy(m["ycum"][1])
        blocks.insert(1, (col(L[0]), 0, ix(xL), y0, y1)); blocks.insert(2, (col(L[0]), ix(xR), nx, y0, y1))
    onod = []
    for i, o in enumerate(m["offsets_mm"]):
        onod.append(f"onod={2 * i} {ix(xR + o)} {iy(m['y_in'])} {nz // 2}")
        onod.append(f"onod={2 * i + 1} {ix(xR + o)} {iy(m['ycum'][-2])} {nz // 2}")

    mat_by_name = {mt.name: mt for mt in model.materials.values()}
    out, i, cur_name, region = [], 0, None, False
    while i < len(lines):
        l = lines[i]
        if l.startswith("nxwi="):
            out += [f"nxwi={nx}"] + [f"xwid={_fmt(v)}" for v in np.diff(xe)]
            i += 1
            while lines[i].startswith("xwid="): i += 1
            continue
        if l.startswith("nywi="):
            out += [f"nywi={ny}"] + [f"ywid={_fmt(v)}" for v in np.diff(ye)]
            i += 1
            while lines[i].startswith("ywid="): i += 1
            continue
        if l.startswith("nblk="):
            out += [f"nblk={len(blocks)}"] + [f"blok={k} {c} {a} {b} {d} {e} 0 {nz}" for k, (c, a, b, d, e) in enumerate(blocks)]
            i += 1
            while lines[i].startswith("blok="): i += 1
            continue
        if l.startswith("nond="):
            out += [f"nond={len(onod)}"] + onod
            i += 1
            while lines[i].startswith("onod="): i += 1
            continue
        if l.startswith("rptf="): region = True
        if region and l.startswith("name="): cur_name = l.split("=", 1)[1].strip()
        if region and l.startswith("colr="): cur_name = None
        if region and cur_name in mat_by_name:
            mt = mat_by_name[cur_name]
            for key, val in (("lamb", mt.lam), ("ro", mt.rho), ("c", mt.cp), ("epsi", mt.eps), ("refs", mt.refl)):
                mm = re.match(rf"^({key}\s*=)(.*)$", l)
                if mm: l = f"{mm.group(1)}{val:g}"
        if calc_s is not None and l.startswith("calc="):
            l = f"calc={int(calc_s)}"          # 계산 기간(초). 3600 이면 첫 스텝(정상상태)과 1시간 뒤만 계산
        out.append(l)
        i += 1
    open(out_path, "wb").write(nl.join(out).encode(enc))
    return out_path
