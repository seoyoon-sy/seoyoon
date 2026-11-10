#!/usr/bin/env python3
"""
voltra_emu.py 검증: 엑셀(VOLTRA 결과)의 케이스별 시트와 모사 결과를 야간(18시~07시) 기준으로 비교한다.

사용:
    python voltra_emu_validate.py template.vtr voltra_30cases_night420_통합_v2.xlsx 비교결과.xlsx \
        --bc T01=wall_TO.FTE T02=Wall_TI.FTE G01=Wall_IG2.FSG D01=Wall_ID2.FSD H01=Wall_HO.FHT H02=Wall_HI.FHT

시트 이름에서 물성치를 읽는다.
    "결로 방지 단열재 X"            -> XPS 열전도율 X
    "단열재 조정 Y", "...등급(Y)"    -> EPS 열전도율 Y
    (지정이 없는 쪽은 .vtr 값 유지. EPS 기본 0.032)
비교 위치: 엑셀의 12개 열(0,10,20,30,40,50,75,100,150,200,300,500 mm)을 .vtr 출력 노드 순서와 1대1로 대응시켰다.
(.vtr 의 노드 x 간격은 0,5,10,20,30,50,75,... mm 이므로 10~40mm 열은 실제 위치가 5~30mm 이다.)
"""
import re, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from voltra_emu import VoltraEmu, _kv

POS = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]


def parse(name):
    eps = xps = None
    m = re.search(r'결로 방지 단열재\s*([0-9.]+)', name)
    if m: xps = float(m.group(1))
    m = (re.search(r'단열재 (?:조정|조절)\s*[가-힣]*등급?\(?\s*([0-9.]+)', name)
         or re.search(r'단열재 (?:조정|조절)\s*[가-힣]*\s*([0-9.]+)', name))
    if m: eps = float(m.group(1))
    return eps, xps


def night_mask(n):
    """18시부터 다음날 07시까지 14개 시각 x 30일 (1월 1일 18시 ~ 1월 31일 07시)."""
    m = np.zeros(n, bool)
    for d in range(30):
        s = 18 + 24 * d
        m[s:s + 14] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('vtr'); ap.add_argument('ref_xlsx'); ap.add_argument('out_xlsx')
    ap.add_argument('--bc', nargs='*'); ap.add_argument('--xps-cp', type=float, default=None,
                    help='XPS 비열 덮어쓰기 (엑셀을 만든 .vtr 에서는 795 였다)')
    ap.add_argument('--skip', nargs='*', default=['Model A', '변동_시간대분포'])
    a = ap.parse_args()
    xl = pd.ExcelFile(a.ref_xlsx)
    rows, perpos = [], []
    for name in xl.sheet_names:
        if name in a.skip:
            continue
        eps, xps = parse(name)
        lam = {'EPS': eps if eps is not None else 0.032}
        if xps is not None: lam['XPS'] = xps
        e = VoltraEmu(a.vtr, lamb=lam, cp={'XPS': a.xps_cp} if a.xps_cp else None)
        e.run(_kv(a.bc) or None)
        df = xl.parse(name)
        n = min(744, len(df)); m = night_mask(n)
        nodes = [(o[1], o[2]) for o in e.d['onod']]
        ix43 = sorted(set(o[1] for o in e.d['onod']))
        ys = sorted(set(o[2] for o in e.d['onod']))
        surf_y, wall_y = ys[-1], ys[0]
        S = np.array([df[f'{p}mm 표면온도(℃)'].values[:n] for p in POS]).T
        W = np.array([df[f'{p}mm 석고보드 벽체 쪽 표면온도(℃)'].values[:n] for p in POS]).T
        M = np.array([e.node_series(ix, surf_y)[:n] for ix in ix43]).T
        Mw = np.array([e.node_series(ix, wall_y)[:n] for ix in ix43]).T
        d, dw = (M - S)[m], (Mw - W)[m]
        rows.append(dict(케이스=name, EPS=lam['EPS'], XPS=lam.get('XPS', np.nan),
                         표면_RMSE=np.sqrt((d ** 2).mean()), 벽체쪽_RMSE=np.sqrt((dw ** 2).mean()),
                         표면_편차_0mm=d[:, 0].mean(), 표면_최대절대오차=np.abs(d).max()))
        perpos.append(np.c_[d.mean(0), np.sqrt((d ** 2).mean(0)), dw.mean(0), np.sqrt((dw ** 2).mean(0))])
        print(name[:48].ljust(48), round(rows[-1]['표면_RMSE'], 3), flush=True)
    r = pd.DataFrame(rows)
    pp = pd.DataFrame(np.mean(perpos, 0), index=[f'{p}mm' for p in POS],
                      columns=['표면 평균편차', '표면 RMSE', '벽체쪽 평균편차', '벽체쪽 RMSE'])
    with pd.ExcelWriter(a.out_xlsx) as w:
        r.round(4).to_excel(w, sheet_name='케이스별', index=False)
        pp.round(4).to_excel(w, sheet_name='위치별(케이스평균)')
    print(r[['표면_RMSE', '벽체쪽_RMSE', '표면_최대절대오차']].describe().round(3))


if __name__ == '__main__':
    main()
