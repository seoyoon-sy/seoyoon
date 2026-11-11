"""정답 Ψ(정상상태_정답.xlsx, TRISCO) 와 VOLTRA ΔTs(야간평균, 평면 모델 - 열교부 0 mm) 의 관계, 그리고 hbcalc ΔTs 오차의 Ψ 환산.
전제: results/_abs_cache.pkl (python examples/compare_surface_temperature.py 로 생성)."""
import sys, pickle, re; import os; ROOT_=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT_)
import numpy as np, pandas as pd
from dataclasses import replace
from hbcalc import Boundary, Solver
from hbcalc.planar import planar_model
from hbcalc.presets import TRY3_MATERIALS
from hbcalc.validate import parse_sheet, night_mask
R=ROOT_+'/'; mk=night_mask(744)
tab=pd.read_excel(R+'정상상태_정답.xlsx'); tab.columns=['case','Phi','Phi2','P2D','psi']
def key(nm):
    e,x=parse_sheet(nm)
    if nm.startswith('Model A'): return ('A',)
    return (e if e is not None else 0.032, x if x is not None else 1.6)
psi_by={key(r.case):r.psi for _,r in tab.iterrows()}
data=pickle.load(open(R+'results/_abs_cache.pkl','rb'))
def planar_night(e,ti,shift):
    mats=dict(TRY3_MATERIALS); mats['EPS']=replace(mats['EPS'],lam=e)
    pm=planar_model([('concrete',200),('EPS',180),('0.5B cement block',90),('gypsum',10)],mats,hrb=5.25)
    bc=Boundary.from_voltra_files(R+'wall_TO.FTE',R+ti,R+'Wall_HO.FHT',R+'Wall_HI.FHT',R+'Wall_IG2.FSG',R+'Wall_ID2.FSD',solar_shift_h=shift)
    s=Solver(pm); s.run(bc,nsteps=744); return s.series('surf')[:744][mk].mean()
cache={}
def P(e,ti,sh):
    k=(e,ti,sh)
    if k not in cache: cache[k]=planar_night(e,ti,sh)
    return cache[k]
for cond,ti in (('난방','Wall_TI.FTE'),('비난방','Wall_TI_2.FTE')):
    rows=[]
    for (k,nm),d in data.items():
        if k!=cond: continue
        ky=key(nm)
        if ky not in psi_by: continue
        e=ky[0]; V0=d['V_surf'][0][mk].mean()
        row=dict(case=nm,psi=psi_by[ky],V_dTs=P(e,ti,0.0)-V0)
        for sh in (0.0,-0.25):
            row[f'M_dTs_{sh}']=P(e,ti,sh)-d[f'M_surf_{sh}'][0][mk].mean()
        rows.append(row)
    r=pd.DataFrame(rows).sort_values('V_dTs')
    rho=r.V_dTs.corr(r.psi,method='spearman')
    print(f'\n===== {cond}: {len(r)}케이스, 정답 Ψ(TRISCO 정상상태) 대 VOLTRA ΔTs(야간평균, 평면모델 - 0 mm)  순위상관 {rho:.3f}')
    print('Ψ 범위',r.psi.min().round(3),r.psi.max().round(3),' ΔTs 범위',r.V_dTs.min().round(2),r.V_dTs.max().round(2))
    # 단조 보간 곡선: ΔTs 로 Ψ 를 예측(정답 Ψ 에 대해). 단조가 아니면 선형 회귀로 대체
    A=np.c_[r.V_dTs,np.ones(len(r))]; k,b=np.linalg.lstsq(A,r.psi,rcond=None)[0]; res=r.psi-(A@[k,b])
    print(f'선형 근사 Ψ = {k:.4f} ΔTs + {b:.4f}, 잔차 RMSE {np.sqrt((res**2).mean()):.4f} (Ψ 범위 {r.psi.max()-r.psi.min():.2f})')
    # 국소 기울기 (VOLTRA 곡선)
    X=r.V_dTs.values; Y=r.psi.values
    from numpy.polynomial import polynomial as Pn
    c=np.polyfit(X,Y,3); r['psi_fit']=np.polyval(c,X); print(f'3차 다항 잔차 RMSE {np.sqrt(((Y-r.psi_fit)**2).mean()):.4f}, 최대 {np.abs(Y-r.psi_fit).max():.4f}')
    dc=np.polyder(c); print('국소 기울기 dΨ/dΔTs: ΔTs 0.5, 2, 4, 5 에서',[round(np.polyval(dc,x),3) for x in (0.5,2,4,5)])
    for sh in (0.0,-0.25):
        # VOLTRA 곡선(3차)에 모사 ΔTs 를 넣어 Ψ 환산; VOLTRA ΔTs 를 넣었을 때의 맞춤 오차를 기준선으로 뺀다
        dpsi=np.polyval(c,r[f'M_dTs_{sh}'])-r.psi_fit
        print(f'  모사 보정 {sh:>5}: ΔTs 오차 평균 {(r[f"M_dTs_{sh}"]-r.V_dTs).mean():+.3f} (최대절대 {(r[f"M_dTs_{sh}"]-r.V_dTs).abs().max():.3f}) -> Ψ 환산오차 평균 {dpsi.mean():+.4f}, 평균절대 {np.abs(dpsi).mean():.4f}, 최대절대 {np.abs(dpsi).max():.4f} | 상대 평균 {100*np.mean(np.abs(dpsi)/r.psi):.2f} %, 최대 {100*np.max(np.abs(dpsi)/r.psi):.2f} %, 3% 초과 {int((np.abs(dpsi)/r.psi>0.03).sum())}개')
    r.round(4).to_csv(R+f'results/psi_true_vs_dTs_{cond}.csv',index=False,encoding='utf-8-sig')
