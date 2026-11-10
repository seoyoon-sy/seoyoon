import sys, time, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hbcalc import Boundary
from hbcalc.validate import *
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
mesh = sys.argv[1]; shift=float(sys.argv[2]); out=sys.argv[3]
bc=Boundary.from_voltra_files(R+'wall_TO.FTE',R+'Wall_TI.FTE',R+'Wall_HO.FHT',R+'Wall_HI.FHT',R+'Wall_IG2.FSG',R+'Wall_ID2.FSD',solar_shift_h=shift)
xl,names=load_cases(R+'voltra_30cases_night420_통합_v2.xlsx')
POS=sorted(set(EXCEL_LABELS)|set(SIM_POS_VOLTRA_ORDER))
rows=[]; perpos_a=[]; perpos_b=[]
for nm in names:
    s,m=run_case(nm,bc,mesh=mesh,positions=POS)
    df=xl.parse(nm)
    ss={p:s.series(f'surf_{p}') for p in POS}; bb={p:s.series(f'back_{p}') for p in POS}
    dA,dAw=compare_case(df,ss,bb,EXCEL_LABELS,SIM_POS_VOLTRA_ORDER)   # 기존 가정: 10~40열 = 5~30 mm
    dB,dBw=compare_case(df,ss,bb,EXCEL_LABELS,EXCEL_LABELS)           # 대안: 열 이름 그대로의 위치
    rows.append(dict(케이스=nm,A_RMSE=np.sqrt((dA**2).mean()),B_RMSE=np.sqrt((dB**2).mean()),A_max=np.abs(dA).max(),B_max=np.abs(dB).max(),
                     A_벽체쪽RMSE=np.sqrt((dAw**2).mean()),B_벽체쪽RMSE=np.sqrt((dBw**2).mean())))
    perpos_a.append(np.c_[dA.mean(0),np.sqrt((dA**2).mean(0))]); perpos_b.append(np.c_[dB.mean(0),np.sqrt((dB**2).mean(0))])
r=pd.DataFrame(rows); r.to_csv(out+'_cases.csv',index=False)
pa=np.mean(perpos_a,0); pb=np.mean(perpos_b,0)
pp=pd.DataFrame({'엑셀열(mm)':EXCEL_LABELS,'A:모사위치(mm)':SIM_POS_VOLTRA_ORDER,'A평균편차':pa[:,0],'A_RMSE':pa[:,1],'B평균편차':pb[:,0],'B_RMSE':pb[:,1]}).round(3)
pp.to_csv(out+'_pos.csv',index=False)
print(mesh,'shift',shift,'cases',len(r)); print(r[['A_RMSE','B_RMSE','A_max','B_max','A_벽체쪽RMSE','B_벽체쪽RMSE']].describe().loc[['mean','min','max']].round(3)); print(pp.to_string(index=False))
