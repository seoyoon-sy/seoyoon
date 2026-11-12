import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_test_walls import *
def dev_strip(key, sx, sy, wx=1.0, wy=1.0, eps_strip=None):
    m=B[key]["mj"]
    if eps_strip is not None: m=m.with_properties(eps={"XPS":eps_strip})
    s=Solver(m,conv_lump=(wx,wy))
    xid=[k for k,v in s.m.materials.items() if v.name=="XPS"][0]
    cid=s.m.cid
    cA={z:np.zeros(s.N) for z in s.zone_ids}
    for f,lo,hi in zip(s.faces,s._face_low_node,s._face_high_node):
        i,j,ax,zn,eps,refl,area,side=f
        if ax=="x": mat=cid[i,j-1] if side=="up" else cid[i,j]
        else: mat=cid[i-1,j] if side=="right" else cid[i,j]
        w=(sx if ax=="x" else sy) if mat==xid else (wx if ax=="x" else wy)
        cA[zn][lo]+=w*area; cA[zn][hi]+=(1-w)*area
    s.conv_A=cA
    s.run(bc,nsteps=1)
    M=np.array([s.series(f"surf_{p}")[0] for p in POS])
    return M-res[key]["junction"]["Vs"][0]
if __name__=="__main__":
    keys=list(res)
    print("띠 면(XPS 접한 면)만 대류 면적 배분 바꾸기 (그 밖은 low): 벽체별 0~150 mm 평균 편차 / RMSE")
    for sx in (0,0.5,1):
        for sy in (0,0.5,1):
            out=[]
            for k in keys:
                d=dev_strip(k,sx,sy); out.append(f"{d[:9].mean():+.3f}/{np.sqrt((d**2).mean()):.3f}")
            print(f"띠 가로면 {sx} 세로면 {sy} |"," ".join(out))
    print("띠 방사율 변경(대류 low): ")
    for e in (0.9,0.5,0.1):
        out=[]
        for k in keys:
            d=dev_strip(k,1,1,eps_strip=e); out.append(f"{d[:9].mean():+.3f}/{np.sqrt((d**2).mean()):.3f}")
        print(f"eps {e} |"," ".join(out))
