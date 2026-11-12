import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_test_walls import *
import scipy.sparse as sp, scipy.sparse.linalg as spl
def system(key,**kw):
    s=Solver(B[key]["mj"],**kw); s.run(bc,nsteps=1)
    A=(s.Kc+s.Krad+sp.diags(s.G_all[0])).tocsr(); b=s.b_all[0]
    return s,A,b
def gs(A,b,x0,tol=1e-4,maxit=10000,omega=1.0):
    L=sp.tril(A,format="csr"); U=sp.triu(A,1,format="csr"); D=A.diagonal()
    M=(sp.diags(D/omega)+sp.tril(A,-1)).tocsr()
    x=x0.copy()
    for it in range(1,maxit+1):
        xn=spl.spsolve_triangular(M,b-(U@x)-(1-1/omega)*D*x*0 ,lower=True) if omega==1.0 else spl.spsolve_triangular(M,b-(U@x)+((1/omega-1)*D)*x,lower=True)
        d=np.abs(xn-x).max(); x=xn
        if d<tol: break
    return x,it
if __name__=="__main__":
    for key in ("N6_noblock_insstrip","N4_weak","N5_noblock","N1_light"):
        s,A,b=system(key)
        xe=spl.spsolve(A.tocsc(),b)
        idx=[s.idx[B[key]["mj"].outputs[f"surf_{p}"]] for p in POS]
        V=res[key]["junction"]["Vs"][0]
        print(key,"정확해 편차 0/50/150mm:",np.round((xe[idx]-V)[[0,5,8]],3))
        for x0v in (0.0,20.0,-3.7):
            for om in (1.0,):
                x,it=gs(A,b,np.full(A.shape[0],x0v),tol=1e-4,maxit=10000,omega=om)
                dv=x[idx]-V
                print(f"   GS 초기 {x0v:5.1f} 반복 {it:5d} 정확해와 최대차 {np.abs(x-xe).max():.3f} | VOLTRA 대비 편차 [0,5,50,150,500]mm",np.round(dv[[0,1,5,8,11]],3))
