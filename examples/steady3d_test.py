"""시험용 3D 정상상태 (z 방향 전도 + z 조각별 복사). hbcalc 2D 솔버의 형상을 재사용."""
import sys; sys.path.insert(0,'/home/user/seoyoon')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from hbcalc import Boundary, Solver
from hbcalc.radiation import grey_network

def K2(t, a):   # d2/dt2 K2 = 1/(a+t^2)^2
    return t*np.arctan(t/np.sqrt(a))/(2*a**1.5)

def steady3d(m, zv, nz=10, ngl=10, hrb=None):
    s = Solver(m, conv_lump="half")
    N, zt = s.N, m.zt; dz = zt/nz
    ext = [(max(k*dz-dz/2,0), min(k*dz+dz/2,zt)) for k in range(nz+1)]
    hk = np.array([b-a for a,b in ext])
    # 전도
    lam_area = np.zeros(N)
    dx, dy = np.diff(m.xe), np.diff(m.ye)
    ii, jj = np.nonzero(s.ismat)
    q = s.lam * dx[:,None]*dy[None,:]/4
    for a in (0,1):
        for b in (0,1): np.add.at(lam_area, s.idx[ii+a,jj+b], q[ii,jj])
    gz = lam_area/dz
    Kc1 = s.Kc/zt            # 단위 깊이당
    blocks = [Kc1*hk[k] for k in range(nz+1)]
    Kc3 = sp.block_diag(blocks, format="csr")
    Dz = sp.diags(gz)
    Lz = sp.lil_matrix((N*(nz+1),)*2)
    rows=[];cols=[];vals=[]
    for k in range(nz):
        a=np.arange(N)+k*N; b=a+N
        rows += [a,b,a,b]; cols += [a,b,b,a]; vals += [gz,gz,-gz,-gz]
    Kz = sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N*(nz+1),)*2)
    # 복사 요소
    E = s.rad_elements; nE=len(E); A2 = s.rad_area/zt     # 단위 깊이당 길이
    pairs = np.argwhere(np.triu(s.rad_W)>0)               # 한 번만
    # 쿼드러처 점
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    # t 값 격자(dz/2 배수)
    tvals = np.arange(-(2*nz+2), 2*nz+3)*dz/2
    W3 = np.zeros((nE*(nz+1),)*2)
    # 조각별 경계
    for ia, ib in np.argwhere(s.rad_W>0):
        if ia > ib: continue
        ea, eb = E[ia], E[ib]
        if ea["ax"] != "x": ea, eb, ia2, ib2 = eb, ea, ib, ia
        else: ia2, ib2 = ia, ib
        u1,u2 = sorted([abs(ea["lo"]-eb["plane"]), abs(ea["hi"]-eb["plane"])])
        v1,v2 = sorted([abs(eb["lo"]-ea["plane"]), abs(eb["hi"]-ea["plane"])])
        u=(u2-u1)/2*xg+(u2+u1)/2; wu=(u2-u1)/2*wg; v=(v2-v1)/2*xg+(v2+v1)/2; wv=(v2-v1)/2*wg
        U,V=np.meshgrid(u,v,indexing="ij"); w=(U*V*np.outer(wu,wv)).ravel(); a_=(U**2+V**2).ravel()
        Ktab = np.array([(w*K2(t,a_)).sum() for t in tvals])/np.pi   # 각 t
        tix = lambda t: int(round(t/(dz/2)))+(2*nz+2)
        for p in range(nz+1):
            ap,bp = ext[p]
            for qd in range(nz+1):
                aq,bq = ext[qd]
                val = Ktab[tix(bq-ap)]-Ktab[tix(bq-bp)]-Ktab[tix(aq-ap)]+Ktab[tix(aq-bp)]
                W3[ia2*(nz+1)+p, ib2*(nz+1)+qd] = val; W3[ib2*(nz+1)+qd, ia2*(nz+1)+p] = val
    return s, E, W3, Kc3, Kz, ext, hk, A2

def solve3d(s, E, W3, Kc3, Kz, ext, hk, A2, zv, nz=10, hrb=5.1, Tair=None):
    N=s.N; nE=len(E)
    A3 = np.array([A2[i]*hk[k] for i in range(nE) for k in range(nz+1)])
    E3 = [E[i] for i in range(nE) for k in range(nz+1)]
    Gt,Gs,vs = grey_network(E3, W3, A3, hrb)
    node3 = np.array([E[i]["node"]+k*N for i in range(nE) for k in range(nz+1)])
    S3 = sp.csr_matrix((np.ones(len(node3)),(node3,np.arange(len(node3)))),shape=(N*(nz+1),len(node3)))
    zone3 = np.array([E[i]["zone"] for i in range(nE) for k in range(nz+1)])
    # 대류 (half 배분 = 요소 면적)
    Gc = np.zeros(N*(nz+1)); b = np.zeros(N*(nz+1))
    for z,(Ta,h) in zv.items():
        sel = zone3==z
        Az = S3[:,np.nonzero(sel)[0]] @ A3[sel]
        Gc += h*Az; b += h*Ta*Az + Ta*(S3 @ (Gs*sel))
    P = (Kc3+Kz+sp.diags(Gc)).tocsc()
    lu = spl.splu(P)
    def mv(x): return P@x + S3@(Gt@(S3.T@x))
    op = spl.LinearOperator((N*(nz+1),)*2, matvec=mv)
    M = spl.LinearOperator((N*(nz+1),)*2, matvec=lambda x: lu.solve(x))
    T, info = spl.cg(op, b, M=M, rtol=1e-11, atol=0, maxiter=500)
    # 열류
    res = {}
    out = (S3@(Gt@(S3.T@T))) + Gc*T - b       # 노드별 순 유출(복사+대류-공급) 
    return T.reshape(nz+1,N), out.reshape(nz+1,N), info, node3, zone3, S3
