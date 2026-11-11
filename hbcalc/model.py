"""계산 모델 자료구조. 모든 길이는 m, 온도는 ℃, 열전도율 W/(m K), 밀도 kg/m3, 비열 J/(kg K)."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    lam: float          # 열전도율
    rho: float          # 밀도
    cp: float           # 비열
    eps: float = 0.9    # 장파 방사율
    refl: float = 0.1   # 일사 반사율 (흡수율 = 1 - refl)


@dataclass(frozen=True)
class Zone:
    """재료가 아닌 셀(경계 셀)의 종류. kind: 'outdoor' 또는 'indoor'.
    같은 zone 안의 표면끼리만 복사 교환을 한다."""
    kind: str
    name: str = ""


@dataclass
class Model:
    xe: np.ndarray                      # x 격자선 좌표 (nx+1), m
    ye: np.ndarray                      # y 격자선 좌표 (ny+1), m
    zt: float                           # 깊이(z 방향 길이), m
    cid: np.ndarray                     # (nx, ny) 셀 번호: >0 재료 번호, <0 zone 번호(-zone id)
    materials: dict                     # {재료 번호: Material}
    zones: dict                         # {zone id(>0): Zone}
    outputs: dict = field(default_factory=dict)   # {이름: (ix, iy)} 격자선 번호
    hrb: float = 5.25                   # 흑체 복사 열전달계수 W/(m2 K)
    meta: dict = field(default_factory=dict)   # 형상 좌표(mm) 등 부가 정보 (.vtr 쓰기용)

    @property
    def shape(self):
        return self.cid.shape

    def material_by_name(self, name):
        for k, m in self.materials.items():
            if m.name == name:
                return k
        raise KeyError(f"없는 재료 이름: {name}. 있는 재료: {[m.name for m in self.materials.values()]}")

    def with_properties(self, lam=None, rho=None, cp=None, eps=None, refl=None):
        """재료 물성을 이름으로 덮어쓴 새 모델. 예: model.with_properties(lam={'EPS': 0.03})"""
        new = {}
        upd = dict(lam=lam, rho=rho, cp=cp, eps=eps, refl=refl)
        names = {m.name for m in self.materials.values()}
        for d in upd.values():
            if d and set(d) - names:
                raise KeyError(f"없는 재료 이름: {set(d) - names}. 있는 재료: {sorted(names)}")
        for k, m in self.materials.items():
            kw = {a: d[m.name] for a, d in upd.items() if d and m.name in d}
            new[k] = replace(m, **kw)
        return replace(self, materials=new)
