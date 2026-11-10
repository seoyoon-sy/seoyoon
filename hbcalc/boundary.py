"""경계조건. 실외온도, 실내온도, 대류계수, 수평 일사를 배열(시간 간격 dt)로 받는다.
VOLTRA 형식 파일(FTE/FHT/FSG/FSD)은 from_voltra_files 로만 읽는다(호환용)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .solar import sun_position, solar_on_surface


def _arr(x, n):
    x = np.asarray(x, float)
    return np.full(n, float(x)) if x.ndim == 0 else x


@dataclass
class Boundary:
    T_out: np.ndarray
    T_in: np.ndarray
    h_out: np.ndarray                   # 실외 대류계수 W/(m2 K)
    h_in: np.ndarray                    # 실내 대류계수 (복사 제외)
    dt: float = 3600.0
    G: np.ndarray | None = None         # 수평면 전일사 W/m2
    D: np.ndarray | None = None         # 수평면 확산일사 W/m2
    lat: float = 37.477
    lon: float = 126.624
    tz: float = 9.0
    day0: int = 1
    solar_shift_h: float = 0.0
    surf_az_deg: float = -90.0          # 외벽 방위: 남 0, 서 +90, 동 -90
    tilt_deg: float = 90.0
    ground_refl: float = 0.2
    # 태양 장애물 (방위min, 방위max, 고도min, 고도max) deg. 기본값: 고도 5도 이하 직달 차단.
    # 직달 = (G-D)/sin(고도) 라서 일출 직후 값이 폭주하는 것을 막는다 (VOLTRA 템플릿의 sobs 와 같은 설정).
    obstacles: tuple = ((-180.0, 180.0, 0.0, 5.0),)

    def __post_init__(self):
        n = max(len(np.atleast_1d(v)) for v in (self.T_out, self.T_in, self.h_out, self.h_in))
        self.n = n
        self.T_out, self.T_in = _arr(self.T_out, n), _arr(self.T_in, n)
        self.h_out, self.h_in = _arr(self.h_out, n), _arr(self.h_in, n)
        if self.G is not None:
            self.G = _arr(self.G, n)
            self.D = _arr(self.D if self.D is not None else 0.0, n)
        for nm in ("T_out", "T_in", "h_out", "h_in"):
            if len(getattr(self, nm)) != n:
                raise ValueError(f"{nm} 길이가 {n} 이 아님")

    def irradiance(self):
        """외벽면 입사 일사 W/m2 (n,)"""
        if self.G is None:
            return np.zeros(self.n)
        alt, az = sun_position(self.n, self.lat, self.lon, self.tz, self.solar_shift_h, self.day0)
        return solar_on_surface(self.G, self.D, alt, az, self.surf_az_deg, self.tilt_deg,
                                self.ground_refl, self.obstacles)

    @classmethod
    def constant(cls, n, T_out=-5.0, T_in=20.0, h_out=17.0, h_in=1.5, dt=3600.0):
        return cls(np.full(n, T_out), np.full(n, T_in), np.full(n, h_out), np.full(n, h_in), dt=dt)

    @classmethod
    def from_voltra_files(cls, T01, T02, H01, H02, G01=None, D01=None, **kw):
        """호환용: VOLTRA 경계조건 파일 읽기 (한 줄에 한 시간 값)."""
        rd = lambda p, tag: _read_series(p, tag)
        T_out, T_in = rd(T01, "FTE"), rd(T02, "FTE")
        h_out, h_in = rd(H01, "FHT"), rd(H02, "FHT")
        n = min(len(T_out), len(T_in), len(h_out), len(h_in))
        G = D = None
        if G01 and D01:
            G, D = rd(G01, "FSG"), rd(D01, "FSD")
            n = min(n, len(G), len(D))
            lat, lon, tz = _read_site(G01)
            kw.setdefault("lat", lat); kw.setdefault("lon", lon); kw.setdefault("tz", tz)
            G, D = G[:n], D[:n]
        return cls(T_out[:n], T_in[:n], h_out[:n], h_in[:n], G=G, D=D, **kw)


def _text(path):
    raw = open(path, "rb").read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp949", errors="replace")


def _read_series(path, tag):
    return np.array([float(l.replace(",", " ").split()[1]) for l in _text(path).splitlines() if l.startswith(tag)])


def _read_site(path):
    for l in _text(path).splitlines():
        if l.startswith("DAT"):
            p = l.replace(",", " ").split()
            return float(p[2]), float(p[3]), float(p[4]) if len(p) > 4 else 9.0
    return 37.5, 127.0, 9.0
