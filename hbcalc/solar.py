"""태양 위치와 경사면 일사. 수평 전일사 G, 수평 확산일사 D (W/m2) 입력."""
from __future__ import annotations
import numpy as np


def sun_position(n, lat, lon, tz, shift=0.0, day0=1):
    """시간 인덱스 k(0이 1월 1일 0시)의 태양 고도, 방위(남 0, 서 +). 단위 rad.
    shift: 시각 보정(시간). 균시차는 Spencer 근사 대신 Cooper/Whitaker 형식의 간단식을 쓴다."""
    k = np.arange(n) + shift
    dayno = day0 + np.floor((k + 24) / 24) - 1
    clock = np.mod(k, 24)
    B = np.radians(360 / 365 * (dayno - 81))
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)      # 분
    ts = clock + (lon - 15 * tz) / 15 + eot / 60
    dec = np.radians(23.45 * np.sin(np.radians(360 / 365 * (284 + dayno))))
    ha = np.radians(15 * (ts - 12))
    ph = np.radians(lat)
    alt = np.arcsin(np.clip(np.sin(ph) * np.sin(dec) + np.cos(ph) * np.cos(dec) * np.cos(ha), -1, 1))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(ph) - np.tan(dec) * np.cos(ph))
    return alt, az


def sky_view_factor(surf_az_deg, tilt_deg, obstacles=(), n=180):
    """등방 하늘에서 장애물(방위min, 방위max, 고도min, 고도max, deg)을 뺀 가시율(0 ~ 1).
    면이 보는 하늘의 입사각 코사인 가중 적분에서 장애물이 가리는 몫을 뺀다. 장애물이 겹치지 않는다고 본다.
    VOLTRA 매뉴얼 D.4: 확산일사는 노드 주변 하늘 가시율(일사 장애물 포함)로 줄인다."""
    if not obstacles:
        return 1.0
    g, t = np.radians(surf_az_deg), np.radians(tilt_deg)
    blocked = 0.0
    for o in obstacles:
        A = np.radians(np.linspace(o[0], o[1], n + 1)); E = np.radians(np.linspace(max(o[2], 0.0), o[3], n + 1))
        Am, Em = (A[1:] + A[:-1]) / 2, (E[1:] + E[:-1]) / 2
        dA, dE = A[1] - A[0], E[1] - E[0]
        AA, EE = np.meshgrid(Am, Em)
        cos_i = np.sin(EE) * np.cos(t) + np.cos(EE) * np.sin(t) * np.cos(AA - g)
        blocked += (np.maximum(cos_i, 0) * np.cos(EE)).sum() * dA * dE
    total = np.pi * (1 + np.cos(t)) / 2
    return float(max(0.0, 1.0 - blocked / total))


def solar_on_surface(G, D, alt, az, surf_az_deg, tilt_deg, rho_g, obstacles=(), diffuse_obstruction=True):
    """경사면 입사 일사(직달 + 확산 + 지면반사), W/m2.
    surf_az_deg: 면의 방위(남 0, 서 +90, 동 -90). tilt_deg: 90 이면 수직.
    obstacles: [(방위min, 방위max, 고도min, 고도max) deg] 태양이 이 범위에 있으면 직달 차단."""
    a, z = np.degrees(alt), np.degrees(az)
    Gh = np.maximum(G, 0)
    Dh = np.minimum(np.maximum(D, 0), Gh)
    blocked = a <= 0
    for o in obstacles:
        blocked |= (z >= o[0]) & (z <= o[1]) & (a >= o[2]) & (a <= o[3])
    Ib = np.where(blocked, 0.0, (Gh - Dh) / np.maximum(np.sin(alt), 1e-3))
    g, t = np.radians(surf_az_deg), np.radians(tilt_deg)
    cos_i = np.sin(alt) * np.cos(t) + np.cos(alt) * np.sin(t) * np.cos(az - g)
    svf = sky_view_factor(surf_az_deg, tilt_deg, obstacles) if diffuse_obstruction else 1.0
    return Ib * np.maximum(cos_i, 0) + Dh * svf * (1 + np.cos(t)) / 2 + rho_g * Gh * (1 - np.cos(t)) / 2
