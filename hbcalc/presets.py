"""TRY3 계열 T자 접합부 프리셋 (TRY3_Plus_QTest2.vtr 에서 읽은 구성). 코드만으로 같은 모델을 만든다."""
from dataclasses import replace
from .model import Material
from .geometry import Layer, TJunction, build_tjunction

TRY3_MATERIALS = {
    "concrete": Material("concrete", 1.6, 2200, 795),
    "EPS": Material("EPS", 0.032, 25, 1500),
    "0.5B cement block": Material("0.5B cement block", 0.6, 1400, 1000),
    "gypsum": Material("gypsum", 0.16, 900, 840),
    "XPS": Material("XPS", 1.6, 2200, 1300),     # 열교 차단 띠. 템플릿 기본값(콘크리트 값). 케이스마다 lam 을 바꾼다.
}


# 사용자의 TRY3 VOLTRA 엑셀 케이스에서 결로 방지 단열재(XPS 판) 물성 (사용자 확인 2026-11-16):
#  - 결로 방지 단열재를 조절한 케이스("결로 방지 단열재 X", X는 1.6이 아님): 밀도 25, 비열 1300, 열전도율만 X.
#  - 조절하지 않은 케이스(외벽 단열재만 조절한 "단열재 조정 Y" 등)와 "결로 방지 단열재 1.6": 판을 콘크리트와 같은 물성(열전도율 1.6, 밀도 2200, 비열 795)으로 둠.
# 위 TRY3_MATERIALS의 XPS는 템플릿 값(1.6, 2200, 1300)이라 어느 쪽과도 비열이 다르다. 밀도, 비열은 시간 변화(열용량)에만 영향을 주고 정상상태는 같다.
TRY3_EXCEL_XPS_RHO = 25.0          # 조절한 케이스의 판 밀도
TRY3_EXCEL_XPS_CP = 1300.0         # 조절한 케이스의 판 비열
TRY3_EXCEL_CONCRETE_STRIP = (2200.0, 795.0)   # 조절하지 않은 케이스의 판 밀도, 비열 (콘크리트와 같음)


def try3_excel_strip_props(xps_lam):
    """엑셀 케이스의 결로 방지 단열재 열전도율(없으면 None)에 맞는 판 (밀도, 비열)."""
    if xps_lam is None or abs(xps_lam - 1.6) < 1e-9:
        return TRY3_EXCEL_CONCRETE_STRIP
    return TRY3_EXCEL_XPS_RHO, TRY3_EXCEL_XPS_CP


def try3_spec(**kw) -> TJunction:
    base = TJunction(
        layers=[Layer("concrete", 200), Layer("EPS", 180), Layer("0.5B cement block", 90), Layer("gypsum", 10)],
        materials=dict(TRY3_MATERIALS),
    )
    return replace(base, **kw)


def try3_model(lam=None, **kw):
    """lam: {'EPS': 0.032, 'XPS': 0.004} 처럼 재료 열전도율을 이름으로 덮어쓴다."""
    m = build_tjunction(try3_spec(**kw))
    return m.with_properties(lam=lam) if lam else m


# VOLTRA TRY3_Plus_QTest2 의 격자선 (mm). 격자를 VOLTRA 와 똑같이 맞춰 비교할 때 쓴다.
VOLTRA_TRY3_X_EDGES_MM = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1005, 1010, 1030, 1050, 1070, 1090, 1110, 1130, 1150, 1170, 1190, 1195, 1200, 1205, 1210, 1220, 1230, 1240, 1250, 1275, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200)
VOLTRA_TRY3_Y_EDGES_MM = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650.0001, 700.0001, 750.0001, 800.0001, 850.0001, 900.0001, 950.0001, 1000.0001, 1020.0001, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300.0001, 1320.0001, 1340.0001, 1360.0001, 1380.0001, 1425.0001, 1470.0001, 1475.0001, 1480.0001, 1536.6668, 1593.3335, 1650.0002, 1698.6669, 1747.3336, 1796.0003, 1844.667, 1893.3337, 1942.0004, 1990.6671, 2039.3338, 2088.0005, 2136.6672, 2185.3339, 2234.0006, 2282.6673, 2331.334, 2380.0007)


def try3_voltra_mesh_model(lam=None, **kw):
    """VOLTRA 와 같은 격자를 쓰는 TRY3 모델 (검증 비교용)."""
    return try3_model(lam, x_edges_mm=VOLTRA_TRY3_X_EDGES_MM, y_edges_mm=VOLTRA_TRY3_Y_EDGES_MM, **kw)


# 수렴한 해(물리적으로 정확한 쪽)를 얻는 격자: 접합부 근처 1 mm 에서 시작해 20 mm 까지 키우고 모든 격자선을 정수 mm 에 맞춘다.
# 노드 약 8500개, 한 케이스(744시간) 약 30초. VOLTRA 격자 대비 접합부 0 mm 에서 0.4 도까지 다른 값을 줄 수 있다 (CLAUDE.md 8.7).
FINE_MESH = dict(snap_mm=1, h_min_mm=1, h_max_mm=20, h_layer_mm=10)
