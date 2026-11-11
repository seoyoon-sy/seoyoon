# -*- coding: utf-8 -*-
"""
열교부 선형열관류율(Ψ) 예측 모델 - 최종 통합본 v6.3 (v6.2 + 비난방 검증 LOCO · 안정화 27일 전체)
================================================================
[v6.2 -> v6.3 변경사항]
1) 안정화 구간을 1일치가 아니라 전체 27일(야간 4~30일차)로 사용: STABLE_NIGHTS_TO_USE = 30
   -> 비난방 검증은 케이스당 30야간 전부(비난방 2 + 도달중 1 + 안정화 27) 사용.
2) 비난방 검증을 LOCO 방식으로 변경 (NH_TRAIN_MODE = "loco", 기본값):
   - fold마다 검증 벽체 1개를 통째로 제외하고, 나머지 29개 벽체의 "정상(20도)" 데이터로 6개 모델
     (2차회귀/RF/KNN/ANN/SVR/RF_PCA)을 다시 학습 -> 빠진 벽체의 비난방 야간(30개)을 예측.
   - 30fold x 30야간 = 900개 (모델별). Phase별 N: 비난방 60 / 도달중 30 / 안정화 810.
   - RF_PCA의 표준화·PCA 회전도 fold의 학습 29개 데이터로만 fit (학습·테스트 같은 변환 적용).
   - NH_TRAIN_MODE = "full30" 으로 두면 v6.2 방식(30개 전체로 1번 학습)으로 돌아감.
   - 비난방 예측 시트에 "Fold", "학습방식" 컬럼 추가.
3) 비난방_Phase별_요약에 "전체합산"(3구간 합산 900개) 행 추가.
4) 정상상태 LOCO·독립테스트 로직은 v6.2와 동일 (변경 없음).
   ※ LOCO 방식은 fold마다 모델을 다시 학습하므로 v6.2보다 실행 시간이 늘어남(약 +5분).

[v6.1 -> v6.2 변경사항]
1) 비난방->도달중->안정화 검증에 "Random_Forest_PCA" 모델을 추가함 (기존 Random_Forest 행도 그대로 유지).
   - 표준화(StandardScaler) + PCA(n_components=2) 회전을 "학습 30개 정상 데이터"로 fit하고,
     학습 데이터와 테스트(비난방) 데이터를 "같은 변환"으로 회전시킨 뒤 RF를 학습/예측함.
     (학습은 원래축, 테스트만 회전하면 좌표계가 어긋나 틀린 예측이 나오므로 반드시 쌍으로 적용)
   - 근거: 축 정렬 분할을 하는 RF는 도달중처럼 C는 회복됐는데 A만 튀는(C-A 결합 패턴이 달라진)
     입력에서 A 하나로 먼저 갈라 잘못된 가지로 가는 문제가 있음. 회전축은 C·A를 섞은 축이라
     같은 분할이 C·A를 동시에 고려하는 효과를 냄. (Rotation Forest, Rodriguez et al. 2006 의 단순화 버전)
   - 정상상태 LOCO·독립테스트에는 적용하지 않음 (정상상태에서는 A 단독 분할이 최적이라 PCA가 오히려 성능 저하)
   - 비난방 결과 시트(비난방_CA_*, 비난방_예측_*, 비난방_Phase별_요약)에 Random_Forest_PCA 행이 함께 출력됨.
     (PCA 행을 RF 행 대신 쓰고 싶으면 NH_MODEL_NAMES에서 "Random_Forest"를 빼면 됨)
2) 그 외 v6.1의 모든 로직은 변경 없음.

[v6 -> v6.1 변경사항]
1) 2변수 ANN·SVR 하이퍼파라미터를 30-fold LOCO(야간단위 n=900) 격자탐색으로 재튜닝해 반영함.
   LOCO·독립테스트(fit_predict_all)와 비난방 검증(nh_fns) 두 곳 모두 같은 설정을 사용.
   (USE_TUNED_ANN_SVR=False 로 두면 v6 설정 그대로 돌아감. 3변수 모델은 기존 설정 유지)
   - ANN : (16,8), alpha=0.1, early_stopping ->
           (32,16), alpha=0.001, 정답(Ψ) 표준화 + 엄격 수렴
           (tol=1e-7, n_iter_no_change=50, early_stopping=False, max_iter=2000)
           * 정답 Ψ가 0.3~0.8로 작아 기본 tol(1e-4)에서 학습이 덜 된 채 조기 종료되던 문제 해결
   - SVR : C=1, gamma='scale'(=0.5) -> C=10, gamma=2, epsilon=0.01
           * gamma가 성능을 좌우함. LOCO 최적은 gamma=50이지만 비난방 검증에서 붕괴(입력분포가 달라지면
             매우 국소적인 커널은 일반화 못 함) -> LOCO 향상 + 비난방 강건성이 모두 유지되는 gamma=2 채택
2) [버그 수정] 정상상태_정답.xlsx가 0.05 단열재를 '다등급(0.05)'로 표기하도록 바뀌어서
   '라등급(0.05)'를 못 찾던 KeyError를 NAME_MAP_TO_ANSWER 매핑 한 줄로 해결.
3) 그 외 v6의 모든 로직은 변경 없음.
   ※ 튜닝은 같은 LOCO 결과로 고른 것이라 다소 낙관적 -> 논문에는 "민감도 분석"으로 표기 권장.

[v3 -> v6 변경사항 요약]
1) 정답(목표값) Ψ를 코드에서 직접 계산(야간비율법)하지 않고,
   "정상상태_정답.xlsx"(TRISCO 등 정상상태 해석 결과)에서 케이스명으로
   조회해서 사용함. 입력변수(C,A) 계산 방식은 기존과 100% 동일.
2) 학습 30개 중 "단열재 조정 0.06+결로 방지 단열재 0.003" ->
   "0.004"로 교체됨(정상상태_정답.xlsx 기준 이름).
3) 독립 테스트셋을 15개 -> 10개로 정리함. case1/case2/case3(실내온도
   변형, 비난방전환) 계열, Model B, 결로방지0.002+조정0.024는
   정상상태_정답.xlsx에 없어서 제외. 대신 단열재조정0.07,
   결로방지단열재12가 새로 추가됨(둘 다 원본 데이터 파일에 시트
   존재 확인함).
4) case3 전용 10일치 분석 로직은 전부 삭제함(해당 시트가 더 이상
   데이터 파일에 없음).
5) [신규] "비난방->도달중->안정화" 검증을 케이스 2개가 아니라
   학습 30개 전체에 적용해서 추가함:
   - Non_heating_Tsi.xlsx(표면온도), Non_heating_totalQ_2d.xlsx
     (비열교부 표면온도) 두 파일을 신규로 사용.
   - 실내외 온도 스케줄은 모든 케이스에 동일하게 적용된 공통
     시나리오(1/3 18시경 난방 시작 -> 1/4 13시경 20도 도달)이므로,
     대표지점은 이 비난방 조건 자체에서 케이스마다 다시 계산함.
   - 학습은 30개 전체의 "정상(20도 고정)" 데이터로 하고(LOCO 아님 -
     테스트 대상 벽체 자체는 학습에 있어도 되지만, "비난방"이라는
     입력 조건 자체는 학습에 전혀 없었다는 논리),
     테스트는 그 30개 케이스의 "비난방" 버전(30야간, 평균 없이
     야간별 개별)을 그대로 넣어봄.
   - 출력에 C, A 값 자체와 실측(정답Ψ)/예측Ψ를 모두 포함함.

[기존 학습 로직 - 문제없음을 확인, 변경 없이 유지]
- 대표지점 고정 방식(야간 420시간 중 최저온도/최대열류 최다빈도 지점)
- 3변수(C,A,B) / 2변수(C,A) 두 버전 계산
- 모델 5개(2차_회귀, Random_Forest, KNN, ANN, SVR), 하이퍼파라미터 동일
- LOCO(학습 30개 교차검증) 로직 동일

[구성 3가지]
1) LOCO: 학습 30개끼리 교차검증 (기존과 동일)
2) 독립 테스트셋 10개: 학습에 전혀 안 쓰인 케이스로 검증 (10개로 정리)
3) [신규] 비난방->도달중->안정화: 학습 30개 전체 대상, 30개 전체
   학습 데이터로 학습한 모델에 각 케이스의 비난방 버전(야간별)을 입력

[출력] 모델성능_최종결과_v6_3.xlsx
  - 요약_LOCO오차율/R², 요약_테스트오차율/R²
  - LOCO_실제값/예측값(3·2변수)/오차율(3·2변수)
  - 테스트_실제값/예측값(3·2변수)/오차율(3·2변수)
  - [신규] 비난방_CA_비난방/도달중/안정화: Phase별로 완전히 나눈 C,A 원자료
    (안정화는 1일치만)
  - [신규] 비난방_예측_비난방/도달중/안정화: Phase별로 완전히 나눈 실측Ψ/예측Ψ/오차율
  - [신규] 비난방_Phase별_요약: 모델x Phase(비난방/도달중/안정화)별 R²/오차율
  - LOCO_원자료/테스트_원자료 (long format)

필요 파일 (본인 경로로 아래 설정 수정):
  1) voltra_30cases_night420_통합_v2.xlsx - 학습30+테스트10, 744시간
  2) voltra_plus_result_30케이스_00mm.xlsx - 비열교부(평면부) 기준값
  3) 정상상태_정답.xlsx - 정답(선형열관류율) 조회용
  4) [신규] Non_heating_Tsi.xlsx - 학습 30개 케이스의 비난방 조건
     표면온도(12개 지점) 시계열, 케이스 이름 그대로의 시트
     (단, "단열재 조절 가등급 0.01"은 이 파일에서도 원래 이름 그대로 씀)
  5) [신규] Non_heating_totalQ_2d.xlsx - 학습 30개 케이스의 비난방 조건
     비열교부 표면온도 등, 케이스 이름 그대로의 시트
     (단, "단열재 조절 가등급 0.01"은 이 파일에서 "단열재 조정 0.01"로
      저장되어 있으므로 주의)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

# ============================================================
# 0. 설정 - 본인 경로로 수정
# ============================================================
FILE_MULTI    = Path("C:/Users/nolyb/Desktop/vol/voltra_30cases_night420_통합_v2.xlsx")
FILE_PLANAR   = Path("C:/Users/nolyb/Desktop/vol/voltra_plus_result_30케이스_00mm.xlsx")
FILE_ANSWER   = Path("C:/Users/nolyb/Desktop/vol/정상상태_정답.xlsx")            # [신규]
FILE_NH_TSI   = Path("C:/Users/nolyb/Desktop/vol/Non_heating_Tsi.xlsx")         # [신규]
FILE_NH_Q2D   = Path("C:/Users/nolyb/Desktop/vol/Non_heating_totalQ_2d.xlsx")   # [신규]
OUTPUT_XLSX   = Path("모델성능_최종결과_v6_3.xlsx")

L_REF_M = 2.2
N_HOURS = 744
N_NIGHTS = 30
HOURS_PER_NIGHT = 14
SEED = 0

# ------------------------------------------------------------
# [v6.1] ANN / SVR 재튜닝 설정 (2변수 기준)
#   True  : 아래 튜닝 설정 사용 (3변수는 기존 설정 유지)
#   False : 기존 v6 설정 그대로 (ANN (16,8)/alpha=0.1/early_stopping, SVR C=1)
# ------------------------------------------------------------
USE_TUNED_ANN_SVR = True

ANN_HIDDEN_2V = (32, 16)          # 후보: (16,8) 기존 / (32,16) 권장 / (64,32) 성능최고이나 탐색범위 끝
ANN_ALPHA_2V = 1e-3
ANN_TOL_2V = 1e-7
ANN_N_ITER_NO_CHANGE_2V = 50
ANN_MAX_ITER_2V = 2000

SVR_C_2V = 10.0                   # 후보: C=3~10 (C>=100은 gamma가 크면 과적합)
# gamma 선택 근거 (C=10): 정상상태 LOCO R²는 gamma가 클수록 좋아지지만(1→0.875, 2→0.903, 50→0.955),
#   비난방 검증(입력분포가 달라지는 상황)은 gamma가 크면 붕괴함(안정화 전체 R²: 2→0.875, 50→0.406).
#   => LOCO 향상과 비난방 강건성이 모두 유지되는 균형점 gamma=2 채택. (LOCO 최고를 원하면 50, 단 비난방 취약)
SVR_GAMMA_2V = 2.0
SVR_EPS = 0.01

# max_iter 도달 시 뜨는 ConvergenceWarning은 의도된 동작이라 숨김
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DEPTHS = ['0mm', '10mm', '20mm', '30mm', '40mm', '50mm', '75mm',
          '100mm', '150mm', '200mm', '300mm', '500mm']

# 학습 30개 ([v6] 0.003 -> 0.004로 교체됨, 기존과 동일하게 항상 고정)
TRAIN_SHEETS = [
    "단열재 조정 0.06+결로 방지 단열재 0.0001", "단열재 조정 0.06+결로 방지 단열재 0.004",
    "결로 방지 단열재 0.0001", "결로 방지 단열재 0.002", "결로 방지 단열재 0.005",
    "결로 방지 단열재 0.009", "결로 방지 단열재 0.014", "결로 방지 단열재 0.022",
    "결로 방지 단열재 0.035", "결로 방지 단열재 0.055", "결로 방지 단열재 0.1",
    "단열재 조정 라등급(0.06)", "단열재 조정 라등급(0.05)", "단열재 조정 나등급(0.04)",
    "결로 방지 단열재 1.6", "결로 방지 단열재 2.2", "결로 방지 단열재 2.8",
    "단열재 조절 가등급 0.01", "결로 방지 단열재 4.3",
    "단열재 조정 0.01+결로 방지 단열재 2.8", "단열재 조정 0.01+결로 방지 단열재 3.5",
    "단열재 조정 0.01+결로 방지 단열재 4.3", "단열재 조정 0.01+결로 방지 단열재 5",
    "단열재 조정 0.01+결로 방지 단열재 5.8", "단열재 조정 0.01+결로 방지 단열재 6.6",
    "단열재 조정 0.01+결로 방지 단열재 7.4", "단열재 조정 0.01+결로 방지 단열재 8.2",
    "단열재 조정 0.01+결로 방지 단열재 9.2", "단열재 조정 0.01+결로 방지 단열재 10.2",
    "단열재 조정 0.01+결로 방지 단열재 11.2",
]
assert len(TRAIN_SHEETS) == 30

# [v6 수정] 독립 테스트셋 10개: (다지점파일 시트명, 00mm파일 시트명)
# 정상상태_정답.xlsx에 실제로 존재하는 것만 사용
TEST_CASES = [
    ("Model A", "Model A  "),
    ("단열재 조정 0.02", "단열재 조정 0.02"),
    ("단열재 조정 0.045", "단열재 조정 0.045"),
    ("단열재 조정 0.07", "단열재 조정 0.07"),
    ("결로 방지 단열재 0.0001 + 단열재 조정 0.024", "결로 방지 단열재 0.0001 + 단열재 조정 0.024"),
    ("결로 방지 단열재 0.005 + 단열재 조정 0.024", "결로 방지 단열재 0.005 + 단열재 조정 0.024"),
    ("결로 방지 단열재 0.012 + 단열재 조정 0.024", "결로 방지 단열재 0.012 + 단열재 조정 0.024"),
    ("결로 방지 단열재 7", "결로 방지 단열재 7"),
    ("결로 방지 단열재 10", "결로 방지 단열재 10"),
    ("결로 방지 단열재 12", "결로 방지 단열재 12"),
]
assert len(TEST_CASES) == 10

# 다지점 파일 시트명 -> 00mm.xlsx 시트명이 다른 경우 매핑
NAME_MAP_TO_PLANAR = {
    "Model A": "Model A  ",
    "단열재 조절 가등급 0.01": "단열재 조정 0.01",
}

# [신규] 데이터 파일 시트명 -> 정상상태_정답.xlsx의 케이스명이 다른 경우 매핑
NAME_MAP_TO_ANSWER = {
    "Model A": "Model A(0.024)",
    # [v6.1] 정답 파일은 0.05 단열재를 '다등급'으로 표기, 데이터 파일 4개는 '라등급(0.05)' 표기
    "단열재 조정 라등급(0.05)": "단열재 조정 다등급(0.05)",
}

# [신규] Non_heating_totalQ_2d.xlsx에서만 이름이 다른 경우 매핑
# (Non_heating_Tsi.xlsx는 TRAIN_SHEETS 이름을 그대로 씀)
NAME_MAP_TO_NH_Q2D = {
    "단열재 조절 가등급 0.01": "단열재 조정 0.01",
}

MODEL_NAMES = ["2차_회귀", "Random_Forest", "KNN", "ANN", "SVR"]

# [v6.2] 비난방 검증 전용 모델 목록: 기존 5개 + RF에 표준화·PCA 회전을 적용한 모델
# (LOCO·독립테스트는 MODEL_NAMES 5개만 사용)
NH_MODEL_NAMES = MODEL_NAMES + ["Random_Forest_PCA"]
VAR_LABELS = ["3변수", "2변수"]

# [수정] Phase 경계(시간, 0=1/1 0시부터 센 시간)
# 실제 스케줄: 1/3 18시경 난방 시작, 1/4 13시경 20도 도달
# 안정화는 전체 기간(4~30일차)이 아니라 요청에 따라 "1일치"(첫 안정화 야간, 4일차)만 사용
PHASE_BOUND_1 = 66   # 이 시각 전까지 = 비난방
PHASE_BOUND_2 = 90   # 이 시각 전까지 = 도달중
STABLE_NIGHTS_TO_USE = 30   # 안정화 구간에서 앞으로 며칠(야간)치를 쓸지 (30 = 안정화 27일 전체)

# [v6.3] 비난방 검증의 학습 방식
#   "loco"  : 검증 벽체 1개를 제외한 29개 '정상' 데이터로 학습 -> 그 벽체의 비난방 야간 예측 (fold마다 재학습)
#   "full30": 30개 전체 '정상' 데이터로 1번 학습 -> 30개 벽체의 비난방 야간 예측 (v6.2 방식)
NH_TRAIN_MODE = "loco"
assert NH_TRAIN_MODE in ("loco", "full30")


# ============================================================
# 1. 야간(18:00~07:00) 인덱스 (기존과 동일)
# ============================================================
def get_night_indices():
    nights = []
    for n in range(N_NIGHTS):
        start = n * 24 + 18
        nights.append(np.arange(start, start + HOURS_PER_NIGHT, dtype=int))
    return nights


NIGHTS = get_night_indices()
ALL_NIGHT_IDX = np.concatenate(NIGHTS)
NIGHT_START_HOURS = [n * 24 + 18 for n in range(N_NIGHTS)]


def get_phase(night_idx):
    h = NIGHT_START_HOURS[night_idx]
    if h < PHASE_BOUND_1:
        return "비난방"
    elif h < PHASE_BOUND_2:
        return "도달중"
    else:
        # 안정화 구간 진입 후 몇 번째 야간인지 계산 -> STABLE_NIGHTS_TO_USE개까지만 "안정화"로 인정
        stable_start_idx = next(n for n in range(N_NIGHTS) if NIGHT_START_HOURS[n] >= PHASE_BOUND_2)
        offset = night_idx - stable_start_idx
        if offset < STABLE_NIGHTS_TO_USE:
            return "안정화"
        return None   # 안정화 1일치를 넘어선 야간은 이번 출력에서 제외


PHASE_BY_NIGHT = [get_phase(n) for n in range(N_NIGHTS)]


# ============================================================
# 2. 케이스 하나 처리: 대표지점 고정 방식 (기존과 동일, 변경 없음)
# ============================================================
def build_case(ef_multi, ef_planar, sheet_multi, sheet_planar, use_b):
    raw = pd.read_excel(ef_multi, sheet_name=sheet_multi, header=0, nrows=N_HOURS)
    raw.columns = [str(c).strip() for c in raw.columns]

    temp_cols = [f"{d} 표면온도(℃)" for d in DEPTHS]
    temps_744 = raw[temp_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    temps_night = temps_744[ALL_NIGHT_IDX]

    argmin_idx = temps_night.argmin(axis=1)
    vals, counts = np.unique(argmin_idx, return_counts=True)
    temp_dom_idx = vals[np.argmax(counts)]
    min_temp_744 = temps_744[:, temp_dom_idx]

    if use_b:
        flux_cols = [f"{d} 열류량" for d in DEPTHS]
        flux_744 = raw[flux_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        flux_night = flux_744[ALL_NIGHT_IDX]
        argmax_idx = flux_night.argmax(axis=1)
        vals2, counts2 = np.unique(argmax_idx, return_counts=True)
        flux_dom_idx = vals2[np.argmax(counts2)]
        max_flux_744 = flux_744[:, flux_dom_idx]

    raw_p = pd.read_excel(ef_planar, sheet_name=sheet_planar, header=0, nrows=N_HOURS)
    raw_p.columns = [str(c).strip() for c in raw_p.columns]
    ts_planar_col = "1000mm 표면온도(℃)" if "1000mm 표면온도(℃)" in raw_p.columns else "비열교부 표면온도(℃)"
    q_planar_col = ("1000mm 석고보드 국부 열류량(W/m2)" if "1000mm 석고보드 국부 열류량(W/m2)" in raw_p.columns
                     else "비열교부 석고보드 국부 열류량(W/m2)")

    t_out_744 = pd.to_numeric(raw_p["실외 온도(℃)"], errors="coerce").to_numpy()
    t_in_744 = pd.to_numeric(raw_p["실내 온도(℃)"], errors="coerce").to_numpy()
    ts_planar_744 = pd.to_numeric(raw_p[ts_planar_col], errors="coerce").to_numpy()
    q_planar_744 = pd.to_numeric(raw_p[q_planar_col], errors="coerce").to_numpy()
    qin1_744 = pd.to_numeric(raw_p["실내로 들어오는 열류량1(W)"], errors="coerce").to_numpy()
    qout1_744 = pd.to_numeric(raw_p["실내에서 나가는 열류량1(W)"], errors="coerce").to_numpy()
    qin2_744 = pd.to_numeric(raw_p["실내로 들어오는 열류량2(W)"], errors="coerce").to_numpy()
    qout2_744 = pd.to_numeric(raw_p["실내에서 나가는 열류량2(W)"], errors="coerce").to_numpy()

    t_out = t_out_744[ALL_NIGHT_IDX]
    t_in = t_in_744[ALL_NIGHT_IDX]
    ts_planar = ts_planar_744[ALL_NIGHT_IDX]
    q_planar = q_planar_744[ALL_NIGHT_IDX]
    min_temp_n = min_temp_744[ALL_NIGHT_IDX]

    c_air = t_in - t_out
    a_surf = ts_planar - min_temp_n

    if use_b:
        max_flux_n = max_flux_744[ALL_NIGHT_IDX]
        b_flux = max_flux_n - q_planar
        seq = np.stack([c_air, a_surf, b_flux], axis=1).reshape(N_NIGHTS, HOURS_PER_NIGHT, 3)
    else:
        seq = np.stack([c_air, a_surf], axis=1).reshape(N_NIGHTS, HOURS_PER_NIGHT, 2)

    return seq


# ============================================================
# [신규] 2-1. 정상상태 정답 테이블 로드 + 조회
# ============================================================
def load_answer_table():
    df = pd.read_excel(FILE_ANSWER, sheet_name="Sheet1")
    df.columns = ["케이스", "전체열류량", "전체열류량2", "2D", "선형열관류율"]
    df["케이스"] = df["케이스"].astype(str).str.strip()
    return df.set_index("케이스")["선형열관류율"]


def get_answer(answer_series, sheet_name):
    key = NAME_MAP_TO_ANSWER.get(sheet_name, sheet_name)
    if key not in answer_series.index:
        raise KeyError(f"정상상태_정답.xlsx에서 '{key}'(원래 시트명: '{sheet_name}')를 찾을 수 없습니다.")
    return float(answer_series.loc[key])


# ============================================================
# [신규] 2-2. 비난방 조건 (C,A) 시퀀스 계산
#        - 대표지점은 비난방 조건 자체에서 다시 계산(케이스별 독립)
#        - 비열교부 표면온도는 Non_heating_totalQ_2d.xlsx에서 가져옴
# ============================================================
def build_case_2var_nonheating(case_name):
    sheet_tsi = case_name  # Tsi파일은 TRAIN_SHEETS 이름 그대로 사용
    sheet_q2d = NAME_MAP_TO_NH_Q2D.get(case_name, case_name)

    df_tsi = pd.read_excel(FILE_NH_TSI, sheet_name=sheet_tsi, header=0, nrows=N_HOURS)
    df_tsi.columns = [str(c).strip() for c in df_tsi.columns]
    temp_cols = [f"{d} 표면온도(℃)" for d in DEPTHS]
    temps_744 = df_tsi[temp_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    temps_night = temps_744[ALL_NIGHT_IDX]
    argmin_idx = temps_night.argmin(axis=1)
    vals, counts = np.unique(argmin_idx, return_counts=True)
    temp_dom_idx = vals[np.argmax(counts)]
    min_temp_744 = temps_744[:, temp_dom_idx]

    t_out_744 = pd.to_numeric(df_tsi["실외 온도(℃)"], errors="coerce").to_numpy()
    t_in_744 = pd.to_numeric(df_tsi["실내 온도(℃)"], errors="coerce").to_numpy()

    df_q = pd.read_excel(FILE_NH_Q2D, sheet_name=sheet_q2d, header=0, nrows=N_HOURS)
    df_q.columns = [str(c).strip() for c in df_q.columns]
    ts_planar_744 = pd.to_numeric(df_q["비열교부 표면온도(℃)"], errors="coerce").to_numpy()

    t_out = t_out_744[ALL_NIGHT_IDX]
    t_in = t_in_744[ALL_NIGHT_IDX]
    ts_planar = ts_planar_744[ALL_NIGHT_IDX]
    min_temp_n = min_temp_744[ALL_NIGHT_IDX]

    c_air = t_in - t_out
    a_surf = ts_planar - min_temp_n
    seq = np.stack([c_air, a_surf], axis=1).reshape(N_NIGHTS, HOURS_PER_NIGHT, 2)
    return seq


# ============================================================
# [추가] 시간별 예측 시트용 설정/함수 (기존 코드는 수정하지 않고 아래 내용만 추가)
#   - 대상: 결로 방지 단열재 0.035 의 비난방 시나리오(비난방 -> 도달중 -> 안정화), 744시간 전체
#   - 모델: Random_Forest, KNN (모델별로 열을 묶어서 출력)
#   - 입력: 야간 14시간 평균이 아니라 '1시간 단위' (ΔT, ΔTs)를 그대로 모델에 넣음
#   - 학습: 비난방 검증과 같은 방식(NH_TRAIN_MODE) -> loco면 이 벽체를 뺀 29개 '정상' 데이터로 학습
# ============================================================
HOURLY_CASE = "결로 방지 단열재 0.035"
HOURLY_MODELS = ["Random_Forest", "KNN"]
HOURLY_SHEET_NAME = "시간별예측_0.035_RF_KNN"


def build_hourly_nonheating_inputs(case_name):
    """build_case_2var_nonheating 과 같은 정의(대표지점·ΔT·ΔTs)로 744시간 전체의 값을 돌려준다."""
    df_tsi = pd.read_excel(FILE_NH_TSI, sheet_name=case_name, header=0, nrows=N_HOURS)
    df_tsi.columns = [str(c).strip() for c in df_tsi.columns]
    temp_cols = [f"{d} 표면온도(℃)" for d in DEPTHS]
    temps_744 = df_tsi[temp_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    argmin_idx = temps_744[ALL_NIGHT_IDX].argmin(axis=1)
    vals, counts = np.unique(argmin_idx, return_counts=True)
    min_temp_744 = temps_744[:, vals[np.argmax(counts)]]

    t_out = pd.to_numeric(df_tsi["실외 온도(℃)"], errors="coerce").to_numpy()
    t_in = pd.to_numeric(df_tsi["실내 온도(℃)"], errors="coerce").to_numpy()

    sheet_q2d = NAME_MAP_TO_NH_Q2D.get(case_name, case_name)
    df_q = pd.read_excel(FILE_NH_Q2D, sheet_name=sheet_q2d, header=0, nrows=N_HOURS)
    df_q.columns = [str(c).strip() for c in df_q.columns]
    ts_planar_744 = pd.to_numeric(df_q["비열교부 표면온도(℃)"], errors="coerce").to_numpy()

    hour_idx = np.arange(N_HOURS)
    is_night = np.zeros(N_HOURS, dtype=bool)
    is_night[ALL_NIGHT_IDX] = True
    night_no = np.full(N_HOURS, np.nan)
    for n, idx in enumerate(NIGHTS):
        night_no[idx] = n + 1
    phase = np.where(hour_idx < PHASE_BOUND_1, "비난방", np.where(hour_idx < PHASE_BOUND_2, "도달중", "안정화"))

    return pd.DataFrame({
        "시간": pd.Timestamp("2026-01-01") + pd.to_timedelta(hour_idx, unit="h"),
        "시간번호(0=1/1 0시)": hour_idx,
        "구간": phase,
        "야간여부": np.where(is_night, "야간", "주간"),
        "야간번호": night_no,
        "실외온도(℃)": t_out, "실내온도(℃)": t_in,
        "ΔT(C)": t_in - t_out,
        "ΔTs(A)": ts_planar_744 - min_temp_744,
    })


def build_hourly_prediction_sheet(case_name, seqs_normal_2v, targets_2v):
    """시간별 (ΔT, ΔTs) -> Ψ 예측. 공통 열(시간/구간/입력) 뒤에 모델별 열을 묶어서 붙인다."""
    df = build_hourly_nonheating_inputs(case_name)
    train_names = [x for x in TRAIN_SHEETS if x != case_name] if NH_TRAIN_MODE == "loco" else TRAIN_SHEETS
    fns = fit_nh_models(train_names, seqs_normal_2v, targets_2v)
    actual = float(targets_2v[case_name])
    X = df[["ΔT(C)", "ΔTs(A)"]].to_numpy(dtype=float)
    df["정답Ψ"] = actual
    for m in HOURLY_MODELS:
        p = np.asarray(fns[m](X), dtype=float)
        df[f"{m}_예측Ψ"] = p
        df[f"{m}_오차율(%)"] = np.abs(p - actual) / abs(actual) * 100
    return df


# ============================================================
# 3. 5개 모델 학습 + 예측 (기존과 동일, 변경 없음)
# ============================================================
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def fit_ann(X_train, y_train, nvar):
    """ANN 학습 후, 원본 스케일 X를 받아 예측값을 돌려주는 함수를 반환"""
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    if USE_TUNED_ANN_SVR and nvar == 2:
        # [v6.1] 정답 표준화 + 엄격 수렴 (Ψ가 작아 기본 tol에서 조기 수렴하던 문제 해결)
        y_arr = np.asarray(y_train, dtype=float)
        y_mean, y_std = float(y_arr.mean()), float(y_arr.std())
        ann = MLPRegressor(hidden_layer_sizes=ANN_HIDDEN_2V, max_iter=ANN_MAX_ITER_2V,
                            alpha=ANN_ALPHA_2V, tol=ANN_TOL_2V,
                            n_iter_no_change=ANN_N_ITER_NO_CHANGE_2V,
                            random_state=SEED, early_stopping=False
                            ).fit(Xtr, (y_arr - y_mean) / y_std)
        return lambda X: ann.predict(sc.transform(X)) * y_std + y_mean
    # 기존 v6 설정
    ann = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=5000, alpha=0.1,
                        random_state=SEED, early_stopping=True).fit(Xtr, y_train)
    return lambda X: ann.predict(sc.transform(X))


def fit_svr(X_train, y_train, nvar):
    """SVR 학습 후, 원본 스케일 X를 받아 예측값을 돌려주는 함수를 반환"""
    sc = StandardScaler().fit(X_train)
    if USE_TUNED_ANN_SVR and nvar == 2:
        svr = SVR(kernel="rbf", C=SVR_C_2V, gamma=SVR_GAMMA_2V, epsilon=SVR_EPS)   # [v6.1]
    else:
        svr_C = 100.0 if nvar == 3 else 1.0                                        # 기존 v6
        svr = SVR(kernel="rbf", C=svr_C, epsilon=0.01)
    svr.fit(sc.transform(X_train), y_train)
    return lambda X: svr.predict(sc.transform(X))


def fit_predict_all(X_train_mean, y_train, X_test_mean, nvar):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    reg = LinearRegression().fit(poly.fit_transform(X_train_mean), y_train)
    pred_reg = reg.predict(poly.transform(X_test_mean))

    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X_train_mean, y_train)
    pred_rf = rf.predict(X_test_mean)

    mm = MinMaxScaler().fit(X_train_mean)
    knn = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(mm.transform(X_train_mean), y_train)
    pred_knn = knn.predict(mm.transform(X_test_mean))

    pred_ann = fit_ann(X_train_mean, y_train, nvar)(X_test_mean)
    pred_svr = fit_svr(X_train_mean, y_train, nvar)(X_test_mean)

    return {
        "2차_회귀": pred_reg, "Random_Forest": pred_rf, "KNN": pred_knn,
        "ANN": pred_ann, "SVR": pred_svr,
    }


# ============================================================
# [v6.3] 비난방 검증용 6개 모델 학습 (fold마다 호출 가능하도록 함수로 분리)
# ============================================================
def fit_nh_models(train_names, seqs_normal_2v, targets_2v):
    """train_names 케이스의 '정상(20도)' 데이터(2변수)로 6개 모델을 학습하고,
    {모델명: 예측함수(원본 스케일 X)} 를 반환한다."""
    X = np.concatenate([seqs_normal_2v[c].mean(axis=1) for c in train_names], axis=0)
    y = np.concatenate([np.full(N_NIGHTS, targets_2v[c]) for c in train_names])

    poly = PolynomialFeatures(degree=2, include_bias=False)
    reg = LinearRegression().fit(poly.fit_transform(X), y)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X, y)
    mm = MinMaxScaler().fit(X)
    knn = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(mm.transform(X), y)

    # RF + 표준화·PCA 회전: 스케일러/PCA는 학습 데이터로만 fit, 학습·테스트에 같은 변환 적용
    sc_p = StandardScaler().fit(X)
    pca = PCA(n_components=2).fit(sc_p.transform(X))
    rf_p = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(
        pca.transform(sc_p.transform(X)), y)

    return {
        "2차_회귀": lambda Z: reg.predict(poly.transform(Z)),
        "Random_Forest": lambda Z: rf.predict(Z),
        "KNN": lambda Z: knn.predict(mm.transform(Z)),
        "ANN": fit_ann(X, y, 2),          # 2변수 튜닝 설정 공유
        "SVR": fit_svr(X, y, 2),
        "Random_Forest_PCA": lambda Z: rf_p.predict(pca.transform(sc_p.transform(Z))),
    }


# ============================================================
# 5. 메인
# ============================================================
def main():
    answer_series = load_answer_table()
    print(f"정답 테이블 로드 완료: {len(answer_series)}개 케이스")
    print(f"ANN·SVR 튜닝 설정 사용(2변수): {USE_TUNED_ANN_SVR}")

    with pd.ExcelFile(FILE_MULTI, engine="openpyxl") as ef_multi, \
         pd.ExcelFile(FILE_PLANAR, engine="openpyxl") as ef_planar:

        test_names = [c[0] for c in TEST_CASES]
        overlap = set(TRAIN_SHEETS) & set(test_names)
        assert not overlap, f"데이터 누수! 겹치는 케이스: {overlap}"

        data = {}
        for use_b, label in [(True, "3변수"), (False, "2변수")]:
            seqs, targets = {}, {}
            for s in TRAIN_SHEETS:
                s0 = NAME_MAP_TO_PLANAR.get(s, s)
                seqs[s] = build_case(ef_multi, ef_planar, s, s0, use_b)
                targets[s] = get_answer(answer_series, s)

            tseqs, tactuals = {}, {}
            for sheet_v2, sheet_0 in TEST_CASES:
                tseqs[sheet_v2] = build_case(ef_multi, ef_planar, sheet_v2, sheet_0, use_b)
                tactuals[sheet_v2] = get_answer(answer_series, sheet_v2)

            data[label] = dict(seqs=seqs, targets=targets, tseqs=tseqs, tactuals=tactuals)
            print(f"{label} 데이터 준비 완료")

        # [신규] 비난방 조건 (C,A) 시퀀스 - 2변수만, 학습 30개 전체
        print("\n비난방 조건 데이터 로딩 중...")
        seqs_nonheat = {}
        for c in TRAIN_SHEETS:
            seqs_nonheat[c] = build_case_2var_nonheating(c)
            print(f"  {c} 완료")

    loco_night_rows, test_case_rows, test_night_rows = [], [], []

    for label in VAR_LABELS:
        nvar = 3 if label == "3변수" else 2
        seqs, targets = data[label]["seqs"], data[label]["targets"]
        tseqs, tactuals = data[label]["tseqs"], data[label]["tactuals"]

        # ---- LOCO ----
        # [v6 수정] 케이스 평균(30개)으로 뭉치지 않고, 30fold x 30야간 = 900개를
        # 평균 없이 개별로 전부 남긴다. 요약 지표(R², 오차율)도 이 900개 기준으로 계산.
        for held_out in TRAIN_SHEETS:
            fold_train = [c for c in TRAIN_SHEETS if c != held_out]
            X_train_mean = np.concatenate([seqs[c].mean(axis=1) for c in fold_train], axis=0)
            y_train = np.concatenate([np.full(N_NIGHTS, targets[c]) for c in fold_train])
            X_test_mean = seqs[held_out].mean(axis=1)   # (30,nvar) 야간별 그대로

            preds = fit_predict_all(X_train_mean, y_train, X_test_mean, nvar)
            actual = targets[held_out]
            for m in MODEL_NAMES:
                arr = preds[m]
                for night_idx in range(N_NIGHTS):
                    p = float(arr[night_idx])
                    loco_night_rows.append({
                        "구분": label, "케이스": held_out, "야간번호": night_idx + 1, "모델": m,
                        "실제Ψ": actual, "예측Ψ": p,
                        "오차율(%)": abs(p - actual) / abs(actual) * 100,
                    })
        print(f"{label} LOCO 완료 (야간 단위, 30fold x 30야간 = {30*30}개)")

        # ---- 독립 테스트셋 10개 (기존과 동일 구조) ----
        X_train_mean_full = np.concatenate([seqs[c].mean(axis=1) for c in TRAIN_SHEETS], axis=0)
        y_train_full = np.concatenate([np.full(N_NIGHTS, targets[c]) for c in TRAIN_SHEETS])

        for sheet_v2, _sheet_0 in TEST_CASES:
            X_test_mean = tseqs[sheet_v2].mean(axis=1)
            preds = fit_predict_all(X_train_mean_full, y_train_full, X_test_mean, nvar)
            actual = tactuals[sheet_v2]
            for m in MODEL_NAMES:
                arr = preds[m]
                p = float(np.mean(arr))
                test_case_rows.append({"구분": label, "케이스": sheet_v2, "모델": m,
                                        "실제Ψ": actual, "예측Ψ": p,
                                        "오차율(%)": abs(p - actual) / abs(actual) * 100})
                # [추가] 30야간 평균 없이 개별로도 남김 (테스트셋 야간단위 산점도용)
                for night_idx in range(N_NIGHTS):
                    pn = float(arr[night_idx])
                    test_night_rows.append({
                        "구분": label, "케이스": sheet_v2, "야간번호": night_idx + 1, "모델": m,
                        "실제Ψ": actual, "예측Ψ": pn,
                        "오차율(%)": abs(pn - actual) / abs(actual) * 100,
                    })
        print(f"{label} 테스트셋 완료")

    df_loco_night = pd.DataFrame(loco_night_rows)
    df_test_case = pd.DataFrame(test_case_rows)
    df_test_night = pd.DataFrame(test_night_rows)   # [추가] 테스트셋 야간단위(10케이스x30야간, 평균없음)

    # ============================================================
    # [신규] 비난방->도달중->안정화 검증 (2변수만, 학습 30개 전체)
    #   - 학습: 30개 전체의 "정상" 데이터
    #   - 테스트: 그 30개 케이스의 "비난방" 데이터, 야간별 개별(평균 없음)
    # ============================================================
    print(f"\n비난방 검증 시작 (학습방식={NH_TRAIN_MODE}, 안정화 {STABLE_NIGHTS_TO_USE}일치)...")
    seqs_normal_2v = data["2변수"]["seqs"]
    targets_2v = data["2변수"]["targets"]

    if NH_TRAIN_MODE == "full30":
        nh_fns_all = fit_nh_models(TRAIN_SHEETS, seqs_normal_2v, targets_2v)

    # [신규] C,A 원자료 (정상 vs 비난방, 30케이스x30야간) + 비난방 예측
    ca_rows = []
    pred_rows = []
    for fold_no, c in enumerate(TRAIN_SHEETS, start=1):
        if NH_TRAIN_MODE == "loco":
            fold_train = [x for x in TRAIN_SHEETS if x != c]        # 검증 벽체 제외 29개
            nh_fns = fit_nh_models(fold_train, seqs_normal_2v, targets_2v)
        else:
            nh_fns = nh_fns_all

        seq_normal_night = seqs_normal_2v[c].mean(axis=1)   # (30,2)
        seq_nonheat_night = seqs_nonheat[c].mean(axis=1)    # (30,2)
        actual = targets_2v[c]

        for n in range(N_NIGHTS):
            ca_rows.append({
                "케이스": c, "야간번호": n + 1, "Phase": PHASE_BY_NIGHT[n],
                "정상_C": seq_normal_night[n, 0], "정상_A": seq_normal_night[n, 1],
                "비난방_C": seq_nonheat_night[n, 0], "비난방_A": seq_nonheat_night[n, 1],
                "정답Ψ": actual,
            })

        X_test_nh = seq_nonheat_night
        for m in NH_MODEL_NAMES:
            preds = nh_fns[m](X_test_nh)
            for n in range(N_NIGHTS):
                p = float(preds[n])
                pred_rows.append({
                    "Fold": fold_no, "학습방식": NH_TRAIN_MODE,
                    "케이스": c, "야간번호": n + 1, "Phase": PHASE_BY_NIGHT[n], "모델": m,
                    "비난방_C": seq_nonheat_night[n, 0], "비난방_A": seq_nonheat_night[n, 1],
                    "실제Ψ": actual, "예측Ψ": p,
                    "오차율(%)": abs(p - actual) / abs(actual) * 100,
                })
        print(f"  [비난방 {NH_TRAIN_MODE} {fold_no}/{len(TRAIN_SHEETS)}] {c} 완료", flush=True)

    df_nh_ca = pd.DataFrame(ca_rows)
    df_nh_pred = pd.DataFrame(pred_rows)

    # [수정] 안정화 1일치를 넘어서는 야간(Phase=None)은 이번 출력에서 제외
    df_nh_ca = df_nh_ca[df_nh_ca["Phase"].notna()].reset_index(drop=True)
    df_nh_pred = df_nh_pred[df_nh_pred["Phase"].notna()].reset_index(drop=True)

    n_nights_used = sum(p is not None for p in PHASE_BY_NIGHT)   # 케이스당 실제 사용 야간 수
    print(f"비난방 검증 완료(학습방식={NH_TRAIN_MODE}, 안정화 {STABLE_NIGHTS_TO_USE}일치 사용): "
          f"C,A 원자료 {len(df_nh_ca)}행, 예측 원자료 {len(df_nh_pred)}행 "
          f"(30케이스x{n_nights_used}야간x{len(NH_MODEL_NAMES)}모델={30 * n_nights_used * len(NH_MODEL_NAMES)} 기대)")

    # [신규] Phase별로 완전히 나눠서 별도 표로도 준비 (비난방끼리/도달중끼리/안정화끼리)
    PHASE_ORDER = ["비난방", "도달중", "안정화"]
    df_nh_pred_by_phase = {p: df_nh_pred[df_nh_pred["Phase"] == p].reset_index(drop=True) for p in PHASE_ORDER}
    df_nh_ca_by_phase = {p: df_nh_ca[df_nh_ca["Phase"] == p].reset_index(drop=True) for p in PHASE_ORDER}

    nh_phase_summary = (
        df_nh_pred.groupby(["모델", "Phase"])
        .apply(lambda g: pd.Series({
            "N": len(g),
            "R2": r2_score(g["실제Ψ"], g["예측Ψ"]),
            "평균오차율(%)": g["오차율(%)"].mean(),
            "중앙값오차율(%)": g["오차율(%)"].median(),
        }))
        .reset_index()
    )
    # [v6.3] 3구간 합산(전체) 행 추가
    pooled_rows = []
    for m in NH_MODEL_NAMES:
        g = df_nh_pred[df_nh_pred["모델"] == m]
        pooled_rows.append({
            "모델": m, "Phase": "전체합산", "N": len(g),
            "R2": r2_score(g["실제Ψ"], g["예측Ψ"]),
            "평균오차율(%)": g["오차율(%)"].mean(),
            "중앙값오차율(%)": g["오차율(%)"].median(),
        })
    nh_phase_summary = pd.concat([nh_phase_summary, pd.DataFrame(pooled_rows)], ignore_index=True)
    print("\n=== 비난방 검증 Phase별 요약 ===")
    print(nh_phase_summary.round(4).to_string(index=False))

    # ---------------- 기존 요약 테이블 ----------------
    def make_summary(df, actual_col="실제Ψ", pred_col="예측Ψ"):
        rows = []
        for label in VAR_LABELS:
            for m in MODEL_NAMES:
                sub = df[(df["구분"] == label) & (df["모델"] == m)]
                rows.append({
                    "구분": label, "모델": m,
                    "R2": r2_score(sub[actual_col], sub[pred_col]),
                    "RMSE": rmse(sub[actual_col], sub[pred_col]),
                    "평균오차율(%)": sub["오차율(%)"].mean(),
                })
        return pd.DataFrame(rows)

    # [v6 수정] LOCO 요약은 df_loco_night(야간단위, 900개)로 계산 -> night 기준 R²/오차율
    loco_summary = make_summary(df_loco_night)
    # [수정] 테스트 요약(R2/오차율)도 LOCO와 같이 야간 단위(10케이스 x 30야간 = 300개) 기준으로 계산
    test_summary = make_summary(df_test_night)
    test_summary_case = make_summary(df_test_case)   # 참고용: 케이스평균 10개 기준

    pivot_loco_err = loco_summary.pivot(index="모델", columns="구분", values="평균오차율(%)")[VAR_LABELS].reindex(MODEL_NAMES)
    pivot_loco_r2 = loco_summary.pivot(index="모델", columns="구분", values="R2")[VAR_LABELS].reindex(MODEL_NAMES)
    pivot_test_err = test_summary.pivot(index="모델", columns="구분", values="평균오차율(%)")[VAR_LABELS].reindex(MODEL_NAMES)
    pivot_test_r2 = test_summary.pivot(index="모델", columns="구분", values="R2")[VAR_LABELS].reindex(MODEL_NAMES)

    print("\n=== 요약: LOCO 평균오차율(%) ===")
    print(pivot_loco_err.round(2).to_string())
    print("\n=== 요약: 테스트셋 평균오차율(%) ===")
    print(pivot_test_err.round(2).to_string())

    # 케이스x모델 피벗 (참고용 - 야간 30개를 평균 내서 보기 좋게 표시)
    # [주의] 요약_LOCO_R2/오차율 시트는 이 케이스평균이 아니라 위쪽 df_loco_night
    # (야간단위 900개) 기준으로 계산됨. 이 피벗은 어디까지나 케이스별 개요 확인용.
    loco_actual = df_loco_night[df_loco_night["구분"] == "2변수"].drop_duplicates("케이스").set_index("케이스")["실제Ψ"].reindex(TRAIN_SHEETS)
    loco_pred, loco_err = {}, {}
    for label in VAR_LABELS:
        sub_label = df_loco_night[df_loco_night["구분"] == label]
        case_mean = sub_label.groupby(["케이스", "모델"], as_index=False)["예측Ψ"].mean()
        loco_pred[label] = case_mean.pivot(index="케이스", columns="모델", values="예측Ψ").reindex(TRAIN_SHEETS)[MODEL_NAMES]
        err_case_mean = sub_label.groupby(["케이스", "모델"], as_index=False)["오차율(%)"].mean()
        loco_err[label] = err_case_mean.pivot(index="케이스", columns="모델", values="오차율(%)").reindex(TRAIN_SHEETS)[MODEL_NAMES]

    test_names_order = [c[0] for c in TEST_CASES]
    test_actual = df_test_case[df_test_case["구분"] == "2변수"].drop_duplicates("케이스").set_index("케이스")["실제Ψ"].reindex(test_names_order)
    test_pred, test_err = {}, {}
    for label in VAR_LABELS:
        test_pred[label] = df_test_case[df_test_case["구분"] == label].pivot(index="케이스", columns="모델", values="예측Ψ").reindex(test_names_order)[MODEL_NAMES]
        test_err[label] = df_test_case[df_test_case["구분"] == label].pivot(index="케이스", columns="모델", values="오차율(%)").reindex(test_names_order)[MODEL_NAMES]

    note = pd.DataFrame({
        "항목": ["정답(목표값)", "학습/테스트", "모델", "[v6] LOCO 집계 기준", "v6 신규: 비난방 검증", "비난방 Phase 경계", "비난방 학습 방식", "[v6.1] ANN·SVR 하이퍼파라미터", "[v6.2] 비난방 검증 Random_Forest_PCA", "[v6.3] 테스트_원자료(야간단위)"],
        "내용": [
            "정상상태_정답.xlsx의 '선형열관류율'을 케이스명으로 조회(TRISCO 등 정상상태 해석)",
            "학습 30개(0.003->0.004 교체) / 테스트 10개(Model A, 조정0.02/0.045/0.07, 결로방지0.0001/0.005/0.012+조정0.024, 결로방지7/10/12)",
            "2차_회귀, Random_Forest, KNN, ANN, SVR (2변수 기준 비난방 검증도 동일 5모델)",
            "요약_LOCO_R2/오차율은 케이스 평균(30개)이 아니라 야간 단위(30fold x 30야간=900개, 평균 없음) 기준으로 계산함. LOCO_실제값/예측값/오차율 피벗 시트는 케이스별 개요 확인용으로 900개를 케이스별 평균 낸 참고 표임",
            "학습 30개 케이스 전체에 비난방(1/3 18시 난방시작~1/4 13시 20도도달) 조건 적용, 야간별(평균없음) 검증",
            f"h<{PHASE_BOUND_1}=비난방, h<{PHASE_BOUND_2}=도달중, 안정화는 그 이후 첫 {STABLE_NIGHTS_TO_USE}일치만 사용(그 뒤 야간은 출력 제외). Phase별로 시트를 완전히 분리해서 출력함(비난방_CA_비난방/도달중/안정화, 비난방_예측_비난방/도달중/안정화)",
            f"비난방 검증 학습방식={NH_TRAIN_MODE}. loco=검증 벽체 1개를 제외한 29개 벽체의 '정상(20도)' 데이터로 fold마다 재학습해 그 벽체의 비난방 야간을 예측(30fold x 30야간=900개/모델) / full30=30개 전체로 1번 학습. 어느 쪽이든 '비난방' 입력 자체는 학습에 쓰이지 않음. Phase별 N(loco, 안정화 30일치 설정): 비난방 60 / 도달중 30 / 안정화 810",
            f"2변수 튜닝={USE_TUNED_ANN_SVR} | ANN: hidden={ANN_HIDDEN_2V}, alpha={ANN_ALPHA_2V}, 정답표준화, tol={ANN_TOL_2V}, n_iter_no_change={ANN_N_ITER_NO_CHANGE_2V}, early_stopping=False, max_iter={ANN_MAX_ITER_2V} | SVR: C={SVR_C_2V}, gamma={SVR_GAMMA_2V}, epsilon={SVR_EPS} | 3변수·미튜닝: ANN (16,8)/alpha=0.1/early_stopping, SVR C=100(3변수)·1(2변수)/gamma='scale'",
            "비난방 검증에만 Random_Forest_PCA 행 추가: StandardScaler+PCA(2)를 학습 30개 정상 데이터로 fit, 학습·테스트를 같은 변환으로 회전 후 RF(n_estimators=300). 정상상태 LOCO·독립테스트에는 미적용(정상상태에서는 원래 C,A가 더 좋음). 기존 Random_Forest 행과 나란히 비교 가능",
            "기존 '테스트_원자료'(30야간 평균, 케이스당 1행)는 그대로 유지. 이 시트는 같은 예측을 30야간 평균 없이 개별로 남긴 것(10케이스x30야간x5모델x2변수구성=3000행): 실제Ψ/예측Ψ/오차율(%)이 야간별로 들어있어 LOCO_원자료(야간단위)와 같은 방식으로 산점도를 그릴 수 있음. 요약_테스트_R2/오차율/RMSE 시트는 이 야간 단위 300개 기준으로 계산함(케이스평균 10개 기준은 요약_테스트_RMSE_케이스평균·성능지표 시트에 참고용으로만 남김)",
        ],
    })

    # ============================================================
    # [추가] 학습 / 비난방 분석 결과를 feature 형태로 정리한 시트 (엑셀 오른쪽 끝에 추가됨)
    #   - 학습_2변수_feature   : 학습 30케이스 x 30야간 = 900개의 (Mean_C, Mean_A, True_Psi)  (v3와 같은 형식)
    #   - 비난방_2변수_feature : 비난방 검증 30케이스 x 30야간의 (Mean_C, Mean_A, True_Psi, 구간)
    #                            + 모델별 예측Ψ(Pred_*)  <- 비난방 검증 결과값
    #   위에서 이미 만들어진 변수를 읽기만 하고, 기존 DataFrame은 수정하지 않는다(.copy() 사용).
    # ============================================================
    case_id_map = {c: i for i, c in enumerate(TRAIN_SHEETS, start=1)}

    feat_rows = []
    for case_name in TRAIN_SHEETS:
        X_case_mean = data["2변수"]["seqs"][case_name].mean(axis=1)      # (30,2) 야간별 평균 C, A
        true_psi = float(data["2변수"]["targets"][case_name])
        for night_idx in range(N_NIGHTS):
            feat_rows.append({
                "Case_ID": case_id_map[case_name], "케이스": case_name, "야간번호": night_idx + 1,
                "Mean_C": float(X_case_mean[night_idx, 0]), "Mean_A": float(X_case_mean[night_idx, 1]),
                "True_Psi": true_psi,
            })
    df_training_feature = pd.DataFrame(feat_rows)
    assert len(df_training_feature) == len(TRAIN_SHEETS) * N_NIGHTS

    nh_feature = df_nh_ca[["케이스", "야간번호", "Phase", "비난방_C", "비난방_A", "정답Ψ"]].copy()
    nh_feature.insert(0, "Case_ID", nh_feature["케이스"].map(case_id_map))
    nh_feature = nh_feature.rename(columns={"Phase": "구간", "비난방_C": "Mean_C",
                                            "비난방_A": "Mean_A", "정답Ψ": "True_Psi"})
    nh_pred_wide = (df_nh_pred.pivot_table(index=["케이스", "야간번호"], columns="모델", values="예측Ψ")
                    [NH_MODEL_NAMES].add_prefix("Pred_").reset_index())
    df_nh_feature = nh_feature.merge(nh_pred_wide, on=["케이스", "야간번호"], how="left")
    assert len(df_nh_feature) == len(df_nh_ca)
    assert df_nh_feature[[f"Pred_{m}" for m in NH_MODEL_NAMES]].notna().all().all()

    # ============================================================
    # [추가] 성능지표 종합 (R2 / RMSE / MAPE)  -> 엑셀 맨 오른쪽 끝 시트로 추가됨
    #   기존 계산 결과와 기존 시트는 전혀 바꾸지 않고, 위에서 만든 DataFrame을 읽기만 한다.
    #   - R2   : 결정계수 (기존 요약_*_R2 시트와 같은 값)
    #   - RMSE : sqrt(mean((예측Ψ-실제Ψ)^2))   [W/(m·K)]
    #   - MAPE : mean(|예측Ψ-실제Ψ| / |실제Ψ|) x 100   [%]  (기존 '평균오차율(%)'과 같은 정의)
    #   계산 기준은 기존 요약 시트와 동일:
    #     LOCO   : 야간 단위 900개 (df_loco_night)
    #     테스트 : 야간 단위 300개 (df_test_night, 10케이스 x 30야간, 평균 없음)  <- LOCO(900개)와 같은 기준
    #              (케이스 평균 10개 기준은 "독립테스트(케이스평균)" 행/시트로 별도 표시)
    #     비난방 : Phase별 + 전체합산 (df_nh_pred, LOCO 방식 학습)
    # ============================================================
    pivot_loco_rmse = loco_summary.pivot(index="모델", columns="구분", values="RMSE")[VAR_LABELS].reindex(MODEL_NAMES)
    pivot_test_rmse_case = test_summary_case.pivot(index="모델", columns="구분", values="RMSE")[VAR_LABELS].reindex(MODEL_NAMES)   # 케이스평균 10개
    test_summary_night = make_summary(df_test_night)                                                                          # 야간 300개
    pivot_test_rmse = test_summary_night.pivot(index="모델", columns="구분", values="RMSE")[VAR_LABELS].reindex(MODEL_NAMES)

    def _metrics(_g):
        _a = _g["실제Ψ"].to_numpy(dtype=float)
        _p = _g["예측Ψ"].to_numpy(dtype=float)
        return {"N": len(_g), "R2": r2_score(_a, _p), "RMSE": rmse(_a, _p),
                "MAPE(%)": float(np.mean(np.abs(_p - _a) / np.abs(_a)) * 100)}

    _metric_rows = []
    for _name, _src in (("LOCO", df_loco_night), ("독립테스트", df_test_night), ("독립테스트(케이스평균)", df_test_case)):
        for _lab in VAR_LABELS:
            for _m in MODEL_NAMES:
                _g = _src[(_src["구분"] == _lab) & (_src["모델"] == _m)]
                _metric_rows.append({"검증": _name, "변수구성": _lab, "Phase": "-", "모델": _m, **_metrics(_g)})
    for _m in NH_MODEL_NAMES:
        for _ph in PHASE_ORDER + ["전체합산"]:
            _g = df_nh_pred[df_nh_pred["모델"] == _m]
            if _ph != "전체합산":
                _g = _g[_g["Phase"] == _ph]
            _metric_rows.append({"검증": "비난방검증", "변수구성": "2변수", "Phase": _ph, "모델": _m, **_metrics(_g)})
    df_metrics = pd.DataFrame(_metric_rows)
    df_metrics_note = pd.DataFrame({
        "지표": ["R2", "RMSE", "MAPE(%)", "N", "LOCO", "독립테스트", "독립테스트(케이스평균)", "비난방검증"],
        "설명": [
            "결정계수 (기존 요약_*_R2 시트와 같은 값)",
            "sqrt(mean((예측Ψ-실제Ψ)^2)), 단위 W/(m·K). 값이 작을수록 좋음",
            "mean(|예측Ψ-실제Ψ|/|실제Ψ|) x 100, 단위 %. 기존 '평균오차율(%)'과 같은 정의",
            "지표를 계산한 데이터 개수",
            "학습 30개 벽체 LOCO, 야간 단위 30fold x 30야간 = 900개 기준",
            "독립 테스트 10개 케이스 x 30야간 = 야간 단위 300개 기준(평균 없음, LOCO 900개와 같은 방식)",
            "참고용: 케이스별 30야간 평균 예측 10개 기준 (요약_테스트_R2/오차율은 이 기준이 아니라 야간 300개 기준)",
            "정상 데이터로 학습 -> 비난방 야간 예측. Phase별(비난방/도달중/안정화)과 전체합산, Random_Forest_PCA 포함",
        ],
    })

    # [추가] 결로 방지 단열재 0.035 비난방 시나리오의 시간별 예측 (RF, KNN)
    df_hourly_pred = build_hourly_prediction_sheet(HOURLY_CASE, data["2변수"]["seqs"], data["2변수"]["targets"])
    print(f"시간별 예측 시트용 데이터 준비 완료: {HOURLY_CASE}, {len(df_hourly_pred)}시간 x {HOURLY_MODELS}")

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        note.to_excel(writer, sheet_name="설명", index=False)
        pivot_loco_err.round(2).to_excel(writer, sheet_name="요약_LOCO오차율")
        pivot_loco_r2.round(4).to_excel(writer, sheet_name="요약_LOCO_R2")
        pivot_test_err.round(2).to_excel(writer, sheet_name="요약_테스트오차율")
        pivot_test_r2.round(4).to_excel(writer, sheet_name="요약_테스트_R2")

        loco_actual.round(4).to_frame("실제Ψ").to_excel(writer, sheet_name="LOCO_실제값")
        for label in VAR_LABELS:
            loco_pred[label].round(4).to_excel(writer, sheet_name=f"LOCO_{label}_예측값")
            loco_err[label].round(2).to_excel(writer, sheet_name=f"LOCO_{label}_오차율")

        test_actual.round(4).to_frame("실제Ψ").to_excel(writer, sheet_name="테스트_실제값")
        for label in VAR_LABELS:
            test_pred[label].round(4).to_excel(writer, sheet_name=f"테스트_{label}_예측값")
            test_err[label].round(2).to_excel(writer, sheet_name=f"테스트_{label}_오차율")

        # [수정] 비난방 검증 결과 - Phase별로 완전히 나눠서 별도 시트
        for p in PHASE_ORDER:
            df_nh_ca_by_phase[p].round(4).to_excel(writer, sheet_name=f"비난방_CA_{p}", index=False)
            df_nh_pred_by_phase[p].round(4).to_excel(writer, sheet_name=f"비난방_예측_{p}", index=False)
        nh_phase_summary.round(4).to_excel(writer, sheet_name="비난방_Phase별_요약", index=False)

        df_loco_night.round(4).to_excel(writer, sheet_name="LOCO_원자료(야간단위)", index=False)
        df_test_case.round(4).to_excel(writer, sheet_name="테스트_원자료", index=False)
        df_test_night.round(4).to_excel(writer, sheet_name="테스트_원자료(야간단위)", index=False)   # [추가]

        # [추가] 학습 / 비난방 분석 feature 시트 (오른쪽 끝)
        df_training_feature.round(6).to_excel(writer, sheet_name="학습_2변수_feature", index=False)
        df_nh_feature.round(6).to_excel(writer, sheet_name="비난방_2변수_feature", index=False)

        # [추가] RMSE / MAPE 성능 시트 (오른쪽 끝)
        pivot_loco_rmse.round(4).to_excel(writer, sheet_name="요약_LOCO_RMSE")
        pivot_test_rmse.round(4).to_excel(writer, sheet_name="요약_테스트_RMSE")                       # 야간 300개 기준
        pivot_test_rmse_case.round(4).to_excel(writer, sheet_name="요약_테스트_RMSE_케이스평균")        # 케이스평균 10개 기준 (참고)
        df_metrics.round(4).to_excel(writer, sheet_name="성능지표_R2_RMSE_MAPE", index=False)
        df_metrics_note.to_excel(writer, sheet_name="성능지표_R2_RMSE_MAPE", index=False, startrow=len(df_metrics) + 3)

        # [추가] 시간별 예측 시트 (맨 오른쪽 끝)
        df_hourly_pred.round(6).to_excel(writer, sheet_name=HOURLY_SHEET_NAME, index=False)

    print(f"\n결과 저장 완료: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()