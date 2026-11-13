"""표면 칸 대류 면적 배분 가중치 w (모든 면 같은 값, 1.0 = 좌표가 작은 쪽 끝 노드에 전부 'low', 0.5 = 절반씩 'half').
VOLTRA 결과와의 경험적 맞춤이다 (CLAUDE.md 8.35, 8.36, 8.46). 결로 방지 단열재(XPS 판)의 열전도율 lam 이 클수록 w = 1.0 이 맞고,
단열재급으로 작아질수록 w 가 낮아진다. log10(lam)에 대한 선형 보간. 최적 w 가 격자 종류에 따라 달라서 표가 두 개다.
 - mesh='auto' (기본): hbcalc 자동 격자(.vtr 생성기가 만드는 격자). calib_steady 14개 VOLTRA 정상상태(EPS 0.032 열 10점)로 맞춘 표.
   EPS 0.024, 0.06 4개 표본 밖 평균 RMSE 0.017, 교차 검증(LOO) 평균 RMSE 0.015.
 - mesh='hand': TRY3 손격자(VOLTRA 원본 격자) 난방 39케이스에서 맞춘 이전 표 (0.0001은 대칭 자동 격자와 절충해 0.4).
"""
import numpy as np

_TABLES = {
    "auto": ([1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 4e-2, 1e-1, 3e-1, 1.0], [0.575, 0.650, 0.775, 0.800, 0.825, 0.850, 0.875, 0.975, 1.0]),
    "hand": ([1e-4, 2e-3, 5e-3, 1.2e-2, 2.2e-2], [0.4, 0.55, 0.70, 0.90, 1.0]),
}


def strip_weight(lam: float, mesh: str = "auto") -> float:
    xs, ws = _TABLES[mesh]
    return float(np.interp(np.log10(max(float(lam), 1e-8)), np.log10(xs), ws))
