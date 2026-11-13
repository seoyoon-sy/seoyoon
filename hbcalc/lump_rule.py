"""표면 칸 대류 면적 배분 가중치 w (모든 면 같은 값, 1.0 = 좌표가 작은 쪽 끝 노드에 전부 'low', 0.5 = 절반씩 'half').
VOLTRA 결과와의 경험적 맞춤이다 (CLAUDE.md 8.35, 8.36). 결로 방지 단열재(XPS 판)의 열전도율 lam 이 클수록 w = 1.0 이 맞고,
단열재급으로 작아질수록 w 가 낮아진다. 점들은 TRY3 손격자 VOLTRA 난방 케이스에서 야간 시간별 오차가 최소인 w 이고
(0.0001은 대칭 자동 격자 G2, G4에서 맞는 0.5와 절충해 0.4), log10(lam)에 대한 선형 보간이다.
블록 없는 벽체(N5b, N6, lam 0.04)는 w 0.8 ~ 0.85가 맞아 이 규칙(w = 1.0)과 어긋난다(알려진 한계)."""
import numpy as np

_L = np.log10([1e-4, 2e-3, 5e-3, 1.2e-2, 2.2e-2])
_W = [0.4, 0.55, 0.70, 0.90, 1.0]


def strip_weight(lam: float) -> float:
    return float(np.interp(np.log10(max(float(lam), 1e-8)), _L, _W))
