# 독립 테스트 T1 ~ T8 (XPS 판 밀도 25, 비열 1300으로 정정, 2026-11-16)

원본(`../independent_test/`)과 난수, 두께, 열전도율, 나머지 재료 물성이 모두 같고 결로 방지 단열재(XPS 판)의 밀도와 비열만 사용자의 TRY3 케이스 규칙(밀도 25, 비열 1300, 열전도율만 변경)으로 통일했다.
원본은 열전도율 0.06 이상인 판(T2, T4, T6, T8)의 밀도를 1000 ~ 2100으로 임의로 정했었고, 열전도율 0.0136 이하 판(T1, T3, T5, T7)은 밀도 40, 비열 1300 ~ 1450이었다.
- 생성: `python examples/make_independent_test_xps25.py`
- VOLTRA에서 다시 돌릴 것: T2, T4, T6, T8 난방과 T2, T4 비난방 6개(밀도가 크게 달랐던 벽체). T1, T3, T5, T7은 밀도 40 대 25, 비열 1300 ~ 1450 대 1300 차이가 작아서 선택 사항.
- 결과는 Report Output 형식으로 `voltra_results/independent_test_xps25/{이름}_{heating,nonheating}.txt` 이름으로 올려 주세요.
