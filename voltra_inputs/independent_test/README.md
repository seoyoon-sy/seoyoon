# 독립 테스트셋 (T1 ~ T8)

보정 규칙(`conv_lump='auto'`, `hbcalc/lump_rule.py`)을 고정한 뒤 한 번만 평가하기 위한 새 벽체 8개입니다. 규칙을 맞추는 데 쓰지 않습니다(보정용은 `voltra_inputs/calib_steady`).

- 생성: `python examples/make_independent_test.py` (난수 시드 20261113). 벽체 정의는 `wall_definitions.csv`.
- 두께, 열전도율, 밀도, 비열, 내벽 두께, 결로 방지 단열재(XPS 판) 두께와 돌출을 모두 무작위로 정했습니다. 블록 있음 4개(T1, T3, T5, T7), 없음 4개(T2, T4, T6, T8).
- XPS 판 열전도율은 구간별로 나눠 뽑았습니다: T3 0.00036, T1 0.0013, T7 0.0085, T5 0.013, T4 0.07, T8 0.084, T6 0.6, T2 3.0.
- 파일: `T1 ~ T8_heating.vtr`(난방 20도 고정, Wall_TI.FTE, 744시간) 8개, `T1 ~ T4_nonheating.vtr`(Wall_TI_2.FTE) 4개. 접합부, hbcalc 자동 격자(정수 mm), 출력 노드는 템플릿과 같은 12위치(표면과 석고보드 벽체쪽).
- 확인한 것: 생성한 파일을 읽어 들인 모델의 물성이 wall_definitions.csv와 같고, 실내온도 경계가 난방은 Wall_TI.FTE, 비난방은 Wall_TI_2.FTE이다. VOLTRA에서 열리는지는 확인하지 못했다(이 환경에 VOLTRA 없음).
- hbcalc 예측값은 일부러 계산하지 않았습니다. 보정을 끝내고 규칙을 고정한 뒤 처음 계산해서 VOLTRA와 비교합니다.
- VOLTRA 결과는 Report Output 형식(출력 노드 온도가 나오는 리포트)으로 올려 주세요. Calculation Results 형식은 쓸 수 없습니다.
