# 시험 벽체 .vtr (열전도율, 밀도, 비열, 두께를 모두 바꾼 새 벽체 4종)

벽체마다 파일 2개: `*_junction.vtr`(VOLTRA 2D, T자 접합부) 와 `*_planar.vtr`(평면 벽체, 비열교부 기준).
정의는 `wall_definitions.csv`, 같은 조건 hbcalc 야간 평균 표면온도는 `expected_hbcalc.csv`.

- 실내온도는 20 도 고정(Wall_TI.FTE). 경계조건 파일 경로는 템플릿 그대로(C:\Users\nolyb\Desktop\어치\BCfile\...)라서 PC 경로에 맞춰야 한다.
- 재료 이름(concrete, EPS 등)은 템플릿 이름이 그대로 있는 라벨이다. 물성만 바뀌었다. 방사율 0.9, 일사반사율 0.1은 그대로.
- 내벽 코어는 외벽 콘크리트와 같은 재료 번호라 물성이 같다.
- 접합부 .vtr는 이전 변형 4종(A ~ D)과 같은 생성기라 VOLTRA에서 열렸다. 평면 .vtr 생성기는 새로 만들었고 VOLTRA에서 여는 것은 아직 못 해 봤다.
- 평면 .vtr는 사용자 템플릿 TRY3_Plus_2D.vtr에서 y 격자(층별 칸 두께 20 mm 이하), 블록, 물성, 출력 노드(표면, 벽체쪽)와 oblk의 y 범위만 바꿨다.
- 재생성: python examples/make_test_walls.py
