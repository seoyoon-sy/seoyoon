# 띠 열전도율 분리 시험 (정상상태)

N6의 접합부 0 mm 과보정이 띠 열전도율 때문인지 다른 구성 때문인지 가리기 위한 VOLTRA 입력 2개.

- `N6b_strip0.5.vtr`: N6 구성(블록 없음, 콘크리트 180, EPS 100, 석고 10, 내벽 180, 띠 10 mm 돌출 170)에 N5의 띠 재료(열전도율 0.5, 밀도 1200, 비열 1000).
- `N5b_strip0.04.vtr`: N5 구성(블록 없음, 콘크리트 220, EPS 160, 석고 12, 내벽 200)에 N6의 띠 재료(열전도율 0.04, 밀도 40, 비열 1450).

원본(`../test_walls/N5_noblock_junction.vtr`, `N6_noblock_insstrip_junction.vtr`) 대비 달라진 줄은 `calc=3600`(첫 스텝이 정상상태, 1시간 뒤까지만 계산)과 띠 재료 3줄(lamb, ro, c)뿐이다(diff로 확인). 난방 20 도 고정(Wall_TI.FTE). 정상상태라 밀도와 비열은 결과에 영향이 없다.

`expected_hbcalc_steady.csv`: 같은 조건 hbcalc 첫 스텝 표면온도와 영역 열류(half, low 두 배분, 원본 N5, N6 포함).
생성: `python examples/make_strip_split_cases.py`

## 확인할 것
VOLTRA 리포트의 시각 0 s 행(출력 노드 온도 24개: 위치마다 표면, 벽체쪽 순서)에서 표면 0 ~ 150 mm를 hbcalc low 값과 비교한다.
기대(가설): 과보정이 띠 열전도율 때문이면 N6b의 0 mm 편차가 원본 N6(+0.142)보다 크게 줄고, N5b가 원본 N5(-0.044)보다 크게 늘어난다. 구성(두께, 블록 없음 등) 때문이면 편차가 원본을 따라간다.
