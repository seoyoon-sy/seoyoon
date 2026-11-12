# 물성만 바꾼 시험 벽체 .vtr (기하는 TRY3 템플릿 그대로)

벽체마다 파일 2개: `*_junction.vtr`(TRY3_Plus_QTest2.vtr 기반, VOLTRA 2D 접합부), `*_planar.vtr`(TRY3_Plus_2D.vtr 기반, 평면 벽체).
템플릿과 줄 수, 격자, 블록, 출력 노드가 완전히 같고 바뀐 것은 재료의 lamb, ro, c 줄과 실내온도 경계(접합부만: Wall_TI_2.FTE -> Wall_TI.FTE, funp 20 2 1440 360 0 0)뿐이다.
- 두께는 TRY3 그대로(콘크리트 200, EPS 180, 블록 90, 석고 10, 내벽 180, 띠 10). 방사율 0.9, 일사반사율 0.1도 그대로.
- 내벽 코어는 외벽 콘크리트와 같은 재료라 물성이 같다.
- 경계조건 파일 경로는 템플릿 그대로.
- hbcalc 기대값(expected_hbcalc.csv)은 접합부는 VOLTRA 손격자, 평면은 템플릿 격자 기준, 난방 20 도 고정, 일사 보정 0.
- 재생성: python examples/make_property_walls.py
