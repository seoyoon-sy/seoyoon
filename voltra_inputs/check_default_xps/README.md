# 저장된 VOLTRA 결과가 현재 설정과 일치하는지 확인하는 한 케이스 (2026-11-16)
- `Z1_try3_xps1.6_nonheating.vtr`: TRY3 손격자(`TRY3_Plus_QTest2.vtr`) 그대로, 결로 방지 단열재(XPS 판) 열전도율 1.6, 밀도 25, 비열 1300, EPS 0.032. 실내온도는 템플릿 기본(Wall_TI_2.FTE, 비난방).
- 이 파일을 VOLTRA에서 돌린 결과(Report Output)를 기존 엑셀(voltra_30cases_night420_통합_v2... Non_heating 쪽 `결로 방지 단열재 1.6` 시트)의 0 mm 열과 비교한다. 같으면 저장된 결과는 현재 설정과 일치하고 오차의 원인은 hbcalc 쪽이다. 다르면 저장된 5개 케이스(가등급, 나등급, 라등급 2개, 1.6)가 다른 설정으로 돌았을 가능성이 크다.
- 결과 파일 이름은 `Z1_try3_xps1.6_nonheating.txt`로 `voltra_results/check_default_xps/`에 올려 주세요.
