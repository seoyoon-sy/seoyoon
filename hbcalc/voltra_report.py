"""VOLTRA 'Report Output' (RTF 텍스트) 읽기. 반환: (열 설명 목록, (N, 31) 배열).
열 1 시간 s, 열 2~7 경계 영역 색 2, 1, 7 의 유입/유출 열류 W, 열 8~31 출력 노드 1~24 온도 ℃."""
import re
import numpy as np


def read_report(path):
    raw = open(path, "rb").read().decode("cp1252", errors="replace")
    lines = []
    for l in raw.replace("\r", "").split("\n"):
        l = re.sub(r"\\'[0-9a-fA-F]{2}", "?", l)
        l = re.sub(r"\\[a-zA-Z]+-?\d*[ ]?", "", l).replace("{", "").replace("}", "").strip()
        lines.append(l)
    cols = [l for l in lines if re.match(r"^Column \d+:", l)]
    first = max(i for i, l in enumerate(lines) if re.match(r"^Column \d+:", l)) + 1
    vals = " ".join(lines[first:]).split()
    arr = np.array([float(v) for v in vals])
    n = len(cols)
    if arr.size % n:
        raise ValueError(f"데이터 개수 {arr.size} 가 열 수 {n} 의 배수가 아님")
    return cols, arr.reshape(-1, n)
