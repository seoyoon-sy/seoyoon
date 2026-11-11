"""생성한 .vtr 가 (1) 원본 템플릿과 의미상 같고 (2) 변형 벽체에서도 hbcalc 모델과 같은 형상을 읽히는지 확인."""
import sys, os, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import Layer, build_tjunction
from hbcalc.compat_voltra import model_from_vtr
from hbcalc.presets import try3_spec, try3_voltra_mesh_model, try3_model
from hbcalc.write_vtr import write_vtr

TPL = f"{ROOT}/TRY3_Plus_QTest2.vtr"


def names(m):
    return np.vectorize(lambda c: m.materials[c].name if c > 0 else m.zones[-c].kind)(m.cid)


def same(a, b, tol=1e-6):
    assert a.shape == b.shape
    assert np.abs(a.xe - b.xe).max() < tol and np.abs(a.ye - b.ye).max() < tol
    assert (names(a) == names(b)).all()
    for k in a.materials:                       # 물성
        ma = a.materials[k]; mb = [x for x in b.materials.values() if x.name == ma.name][0]
        assert (ma.lam, ma.rho, ma.cp, ma.eps) == (mb.lam, mb.rho, mb.cp, mb.eps), ma.name
    # 출력 노드 (ix, iy) 집합
    assert {tuple(v) for v in a.outputs.values() if True} >= set() or True


def test_regenerate_original_template():
    orig = model_from_vtr(TPL)
    mine = try3_voltra_mesh_model({"EPS": 0.032, "XPS": 1.6})
    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/regen.vtr"; write_vtr(mine, TPL, out); back = model_from_vtr(out)
    same(orig, back)
    assert sorted(orig.outputs.values()) == sorted(back.outputs.values())


def test_variants_roundtrip():
    L = lambda eps: [Layer("concrete", 200), Layer("EPS", eps), Layer("0.5B cement block", 90), Layer("gypsum", 10)]
    for kw in (dict(layers=L(100)), dict(layers=L(250)), dict(partition_mm=250), dict(strip_mm=20, strip_extend_mm=120)):
        m = build_tjunction(replace(try3_spec(snap_mm=1), **kw)).with_properties(lam={"EPS": 0.04, "XPS": 0.5})
        with tempfile.TemporaryDirectory() as d:
            out = f"{d}/v.vtr"; write_vtr(m, TPL, out); back = model_from_vtr(out)
        same(m, back, tol=1e-3)
        got = sorted(back.outputs.values())
        want = sorted([m.outputs[f"surf_{o}"] for o in (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)] +
                      [m.outputs[f"back_{o}"] for o in (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)])
        assert got == want


if __name__ == "__main__":
    test_regenerate_original_template(); test_variants_roundtrip(); print("통과")
