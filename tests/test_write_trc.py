import sys, os, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from dataclasses import replace
from hbcalc import build_tjunction
from hbcalc.compat_trisco import model_from_trc
from hbcalc.presets import try3_spec
from hbcalc.write_trc import write_trc

TPL = f"{ROOT}/trisco_inputs/TRY3_Plus.trc"


def test_trc_roundtrip_original_mesh_geometry():
    """TRISCO 템플릿의 격자 그대로 재생성하면 원본과 같은 셀 배치와 재료가 읽혀야 한다."""
    orig, _ = model_from_trc(TPL)
    xe, ye = tuple(orig.xe * 1000), tuple(orig.ye * 1000)
    spec = replace(try3_spec(x_edges_mm=xe, y_edges_mm=ye, strip_extend_mm=170.0, offsets_mm=(0, 50, 100, 150, 200, 300, 500)))
    m = build_tjunction(spec).with_properties(lam={"EPS": 0.06, "XPS": 0.004})
    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/r.trc"; write_trc(m, TPL, out); back, zv = model_from_trc(out)
    assert back.shape == orig.shape
    assert np.abs(back.xe - orig.xe).max() < 1e-6 and np.abs(back.ye - orig.ye).max() < 1e-6
    assert (back.cid == orig.cid).all()
    assert {k: v.lam for k, v in back.materials.items()} == {k: v.lam for k, v in orig.materials.items()}
