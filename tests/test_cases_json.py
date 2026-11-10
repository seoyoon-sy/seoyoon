import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from hbcalc import Solver, Boundary
from hbcalc.cases import model_from_json, model_from_dict
from hbcalc.presets import try3_model, FINE_MESH


def test_json_equals_preset():
    a = model_from_json(f"{ROOT}/examples/case_example.json")
    b = try3_model({"XPS": 0.004}, **FINE_MESH)
    assert a.shape == b.shape and np.abs(a.xe - b.xe).max() < 1e-12 and np.abs(a.ye - b.ye).max() < 1e-12
    assert (a.cid == b.cid).all()


def test_asymmetric_wall_lengths_and_bad_key():
    d = json.load(open(f"{ROOT}/examples/case_example.json", encoding="utf-8"))
    d.update(wall_left_mm=600, wall_right_mm=1400, mesh="auto")
    m = model_from_dict(d)
    assert abs(m.xe[-1] - (0.6 + 0.2 + 1.4)) < 1e-9
    s = Solver(m)
    s.run(Boundary.constant(5, -5, 20), nsteps=3)
    assert np.isfinite(s.series("surf_0")).all()
    d["wall_lenght_mm"] = 1
    try:
        model_from_dict(d); assert False
    except KeyError:
        pass
