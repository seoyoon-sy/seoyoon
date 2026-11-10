"""숫자만으로 케이스 정의 (캐드 불필요). dict 또는 JSON 파일 -> TJunction -> Model.

JSON 예 (examples/case_example.json):
{
  "materials": {"concrete": {"lam": 1.6, "rho": 2200, "cp": 795}, ...},
  "layers": [["concrete", 200], ["EPS", 180], ["0.5B cement block", 90], ["gypsum", 10]],
  "partition_material": "concrete", "partition_mm": 180,
  "strip_material": "XPS", "strip_mm": 10, "strip_from_layer": 1, "strip_extend_mm": 170,
  "wall_left_mm": 1000, "wall_right_mm": 1000, "room_depth_mm": 900, "z_mm": 1000,
  "mesh": "fine"
}
mesh: "fine"(FINE_MESH, 수렴한 해), "auto"(자동), 또는 생략. 그 밖의 TJunction 필드(h_min_mm 등)도 그대로 쓸 수 있다.
"""
import json
from dataclasses import fields
from .model import Material
from .geometry import Layer, TJunction, build_tjunction
from .presets import FINE_MESH


def spec_from_dict(d):
    d = dict(d)
    mats = {}
    for name, p in d.pop("materials").items():
        mats[name] = Material(name, p["lam"], p["rho"], p["cp"], p.get("eps", 0.9), p.get("refl", 0.1))
    layers = [Layer(n, float(t)) for n, t in d.pop("layers")]
    mesh = d.pop("mesh", None)
    if mesh == "fine":
        d = {**FINE_MESH, **d}
    elif mesh not in (None, "auto"):
        raise ValueError(f"mesh 는 'fine' 또는 'auto': {mesh}")
    ok = {f.name for f in fields(TJunction)}
    bad = set(d) - ok
    if bad:
        raise KeyError(f"알 수 없는 항목 {sorted(bad)}. 가능한 항목: {sorted(ok)}")
    if "offsets_mm" in d:
        d["offsets_mm"] = tuple(d["offsets_mm"])
    return TJunction(layers=layers, materials=mats, **d)


def model_from_dict(d):
    return build_tjunction(spec_from_dict(d))


def model_from_json(path):
    with open(path, encoding="utf-8") as f:
        return model_from_dict(json.load(f))
