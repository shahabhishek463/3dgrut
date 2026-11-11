"""office_twin_scene.usda: the NuRec splat (what you see) + the static collider (what robots hit), one frame, Z up, metres.
Usage: python mantra_twin/compose_twin.py <final_dir>   (needs office_twin.usd and office_splat.usdz there)"""
import argparse
from pathlib import Path

from pxr import Usd, UsdGeom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final_dir", type=Path)
    ap.add_argument("--show-mesh", action="store_true", help="keep the coloured scan mesh visible as well")
    a = ap.parse_args()
    d = a.final_dir.expanduser().resolve()
    out = d / "office_twin_scene.usda"
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/OfficeTwin")
    stage.SetDefaultPrim(root.GetPrim())

    geo = stage.DefinePrim("/OfficeTwin/Geometry", "Xform")
    geo.GetReferences().AddReference("./office_twin.usd")  # /OfficeTwin: visual mesh + invisible collider
    if not a.show_mesh:
        UsdGeom.Imageable(stage.GetPrimAtPath("/OfficeTwin/Geometry/visual")).MakeInvisible()
    if (d / "office_splat.usdz").exists():
        splat = stage.DefinePrim("/OfficeTwin/Splat", "Xform")
        splat.GetReferences().AddReference("./office_splat.usdz")  # trained in this frame: no transform needed
    else:
        print("no office_splat.usdz yet: scene has the mesh only")
        UsdGeom.Imageable(stage.GetPrimAtPath("/OfficeTwin/Geometry/visual")).MakeVisible()
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
