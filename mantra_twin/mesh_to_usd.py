"""Write office_twin.usd: coloured visual mesh (no physics) + invisible static triangle-mesh collider. Z-up, metres.
Usage: python mantra_twin/mesh_to_usd.py <final_dir> [--name office_twin]"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt


def add_mesh(stage, path, mesh, colors: bool):
    m = UsdGeom.Mesh.Define(stage, path)
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.triangles, dtype=np.int32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr([Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())])
    if colors and mesh.has_vertex_colors():
        pv = m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_colors, dtype=np.float32)))
    if mesh.has_vertex_normals():
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype=np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final_dir", type=Path)
    ap.add_argument("--name", default="office_twin")
    a = ap.parse_args()
    d = a.final_dir.expanduser()
    out = d / f"{a.name}.usd"
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/OfficeTwin")
    stage.SetDefaultPrim(root.GetPrim())

    add_mesh(stage, "/OfficeTwin/visual", o3d.io.read_triangle_mesh(str(d / "office_visual.ply")), True)
    col = add_mesh(stage, "/OfficeTwin/collision", o3d.io.read_triangle_mesh(str(d / "office_collision.ply")), False)
    col.CreateVisibilityAttr(UsdGeom.Tokens.invisible)  # PhysX still collides with invisible prims
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)  # static: exact tris
    root.GetPrim().SetCustomDataByKey("source", "OAK-D scan, digital_twin/ pipeline (PI-3)")
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
