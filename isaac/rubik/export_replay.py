"""Export the same measured bodies to the browser; no motion synthesis."""

import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdShade


def export_geometry(stage, names, output):
    paths = {
        p.GetName(): p
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in names
    }
    meshes = []
    for i, name in enumerate(names):
        body = paths[name]
        world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rigid = Gf.Matrix4d(1)
        rigid.SetRotate(Gf.Transform(world).GetRotation())
        rigid.SetTranslateOnly(world.ExtractTranslation())
        for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if "collision" in str(p.GetPath()).lower():
                continue
            if not (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube)):
                continue
            if UsdGeom.Imageable(p).ComputeVisibility() == "invisible":
                continue
            if p.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(p)
                points = mesh.GetPointsAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                if not points:
                    continue
                faces = []
                offset = 0
                for count in counts:
                    for j in range(1, count - 1):
                        faces.extend(
                            [
                                int(indices[offset]),
                                int(indices[offset + j]),
                                int(indices[offset + j + 1]),
                            ]
                        )
                    offset += count
            else:
                half = UsdGeom.Cube(p).GetSizeAttr().Get() / 2
                points = [
                    (x * half, y * half, z * half)
                    for x, y, z in [
                        (-1, -1, -1),
                        (1, -1, -1),
                        (1, 1, -1),
                        (-1, 1, -1),
                        (-1, -1, 1),
                        (1, -1, 1),
                        (1, 1, 1),
                        (-1, 1, 1),
                    ]
                ]
                faces = [
                    0,
                    2,
                    1,
                    0,
                    3,
                    2,
                    4,
                    5,
                    6,
                    4,
                    6,
                    7,
                    0,
                    1,
                    5,
                    0,
                    5,
                    4,
                    3,
                    7,
                    6,
                    3,
                    6,
                    2,
                    0,
                    4,
                    7,
                    0,
                    7,
                    3,
                    1,
                    2,
                    6,
                    1,
                    6,
                    5,
                ]
            trans = (
                UsdGeom.Xformable(p).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                * rigid.GetInverse()
            )
            vertices = [
                round(float(v), 6)
                for pt in points
                for v in trans.Transform(Gf.Vec3d(*pt))
            ]
            color = [0.75, 0.78, 0.72]
            display = UsdGeom.Gprim(p).GetDisplayColorAttr().Get()
            if display:
                color = list(display[0])
            else:
                try:
                    material = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
                    shader = material.ComputeSurfaceSource()[0]
                    for key in ("diffuse_color_constant", "diffuseColor"):
                        value = shader.GetInput(key).Get()
                        if value is not None:
                            color = list(value)[:3]
                            break
                except Exception:
                    pass
            meshes.append(dict(body=i, vertices=vertices, indices=faces, color=color))
    (output / "scene.json").write_text(
        json.dumps(
            dict(
                kind="rubik",
                bodies=names,
                meshes=meshes,
                cube_size=0,
                table=None,
                follow_body=0,
            ),
            separators=(",", ":"),
        )
    )
