"""Maximal-coordinate PhysX cube with passive joints and no cube actuators."""

import numpy as np
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, PhysxSchema
from cube_state import COORDS, NORMALS, state_after
from scipy.spatial.transform import Rotation

PITCH = 0.01905
COLORS = [
    (1, 0.95, 0.85),
    (0.85, 0.04, 0.03),
    (0.02, 0.55, 0.13),
    (1, 0.72, 0.02),
    (1, 0.25, 0.02),
    (0.02, 0.15, 0.8),
]


def make_cube(
    stage, center=(0, 0, 0.5), fixture=True, scramble="", orientation=None, locked=False
):
    frame = np.eye(3) if orientation is None else np.asarray(orientation)
    initial = state_after(scramble)
    root = "/World/Cube"
    UsdGeom.Xform.Define(stage, root)
    mat = UsdShade.Material.Define(stage, root + "/Plastic")
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(0.08)
    physics.CreateDynamicFrictionAttr(0.05)
    physics.CreateRestitutionAttr(0.0)
    names = ["core"] + [f"cubie_{i:02d}" for i in range(26)]
    for i, name in enumerate(names):
        path = root + ("/core/" if locked and i else "/") + name
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*((0, 0, 0) if locked and i else center)))
        q = (
            Rotation.from_matrix(frame).as_quat()
            if i == 0
            else Rotation.from_matrix(
                initial[i - 1] if locked else frame @ initial[i - 1]
            ).as_quat()
        )
        body.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        coord = np.zeros(3) if i == 0 else COORDS[i - 1] * PITCH
        if not locked or i == 0:
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(
                0.0
            )
            phys = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
            phys.CreateSolverPositionIterationCountAttr(32)
            phys.CreateSolverVelocityIterationCountAttr(8)
            phys.CreateMaxDepenetrationVelocityAttr(0.2)
            mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass.CreateMassAttr(0.093 if locked else 0.015 if i == 0 else 0.003)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*coord))
            inertia = 27 * 2e-7 + 36 * 0.003 * PITCH**2 if locked else 2e-7
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*([inertia] * 3)))
        if i == 0:
            continue
        shape = UsdGeom.Cube.Define(stage, path + "/plastic")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*coord))
        shape.AddScaleOp().Set(Gf.Vec3f(0.0185, 0.0185, 0.0185))
        shape.CreateDisplayColorAttr([Gf.Vec3f(0.025, 0.028, 0.032)])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        cp = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        cp.CreateContactOffsetAttr(0.0002)
        cp.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        c = COORDS[i - 1]
        for f, n in enumerate(NORMALS):
            if c @ n != 1:
                continue
            sticker = UsdGeom.Cube.Define(stage, path + f"/sticker_{f}")
            sticker.CreateSizeAttr(1.0)
            sticker.AddTranslateOp().Set(Gf.Vec3d(*(coord + n * 0.0093)))
            size = np.array([0.0168] * 3)
            size[np.argmax(abs(n))] = 0.00015
            sticker.AddScaleOp().Set(Gf.Vec3f(*size))
            sticker.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[f])])
        if locked:
            continue
        jp = root + f"/joint_{i-1:02d}"
        if np.count_nonzero(c) == 1:
            joint = UsdPhysics.RevoluteJoint.Define(stage, jp)
            joint.CreateAxisAttr(["X", "Y", "Z"][np.argmax(abs(c))])
        else:
            joint = UsdPhysics.SphericalJoint.Define(stage, jp)
        joint.CreateBody0Rel().SetTargets([root + "/core"])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateCollisionEnabledAttr(False)
    if fixture:
        joint = UsdPhysics.FixedJoint.Define(stage, root + "/fixture")
        joint.CreateBody1Rel().SetTargets([root + "/core"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*center))
        fq = Rotation.from_matrix(frame).as_quat()
        joint.CreateLocalRot0Attr(Gf.Quatf(float(fq[3]), Gf.Vec3f(*fq[:3])))
    return ["core"] if locked else names
