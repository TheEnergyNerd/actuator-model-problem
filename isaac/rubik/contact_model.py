"""Author material/contact settings on native USD collision instances explicitly."""


def make_collisions_editable(stage, root):
    from pxr import UsdPhysics

    while True:
        instances = [
            p
            for p in stage.Traverse()
            if str(p.GetPath()).startswith(root + "/") and p.IsInstance()
        ]
        if not instances:
            break
        for p in instances:
            p.SetInstanceable(False)
    colliders = [
        p
        for p in stage.Traverse()
        if str(p.GetPath()).startswith(root + "/") and p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(colliders) != 21:
        raise RuntimeError(
            f"Expected 21 pinned Wuji colliders, found {len(colliders)} under {root}"
        )
    return colliders


def configure_hand_contacts(stage, root, profile="stock"):
    if profile not in ("stock", "rigid", "pad"):
        raise ValueError(profile)
    mu_s, mu_d, combine, offset = (
        (0.5, 0.5, "average", 0.02) if profile == "stock" else (2.0, 1.5, "max", 0.0002)
    )
    from pxr import UsdPhysics, PhysxSchema, UsdShade

    colliders = make_collisions_editable(stage, root)
    import json
    from pathlib import Path

    exclusions = json.loads(
        (Path(__file__).parent / "native_collision_exclusions.json").read_text()
    )
    side = "left" if stage.GetPrimAtPath(root + "/l_wrist").IsValid() else "right"
    for pair in exclusions[side]:
        first = stage.GetPrimAtPath(root + "/" + pair["body1"])
        second = root + "/" + pair["body2"]
        if not first.IsValid() or not stage.GetPrimAtPath(second).IsValid():
            raise RuntimeError("Missing native collision-pair body")
        UsdPhysics.FilteredPairsAPI.Apply(first).CreateFilteredPairsRel().AddTarget(
            second
        )
    material = UsdShade.Material.Define(stage, root + "/GripMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(mu_s)
    api.CreateDynamicFrictionAttr(mu_d)
    PhysxSchema.PhysxMaterialAPI.Apply(
        material.GetPrim()
    ).CreateFrictionCombineModeAttr(combine)
    pad = UsdShade.Material.Define(stage, root + "/PadMaterial")
    pad_api = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
    pad_api.CreateStaticFrictionAttr(mu_s)
    pad_api.CreateDynamicFrictionAttr(mu_d)
    compliant = PhysxSchema.PhysxMaterialAPI.Apply(pad.GetPrim())
    compliant.CreateFrictionCombineModeAttr(combine)
    compliant.CreateCompliantContactStiffnessAttr(5000.0 if profile == "pad" else 0.0)
    compliant.CreateCompliantContactDampingAttr(3.0 if profile == "pad" else 0.0)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(root + "/") and prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    for prim in colliders:
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(offset)
        api.CreateRestOffsetAttr(0.0)
        is_pad = profile != "stock" and "_distal/" in str(prim.GetPath())
        api.CreateTorsionalPatchRadiusAttr(0.004 if is_pad else 0.0)
        api.CreateMinTorsionalPatchRadiusAttr(0.002 if is_pad else 0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            pad if is_pad else material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
        assert abs(api.GetContactOffsetAttr().Get() - offset) < 1e-6
    return {
        "profile": profile,
        "colliders": len(colliders),
        "native_collision_exclusions": len(exclusions[side]),
        "static_friction": mu_s,
        "dynamic_friction": mu_d,
        "friction_combine": combine,
        "contact_offset_m": offset,
        "torsional_patch_radius_m": 0.004 if profile != "stock" else 0.0,
        "minimum_torsional_patch_radius_m": 0.002 if profile != "stock" else 0.0,
        "pad_stiffness_n_m": 5000.0 if profile == "pad" else 0.0,
        "pad_damping_ns_m": 3.0 if profile == "pad" else 0.0,
    }
