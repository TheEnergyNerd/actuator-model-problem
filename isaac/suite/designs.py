"""Apply the existing physical design interventions to one course robot."""

import torch


def apply_design(robot, name, candidate=None):
    from atlas_actuators.design_study import (
        Design,
        designs,
        parameter_batch,
        describe,
        add_carrier_mass,
    )
    import omni.usd
    from pxr import Usd, UsdPhysics

    case = (
        Design(**candidate)
        if candidate
        else next(d for d in designs() if d.name == name)
    )
    specifications = {}
    for key, actuator in robot.actuators.items():
        specifications[key] = describe(actuator._p, case, 0.25, 1e-4)
        actuator._p = parameter_batch(
            actuator._p, [case], robot.num_instances, robot.device
        )
    stage = omni.usd.get_context().get_stage()
    carriers = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() in robot.joint_names:
            parent = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
            if parent and parent[0].name in robot.body_names:
                carriers[prim.GetName()] = robot.body_names.index(parent[0].name)
    if set(carriers) != set(robot.joint_names):
        raise ValueError("Every actuator must have a resolved physical mass carrier")
    view = robot.root_physx_view
    original = view.get_masses().clone()
    increments = torch.full((robot.num_instances,), 0.25 * case.added_mass_factor)
    masses, inertias = add_carrier_mass(
        original,
        view.get_inertias(),
        [carriers[j] for j in robot.joint_names],
        increments,
        0.04,
    )
    ids = torch.arange(robot.num_instances)
    view.set_masses(masses, ids)
    view.set_inertias(inertias, ids)
    added = view.get_masses().sum(-1) - original.sum(-1)
    torch.testing.assert_close(
        added, increments * robot.num_joints, atol=1e-4, rtol=1e-4
    )
    return dict(
        name=case.name,
        inertia_policy="Added carrier mass changes body inertia; native joint armature remains fixed; no added rotor inertia in this course intervention",
        specifications=specifications,
        measured_added_mass_kg=added.tolist(),
        carrier_mapping={j: robot.body_names[b] for j, b in carriers.items()},
        placement="Added spherical-equivalent stator mass at each carrier COM, radius 0.04 m; native COM unchanged",
    )
