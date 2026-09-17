"""Apply measured, isolated hardware interventions after a shared prepared grasp."""

import json
from pathlib import Path
import torch


def apply_intervention(env, out, effort_scale=1.0, mass_scale=1.0):
    robot = env.robot
    before_effort = robot.data.joint_effort_limits.clone()
    view = robot.root_physx_view
    before_mass = view.get_masses().clone()
    before_inertia = view.get_inertias().clone()
    if effort_scale != 1:
        robot.write_joint_effort_limit_to_sim(before_effort * effort_scale)
        # Keep the implicit actuator's torque estimate consistent with the new cap.
        for actuator in robot.actuators.values():
            actuator.effort_limit *= effort_scale
    if mass_scale != 1:
        ids = torch.arange(env.num_envs, dtype=torch.int32, device="cpu")
        view.set_masses(before_mass * mass_scale, ids)
        view.set_inertias(before_inertia * mass_scale, ids)
    actual_mass = view.get_masses()
    actual_effort = view.get_dof_max_forces()
    if not torch.allclose(actual_mass.cpu(), (before_mass * mass_scale).cpu()):
        raise RuntimeError("PhysX mass intervention did not apply")
    if not torch.allclose(actual_effort.cpu(), (before_effort * effort_scale).cpu()):
        raise RuntimeError("PhysX torque ceiling intervention did not apply")
    (Path(out) / "intervention.json").write_text(
        json.dumps(
            dict(
                effort_scale=effort_scale,
                hand_mass_scale=mass_scale,
                applied_after="Shared seeded grasp settling, before policy rollout",
                joint_names=robot.joint_names,
                body_names=robot.body_names,
                baseline_joint_effort_limits_nm=before_effort[0].cpu().tolist(),
                joint_effort_limits_nm=actual_effort[0].cpu().tolist(),
                baseline_body_mass_kg=before_mass[0].cpu().tolist(),
                body_mass_kg=actual_mass[0].cpu().tolist(),
                inertia_scale=mass_scale,
                interpretation="Torque-ceiling and uniformly scaled hand-mass sensitivity; not a Kv, winding or thermal model. Fixed wrist unchanged.",
            ),
            indent=2,
        )
    )
