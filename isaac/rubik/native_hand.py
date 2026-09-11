"""Native Wuji articulation mounted to a force-limited Cartesian wrist servo."""

from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import UsdPhysics, PhysxSchema, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import matrix_from_quat, quat_rotate_inverse
from wuji_kinematics import Hand
from contact_model import configure_hand_contacts


class MountedHand:
    def __init__(
        self,
        stage,
        side,
        layer,
        frame,
        closure,
        cube_pos=(0, 0, 0.5),
        contact_profile="pad",
        initial_joints=None,
    ):
        self.hand = Hand(side, layer)
        self.frame = frame
        self.q, self.ik_error = self.hand.grasp(closure)
        if side == "left":
            for finger in ("middle_finger", "ring_finger", "pinky"):
                self.q[f"l_{finger}_mcp_flex"] = -0.65
        if initial_joints:
            self.q.update(initial_joints)
        self.pos, self.rot = self.hand.wrist_pose(cube_pos, frame)
        self.quat = Rotation.from_matrix(self.rot).as_quat()[[3, 0, 1, 2]]
        self.path = "/World/" + side.title() + "Hand"
        usd = str(
            Path(__file__).parent
            / f"assets/wuji/hand2/hand2_beta1/body/usd/{side}/wujihand2.usd"
        )
        cfg = ArticulationCfg(
            prim_path=self.path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, max_depenetration_velocity=0.2
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=8,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(self.pos), rot=tuple(self.quat), joint_pos=self.q
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=3.0,
                    damping=0.08,
                    effort_limit_sim=1.5,
                )
            },
        )
        self.robot = Articulation(cfg)
        for prim in list(stage.Traverse()):
            if (
                str(prim.GetPath()).startswith(self.path)
                and prim.IsA(UsdPhysics.FixedJoint)
                and not UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
            ):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                prim.SetActive(False)
        UsdPhysics.ArticulationRootAPI.Apply(
            stage.GetPrimAtPath(self.path + "/" + side[0] + "_wrist")
        )
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(self.path + "/") and prim.IsA(
                UsdPhysics.RevoluteJoint
            ):
                state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                state.CreatePositionAttr(
                    float(np.rad2deg(self.q.get(prim.GetName(), 0.0)))
                )
                state.CreateVelocityAttr(0.0)
        self.contact_model = configure_hand_contacts(
            stage, self.path, profile=contact_profile
        )

    def initialize(self, device):
        self.robot.reset()
        self.device = device
        self.robot.write_root_pose_to_sim(
            torch.tensor(
                [list(self.pos) + list(self.quat)], device=device, dtype=torch.float
            )
        )
        self.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
        )
        self.wrist = self.robot.body_names.index(self.hand.prefix + "wrist")
        self.target = self.robot.data.default_joint_pos.clone()

    def command(self, frame=None, joints=None, cube_pos=(0, 0, 0.5)):
        if frame is None:
            frame = self.frame
        pos, rot = self.hand.wrist_pose(cube_pos, frame)
        if joints:
            for name, q in joints.items():
                self.target[0, self.robot.joint_names.index(name)] = q
        self.robot.set_joint_position_target(self.target)
        body = self.robot.data.body_state_w[0, self.wrist]
        actual = matrix_from_quat(body[3:7]).cpu().numpy()
        er = Rotation.from_matrix(rot @ actual.T).as_rotvec()
        force = (torch.tensor(pos, device=self.device) - body[:3]) * 2000 - body[
            7:10
        ] * 35
        torque = torch.tensor(er, device=self.device) * 15 - body[10:13] * 0.25
        self.robot.set_external_force_and_torque(
            quat_rotate_inverse(body[3:7], force.clamp(-30, 30).float())[None, None],
            quat_rotate_inverse(body[3:7], torque.clamp(-2, 2).float())[None, None],
            body_ids=[self.wrist],
        )
        self.robot.write_data_to_sim()
