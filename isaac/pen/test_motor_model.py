import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from motor_model import attach


class ModelTests(unittest.TestCase):
    def run_drive(self, mode, speed):
        actuator = SimpleNamespace(
            stiffness=torch.ones(2, 3) * 10, damping=torch.ones(2, 3) * 0.1
        )
        robot = SimpleNamespace(
            actuators={"hand": actuator},
            data=SimpleNamespace(
                joint_effort_limits=torch.tensor([[1.0, 2.0, 0.3]])
                .expand(2, -1)
                .clone()
            ),
            joint_names=["a", "b", "c"],
        )
        robot.write_joint_stiffness_to_sim = lambda value: setattr(
            robot, "sim_kp", value
        )
        robot.write_joint_damping_to_sim = lambda value: setattr(robot, "sim_kd", value)
        env = SimpleNamespace(robot=robot, physics_dt=1 / 240)
        with tempfile.TemporaryDirectory() as d:
            attach(env, Path(d), mode)
        action = SimpleNamespace(
            joint_positions=torch.ones(2, 3) * 10,
            joint_velocities=torch.zeros(2, 3),
            joint_efforts=torch.zeros(2, 3),
        )
        out = actuator.compute(action, torch.zeros(2, 3), torch.full((2, 3), speed))
        self.assertTrue(torch.equal(robot.sim_kp, torch.zeros(2, 3)))
        self.assertTrue(torch.equal(robot.sim_kd, torch.zeros(2, 3)))
        self.assertIsNone(out.joint_positions)
        self.assertFalse(robot._has_implicit_actuators)
        return out.joint_efforts, env

    def test_matched_cold_stall(self):
        ideal, _ = self.run_drive("ideal", 0.0)
        real, _ = self.run_drive("motor", 0.0)
        torch.testing.assert_close(real, ideal, rtol=1e-4, atol=1e-5)

    def test_speed_reduces_delivery_only_in_motor_model(self):
        ideal, _ = self.run_drive("ideal", 100.0)
        real, env = self.run_drive("motor", 100.0)
        self.assertTrue(torch.all(real.abs() < ideal.abs() * 0.5))
        self.assertTrue(torch.isfinite(env.motor_state["temperature"]).all())
        self.assertTrue(torch.all(env.motor_state["energy"] >= 0))

    def test_hot_start_reduces_current(self):
        cold, _ = self.run_drive("motor", 0.0)
        hot, env = self.run_drive("motor-hot", 0.0)
        self.assertTrue(torch.all(hot.abs() < cold.abs() * 0.6))
        self.assertTrue(torch.all(env.motor_state["temperature"] > 99))


if __name__ == "__main__":
    unittest.main()
