from types import SimpleNamespace
import torch
from train_matched import install_objective


class FakeEnv:
    device = "cpu"
    num_envs = 2

    def __init__(self):
        self.cfg = SimpleNamespace()
        self.best = torch.zeros(2)
        self.holding = torch.zeros(2, dtype=torch.bool)
        self.action = torch.zeros(2, 22)
        self.ref_pos = torch.zeros(2, 3)
        self.pos = torch.zeros(2, 3)
        self.axis = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
        self.forces = torch.ones(2, 5) * 0.1
        self.pen = SimpleNamespace(
            data=SimpleNamespace(
                root_ang_vel_w=torch.zeros(2, 3), root_lin_vel_w=torch.zeros(2, 3)
            )
        )
        self.sim = SimpleNamespace(step=lambda **kw: None)
        self.scene = SimpleNamespace(update=lambda **kw: None)
        self.motor_state = {
            k: torch.ones(2, 22) * 9 for k in ["id", "iq", "energy", "temperature"]
        }

    def _reset_idx(self, ids):
        self.best[ids] = 0
        self.holding[ids] = False

    def _get_dones(self):
        self.best += 0.05
        return torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)

    def _pen_state(self):
        return self.pos, None, self.axis

    def _finger_forces(self):
        return self.forces, torch.zeros(2)


def test_drop_terminates_and_no_airborne_rotation_reward(monkeypatch, tmp_path):
    monkeypatch.setattr("motor_model.attach", lambda *args: None)
    env = FakeEnv()
    install_objective(env, "motor", tmp_path)
    env.pos[0, 2] = -0.04
    env.forces[0] = 0
    done, _ = env._get_dones()
    assert done.tolist() == [True, False]
    reward = env._get_rewards()
    assert reward[0] < -20
    assert reward[1] > 0


def test_reset_clears_electrical_state_and_requires_full_hold(monkeypatch, tmp_path):
    monkeypatch.setattr("motor_model.attach", lambda *args: None)
    env = FakeEnv()
    install_objective(env, "motor", tmp_path)
    assert (env.motor_state["temperature"] >= 25).all()
    assert (env.motor_state["temperature"] <= 100).all()
    assert all((env.motor_state[k] == 0).all() for k in ["id", "iq", "energy"])
    env.holding[:] = True
    for _ in range(59):
        done, _ = env._get_dones()
        assert not done.any()
    done, _ = env._get_dones()
    assert done.all()
    env._reset_idx(torch.arange(2))
    assert (env.hold_steps == 0).all()
