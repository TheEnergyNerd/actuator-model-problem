import math
import torch
from bench_profile import MEASURED, PHASE_KV, KT, electrical_parameters


def test_bench_conversion_preserves_electrical_mechanical_power():
    # Three-phase sinusoidal power, at iq=3 A and 100 rad/s, must equal
    # torque*speed. Catches the tempting but incorrect 60/(2*pi*Kv) mapping.
    omega, iq = 100.0, 3.0
    phase_emf = omega * 60 / (2 * math.pi * PHASE_KV)
    assert math.isclose(1.5 * phase_emf * iq, KT * iq * omega)
    assert math.isclose(
        PHASE_KV, math.sqrt(3) * MEASURED["Kv_line_line_peak_rpm_per_V"]
    )
    assert math.isclose(KT, 0.026619239323403716, rel_tol=1e-7)
    p = electrical_parameters()
    assert p["Lq"] > p["Ld"] and p["Rs"] == MEASURED["Rs"]


def test_fixed_temperature_reset_removes_hidden_initial_temperature(
    monkeypatch, tmp_path
):
    from test_training import FakeEnv
    from train_matched import install_objective

    monkeypatch.setattr("motor_model.attach", lambda *args: None)
    env = FakeEnv()
    install_objective(env, "motor", tmp_path, temperature_range=(25, 25), seed=17)
    assert torch.equal(env.motor_state["temperature"], torch.full((2, 22), 25.0))


def test_bench_profile_has_distinct_axes_and_hot_current_limit(monkeypatch, tmp_path):
    from test_motor_model import ModelTests

    monkeypatch.setenv("ATLAS_MOTOR_PROFILE", "mj5208-virtual")
    harness = ModelTests()
    cold, _ = harness.run_drive("motor", 0.0)
    hot, env = harness.run_drive("motor-hot", 0.0)
    assert torch.isfinite(hot).all()
    assert torch.all(hot.abs() < cold.abs() * 0.6)
    assert torch.all(
        torch.sqrt(env.motor_state["id"] ** 2 + env.motor_state["iq"] ** 2) <= 1.000001
    )
