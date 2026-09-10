"""FOC invariants and backend parity; simulator-independent CPU tests."""
import math
import numpy as np
import pytest
import torch
from atlas_actuators.foc_core import (
    foc_actuator_params, step_current_lag, step_dq, _torque_motor,
    project_pack_current, _Rs_at,
)

CTRL = {"Kt_Nm_per_A": 0.25, "R_phase_mohm": 30, "L_phase_uH": 85,
        "pole_pairs": 20, "I_peak_A": 55, "I_cont_A": 28}


def hand_params(mode="projected"):
    p = foc_actuator_params({"Kt_Nm_per_A": .1, "R_phase_mohm": 2000,
        "L_phase_uH": 250, "pole_pairs": 7, "I_peak_A": 5., "I_cont_A": 1.5},
        gear_ratio=5, gear_eff=.75, V_bus_V=24)
    p.T_derate_C = 70
    p.thermal_C_JperK = 6
    p.thermal_R_KperW = 5
    p.I_pack_max_A = 30
    p.R_pack_ohm = .05
    p.pack_limit_mode = mode
    return p


@pytest.mark.parametrize("mode", ["legacy", "projected"])
def test_numpy_torch_parity(mode):
    p = hand_params(mode)
    rng = np.random.default_rng(42)
    shape = (64, 16)
    args = [rng.uniform(-3, 3, shape), rng.uniform(-10, 10, shape),
            rng.uniform(-1, 0, shape), rng.uniform(-5, 5, shape),
            rng.uniform(25, 70, shape)]
    expected = step_current_lag(p, *args, 1/120)
    actual = step_current_lag(p, *(torch.from_numpy(x) for x in args), 1/120)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.numpy(), b, rtol=1e-12, atol=1e-12)


def test_lag_response():
    p = foc_actuator_params(CTRL, gear_ratio=1, gear_eff=1)
    z = np.zeros((1, 1))
    dt = 1e-5
    _, _, iq, _ = step_current_lag(p, z + p.Kt*10, z, z, z, z+25, dt)
    assert iq.item() == pytest.approx((1-math.exp(-2*math.pi*p.current_bw_hz*dt))*10)


def test_legacy_pack_oscillation_is_reproducible():
    p = hand_params("legacy")
    z = np.zeros((1, 16)); id_, iq = z.copy(), z.copy()
    currents = []
    for _ in range(4):
        _, id_, iq, _ = step_current_lag(p, z+100, z, id_, iq, z+25, 1/120)
        currents.append(float((1.5*p.Rs*(id_**2+iq**2)).sum()/p.V_bus))
    assert currents == pytest.approx([50,18,50,18])


def test_projected_pack_ceiling_is_enforced_without_oscillation():
    p = hand_params()
    z = np.zeros((2,16)); id_, iq = z.copy(), z.copy()
    currents = []
    for _ in range(20):
        _, id_, iq, _ = step_current_lag(p, z+100, z, id_, iq, z+25, 1/120)
        power = (1.5*p.Rs*(id_**2+iq**2)).sum(-1)
        voltage_at_ceiling = p.V_bus-p.R_pack_ohm*p.I_pack_max_A
        currents.append(power/voltage_at_ceiling)
    np.testing.assert_allclose(currents, 30, atol=1e-8)


def test_pack_projection_bounds_draw_in_motoring_and_regeneration():
    p = hand_params()
    rng = np.random.default_rng(0)
    id_ = rng.uniform(-5,0,(64,16)); iq = rng.uniform(-5,5,(64,16))
    T = rng.uniform(25,70,(64,16)); w = rng.uniform(-100,100,(64,16))
    id_, iq = project_pack_current(p,id_,iq,T,w)
    power = 1.5*_Rs_at(p,T)*(id_**2+iq**2)+_torque_motor(p,id_,iq,T)*w
    positive = np.maximum(power,0).sum(-1)
    assert np.all(positive <= p.I_pack_max_A*(p.V_bus-p.R_pack_ohm*p.I_pack_max_A)+1e-8)


def test_pack_limit_does_not_couple_different_environments():
    p = hand_params(); z=np.zeros((2,16)); iq=z.copy(); iq[0]=5; iq[1]=.1
    _, result = project_pack_current(p,z,iq,z+25,z)
    np.testing.assert_allclose(result[1], iq[1])
    assert np.all(result[0]<iq[0])


def test_thermal_cooling_at_zero_current():
    p=hand_params(); z=np.zeros((1,16))
    _,_,_,T=step_current_lag(p,z,z,z,z,z+60,1/120)
    assert np.all(T<60) and np.all(T>25)


def test_dq_hot_motor_settles_to_hot_torque():
    p=foc_actuator_params(CTRL,gear_ratio=1,gear_eff=1)
    z=np.zeros((1,1)); id_,iq,integ_d,integ_q=[z.copy() for _ in range(4)]
    for _ in range(200):
        tau,id_,iq,integ_d,integ_q,_=step_dq(p,z+1,z,id_,iq,integ_d,integ_q,z+80,dt=.001,substeps=100)
    assert tau.item()==pytest.approx(1,rel=.01)
    np.testing.assert_allclose(tau,_torque_motor(p,id_,iq,z+80))


def test_scalar_reference_parity_when_parent_checkout_available():
    scalar = pytest.importorskip("foc_motor", reason="Original scalar reference is absent from the archive")
    p=foc_actuator_params(CTRL,gear_ratio=1,gear_eff=1)
    z=np.zeros((1,1))
    tau,*_=step_current_lag(p,z+1e6,z,z,z,z+25,.01)
    model=scalar.FOCMotor(scalar.foc_params_from_design(CTRL,V_bus_V=48),omega_m=0)
    for _ in range(8000): model.step(p.Kt*p.I_peak,dt=1e-5,free=False)
    assert tau.item()==pytest.approx(abs(model.torque),rel=.03)


def test_sense_passthrough_when_disabled():
    import torch
    from atlas_actuators.foc_core import sense_encoder_backlash
    pj = torch.tensor([[0.123]]); pm = torch.tensor([[0.0]])
    prev = torch.tensor([[0.1]]); eff = torch.tensor([[5.0]])
    mp, mv, tr, pm2, prev2 = sense_encoder_backlash(pj, pm, prev, eff, 0, 9.0, 0.0, 0.02)
    assert torch.allclose(mp, pj) and float(tr) == 1.0
    assert abs(float(mv) - (0.123 - 0.1) / 0.02) < 1e-6

def test_sense_encoder_quantization_step():
    import torch
    from atlas_actuators.foc_core import sense_encoder_backlash
    counts, gear = 16384, 12.0
    step = 2 * 3.141592653589793 / (counts * gear)
    pj = torch.tensor([[step * 3.4]])
    pm = pj.clone(); prev = torch.zeros_like(pj); eff = torch.ones_like(pj)
    mp, _, _, _, _ = sense_encoder_backlash(pj, pm, prev, eff, counts, gear, 0.0, 0.02)
    assert abs(float(mp) - step * 3.0) < 1e-9

def test_backlash_reversal_partial_transmission():
    import torch
    from atlas_actuators.foc_core import sense_encoder_backlash
    b, dt, v = 0.01, 0.02, 2.5  # traversal of full gap: 0.004 s = 20% of dt
    pj = torch.zeros(1, 1); pm = torch.zeros(1, 1); prev = torch.zeros(1, 1)
    push = torch.ones(1, 1)
    # step 1: crossing half the gap costs 10% of the step
    _, _, tr1, pm, prev = sense_encoder_backlash(pj, pm, prev, push, 0, 9.0, b, dt, v)
    assert abs(float(tr1) - 0.9) < 1e-6 and abs(float(pm) - b / 2) < 1e-9
    # step 2: engaged on the +face, full transmission
    _, _, tr2, pm, prev = sense_encoder_backlash(pj, pm, prev, push, 0, 9.0, b, dt, v)
    assert float(tr2) == 1.0
    # torque reversal: crossing the full gap costs 20% of the step
    _, _, tr3, pm, prev = sense_encoder_backlash(pj, pm, prev, -push, 0, 9.0, b, dt, v)
    assert abs(float(tr3) - 0.8) < 1e-6 and abs(float(pm) + b / 2) < 1e-9
    _, _, tr4, pm, prev = sense_encoder_backlash(pj, pm, prev, -push, 0, 9.0, b, dt, v)
    assert float(tr4) == 1.0

def test_backlash_joint_drags_motor_through_contact():
    import torch
    from atlas_actuators.foc_core import sense_encoder_backlash
    b, dt = 0.01, 0.02
    pj = torch.zeros(1, 1); pm = torch.zeros(1, 1); prev = torch.zeros(1, 1)
    push = torch.ones(1, 1)
    for _ in range(2):  # engage on +face
        _, _, _, pm, prev = sense_encoder_backlash(pj, pm, prev, push, 0, 9.0, b, dt)
    pj = pj - 0.02  # joint retreats: face pushes motor back, contact held
    mp, _, tr, pm, prev = sense_encoder_backlash(pj, pm, prev, push, 0, 9.0, b, dt)
    assert abs(float(pm) - (float(pj) + b / 2)) < 1e-9
    assert float(tr) == 1.0
