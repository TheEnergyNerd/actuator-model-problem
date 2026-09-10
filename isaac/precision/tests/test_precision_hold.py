import torch
from atlas_actuators.precision_metrics import stable_hold_mask, advance_hold


def test_brief_hit_never_counts_and_one_bad_frame_breaks_dwell():
    elapsed = torch.zeros(1)
    for _ in range(14):
        elapsed, event = advance_hold(elapsed, torch.tensor([True]), 1 / 30)
        assert not event.item()
    elapsed, event = advance_hold(elapsed, torch.tensor([False]), 1 / 30)
    assert elapsed.item() == 0 and not event.item()
    for _ in range(14):
        elapsed, event = advance_hold(elapsed, torch.tensor([True]), 1 / 30)
    assert not event.item()
    elapsed, event = advance_hold(elapsed, torch.tensor([True]), 1 / 30)
    assert event.item()
    elapsed, event = advance_hold(elapsed, torch.tensor([True]), 1 / 30)
    assert not event.item(), "A held goal must not count repeatedly"


def test_alignment_alone_is_insufficient_for_stable_success():
    e = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.11, float("nan")])
    w = torch.tensor([0.1, 0.6, 0.1, 0.1, 0.1, 0.1])
    v = torch.tensor([0.01, 0.01, 0.06, 0.01, 0.01, 0.01])
    p = torch.tensor([0.01, 0.01, 0.01, 0.13, 0.01, 0.01])
    assert stable_hold_mask(e, w, v, p).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_goal_planner_limits_rotation_and_handles_quaternion_sign():
    from atlas_actuators.precision_metrics import bounded_goal_step

    current = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    target = torch.tensor([[0.0, 0, 1, 0], [-1.0, 0, 0, 0]])
    step = bounded_goal_step(current, target, 0.4, 1 / 30)
    angle = 2 * torch.acos(step[:, 0].abs().clamp(max=1.0))
    assert angle[0] < 0.4 / 30 + 1e-4 and angle[0] > 0.01
    torch.testing.assert_close(step.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(step[1], current[1])
