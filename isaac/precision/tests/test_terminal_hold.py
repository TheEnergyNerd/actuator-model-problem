import torch
from atlas_actuators.hold_controller import TerminalHold


def test_terminal_hold_hysteresis_and_independent_reset():
    proposal = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    previous = proposal - 0.5
    c = TerminalHold(proposal)
    assert torch.equal(
        c.command(proposal, previous, torch.tensor([0.07, 0.2]), torch.zeros(2)),
        torch.stack([previous[0], proposal[1]]),
    )
    assert torch.equal(
        c.command(proposal, previous + 10, torch.tensor([0.1, 0.2]), torch.zeros(2))[0],
        previous[0],
    )
    c.reset(torch.tensor([False, True]))
    assert c.latched.tolist() == [True, False]
    assert torch.equal(
        c.command(proposal, previous, torch.tensor([0.15, 0.2]), torch.zeros(2)),
        proposal,
    )
    c.command(proposal, previous, torch.tensor([0.07, 0.07]), torch.zeros(2))
    assert (
        not c.command(
            proposal, previous, torch.tensor([float("nan"), 0.2]), torch.zeros(2)
        )
        .isnan()
        .any()
    )
    c.reset()
    assert not c.latched.any()
