"""residual.py — a learned residual on a design-derived physics prior.

NeRD's architecture, used as a CORRECTION rather than a REPLACEMENT.

NeRD (Heiden et al., CoRL 2025) learns robot dynamics outright, replacing the
low-level solver. That needs a transformer because it must model everything.
Here the analytical model already supplies the actuator envelope, the thermal
derate, the shared DC bus and the rigid-body dynamics, so the network only has
to predict what physics cannot: contact-mediated coupling, frame compliance,
and cross-body effects with no closed form. Less to learn, less data needed,
and every structured parameter underneath stays a design quantity.

Three deliberate choices, each from a property of the problem:

1. ATTENTION OVER JOINTS, not over time.  The couplings that need learning are
   content-dependent and all-pairs -- which finger matters depends on where it
   is touching, not merely how many are touching. Temporal context is supplied
   as a short flattened history because the leftover dynamics are short-memory
   once the prior handles current lag and thermal state.

2. AGGREGATES ARE FED IN, NOT LEARNED.  Softmax attention computes a weighted
   MEAN; its weights sum to one, so it cannot represent a SUM without spending
   capacity to encode the count (Xu et al., "How Powerful are GNNs?" -- sum
   pooling strictly dominates mean for multiset functions). Total bus current
   is a sum. So the aggregate is computed exactly and appended as a feature
   rather than left for attention to approximate.

3. THE RESIDUAL IS GATED AND ZERO-INITIALISED.  Output starts at exactly zero,
   so an untrained model reproduces the physics prior bit-for-bit, and a
   learned tanh gate bounds it to +/- max_residual. A residual that cannot run
   away is one you can ship: the prior remains the guarantee.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                                    # importable without torch
    _TORCH = False
    nn = object  # type: ignore


if _TORCH:

    class JointResidual(nn.Module):
        """Per-joint torque residual, attention-coupled across joints.

        forward(feats, aggregates=None, mask=None) -> residual torque [B, J]

        feats      [B, J, F]  per-joint history-flattened features
        aggregates [B, A]     exact robot-level scalars (bus current/voltage,
                              total copper loss, hottest winding, ...). These
                              are SUMS and MAXES the attention cannot form.
        mask       [B, J]     True for joints that exist (ragged DOF support)
        """

        def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                     n_layers: int = 2, n_aggregates: int = 0,
                     max_joints: int = 64, max_residual: float = 0.25,
                     dropout: float = 0.0):
            super().__init__()
            self.max_residual = float(max_residual)
            self.embed = nn.Linear(n_features, d_model)
            # Learned joint identity: a hand's joints are not interchangeable
            # (a thumb is not a little finger) even though the operator is
            # permutation-equivariant without it.
            self.joint_id = nn.Parameter(torch.zeros(max_joints, d_model))
            nn.init.normal_(self.joint_id, std=0.02)
            self.agg = (nn.Linear(n_aggregates, d_model)
                        if n_aggregates > 0 else None)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)
            self.gate = nn.Linear(d_model, 1)
            # zero-init the head so an untrained residual is exactly the prior
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.0)

        def forward(self, feats, aggregates=None, mask=None):
            B, J, _ = feats.shape
            h = self.embed(feats) + self.joint_id[:J].unsqueeze(0)
            if self.agg is not None and aggregates is not None:
                # broadcast the exact robot-level scalars to every joint
                h = h + self.agg(aggregates).unsqueeze(1)
            pad = (~mask) if mask is not None else None
            h = self.blocks(h, src_key_padding_mask=pad)
            # bounded, gated residual -> cannot exceed +/- max_residual
            r = torch.tanh(self.head(h)).squeeze(-1)
            g = torch.sigmoid(self.gate(h)).squeeze(-1)
            out = self.max_residual * g * r
            if mask is not None:
                out = out * mask.to(out.dtype)
            return out

        @classmethod
        def load(cls, path):
            """Rebuild from a checkpoint saved by `save` (architecture + weights)."""
            blob = torch.load(path, map_location="cpu", weights_only=True)
            m = cls(**blob["cfg"])
            m.load_state_dict(blob["state"])
            m.eval()
            return m

        def save(self, path, cfg: dict):
            """Store architecture kwargs beside the weights so `load` is exact."""
            torch.save({"cfg": cfg, "state": self.state_dict()}, path)


    def build_features(q_err, qd, tau_cmd, tau_prior, winding_T, history=None):
        """Assemble [B, J, F] from the quantities the prior already computes.

        `history` is an optional list of earlier (q_err, qd, tau_cmd) tuples,
        oldest first; short memory is enough once the prior models current lag
        and thermal state, so 3-5 steps is typically plenty.
        """
        parts = [q_err, qd, tau_cmd, tau_prior, winding_T]
        if history:
            for h in history:
                parts.extend(h)
        return torch.stack(parts, dim=-1)


    def exact_aggregates(i_total, v_bus, p_copper_total, t_hottest):
        """Robot-level scalars computed EXACTLY, not learned.

        Sums and maxima over joints; softmax attention normalises these away,
        so they are handed to the network rather than approximated by it.
        """
        return torch.stack([i_total, v_bus, p_copper_total, t_hottest], dim=-1)


def load_residual_checkpoint(path, device="cpu"):
    """Load either supported format; a .pt suffix does not distinguish them."""
    if not _TORCH:
        raise ImportError("Residual inference requires PyTorch")
    try:
        model = torch.jit.load(str(path), map_location=device)
    except RuntimeError:
        model = JointResidual.load(path).to(device)
    return model.eval()
