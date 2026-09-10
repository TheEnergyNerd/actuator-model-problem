"""Bounded orientation integral feedback around a goal-conditioned hand policy.

The servo adjusts the orientation command seen by the policy. It never writes
object/joint state or bypasses the actuator. Quaternions use (w, x, y, z).
"""
import torch


def multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


def rotation_error(target, actual):
    conjugate = actual * actual.new_tensor([1, -1, -1, -1])
    q = multiply(target, conjugate)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-9)
    q = torch.where(q[..., :1] < 0, -q, q)
    xyz = q[..., 1:]
    n = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n, q[..., :1].clamp_min(0))
    return xyz * torch.where(n > 1e-7, angle / n.clamp_min(1e-9), torch.full_like(n, 2))


class OrientationServo:
    def __init__(self, target, gain=0.6, max_bias_rad=0.35, capture_rad=0.5):
        if gain < 0 or max_bias_rad <= 0 or capture_rad <= 0:
            raise ValueError('Invalid servo gains or limits')
        self.bias = torch.zeros_like(target[..., 1:])
        self.gain = gain
        self.max_bias = max_bias_rad
        self.capture = capture_rad

    def reset(self):
        self.bias.zero_()

    def command(self, target, actual, dt):
        if dt <= 0:
            raise ValueError('dt must be positive')
        error = rotation_error(target, actual)
        near = torch.linalg.vector_norm(error, dim=-1, keepdim=True) < self.capture
        self.bias.add_(self.gain * dt * torch.where(near, error, torch.zeros_like(error)))
        norm = torch.linalg.vector_norm(self.bias, dim=-1, keepdim=True)
        self.bias.mul_((self.max_bias / norm.clamp_min(1e-9)).clamp(max=1))
        angle = torch.linalg.vector_norm(self.bias, dim=-1, keepdim=True)
        xyz = self.bias * (torch.sin(angle/2) / angle.clamp_min(1e-9))
        correction = torch.cat((torch.cos(angle/2), xyz), -1)
        return multiply(correction, target)
