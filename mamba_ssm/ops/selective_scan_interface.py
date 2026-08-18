import torch
import torch.nn.functional as F


def selective_scan_fn(
    x,
    delta,
    A,
    B,
    C,
    D,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=None,
):
    """Lightweight fallback when CUDA selective scan kernels are unavailable.

    This keeps tensor shapes/API compatible with mamba_ssm so models can run,
    but it is not a numerically equivalent replacement for the fused kernel.
    """
    y = x

    if delta is not None:
        gate = delta
        if delta_bias is not None:
            gate = gate + delta_bias.view(1, -1, 1)
        if delta_softplus:
            gate = F.softplus(gate)
        y = y * torch.sigmoid(gate)

    if D is not None:
        y = y + D.view(1, -1, 1) * x

    if z is not None:
        y = y * torch.sigmoid(z)

    if return_last_state:
        last = y[:, :, -1]
        return y, last

    return y