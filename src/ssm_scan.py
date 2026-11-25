"""Parallel Associative Prefix Scan (Hillis-Steele log2-depth) for exact ZOH continuous-time SSM recurrences."""

import math
import torch


def parallel_first_order_scan(alpha: torch.Tensor, b: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Solve h[:, t, :] = alpha[:, t, :] * h[:, t-1, :] + b[:, t, :] for t = 0..L-1 in O(log2 L) parallel GPU steps.

    Runs strictly in float32 even under AMP autocast to preserve exact continuous-time ODE integration.
    """
    orig_dtype = b.dtype
    with torch.amp.autocast("cuda", enabled=False):
        alpha = alpha.float()
        b = b.float()
        h0 = h0.float()
        bsz, seq_len, dim = alpha.shape
        if seq_len == 0:
            return b.to(orig_dtype)

        # Fold initial condition h0 into step t=0 strictly out-of-place (autograd-safe):
        b0 = alpha[:, :1, :] * h0.unsqueeze(1) + b[:, :1, :]
        if seq_len > 1:
            b_cur = torch.cat([b0, b[:, 1:, :]], dim=1)
            a_cur = torch.cat([torch.zeros_like(alpha[:, :1, :]), alpha[:, 1:, :]], dim=1)
        else:
            return b0.to(orig_dtype)

        num_steps = int(math.ceil(math.log2(max(seq_len, 2))))
        for step in range(num_steps):
            stride = 1 << step
            if stride >= seq_len:
                break
            a_prev = a_cur[:, :-stride, :]
            b_prev = b_cur[:, :-stride, :]
            a_right = a_cur[:, stride:, :]
            b_right = b_cur[:, stride:, :]

            new_a = a_right * a_prev
            new_b = a_right * b_prev + b_right

            a_cur = torch.cat([a_cur[:, :stride, :], new_a], dim=1)
            b_cur = torch.cat([b_cur[:, :stride, :], new_b], dim=1)

        return b_cur.to(orig_dtype)
