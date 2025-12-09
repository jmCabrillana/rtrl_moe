"""a^n b^n counting task utilities."""
import random
from typing import Tuple

import torch
import torch.nn.functional as F

# Token definitions
PAD, A, B, Q = 0, 1, 2, 3
VOCAB_SIZE = 4
OUTPUT_DIM = 2


def sample(seq_len: int, device: torch.device, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate padded a^n b^n sequences ending with a query token.

    Sequences are fixed length (seq_len). Target is 1 if counts match, else 0.
    """
    max_tokens = max(2, seq_len - 1)  # tokens available before the query
    x = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    y = torch.empty(batch_size, dtype=torch.long)

    for b in range(batch_size):
        if random.random() < 0.5:
            # Positive example: choose n_a so that 2 * n_a fits
            max_a_pos = max_tokens // 2
            max_a_pos = max(1, max_a_pos)
            n_a = random.randint(1, max_a_pos)
            n_b = n_a
            y[b] = 1
        else:
            # Negative example: choose n_a with room for a different n_b
            n_a = random.randint(1, max_tokens - 1)
            possible_n_b = [k for k in range(1, max_tokens - n_a + 1) if k != n_a]
            if possible_n_b:
                n_b = random.choice(possible_n_b)
            else:
                # If no different count fits, fall back to closest different count within budget
                n_b = max(1, min(max_tokens - n_a, n_a - 1 if n_a > 1 else 1))
            # If still equal (can happen for very short seq_len), treat as positive to avoid random labels
            if n_b == n_a:
                y[b] = 1
            else:
                y[b] = 0

        # Build sequence and ensure it fits before the query token
        seq = [A] * n_a + [B] * n_b
        seq = seq[: seq_len - 1]  # leave room for query token

        for idx, token in enumerate(seq):
            x[b, idx] = token

        x[b, seq_len - 1] = Q

    x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float().to(device)
    return x_onehot, y.to(device)
