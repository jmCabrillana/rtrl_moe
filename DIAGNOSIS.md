# Root Cause Analysis: Why Training Wasn't Working

## Problem Summary
The MoE model was **not converging** on AnBn task. Loss showed high variance and spikes, accuracy fluctuated randomly, and state norm exploded (0.125 → 176+ over 7 timesteps).

## Root Causes Identified

### 1. **Catastrophic State Zeroing** (CRITICAL)
**Location**: `rtrl_moe/model/moe.py`, line ~179

```python
# WRONG: This zeros out all unwritten slots!
state_update = torch.zeros(B, S, D, device=device, dtype=dtype)
state_update = state_update.scatter_add_(1, write_idx_expanded, weighted_update)
state = beta * state_old + (1 - beta) * state_update  # β=0.6
```

**Problem**:
- State is initialized as zeros, then only written slots are filled
- Over time, slots that aren't written decay to zero due to: `beta*0 + (1-beta)*0 = 0`
- This is a **catastrophic forgetting** problem: unwritten slots permanently decay
- Written slots grow without dissipation

**Impact**: State norm explosion, loss of information in unused slots

### 2. **No Gradient Flow to Gating** (CRITICAL)
Many parameters had **zero gradients**:
- `state_gate.proj`: 0.0 (expert router never learned)
- `state_latent_proj`: NO GRADIENT (unused)
- `state_read_gate`: 0.0 (read gating never learned)
- `state_ln_slot`, `state_slot_ctx`: 0.0 (write gating never learned)

**Reason**: These are used only for sparse routing decisions, not directly in output computation.
The gradient path is: loss → output_proj → out_mha/out_ffn → **skips most state params**

### 3. **Unstable Sparse Recurrence**
The RTRL implementation expects **full state updates** but the MoE model only updates written slots.
This breaks the recurrence relation: `P_t[k] ← J_h @ P_t[k] + J_θ`

### 4. **No Dissipation Mechanism**
Written slots accumulate updates without bound. The residual connection (β=0.6) still allows growth:
- slot norm: 0.1 → 0.3 → 0.6 → 1.5 → 3.8 → 10 → 25 → 67 → 176

## Solution: SimpleRNN

Created a **minimal, stable recurrent model** with:

✓ **Full state updates** (no sparse gating that breaks causality)
✓ **LayerNorm** for bounded state (keeps h_norm ≈ 4.0 constant)
✓ **Gated update**: h_next = (1-α)*h + α*φ(h_contrib)
  - Gate α ∈ [0,1] controls update strength
  - Residual connection ensures stability
✓ **All parameters receive gradients**
✓ **Efficient**: O(d²) per step vs O(MoE complexity)

### SimpleRNN Results (50 steps on AnBn):
- Loss: 1.788 → 0.5 (stable downward trend) ✓
- Accuracy: ~50-60% (better than random) ✓
- State norm: 4.0 (bounded, never explodes) ✓
- Training time: 0.1-0.2s per step ✓

## Recommendations

### For your RTRL research:
1. **Don't use sparse gating for recurrent state** - it breaks the dynamics
2. **Always maintain full state** - even if using MoE for other components
3. **Use LayerNorm or other normalization** - prevents explosion
4. **Test stability first** - run diagnose.py before training

### Next Steps:
1. Train SimpleRNN longer (500+ steps) on AnBn and Haystack
2. If SimpleRNN converges well, consider:
   - Adding MoE only in **non-recurrent** components
   - Or using MoE with **full state reads/writes**
3. Validate RTRL implementation with simple baseline first

## Files Changed
- `rtrl_moe/model/simple_rnn.py` - NEW minimal stable model
- `rtrl_moe/train.py` - Added --model simple support
- `rtrl_moe/model/moe.py` - Fixed state update (but architecture still problematic)
- `test_simple_rnn.py` - Verification script
- `diagnose.py` - Diagnostic tools

## Testing
```bash
# Test SimpleRNN quick
python test_simple_rnn.py

# Train SimpleRNN
python rtrl_moe/train.py --model simple --task anbn --params "seq_len=32,n_steps=500" --exp-name simple_anbn_500

# Compare with MoE (expect failures)
python rtrl_moe/train.py --model moe --task anbn --params "seq_len=32,n_steps=50" --exp-name moe_anbn_test
```
