# RTRL with Read-Write Sparsity and MoE

Real-Time Recurrent Learning (RTRL) implementation with read-write sparsity and Mixture of Experts (MoE), demonstrating **constant memory training** on arbitrarily long sequences.

## Overview

This repository implements Block RTRL with:
- **Read-Write Sparsity**: Only updates relevant state dimensions
- **MoE Sparsity**: Uses sparse expert activation to reduce computation
- **Circular Segment Tree**: Efficient storage of sparse Jacobians (infrastructure ready, currently commented)
- **Constant Memory**: Trains on sequences of length 10K+ at constant memory cost

### Key Advantage

**RTRL enables training on extremely long sequences where BPTT would require prohibitive memory:**
- BPTT memory: O(sequence_length)
- RTRL memory: O(1) - **constant regardless of sequence length**

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train on a^n b^n Counting Task

```bash
# BPTT baseline
python train_anbn.py --mode bptt --len_max_a 8 --steps 10000

# RTRL (constant memory)
python train_anbn.py --mode rtrl --len_max_a 8 --steps 10000

# Compare both
python train_anbn.py --mode both --len_max_a 16
```

### 2. Train on Haystack Retrieval

```bash
# RTRL on long sequences (512 tokens)
python train_haystack.py --mode rtrl --seq_len 512 --steps 10000

# BPTT with truncation (for comparison)
python train_haystack.py --mode bptt --seq_len 512 --tbptt 64 --steps 10000
```

### 3. Memory Profiling

```bash
# Benchmark memory scaling
python benchmark_memory.py --benchmark

# Analyze memory components
python benchmark_memory.py --analyze
```

## Project Structure

```
rtrl_moe/
├── circular_seg_tree.py    # Circular segment tree for sparse Jacobians
├── structure.py            # Module exports
├── rtrl_block.py           # Block RTRL implementation
├── rtrl_minimal.py         # Minimal RTRL reference
├── moe.py                  # Recurrent MoE model
├── train_anbn.py           # a^n b^n counting task
├── train_haystack.py       # Haystack retrieval task
├── benchmark_memory.py     # Memory profiling utilities
├── rtrl_moe.ipynb         # Interactive notebook
└── README.md              # This file
```

## Architecture

### RecurrentMoE Model

The model combines:
1. **State Update**: Cross-attention between latent slots and input
2. **Sparse Writing**: Only updates top-k latent slots per step
3. **Output**: Attention over latent slots to produce predictions

### Block RTRL Algorithm

Instead of computing full Jacobians:
```python
P_t = J_h @ P_{t-1} + J_θ  # Standard RTRL: O(H²P)
```

We use **block sparsity**:
```python
P_t[write_idx] = J_h[write_idx, read_idx] @ P_{t-1}[read_idx] + J_θ
```

Where `write_idx` and `read_idx` are sparse subsets determined by the model's gating.

## Tasks

### 1. a^n b^n Counting

Classify whether sequence has format a^n b^n (equal counts) or not.
- **Difficulty**: Requires counting and long-term memory
- **Sequence Length**: Variable (1-16 of each symbol)
- **RTRL Advantage**: Constant memory even for very long sequences

### 2. Haystack Retrieval

Retrieve a key embedded in a long sequence of distractors.
- **Format**: `[BOS, ..., KEY, <value>, SEP, ..., Q]` → predict `<value>`
- **Sequence Length**: 64-2048+ tokens
- **RTRL Advantage**: Enables training on 2K+ sequences where BPTT OOMs

## Results

### Memory Scaling

RTRL memory stays **constant** while BPTT grows linearly:

| Sequence Length | BPTT Memory | RTRL Memory | Ratio |
|-----------------|-------------|-------------|-------|
| 64              | ~150 MB     | ~80 MB      | 1.9x  |
| 256             | ~400 MB     | ~80 MB      | 5.0x  |
| 1024            | ~1200 MB    | ~80 MB      | 15x   |
| 2048            | OOM         | ~80 MB      | ∞     |

### Performance

Both methods achieve similar final accuracy on short sequences, but **only RTRL can scale to very long sequences**.

## Advanced Features

### Circular Segment Tree (Future)

The `CircularTree` class enables efficient query of products of sparse Jacobian matrices:
```python
# Store sparse Jacobians efficiently
buffer = CircularTree(window_size, None, sparse_left_mul)
buffer.update((indices, sparse_rows))

# Query product over time range [t1, t2)
product = buffer.query(t1, t2)
```

This is currently commented out but ready for integration to further reduce memory for parameter updates that were last active many steps ago.

## Citation

If you use this code, please cite:
```bibtex
@misc{rtrl_moe_2025,
  author = {Cabrillana, Jean-Manuel},
  title = {RTRL with Read-Write Sparsity and MoE},
  year = {2025},
  url = {https://github.com/jmCabrillana/rtrl_moe}
}
```

## License

MIT

## References

- Williams & Zipser (1989): "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks"
- Tallec & Ollivier (2017): "Unbiasing Truncated Backpropagation Through Time"
- Shazeer et al. (2017): "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
