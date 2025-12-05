"""
Memory profiling and benchmarking script
Demonstrates RTRL's constant memory advantage over BPTT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

from rtrl_block import BlockRTRL
from moe import RecurrentMoE


def profile_memory_and_time(model, seq_len, vocab_size, hidden_dim, device, mode='bptt', tbptt=None):
    """
    Profile both memory usage and computation time
    Returns: (peak_memory_mb, elapsed_time_sec)
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Generate dummy data
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    y = torch.randint(0, vocab_size - 4, (1,), device=device)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    if mode == 'bptt':
        if tbptt is None:
            # Full BPTT
            x_onehot = F.one_hot(x, num_classes=vocab_size).float()
            B, H = 1, hidden_dim
            h_t = model.init_state(B, device=device).requires_grad_()
            logits, _, h_next = model(x_onehot, h_t)
            loss = criterion(logits, y)
            loss.backward()
        else:
            # Truncated BPTT
            B, H = 1, hidden_dim
            h_t = model.init_state(B, device=device)
            
            for s in range(0, seq_len - 1, tbptt):
                chunk = x[:, s:min(s+tbptt, seq_len)]
                x_chunk = F.one_hot(chunk, num_classes=vocab_size).float()
                logits, _, h_next = model(x_chunk, h_t)
                h_t = h_next.detach()
            
            loss = criterion(logits, y)
            loss.backward()
    
    elif mode == 'rtrl':
        state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
        B, H = 1, model.d * model.n_slots
        rtrl = BlockRTRL(state_params, B, H)
        h_t = model.init_state(B, device=device).requires_grad_()
        
        # Process sequence step by step
        for s in range(seq_len - 1):
            x_t = F.one_hot(x[:, s:s+1], num_classes=vocab_size).float()
            logits, info, h_next = model(x_t, h_t)
            rtrl.step(model, x_t, h_t, None, state_params)
            h_t = h_next.detach().requires_grad_()
        
        # Final step with loss
        x_final = F.one_hot(x[:, -1:], num_classes=vocab_size).float()
        logits, info, h_next = model(x_final, h_t)
        loss = criterion(logits, y)
        rtrl.step(model, x_final, h_t, loss, state_params)
    
    elapsed_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    
    return peak_memory, elapsed_time


def benchmark_scaling():
    """Benchmark memory and time scaling with sequence length"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_dim = 64
    vocab_size = 8
    n_slots = 16
    
    # Test different sequence lengths
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    
    results = {
        'seq_lengths': seq_lengths,
        'bptt_full_memory': [],
        'bptt_full_time': [],
        'bptt_trunc_memory': [],
        'bptt_trunc_time': [],
        'rtrl_memory': [],
        'rtrl_time': []
    }
    
    tbptt = 64  # Truncated BPTT window
    
    print(f"\n{'='*80}")
    print(f"Memory & Time Scaling Benchmark")
    print(f"Hidden Dim: {hidden_dim}, Vocab: {vocab_size}, Slots: {n_slots}")
    print(f"{'='*80}\n")
    print(f"{'SeqLen':>8} | {'BPTT Full':^20} | {'BPTT Trunc':^20} | {'RTRL':^20} | {'Speedup':>10}")
    print(f"{'':>8} | {'Mem(MB)':>9} {'Time(s)':>9} | {'Mem(MB)':>9} {'Time(s)':>9} | {'Mem(MB)':>9} {'Time(s)':>9} | {'BPTT/RTRL':>10}")
    print(f"{'-'*80}")
    
    for seq_len in seq_lengths:
        try:
            # BPTT Full
            model = RecurrentMoE(d_model=hidden_dim, n_heads=2, n_slots=n_slots,
                                n_experts=4, topk=2, d_in=vocab_size, d_out=vocab_size-4).to(device)
            mem_bptt_full, time_bptt_full = profile_memory_and_time(
                model, seq_len, vocab_size, hidden_dim, device, 'bptt', None)
            results['bptt_full_memory'].append(mem_bptt_full)
            results['bptt_full_time'].append(time_bptt_full)
            del model
            torch.cuda.empty_cache()
            
            # BPTT Truncated
            model = RecurrentMoE(d_model=hidden_dim, n_heads=2, n_slots=n_slots,
                                n_experts=4, topk=2, d_in=vocab_size, d_out=vocab_size-4).to(device)
            mem_bptt_trunc, time_bptt_trunc = profile_memory_and_time(
                model, seq_len, vocab_size, hidden_dim, device, 'bptt', tbptt)
            results['bptt_trunc_memory'].append(mem_bptt_trunc)
            results['bptt_trunc_time'].append(time_bptt_trunc)
            del model
            torch.cuda.empty_cache()
            
            # RTRL
            model = RecurrentMoE(d_model=hidden_dim, n_heads=2, n_slots=n_slots,
                                n_experts=4, topk=2, d_in=vocab_size, d_out=vocab_size-4).to(device)
            mem_rtrl, time_rtrl = profile_memory_and_time(
                model, seq_len, vocab_size, hidden_dim, device, 'rtrl')
            results['rtrl_memory'].append(mem_rtrl)
            results['rtrl_time'].append(time_rtrl)
            del model
            torch.cuda.empty_cache()
            
            speedup = time_bptt_full / time_rtrl
            
            print(f"{seq_len:>8} | {mem_bptt_full:>9.2f} {time_bptt_full:>9.3f} | "
                  f"{mem_bptt_trunc:>9.2f} {time_bptt_trunc:>9.3f} | "
                  f"{mem_rtrl:>9.2f} {time_rtrl:>9.3f} | {speedup:>10.2f}x")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{seq_len:>8} | {'OOM':>9} {'---':>9} | {'---':>9} {'---':>9} | {'---':>9} {'---':>9} | {'---':>10}")
                results['bptt_full_memory'].append(None)
                results['bptt_full_time'].append(None)
                results['bptt_trunc_memory'].append(None)
                results['bptt_trunc_time'].append(None)
                results['rtrl_memory'].append(None)
                results['rtrl_time'].append(None)
                torch.cuda.empty_cache()
            else:
                raise
    
    print(f"{'-'*80}\n")
    
    # Plot results
    plot_results(results)
    
    return results


def plot_results(results):
    """Plot memory and time scaling results"""
    seq_lengths = results['seq_lengths']
    
    # Filter out None values
    def filter_none(data):
        return [(s, m) for s, m in zip(seq_lengths, data) if m is not None]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Memory plot
    if results['bptt_full_memory']:
        data = filter_none(results['bptt_full_memory'])
        if data:
            s, m = zip(*data)
            ax1.plot(s, m, 'o-', label='BPTT (Full)', linewidth=2)
    
    if results['bptt_trunc_memory']:
        data = filter_none(results['bptt_trunc_memory'])
        if data:
            s, m = zip(*data)
            ax1.plot(s, m, 's-', label='BPTT (Truncated)', linewidth=2)
    
    if results['rtrl_memory']:
        data = filter_none(results['rtrl_memory'])
        if data:
            s, m = zip(*data)
            ax1.plot(s, m, '^-', label='RTRL', linewidth=2, color='green')
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax1.set_title('Memory Scaling', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # Time plot
    if results['bptt_full_time']:
        data = filter_none(results['bptt_full_time'])
        if data:
            s, t = zip(*data)
            ax2.plot(s, t, 'o-', label='BPTT (Full)', linewidth=2)
    
    if results['bptt_trunc_time']:
        data = filter_none(results['bptt_trunc_time'])
        if data:
            s, t = zip(*data)
            ax2.plot(s, t, 's-', label='BPTT (Truncated)', linewidth=2)
    
    if results['rtrl_time']:
        data = filter_none(results['rtrl_time'])
        if data:
            s, t = zip(*data)
            ax2.plot(s, t, '^-', label='RTRL', linewidth=2, color='green')
    
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Time Scaling', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('rtrl_vs_bptt_scaling.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: rtrl_vs_bptt_scaling.png")
    plt.close()


def analyze_memory_components():
    """Analyze memory components of RTRL"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_dim = 64
    n_slots = 16
    vocab_size = 8
    seq_len = 512
    
    model = RecurrentMoE(d_model=hidden_dim, n_heads=2, n_slots=n_slots,
                        n_experts=4, topk=2, d_in=vocab_size, d_out=vocab_size-4).to(device)
    
    state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
    B, H = 1, model.d * model.n_slots
    
    # Calculate memory components
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    rtrl = BlockRTRL(state_params, B, H)
    P_t_memory = sum(p.numel() * p.element_size() for p in rtrl.P_t.values()) / 1024**2
    activation_memory = B * H * 4 / 1024**2  # rough estimate
    
    print(f"\n{'='*60}")
    print(f"RTRL Memory Component Analysis")
    print(f"{'='*60}")
    print(f"Model Parameters:     {param_memory:.2f} MB")
    print(f"RTRL Sensitivity (P): {P_t_memory:.2f} MB")
    print(f"Activations (est):    {activation_memory:.2f} MB")
    print(f"Total (est):          {param_memory + P_t_memory + activation_memory:.2f} MB")
    print(f"\nNote: RTRL memory is CONSTANT regardless of sequence length!")
    print(f"      BPTT memory grows linearly with sequence length.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='Run full benchmark')
    parser.add_argument('--analyze', action='store_true', help='Analyze memory components')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Memory profiling requires GPU.")
        return
    
    if args.analyze or (not args.benchmark and not args.analyze):
        analyze_memory_components()
    
    if args.benchmark or (not args.benchmark and not args.analyze):
        results = benchmark_scaling()


if __name__ == "__main__":
    main()
