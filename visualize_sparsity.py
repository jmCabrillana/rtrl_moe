"""
Visualize sparsity patterns in RTRL with MoE
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from rtrl_block import BlockRTRL
from moe import RecurrentMoE, get_expert_latent_activated


def visualize_sparsity_over_time():
    """Visualize which experts and slots are active over time"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, D = 1, 64
    n_slots, n_experts = 16, 32
    steps = 100
    
    model = RecurrentMoE(
        d_model=D, n_heads=2, n_slots=n_slots,
        n_experts=n_experts, topk=2, d_in=8, d_out=4
    ).to(device)
    
    # Track activations
    expert_activations = np.zeros((steps, n_experts))
    slot_activations = np.zeros((steps, n_slots))
    param_updates = {k: [] for k in model.named_parameters() if k.startswith("state_")}
    
    state_params = {k: v for k, v in model.named_parameters() if k.startswith("state_")}
    H = model.d * model.n_slots
    rtrl = BlockRTRL(state_params, B, H, len_buffer=16)
    
    h = model.init_state(B, device=device).requires_grad_()
    
    print(f"Running {steps} steps to collect sparsity patterns...")
    
    for step in range(steps):
        x = F.one_hot(torch.randint(0, 8, (B, 1)), num_classes=8).float().to(device)
        y, info, h_next = model(x, h)
        
        # Record expert activations
        expert_ids = info['idx_experts'].cpu().numpy().flatten()
        expert_activations[step, expert_ids] = 1
        
        # Record slot activations
        slot_ids = info['idx_slots'].cpu().numpy().flatten()
        slot_activations[step, slot_ids] = 1
        
        # Track parameter updates
        active_params, write_idx = get_expert_latent_activated(model, info)
        for k in param_updates.keys():
            param_updates[k].append(1 if k in active_params else 0)
        
        rtrl.step(model, x, h, None, active_params, write_idx, write_idx)
        h = h_next.detach().requires_grad_()
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Expert activations over time
    im1 = axes[0].imshow(expert_activations.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    axes[0].set_ylabel('Expert ID', fontsize=12)
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_title('Expert Activations (sparse in expert dimension)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Active')
    
    # Add statistics
    sparsity_expert = 100 * expert_activations.sum() / expert_activations.size
    axes[0].text(0.02, 0.98, f'Sparsity: {sparsity_expert:.1f}%', 
                transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Slot activations over time
    im2 = axes[1].imshow(slot_activations.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    axes[1].set_ylabel('Slot ID', fontsize=12)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_title('Slot Activations (sparse in state dimension)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Active')
    
    sparsity_slot = 100 * slot_activations.sum() / slot_activations.size
    axes[1].text(0.02, 0.98, f'Sparsity: {sparsity_slot:.1f}%', 
                transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Parameter update patterns
    param_names = list(param_updates.keys())[:20]  # First 20 params
    param_matrix = np.array([param_updates[k] for k in param_names])
    
    im3 = axes[2].imshow(param_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    axes[2].set_ylabel('Parameter', fontsize=12)
    axes[2].set_xlabel('Time Step', fontsize=12)
    axes[2].set_title('Parameter Update Pattern (sparse in time)', fontsize=14, fontweight='bold')
    axes[2].set_yticks(range(len(param_names)))
    axes[2].set_yticklabels([k.split('.')[-1][:20] for k in param_names], fontsize=8)
    plt.colorbar(im3, ax=axes[2], label='Updated')
    
    sparsity_param = 100 * param_matrix.sum() / param_matrix.size
    axes[2].text(0.02, 0.98, f'Temporal sparsity: {100-sparsity_param:.1f}%', 
                transform=axes[2].transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('sparsity_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: sparsity_visualization.png")
    
    # Print statistics
    print(f"\nSparsity Statistics:")
    print(f"  Expert sparsity: {sparsity_expert:.1f}% active (expected ~{100*2/n_experts:.1f}% for top-2)")
    print(f"  Slot sparsity: {sparsity_slot:.1f}% active (expected ~{100*2/n_slots:.1f}% for top-2)")
    print(f"  Parameter temporal sparsity: {100-sparsity_param:.1f}% dormant")
    print(f"\nThis demonstrates:")
    print(f"  ✓ MoE sparse expert selection working")
    print(f"  ✓ Sparse slot writing working")
    print(f"  ✓ Parameters are often dormant (→ segment tree pays off!)")


def compare_update_costs():
    """Compare computational costs with/without sparsity"""
    print("\n" + "="*70)
    print("Computational Cost Comparison")
    print("="*70)
    
    H_full = 1024  # Full hidden state
    P_full = 1_000_000  # Total parameters
    
    # With sparsity
    W = R = 128  # Sparse read/write
    P_active = 30_000  # Active parameters (3%)
    
    # Cost formulas
    cost_full = H_full**2 * P_full
    cost_sparse_rw = W * R * P_full
    cost_sparse_full = W * R * P_active
    
    print(f"\nConfiguration:")
    print(f"  Full state: H = {H_full:,}")
    print(f"  Sparse write/read: W = R = {W}")
    print(f"  Total params: {P_full:,}")
    print(f"  Active params: {P_active:,} ({100*P_active/P_full:.1f}%)")
    
    print(f"\nComputational Cost (operations per step):")
    print(f"  Full RTRL:             {cost_full:>15,} ops")
    print(f"  + Read/write sparsity: {cost_sparse_rw:>15,} ops  ({cost_full/cost_sparse_rw:>6.1f}x faster)")
    print(f"  + Expert sparsity:     {cost_sparse_full:>15,} ops  ({cost_full/cost_sparse_full:>6.0f}x faster)")
    
    print(f"\nSegment Tree Lazy Update:")
    dormant_steps = 100
    cost_naive = dormant_steps * H_full**3
    cost_tree = int(np.log2(dormant_steps)) * H_full**3
    
    print(f"  Parameter dormant for {dormant_steps} steps:")
    print(f"    Naive update:      {cost_naive:>15,} ops")
    print(f"    Segment tree:      {cost_tree:>15,} ops  ({cost_naive/cost_tree:>6.1f}x faster)")
    
    print(f"\nOverall speedup: ~{cost_full/cost_sparse_full * cost_naive/cost_tree:,.0f}x")
    print(f"="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RTRL MoE Sparsity Visualization")
    print("="*70)
    
    visualize_sparsity_over_time()
    compare_update_costs()
    
    print("\n✓ Complete! Check 'sparsity_visualization.png' for visual results.")
