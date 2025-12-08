import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d, mult=1, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class ClassicRNN(nn.Module):
    """
    Classic RNN with same input/output interface as RecurrentMoE.
    
    Forward signature:
        input: (B, T, d_in)
        state: (B, n_slots * d_model) flat representation of hidden state
        output: (y, info, new_state)
            y: (B, d_out) prediction at last timestep
            info: dict with routing info (empty for RNN)
            new_state: (B, n_slots * d_model) flat representation
    """
    def __init__(self, d_model=512, n_heads=2, n_slots=4, n_experts=4, topk=2, topk_read=2, topk_write=2,
                  dropout=0.0, d_in=None, d_out=None, orthogonalize=False):
        super().__init__()
        d = d_model
        self.d = d
        self.n_slots = n_slots
        
        if not d_in: 
            d_in = d
        self.d_in = d_in
        
        if not d_out: 
            d_out = d
        self.d_out = d_out
        
        # Store signature parameters for compatibility
        self.topk = topk
        self.topk_read = topk_read
        self.topk_write = topk_write
        self.orthogonalize = orthogonalize
        
        # Input embedding
        self.input_proj = nn.Linear(d_in, d)
        
        # RNN cell (GRU for stable gradients)
        self.rnn_cell = nn.GRUCell(d, d)
        
        # Processing layers
        self.ln_rnn = nn.LayerNorm(d)
        self.ffn = MLP(d, dropout=dropout)
        self.ln_ffn = nn.LayerNorm(d)
        
        # Output projection
        self.output_proj = nn.Linear(d, d_out)
    
    @torch.no_grad()
    def init_state(self, batch_size, device=None, dtype=None):
        """Initialize hidden state as (B, n_slots * d) flat tensor."""
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        # Initialize with small random values
        state = torch.randn(batch_size, self.n_slots * self.d, device=device, dtype=dtype) * 0.02
        return state
    
    def forward(self, x, state_flat):
        """
        Forward pass processing sequence x with given state.
        
        Args:
            x: (B, T, d_in) input sequence
            state_flat: (B, n_slots * d) flat hidden state
        
        Returns:
            y: (B, d_out) output prediction
            info: dict with empty routing info (for compatibility)
            new_state: (B, n_slots * d) new hidden state
        """
        B, T, _ = x.shape
        D = self.d
        S = self.n_slots
        
        device = x.device
        dtype = x.dtype
        
        # Reshape state to (B, D)
        h = state_flat.reshape(B, D).contiguous()
        
        # Process sequence timestep by timestep
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_in)
            
            # Embed input
            x_emb = self.input_proj(x_t)  # (B, d)
            
            # GRU cell update
            h = self.rnn_cell(x_emb, h)  # (B, d)
            
            # Process with residual layers
            h_ln = self.ln_rnn(h)
            h_ffn = self.ffn(h_ln)
            h = h + h_ffn
        
        # Output from final hidden state
        y = self.output_proj(h)  # (B, d_out)
        
        # Reshape state back to flat representation (B, n_slots * d)
        # Tile h across slots for consistency with MoE interface
        new_state = h.unsqueeze(1).expand(B, S, D).reshape(B, S * D).contiguous()
        
        # Empty info dict for compatibility
        info = {
            "idx_experts": torch.tensor([], dtype=torch.long, device=device),
            "idx_slot_write": torch.tensor([], dtype=torch.long, device=device),
            "idx_slots_read": torch.tensor([], dtype=torch.long, device=device),
        }
        
        return y, info, new_state


def get_expert_latent_activated_rnn(model, info):
    """
    Extract active parameters for RNN (all parameters active).
    
    Returns:
        active_params: dict of all RNN parameters
        write_indices: list of all state indices (all slots written)
        read_indices: list of all state indices (all slots read)
    """
    D = model.d
    S = model.n_slots
    
    # All parameters are always active in RNN
    state_params = {k: v for k, v in model.named_parameters() if not k.startswith("output_")}
    active_params = state_params
    
    # All slots are read and written for RNN
    all_indices = list(range(S * D))
    write_indices = all_indices
    read_indices = all_indices
    
    return active_params, write_indices, read_indices
