import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        h = self.act_fn(self.gate_proj(x))
        h = self.up_proj(x) * h
        out = self.down_proj(h)
        return out

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, scaling_factor=1.0, device=None):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config, is_causal):
        super(Attention, self).__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = is_causal
        
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def forward(self, input_ids, position_ids):
        q = self.q_proj(input_ids)
        k = self.k_proj(input_ids)
        v = self.v_proj(input_ids)
        
        batch, seq_len, _ = q.shape
        
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            cos, sin = self.rotary(v, position_ids)
            q, k = self.rotary.apply_rotary_pos_emb(q, k, cos, sin)

        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)

        out = rearrange(
            out,
            "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)",
        ).contiguous()
        
        out = self.o_proj(out)
    
        return out

    def _repeat_kv(self, x, n_rep):
        batch, num_key_value_heads, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

class DecoderLayer(nn.Module):
    def __init__(self, config, is_causal):
        super(DecoderLayer, self).__init__()
        
        self.attention = Attention(config, is_causal)
        self.norm_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.norm_mlp = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, x, position_ids):
        x_norm = self.norm_attn(x) 
        h = x + self.attention(x_norm, position_ids)
        h_norm = self.norm_mlp(h)
        out = h + self.mlp(h_norm)
        return out

class Llama(nn.Module):
    def __init__(self, config, device, is_causal: bool = True):
        super(Llama, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size).to(device)
        self.decoder_layers = nn.ModuleList()
        
        for _ in range(config.num_hidden_layers):
            self.decoder_layers.append(DecoderLayer(config, is_causal).to(device))

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)
        self.final_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device) 

    def forward(self, input_ids, position_ids, hidden_states=None):
        x = hidden_states if hidden_states is not None else input_ids
        
        h = self.embedding(x)
        
        for layer in self.decoder_layers:
            h = layer(h, position_ids)
        
        h = self.final_norm(h)
        out = self.final_proj(h)
        
        return out.float()