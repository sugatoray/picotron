import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb
import src.distributed.process_group_manager as pgm
from src.parallel.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else torch.float32
init_method = init.xavier_normal_

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def apply_rotary_pos_emb(x, cos, sin):
    batch_size, num_head, seq_length, head_dim = x.size()
    x1 = x[..., : head_dim // 2]  
    x2 = x[..., head_dim // 2 :]  
    rotate_half = torch.cat([-x2, x1], dim=-1)
    x = x * cos + rotate_half * sin
    return x

def get_cos_sin(seq_length, head_dim, base=500000.0):
    assert head_dim%2==0
    # Results on CUDA and CPU are different even with the same formula, To match transformers implementation. frequency should be computed on CPU
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))
    position = torch.arange(seq_length).unsqueeze(1).to(device).float() # [seq_length, 1]
    # To match transformers implementation. m * theta should be computed on GPU
    theta = theta.to(device)
    return torch.cos(position.float()*theta.float()).to(dtype).repeat(1,2), torch.sin(position.float()*theta.float()).to(dtype).repeat(1,2) # [seq_length, head_dim], [seq_length, head_dim]

def flash_attention(q, k, v, causal = True):
    q = q.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    k = k.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    v = v.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    return flash_attn_func(q, k, v, causal=causal)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size//self.num_heads
        model_parallel_size = pgm.process_group_manager.tp_world_size
        self.num_local_heads = config.num_attention_heads // model_parallel_size # TP parallelism
        self.num_local_kv_heads = config.num_key_value_heads // model_parallel_size # TP parallelism
        self.is_merged_qkv_weight = os.getenv('MERGED_QKV_WEIGHT', '1')
        if self.is_merged_qkv_weight  == '1': 
            self.qkv_proj = nn.Linear(config.hidden_size, self.num_heads*self.head_dim + 2*self.num_key_values*self.head_dim, bias=False)
            # self.qkv_proj = ColumnParallelLinear(config.hidden_size, self.num_heads*self.head_dim + 2*self.num_key_values*self.head_dim, bias=False, gather_output=False, init_method=init_method)
        else:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads*self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, self.num_key_values*self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, self.num_key_values*self.head_dim, bias=False)
            # self.q_proj = ColumnParallelLinear(config.hidden_size, self.num_heads*self.head_dim, bias=False, gather_output=False, init_method=init_method) # why the init method is x? Xavier is better?
            # self.k_proj = ColumnParallelLinear(config.hidden_size, self.num_key_values*self.head_dim, bias=False, gather_output=False, init_method=init_method)
            # self.v_proj = ColumnParallelLinear(config.hidden_size, self.num_key_values*self.head_dim, bias=False, gather_output=False, init_method=init_method)
        # if os.getenv('FLASH_ROPE', '1') == '1':
        #     self.flash_rope = FlashRotaryEmbedding(dim=self.head_dim, interleaved=False, base=500000.0)
        
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.out_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False, input_is_parallel=True, init_method=init_method)
        self.kv_cache = None
        self.layer_idx = layer_idx
        
        ## TODO support mask
    
    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_dim = x.size()
        if self.is_merged_qkv_weight == '1':
            qkv = self.qkv_proj(x) # [batch_size, seq_length, num_heads*head_dim + 2*num_key_values*head_dim]
            q, k, v = torch.split(qkv,
                [
                    self.num_local_heads * self.head_dim,
                    self.num_local_kv_heads * self.head_dim,
                    self.num_local_kv_heads * self.head_dim,
                ],
                dim=-1,
            ) # [batch_size, seq_length, num_heads*head_dim] / [batch_size, seq_length, num_key_values*head_dim] / [batch_size, seq_length, num_key_values*head_dim]
        else:
            q = self.q_proj(x) # [batch_size, seq_length, num_heads*head_dim]
            k = self.k_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
            v = self.v_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
        if os.getenv('FLASH_ROPE', '0') != '1':
            q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim).transpose(1, 2)       # [batch_size, num_heads, seq_length, head_dim]
            k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_key_values, seq_length, head_dim]
            v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_key_values, seq_length, head_dim]
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        else:
            q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim)       # [batch_size, seq_length, num_heads, head_dim]
            k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim)  # [batch_size, seq_length, num_key_values, head_dim]
            q = apply_rotary_emb(q,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_heads, head_dim]
            k = apply_rotary_emb(k,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_key_values, head_dim]
            q = q.transpose(1, 2)                                                                   # [batch_size, num_heads, seq_length, head_dim]
            k = k.transpose(1, 2)                                                                   # [batch_size, num_key_values, seq_length, head_dim]
            v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1,2)   # [batch_size, num_key_values, seq_length, head_dim]
        if self.kv_cache is not None:
            # update kv_cache, and get stored k, v
            assert position_ids is not None, "position_ids should be provided to update kv_cache"
            k, v = self.kv_cache.update_cache_get_kv(k, v, position_ids)
        k = k.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        v = v.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        if os.getenv('ATTENTION', 'SDPA') == 'SDPA':
            causal = True if q.size(2) == k.size(2) else False # During decoding phase. The lenghth of q is usually 1. 
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal) # [batch_size, num_heads, seq_length, head_dim]
            out = out.transpose(1, 2) # [batch_size, seq_length, num_heads, head_dim]
        else:
            causal = True if q.size(2) == k.size(2) else False # During decoding phase. The lenghth of q is usually 1. 
            out = flash_attention(q, k, v, causal = causal) # [batch_size, seq_length, num_heads, head_dim] 
        out = out.reshape(batch_size, seq_length, self.num_local_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
        out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
        return out


class LLaMAMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.merged_gate_up = os.getenv('MERGED_GATE_UP_WEIGHT', '1') == '1'
        if self.merged_gate_up:
            self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size*2, bias=False)
            # self.gate_up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size*2, bias=False, gather_output=False, init_method=init_method)
        else:
            # self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            # self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False, init_method=init_method)
            self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False, init_method=init_method)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, input_is_parallel=True, init_method=init_method)
        
    def forward(self, x):
        if  self.merged_gate_up:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    # RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
    def __init__(self, config, layer_idx):
        super().__init__()
        RMSNorm = LlamaRMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = CausalSelfAttention(config, layer_idx = layer_idx)
        self.mlp = LLaMAMLP(config)
        self.layer_idx = layer_idx
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = get_cos_sin(config.max_position_embeddings, head_dim=head_dim , base=config.rope_theta) # [max_position_embeddings, head_dim]

    def forward(self, x, attention_mask = None, position_ids = None):
        #TODO: Use the default position_ids for RoPE during training. If we have time, work on generation
        _, seq_length, _ = x.size()
        cos, sin = self.cos[:seq_length], self.sin[:seq_length]
        x = x + self.attention(self.input_layernorm(x), cos, sin, attention_mask, position_ids) # Attention 
        x = x + self.mlp(self.post_attention_layernorm(x)) # MLP
        return x
    
class LLaMA(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # sanity check 
        assert config.hidden_size % config.num_attention_heads==0
        assert config.num_attention_heads % config.num_key_value_heads==0 
        
        # params
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads 
        self.head_dim = self.hidden_size//self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config
        
        # modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        # self.embedding = VocabParallelEmbedding(self.vocab_size, self.hidden_size, init_method=init_method)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config,layer_idx = i) for i in range(self.num_layers)])
        self.final_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        # self.final_proj = ColumnParallelLinear(self.hidden_size, self.vocab_size, bias=False, gather_output=True, init_method=init_method) # we can also not gather the output. TODO: add vocab_parallel_cross_entropy
        self.final_norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor = None):
        batch_size, seq_length = input_ids.size()
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x)  # [batch_size, seq_length, hidden_dim]
        x = self.final_norm(x)
        logits = self.final_proj(x)
        
        return logits  # [batch_size, seq_length, vocab_size]
    
    # https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L289-L303
    # TODO: Need to check the formula.
    def get_flops(self, fwdbwd_per_iter, dt, num_params):
        L, H, T = self.num_layers , self.hidden_size, self.max_position_embeddings
        flops_per_fwdbwd = 6 * num_params * T + 12* L* H* T ** 2
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        return flops_achieved