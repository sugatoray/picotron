import os
import re
import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors import safe_open
import contextlib

from picotron.utils import assert_no_meta_tensors
import picotron.process_group_manager as pgm

@contextlib.contextmanager
def init_model_with_dematerialized_weights(include_buffers: bool = False):
    """
    From Accelerate library: https://github.com/huggingface/accelerate/blob/v0.11.0/src/accelerate/big_modeling.py#L254
    Context manager that initializes models with empty weights (no memory allocation).
    
    Args:
        include_buffers (bool): Whether to also skip buffer initialization.
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(torch.device("meta")), **kwargs)

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer


def initialize_model_with_materialized_weights(model, model_config, checkpoint_path, initialize_weight_tensor_func = None):
    """Initialize model with correct tensor shapes but random weights"""

    initialization_manager = InitializationManager(model, model_config)

    # convert layer distribution ids to layer_name (using the same naming convention as in safetensors)
    model_layer_name_sft_format = initialization_manager.get_layer_names_in_sft_format()
    print(f"Rank {pgm.process_group_manager.pp_rank} responsible for layers: {model_layer_name_sft_format}")
    
    safetensors_checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")
    with safe_open(safetensors_checkpoint_path, framework="pytorch", device="cpu") as f:
        safetensors_names = f.keys()
        
        if len(safetensors_names) > len(model_layer_name_sft_format):
            print(f"Warning: Checkpoint has {len(safetensors_names)} layers but model only has {len(model_layer_name_sft_format)} layers.")

        # Create state dict with random tensors
        state_dict = {}
        for sft_name in model_layer_name_sft_format:
            # if is_tensor_belongs_to_current_pp_rank(sft_name, model_layer_name_sft_format):
            hf_name = initialization_manager.convert_safetensors_to_hf_name(sft_name)
            tensor = f.get_tensor(sft_name)
            tensor = initialization_manager.adjust_tensor_size(tensor, hf_name)

            #TODO: initialize_weight_tensor_func
            #TODO: is layernorm init the same way as q k v ?
            state_dict[hf_name] = torch.randn_like(tensor)
    
    #TODO: Handle Tensor Parallel splitting if needed

    dist.barrier()
    model.load_state_dict(state_dict, strict=True, assign=True)
    dist.barrier()
    assert_no_meta_tensors(model)
    return model

class InitializationManager:
    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        
    def get_layer_names_in_sft_format(self):
        """Get layer names in safetensors format based on model's layer distribution."""
        decoder_components = [
            "input_layernorm",
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "post_attention_layernorm",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        ]
        
        # Generate base layer names
        layer_names = []
        base_names = [f"model.layers.{id}" for id in self.model.layer_distribution]
        for layer in base_names:
            layer_names.extend(f"{layer}.{component}.weight" for component in decoder_components)
        
        # Add special layers based on pipeline stage
        if pgm.process_group_manager.pp_is_first_stage:
            layer_names.insert(0, "model.embed_tokens.weight")
        elif pgm.process_group_manager.pp_is_last_stage:
            layer_names.extend(["model.norm.weight", "lm_head.weight"])
        
        return layer_names

    def adjust_tensor_size(self, tensor, name):
        """Resize tensor based on architecture changes."""
        if 'attention' not in name:
            return tensor
            
        hidden_size = self.model_config.hidden_size
        head_dim = hidden_size // self.model_config.num_attention_heads
        
        if 'q_proj.weight' in name:
            target_dim = self.model_config.num_attention_heads * head_dim
        elif 'k_proj.weight' in name or 'v_proj.weight' in name:
            target_dim = self.model_config.num_key_value_heads * head_dim
        else:
            return tensor
        
        # Adjust tensor size if needed
        if tensor.shape[0] != target_dim:
            if target_dim > tensor.shape[0]:
                pad_tensor = torch.empty(target_dim - tensor.shape[0], tensor.shape[1], 
                                       dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad_tensor], dim=0)
            else:
                tensor = tensor[:target_dim, :]
                
        return tensor

    def convert_safetensors_to_hf_name(self, sft_name):
        """Convert safetensors naming convention to HuggingFace naming convention."""
        name_mapping = {
            "model.": "",
            "layers.": "decoder_layers.",
            "embed_tokens": "embedding",
            "self_attn.": "attention.",
            "o_proj": "out_proj",
            "lm_head": "final_proj",
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            r'^norm': 'final_norm'
        }
        
        result = sft_name
        for pattern, replacement in name_mapping.items():
            result = re.sub(pattern, replacement, result)
        return result

#TODO: Implement and Move save/load checkpoint here
# class CheckpointManager:
#     pass
