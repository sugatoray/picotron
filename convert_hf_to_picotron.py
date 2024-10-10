"""
torchrun --nproc_per_node=1 convert_hf_to_picotron.py --save_path smollm.pth
"""
import os
import argparse
from tqdm import tqdm
import torch, torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_all_seed
import lovely_tensors as lt; lt.monkey_patch()

from model import Llama
from process_group_manager import setup_process_group_manager

def sanity_check_weights(model, model_hf, picotron_to_hf):
    
    total, fail = 0, 0
    
    state_dict = model.state_dict()
    state_dict_hf = model_hf.state_dict()
    
    for name, name_hf in picotron_to_hf.items():
        
        param_hf = state_dict_hf[name_hf]
        param = state_dict[name]
        
        total += 1
        try:
            torch.testing.assert_close(param_hf, param, rtol=1e-10, atol=1e-10)
        except AssertionError as e:
            print(f"{name_hf} and {name} are not equal")
            fail += 1
    
    if fail == 0:
        print("All parameters are equal")
    else:
        AssertionError(f"{fail}/{total} parameters are not equal")    

def sanity_check_generation(model, model_hf, model_name, prompt, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids_hf = tokenizer.encode(prompt, return_tensors="pt").to(device=model_hf.device)
    input_ids = input_ids_hf.clone().to(device=model_hf.device)

    for _ in range(max_new_tokens):
        # picotron model
        seq_len = input_ids.shape[1]
        position_index = torch.arange(seq_len).view(1, -1).to(device=model_hf.device)
    
        logits = model(input_ids=input_ids, position_ids=position_index)
        next_token = torch.argmax(logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token[:, -1].unsqueeze(-1)], dim=-1)

        # HF model
        logits_hf = model_hf(input_ids_hf).logits
        next_token_hf = torch.argmax(logits_hf[:, -1, :], dim=-1)
        input_ids_hf = torch.cat([input_ids_hf, next_token_hf.unsqueeze(0)], dim=-1)

        # Assert logits are equal
        torch.testing.assert_close(logits, logits_hf, atol=1e-4, rtol=1e-4)
    
    print("Input prompt:\n", prompt)
    print("Reference model output:\n", tokenizer.decode(input_ids_hf[0], skip_special_tokens=True))
    print("picotron model output:\n", tokenizer.decode(input_ids[0], skip_special_tokens=True))

def get_weights_mapping(model_hf, to_hf):
    
    hf_to_picotron = {}
    
    hf_to_picotron["model.embed_tokens.weight"] = "embedding.weight"
    hf_to_picotron["model.norm.weight"] = "final_norm.weight"
    hf_to_picotron["lm_head.weight"] = "final_proj.weight"
    
    for i in range(model_hf.config.num_hidden_layers):
        # Attention
        hf_to_picotron[f"model.layers.{i}.self_attn.q_proj.weight"] = f"decoder_layers.{i}.attention.q_proj.weight"
        hf_to_picotron[f"model.layers.{i}.self_attn.k_proj.weight"] = f"decoder_layers.{i}.attention.k_proj.weight"
        hf_to_picotron[f"model.layers.{i}.self_attn.v_proj.weight"] = f"decoder_layers.{i}.attention.v_proj.weight"
        hf_to_picotron[f"model.layers.{i}.self_attn.o_proj.weight"] = f"decoder_layers.{i}.attention.o_proj.weight"
        # MLP
        hf_to_picotron[f"model.layers.{i}.mlp.gate_proj.weight"] = f"decoder_layers.{i}.mlp.gate_proj.weight"
        hf_to_picotron[f"model.layers.{i}.mlp.up_proj.weight"] = f"decoder_layers.{i}.mlp.up_proj.weight"
        hf_to_picotron[f"model.layers.{i}.mlp.down_proj.weight"] = f"decoder_layers.{i}.mlp.down_proj.weight"

        hf_to_picotron[f"model.layers.{i}.input_layernorm.weight"] = f"decoder_layers.{i}.norm_attn.weight"
        hf_to_picotron[f"model.layers.{i}.post_attention_layernorm.weight"] = f"decoder_layers.{i}.norm_mlp.weight"

    # check if we have takens all keys from the reference model
    for key in hf_to_picotron:
        assert key in model_hf.state_dict(), f"{key} not found in reference model"

    if to_hf:
        # Mapping from picotron to hf
        picotron_to_hf = {v: k for k, v in hf_to_picotron.items()}
        return picotron_to_hf
    
    return hf_to_picotron
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF llama weights to picotron")
    parser.add_argument("--save_path", type=str, default="smollm.pth")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--prompt", type=str, default="My name is")
    parser.add_argument("--max_new_tokens", type=int, default=50)

    args = parser.parse_args()
    
    local_rank, world_size  = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

    #TODO: add gloo backend for generation
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    setup_process_group_manager(tp_size=1, pp_size=1, dp_size=1, cp_size=1)
    set_all_seed(seed=42)
    
    model_hf = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
   
    model = Llama(
        config=model_hf.config,
        device=device,
    )
    
    picotron_to_hf = get_weights_mapping(model_hf, to_hf=True)
    
    ref_state_dict = model_hf.state_dict()
    
    for name, param in tqdm(
        model.named_parameters(),
        total=len(list(model.named_parameters())),
        desc="Converting",
    ):
        if name in picotron_to_hf:
            ref_name = picotron_to_hf[name]
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)

    torch.save(model.state_dict(), args.save_path)

    new_model = Llama(
        config=model_hf.config,
        device=device,
    )
    new_model.load_state_dict(torch.load(args.save_path))

    print("Sanity check weight ...")
    sanity_check_weights(new_model, model_hf, picotron_to_hf)
    print("Sanity check generation ...")
    sanity_check_generation(new_model, model_hf, args.model_name, args.prompt, args.max_new_tokens)
    print("Conversion successful")