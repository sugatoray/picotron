from copy import deepcopy
import numpy as np
from template.template_base_configs import template_base_config
import itertools
import yaml
import os
from transformers import AutoTokenizer
import math
import shutil
import argparse

def update_config_based_on_model(model: str, config: dict):
    
    # Setting num_attention_heads = num_key_value_heads for all models <=> using MHA for all layers
    
    if model == "small-llama":
        config["model"]["model_config"]["hidden_size"] = 512
        config["model"]["model_config"]["intermediate_size"] = 1024
        config["model"]["model_config"]["num_attention_heads"] = 16
        config["model"]["model_config"]["num_hidden_layers"] = 10
        config["model"]["model_config"]["num_key_value_heads"] = 16
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]
    elif model == "llama-1M":
        config["model"]["model_config"]["hidden_size"] = 768
        config["model"]["model_config"]["intermediate_size"] = 3072
        config["model"]["model_config"]["num_attention_heads"] = 16
        config["model"]["model_config"]["num_hidden_layers"] = 12
        config["model"]["model_config"]["num_key_value_heads"] = 16
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]
    elif model == "llama-1B":
        # HuggingFaceFW/ablation-model-fineweb-v1
        config["model"]["model_config"]["hidden_size"] = 2048
        config["model"]["model_config"]["intermediate_size"] = 4096
        config["model"]["model_config"]["num_attention_heads"] = 32
        config["model"]["model_config"]["num_hidden_layers"] = 24
        config["model"]["model_config"]["num_key_value_heads"] = 32
        config["model"]["model_config"]["max_position_embeddings"] = config["tokens"]["sequence_length"]

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name_or_path"])
    config["model"]["model_config"]["vocab_size"] = tokenizer.vocab_size

def create_single_config(
    out_dir: str,
    model: str,
    gpus: int,
    dp: int,
    tp: int,
    pp: int,
    bapr: int,
    mbs: int,
    no_profiler: bool = False,
    cluster: str = "hf",
    exp_name: str = None,
    seq_len: int = 4096,
    lighteval: bool = False,
    s3: bool = False,
    # recompute_layer: bool = False,
    dry_run: bool = False
):

    run_path = os.path.join(out_dir, exp_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Creating single config for {model} given {gpus} GPUs")
    config_content = deepcopy(base_config)
    config_content["tokens"]["sequence_length"] = seq_len
    # config_content["parallelism"]["recompute_layer"] = recompute_layer
    config_content["checkpoints"]["checkpoints_path"] = run_path
    update_config_based_on_model(model, config_content)
    
    if cluster == "hf":
        tp_max_cluster = 8
    elif cluster == "swiss-ai":
        tp_max_cluster = 4 # GH200

    config_content['parallelism']['dp'] = dp
    config_content['parallelism']['tp'] = tp
    config_content['parallelism']['pp'] = pp
    
     # Compute global batch_size and print
    gbs = dp * mbs * bapr
    gbs_token = gbs * seq_len
    # Print in human readable format
    print(f"Gbs_token: {gbs_token:,}, Gbs: {gbs}, dp: {dp}, seq_len: {seq_len}, bapr: {bapr}, mbs: {mbs}")
    
    config_content['tokens']['batch_accumulation_per_replica'] = bapr
    config_content['tokens']['micro_batch_size'] = mbs
    
    # Create a directory for each combination of parallelism
    # if recompute_layer:
    #     run_path += "_recompute_layer"
    
    # Get absoulte path for run_path
    if no_profiler:
        config_content['profiler'] = None
    else:
        config_content['profiler']['profiler_export_path'] = os.path.abspath(run_path)
    
    if s3:
        config_content["general"]["is_s3_available"] = True
        config_content['s3_upload'] = {
            "remove_after_upload": True,
            "s5cmd_concurrency": 5,
            "s5cmd_numworkers": 16,
            "s5cmd_path": "/fsx/elie_bakouch/miniconda3/envs/smollm/bin/s5cmd",
            "upload_s3_path": f"s3://huggingface-brrr-us-east-1/fmom/nanotron_pr/{exp_name}"
        }
    
    if lighteval:
        config_content['lighteval'] = {
            "batch_size": 16,
            "generation": None,
            "logging": {
                "output_dir": None,
                "public_run": False,
                "push_to_hub": True,
                "push_to_tensorboard": True,
                "results_org": "HuggingFaceSmol",
                "save_details": True,
                "tensorboard_metric_prefix": "eval"
            },
            "parallelism": {
                "dp": dp,
                "expert_parallel_size": 1,
                "pp": pp,
                "pp_engine": "1f1b",
                "recompute_layer": False,
                "tp": tp,
                "tp_linear_async_communication": False,
                "tp_mode": "ALL_REDUCE",
                "tp_recompute_allgather": True
            },
            "tasks": {
                "custom_tasks": "nanotron.lighteval.evaluation_tasks",
                "dataset_loading_processes": 8,
                "max_samples": 1000,
                "multichoice_continuations_start_space": None,
                "num_fewshot_seeds": None,
                "pair_wise_tokenization": False,
                "tasks": "early-signal"
            }
        }
    
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    
    if not dry_run:
        os.makedirs(run_path)
        with open(os.path.join(run_path, "config.yaml"), "w") as new_config:
            yaml.dump(config_content, new_config, default_flow_style=False, sort_keys=False)
    
    del config_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="Output directory to store the configs")
    parser.add_argument("--model", type=str, help="Model to create configs for")
    parser.add_argument("--gpus", type=int, help="Number of GPUs")
    parser.add_argument("--dp", type=int, required=True, help="Max number of data parallelism")
    parser.add_argument("--tp", type=int, required=True, help="Max number of tensor parallelism")
    parser.add_argument("--pp", type=int, required=True, help="Max number of pipeline parallelism")
    parser.add_argument("--bapr", type=int, help="Max batch accumulation per replica")
    parser.add_argument("--mbs", type=int, help="Max micro batch size")
    parser.add_argument("--seq_len", type=int, help="Sequence length", default=4096)
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--recompute_layer", action="store_true", help="Enable recompute allgather for tensor parallelism")
    parser.add_argument("--use_async", action="store_true", help="Enable async communication for tensor parallelism")
    parser.add_argument("--lighteval", action="store_true", help="Enable light evaluation")
    parser.add_argument("--s3", action="store_true", help="Enable light evaluation")
    
    args=parser.parse_args()
    
    create_single_config(
        out_dir=args.out_dir,
        model=args.model,
        gpus=args.gpus,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        bapr=args.bapr,
        mbs=args.mbs,
        cluster="hf",
        exp_name=args.exp_name,
        seq_len=args.seq_len,
        # recompute_layer=args.recompute_layer,
        lighteval=args.lighteval,
        s3=args.s3,
        dry_run=False,
        no_profiler=True
    )    
