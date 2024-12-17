"""
torchrun --nproc_per_node 1 test_meta_device.py --hf_token <HF_TOKEN>
"""
import os
import torch
import requests
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import safe_open
import requests
import json
import shutil
import subprocess
import argparse
import picotron.process_group_manager as pgm
from picotron.process_group_manager import setup_process_group_manager
from picotron.model import Llama
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from picotron.pipeline_parallel.pipeline_parallel import PipelineParallel
from picotron.checkpoint import init_model_with_materialized_weights, init_model_with_dematerialized_weights

def launch_distributed(tp_size, pp_size):
    """Launch the distributed processes"""
    nproc_per_node = tp_size * pp_size
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    assert gpu_count >= nproc_per_node, f"Number of GPUs ({gpu_count}) is less than nproc_per_node ({nproc_per_node})"

    if "RANK" not in os.environ:
        # Set required environment variables for distributed training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        print(f"Launching distributed training with {nproc_per_node} processes")
        os.environ["WORLD_SIZE"] = str(nproc_per_node)
        
        current_file = os.path.abspath(__file__)
        cmd = f"torchrun --nproc_per_node {nproc_per_node} {current_file}"
        if "HF_TOKEN" in os.environ:
            cmd += f" --hf_token {os.environ['HF_TOKEN']}"
        subprocess.run(cmd.split())
        exit()

def create_tmp_dir():
    """Create temporary directory in current working directory"""
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    if os.path.exists(tmp_dir):
        return tmp_dir
    os.makedirs(tmp_dir)
    return tmp_dir

def test_model_files_existence(model_name, hf_token):
    """Test if model files are available on HuggingFace"""
    print(f"\n1. Testing model files availability for {model_name}")
    
    files_to_check = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json"
    ]
    
    # Prepare headers with authentication token
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    found_files = []
    for file in files_to_check:
        url = f'https://huggingface.co/{model_name}/resolve/main/{file}'
        try:
            # Use GET request with stream=True and authentication headers
            response = requests.get(url, stream=True, headers=headers)
            if response.status_code == 200:
                found_files.append(file)
                print(f"✅ Found {file}")
                response.close()
            elif response.status_code == 401:
                print(f"❌ Authentication required for {file} (Status: {response.status_code})")
            elif response.status_code == 403:
                print(f"❌ Access denied for {file} (Status: {response.status_code})")
            else:
                print(f"❌ Not found {file} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ Error checking {file}: {str(e)}")
    
    return found_files

def test_model_download(model_name, hf_token, save_dir):
    """Download model using huggingface-cli"""
    print(f"\n2. Testing model download")
    
    os.makedirs(save_dir, exist_ok=True)
    
    files_to_download = ["config.json", "model.safetensors", "model.safetensors.index.json"]
    downloaded_files = []

    for file in files_to_download:
        if os.path.exists(os.path.join(save_dir, file)):
            print(f"✅ {file} already exists")
            downloaded_files.append(file)
            break

        model_cmd = f"huggingface-cli download {model_name} {file} --local-dir {save_dir} --token {hf_token}"
        print(f"Downloading {file}...")
        result = subprocess.run(model_cmd, shell=True, check=False, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            print(f"✅ {file} downloaded successfully")
            downloaded_files.append(file)
            
            # Verify files based on their type
            file_path = os.path.join(save_dir, file)
            if file.endswith('.safetensors'):
                try:
                    with safe_open(file_path, framework="pytorch", device="cpu") as f:
                        keys = list(f.keys())
                        print(f"✅ Safetensors file is valid")
                        print(f"- Number of tensors: {len(keys)}")
                except Exception as e:
                    print(f"❌ Error validating safetensors file: {str(e)}")
                    continue
            elif file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        index_data = json.load(f)
                        print(f"✅ Index JSON file is valid")
                        print(f"- Number of weight shards: {len(index_data.get('weight_map', {}))}")
                except Exception as e:
                    print(f"❌ Error validating index JSON file: {str(e)}")
                    continue
        else:
            error_message = result.stderr.decode('utf-8', errors='replace')
            if "404 Client Error" in error_message or "Entry Not Found" in error_message:
                print(f"❌ File {file} not found in repository")
            else:
                print(f"❌ Download failed: {error_message.strip()}")

    if len(downloaded_files) == 0:
        print("❌ No files were downloaded")
        return False

    print(f"\nSuccessfully downloaded files: {', '.join(downloaded_files)}")
    return True

def test_model_instantiation(model_name, tp_size, pp_size, save_dir):
    """Test loading the model into memory"""
    print(f"\n3. Testing model instantiation")

    dist.init_process_group(rank=int(os.environ["LOCAL_RANK"]), world_size=int(os.environ["WORLD_SIZE"]), backend="nccl", init_method=f"env://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    setup_process_group_manager(
        tp_size=tp_size,
        cp_size=1,
        pp_size=pp_size,
        dp_size=1
    )
    # Test model loading
    model_config = AutoConfig.from_pretrained(f"{save_dir}/config.json")

    with init_model_with_dematerialized_weights():
        model = Llama(config=model_config)

        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    model = init_model_with_materialized_weights(model, model_config, save_dir)
    return True

def run_test(test_name, model_name, hf_token, tp_size=1, pp_size=1):

    launch_distributed(tp_size, pp_size)

    print(f"Running Test for {model_name}")
    
    # Create tmp directory
    tmp_dir = create_tmp_dir()
    print(f"Created temporary directory: {tmp_dir}")
  
    # Test 1: Check files existence
    available_files = test_model_files_existence(model_name, hf_token)
    
    # Test 2: Test download
    if len(available_files) > 0:
        download_success = test_model_download(model_name, hf_token, save_dir=f"{tmp_dir}/{model_name}")
    else:
        print("Skipping download test as no files were found")
        return
    
    # Test 3: Test model instantiation
    if download_success:
        instantiation_success = test_model_instantiation(model_name, tp_size, pp_size, f"{tmp_dir}/{model_name}")
    else:
        print("Skipping instantiation test as download failed")
        return
    
    # Final results
    print(f"\n=== Test: {test_name} ===")
    print(f"Files found: {len(available_files)}")
    print(f"Download: {'Success ✅' if download_success else 'Failed ❌'}")
    print(f"Instantiation: {'Success ✅' if instantiation_success else 'Failed ❌'}")

    dist.destroy_process_group()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, required=True, help="HF token")
    args = parser.parse_args()

    # Set HF token in environment if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # run_test(test_name="No safetensors file", model_name="microsoft/phi-1")
    # run_test(test_name="Corrupted safetensors file", model_name="microsoft/phi-1")

    #TODO: create a test that spawn different process
    run_test(test_name="Single safetensors file", model_name="meta-llama/Llama-3.2-1B", hf_token=args.hf_token)
    # run_test(test_name="Already downloaded safetensors file", model_name="meta-llama/Llama-3.2-1B", hf_token=args.hf_token)
    run_test(test_name="Single safetensors file with TP", model_name="meta-llama/Llama-3.2-1B", hf_token=args.hf_token, tp_size=2)
    # run_test(test_name="Single safetensors file with PP", model_name="microsoft/phi-1", hf_token=args.hf_token, pp_size=2)
    # run_test(test_name="Single safetensors file with TP and PP", model_name="microsoft/phi-1", hf_token=args.hf_token, tp_size=2, pp_size=2)
    
    # run_test(test_name="Sharded safetensors file", model_name=??)
    # run_test(test_name="Already downloaded sharded safetensors file", model_name=??)
    # run_test(test_name="Sharded safetensors file with TP", model_name=??, tp_size=2)
    # run_test(test_name="Sharded safetensors file with PP", model_name="microsoft/phi-1", pp_size=2)

