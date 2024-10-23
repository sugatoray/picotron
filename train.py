"""Training script for LLaMA model.
torchrun --nproc_per_node 1 --master_addr localhost --master_port 25500 train.py --use_wandb
torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --tp_size 4 
torchrun --nproc_per_node 2 --master_addr localhost --master_port 25500 train.py --pp_size 2
torchrun --nproc_per_node 2 --master_addr localhost --master_port 25500 train.py --pp_size 1 --dp_size 2
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --pp_size 2
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    train.py
#VERBOSE=0 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --pp_size 2 --dp_size 2
"""

import os
import time
import argparse
import numpy as np
import torch.nn.functional as F
import torch, torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset,Features, Sequence, Value
from functools import partial
from datasets import Features, Sequence, Value
import numpy as np
from src.parallel.tensor_parallel.tensor_parallel import TensorParallel
import src.distributed.process_group_manager as pgm
from utils import set_all_seed, print, to_readable_format
from src.distributed.process_group_manager import setup_process_group_manager
from src.parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from src.parallel.data_parallel.data_parallel_bucket import DataParallel
from src.parallel.context_parallel import ContextParallel
from model import Llama
import wandb
from src.distributed.distributed_primtives import all_reduce_loss_across_dp_ranks

class MicroBatchDataLoader(DataLoader):
    def __init__(self, global_batch_size, micro_batch_size, seq_length, dataset_name, tokenizer_name, num_workers, num_proc, grad_acc=1, split="train", num_samples=None):
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.local_batch_size = self.global_batch_size // pgm.process_group_manager.dp_world_size # each DP rank gets a local batch
        self.num_local_micro_batches = self.local_batch_size // self.micro_batch_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        self.grad_acc = grad_acc
        
        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)
        if num_samples:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        dist.barrier()
        
        # Tokenize and chunk the dataset
        self.tokenized_dataset = self.tokenize_dataset(self.dataset, "text", self.seq_length, num_proc)
        
        self.sampler = DistributedSampler(
            self.tokenized_dataset, 
            num_replicas=pgm.process_group_manager.dp_world_size, 
            rank=pgm.process_group_manager.dp_rank, 
            shuffle=False
        )
        
        super().__init__(
            self.tokenized_dataset, 
            batch_size=micro_batch_size if pgm.process_group_manager.pp_world_size > 1 else self.local_batch_size, # in PP we split a single batch into multiple micro-batches
            collate_fn=self.collate_batch, 
            pin_memory=True, 
            num_workers=num_workers, 
            sampler=self.sampler, 
            shuffle=False
        )

    @staticmethod
    def tokenizer_group_text(examples, tokenizer, sequence_length):
        """Tokenize a list of texts and group them in chunks of sequence_length + 1"""
        tokenized_text_batch = tokenizer.batch_encode_plus(
            examples,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='np'
        )
        concatenated_tokens = {'input_ids': np.concatenate(tokenized_text_batch['input_ids'])}
        total_length = len(concatenated_tokens['input_ids'])
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        result = {
            'input_ids': [
                concatenated_tokens['input_ids'][i : i + sequence_length + 1]
                for i in range(0, total_length - sequence_length, sequence_length)
            ]
        }
        return result

    def tokenize_dataset(self, dataset, text_column_name, sequence_length, num_proc):
        """Tokenize the dataset and group texts in chunks of sequence_length + 1"""
        # Create a partial function with fixed arguments
        tokenizer_func = partial(
            self.tokenizer_group_text,
            tokenizer=self.tokenizer,
            sequence_length=sequence_length
        )

        tokenized_dataset = dataset.map(
            tokenizer_func,
            input_columns=text_column_name,
            remove_columns=dataset.column_names,
            features=Features({
                "input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)
            }),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {sequence_length+1}",
        )

        return tokenized_dataset

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        batch_size = batch_input_ids.size(0)
        start_idx = pgm.process_group_manager.cp_rank * self.seq_length_per_gpu
        end_idx = start_idx + self.seq_length_per_gpu
        input_ids = batch_input_ids[:, start_idx:end_idx].contiguous()
        target_ids = batch_input_ids[:, start_idx+1:end_idx+1].contiguous()
        position_ids = torch.arange(start_idx, end_idx, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous() 
        local_attn_mask = torch.tril(torch.ones((self.seq_length_per_gpu, self.seq_length_per_gpu), dtype=torch.bool))
        attn_mask = local_attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
            "hidden_states": None
        }
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise StopIteration
        return batch

def train_step(model, data_loader, device):
    acc_loss = 0.0
    
    # get the next batch
    batch = next(data_loader)
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    
    for i in range(data_loader.grad_acc):
        outputs = model(input_ids=input_ids)

        # compute the loss
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len*batch_size, -1)
        loss = F.cross_entropy(outputs, target_ids, reduction='mean')
        
        loss.backward()

        acc_loss += loss.item()
    acc_loss /= data_loader.grad_acc

    return acc_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--load_path", type=str, default="smollm.pth")
    
    args = parser.parse_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    os.environ["DTYPE"] = "bfloat16" if dtype == torch.bfloat16 else "float32"
    os.environ["FLASH_ATTEN"] = "1" # Use cuda kernels from flash attention repo to accelerate the training. Model dtype should be torch.float16!
    assert (dtype == torch.bfloat16 and os.getenv("FLASH_ATTEN") == "1") or os.getenv("FLASH_ATTEN") != "1", "Kernel operations requires dtype=torch.bfloat16"

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    
    # SEQ_LEN, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS, SEED = 10, 6, 2, 1e-4, 20, 1800, 42
    ## hyperparameters
    SEQ_LEN, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS, SEED = 1024, 32, 1, 3e-4, 100000, int(10e8), 42
    grad_acc = 16

    assert SEQ_LEN % args.cp_size == 0, "SEQ_LEN must be divisible by cp_size for Context Parallelism"

    backend = "gloo" if args.use_cpu else "nccl"
    
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    
    dist.init_process_group(rank=local_rank, world_size=world_size, backend=backend, init_method=f"tcp://{host}:{port}")
    
    setup_process_group_manager(tp_size=args.tp_size, cp_size=args.cp_size, pp_size=args.pp_size, dp_size=args.dp_size)

    # if pgm.process_group_manager.global_rank == 0:
        # display_4D_parallelism_grid()
    
    set_all_seed(SEED)

    dataset_name = "roneneldan/TinyStories"
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    # model_name = "meta-llama/Llama-2-7b-hf"
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 16
    config.num_attention_heads = 16
    config.num_key_value_heads = 4

    model = Llama(config=config)
    
    if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
        wandb.init(
            project="picotron",
            name=f"test_convergence_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_size,
                "data_parallel_size": pgm.process_group_manager.dp_size,
                "model": model_name,
                "dataset": dataset_name,
                "max_tokens": MAX_TOKENS,
                "learning_rate": LEARNING_RATE,
                "seed": SEED,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "global_batch_size": GLOBAL_BATCH_SIZE,
            },
        )

    if pgm.process_group_manager.tp_world_size > 1:
        TensorParallel(model)

    # if pgm.process_group_manager.cp_size > 1:
        #TODO: do at the very end when we have fix convergence issue
        # model = ContextParallel(model, config)

    if pgm.process_group_manager.pp_world_size > 1:
        model = PipelineParallel(model, config)
    
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallel(model)

    model.to(dtype).to(device)
    model.train()
    
    data_loader = MicroBatchDataLoader(global_batch_size=GLOBAL_BATCH_SIZE, micro_batch_size=MICRO_BATCH_SIZE, seq_length=SEQ_LEN, dataset_name=dataset_name, tokenizer_name=model_name, grad_acc = grad_acc,num_workers=4, num_proc=4, num_samples=NUM_SAMPLES)
    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, config.hidden_size)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    trained_tokens, step = 0, 0
    tokens_per_step = data_loader.num_global_micro_batches * data_loader.micro_batch_size * SEQ_LEN * grad_acc

    dist.barrier()

    #TODO: Double-check consumed tokens after each steps (for example, MICRO_BATCH_SIZE=2 and using only dp_size=4, num_local_micro_batches=0 => division by 0)
    #TODO: Check convergence
    #TODO: Try multi-nodes
    #TODO: Add activation checkpointing
    #TODO: add gradient accumulation
    
    while trained_tokens < MAX_TOKENS:        
        #TODO: Add epoch support
        # data_loader.set_epoch(step)
        step_start_time = time.time()
        optimizer.zero_grad()
        
        if pgm.process_group_manager.pp_world_size > 1:
            loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device)
        else:
            loss = train_step(model, data_loader, device)
        
        loss = all_reduce_loss_across_dp_ranks(loss, device)

        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        
        # In DDP implementation I need to reset the gradient buffers
        if hasattr(model, 'reset'):
            model.reset()

        step_duration = time.time() - step_start_time
        
        if pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.pp_is_last_stage:
            print(f"[rank {pgm.process_group_manager.global_rank}] Step: {step}, Loss: {loss:.4f}, "
                f"Global batch size: {to_readable_format(tokens_per_step)}, "
                f"Tokens/s: {to_readable_format(tokens_per_step / step_duration)}, "
                f"Tokens/s/GPU: {to_readable_format(tokens_per_step / step_duration / world_size)}, "
                f"Tokens: {to_readable_format(trained_tokens)}/{to_readable_format(MAX_TOKENS)}"
            )
        
        if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
            wandb.log({"loss": loss, "trained_tokens": trained_tokens})
    
    if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
