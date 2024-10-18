"""Training script for LLaMA model.
torchrun --nproc_per_node 1 --master_addr localhost --master_port 25500 train.py --use_wandb
torchrun --nproc_per_node 2 --master_addr localhost --master_port 25500 train.py --tp_size 2 
torchrun --nproc_per_node 2 --master_addr localhost --master_port 25500 train.py --pp_size 2
torchrun --nproc_per_node 2 --master_addr localhost --master_port 25500 train.py --pp_size 1 --dp_size 2
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --tp_size 2
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    train.py
#VERBOSE=0 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --pp_size 2 --dp_size 2
"""

import os
import torch.nn.functional as F
import torch, torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
import argparse
from datasets import Features, Sequence, Value
import numpy as np
from src.parallel.tensor_parallel.tensor_parallel import TensorParallel
import src.distributed.process_group_manager as pgm
from utils import set_all_seed, print
from src.distributed.process_group_manager import setup_process_group_manager
from src.parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from src.parallel.data_parallel.data_parallel_bucket import DataParallel
from src.parallel.context_parallel import ContextParallel
from model import Llama
import wandb

class MicroBatchDataLoader(DataLoader):
    def __init__(self, global_batch_size, micro_batch_size, seq_length, dataset_name, tokenizer_name, grad_acc = 1, split="train", num_samples=None, num_workers=0):
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
        self.tokenized_dataset = self.tokenize_dataset(self.dataset, "text", self.seq_length)
        
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

    def tokenize_dataset(self, dataset, text_column_name, sequence_length, num_proc=48):
        def _tokenizer_group_text(texts):
            tokenized_text_batch = self.tokenizer.batch_encode_plus(
                texts,
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

        tokenized_dataset = dataset.map(
            _tokenizer_group_text,
            input_columns=text_column_name,
            remove_columns=dataset.column_names,
            features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
            batched=True,
            num_proc=num_proc,  # Adjust this based on your system capabilities
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {sequence_length+1}",
        )

        return tokenized_dataset

    def collate_batch(self, batch):
        input_ids = [item['input_ids'][:-1] for item in batch]
        label_ids = [item['input_ids'][1:] for item in batch]
        attention_mask = [[1] * len(input_id) for input_id in input_ids]
        label_mask = [[1] * len(label_id) for label_id in label_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(label_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label_mask': torch.tensor(label_mask, dtype=torch.long),
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
    
    args = parser.parse_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    
    # SEQ_LEN, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS, SEED = 10, 6, 2, 1e-4, 20, 1800, 42
    ## hyperparameters
    SEQ_LEN, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS, SEED = 1024, 16, 4, 3e-4, 100000, int(10e8), 42
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
    config = AutoConfig.from_pretrained(model_name)
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

    model.to(device)
    model.train()
    
    data_loader = MicroBatchDataLoader(GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LEN, dataset_name, model_name, grad_acc = grad_acc, num_samples=NUM_SAMPLES)
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
        optimizer.zero_grad()
        
        if pgm.process_group_manager.pp_world_size > 1:
            loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device)
        else:
            loss = train_step(model, data_loader, device)
        
        # average the loss across all DP/CP ranks
        if pgm.process_group_manager.dp_world_size > 1 or pgm.process_group_manager.cp_world_size > 1:
            #TODO: use all_reduce function from distributed_primitives.py
            loss_tensor = torch.tensor([loss], dtype=torch.float32, device=device)
            handle = dist.all_reduce(loss_tensor, group=pgm.process_group_manager.cp_dp_group, async_op=True, op=dist.ReduceOp.AVG)
        
        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        
        # In DDP implementation I need to reset the gradient buffers
        if hasattr(model, 'reset'):
            model.reset()
        
        if pgm.process_group_manager.global_rank == 0:
            if pgm.process_group_manager.dp_world_size > 1 or pgm.process_group_manager.cp_world_size > 1:
                handle.wait()
                loss = loss_tensor.item()
            print(f"[rank {pgm.process_group_manager.global_rank}] Step: {step}, Loss: {loss:.4f}, "
                f"Global batch size: {tokens_per_step}, "
                f"Tokens: {trained_tokens}/{MAX_TOKENS}"
                )
        
        if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
            wandb.log({"loss": loss, "trained_tokens": trained_tokens})
    
    if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
