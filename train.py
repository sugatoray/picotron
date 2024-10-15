#VERBOSE=0 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --pp_size 2 --dp_size 2
import os
import torch.nn.functional as F
import torch, torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
import argparse

import distributed.process_group_manager as pgm
from distributed.distributed_primtives import all_reduce_gradients_across_dp_cp_ranks
from utils import set_all_seed, print
from distributed.process_group_manager import setup_process_group_manager
from parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from parallel.data_parallel import DataParallel
from parallel.context_parallel import ContextParallel
from model import Llama
import wandb

class MicroBatchDataLoader(DataLoader):
    def __init__(self, global_batch_size, micro_batch_size, seq_length, dataset_name, tokenizer_name, split="train", num_samples=None):
        self.global_batch_size, self.micro_batch_size, self.seq_length = global_batch_size, micro_batch_size, seq_length
        self.local_batch_size = self.global_batch_size // pgm.process_group_manager.dp_world_size
        self.num_local_micro_batches = self.local_batch_size // self.micro_batch_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        
        self.seq_length_per_gpu = seq_length // pgm.process_group_manager.cp_world_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)
        if num_samples: self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        dist.barrier()
        self.dataset = self.dataset.map(lambda examples: self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.seq_length + 1, return_special_tokens_mask=False), batched=True, remove_columns=self.dataset.column_names).with_format("torch", columns=["input_ids"])
        
        self.sampler = DistributedSampler(self.dataset, num_replicas=pgm.process_group_manager.dp_world_size, rank=pgm.process_group_manager.dp_rank, shuffle=False)
        
        super().__init__(self.dataset, batch_size=micro_batch_size, collate_fn=self.collate_batch, pin_memory=True, num_workers=3, sampler=self.sampler, shuffle=False)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def collate_batch(self, batch_data):
        batch_input_ids = torch.stack([item['input_ids'] for item in batch_data])
        batch_size, seq_len = batch_input_ids.shape
        start_idx = pgm.process_group_manager.cp_rank * self.seq_length_per_gpu
        end_idx = start_idx + self.seq_length_per_gpu
        input_ids = batch_input_ids[:, start_idx:end_idx].contiguous()
        target_ids = batch_input_ids[:, start_idx+1:end_idx+1].contiguous()
        position_index = torch.arange(start_idx, end_idx, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()        
        local_attn_mask = torch.tril(torch.ones((self.seq_length_per_gpu, self.seq_length_per_gpu), dtype=torch.bool))
        attn_mask = local_attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_index": position_index,
            "attn_mask": attn_mask,
            "hidden_states": None
        }

def train_step(model, data_loader, device):
    total_loss = 0.0

    for _ in range(data_loader.num_local_micro_batches):
        batch = next(iter(data_loader))
        
        input_ids = batch["input_ids"].to(device)
        position_ids = batch["position_index"].to(device)
        target_ids = batch["target_ids"].to(device)

        batch_size, seq_len = input_ids.shape

        outputs = model(input_ids=input_ids, position_ids=position_ids)

        loss = F.cross_entropy(outputs.view(batch_size * seq_len, -1), target_ids.view(-1), reduction="mean")

        loss.backward()

        total_loss += loss.item()

    avg_loss = total_loss / data_loader.num_local_micro_batches
    return avg_loss

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
    
    SEQ_LEN, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS, SEED = 10, 6, 2, 1e-4, 20, 1800, 42

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
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    dataset_name = "roneneldan/TinyStories"
    config = AutoConfig.from_pretrained(model_name)
    
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
    
    #TODO: find a better way (should need to specify model_name + path to .pth)
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    config = AutoConfig.from_pretrained(model_name)

    model = Llama(
        config=config,
        device=device,
    ).to(device)

    model.load_state_dict(torch.load("smollm.pth"))

    # if pgm.process_group_manager.tp_world_size > 1:
        # model = TensorParallel(model, config).to(device)

    if pgm.process_group_manager.cp_size > 1:
        model = ContextParallel(model, config).to(device)

    if pgm.process_group_manager.pp_world_size > 1:
        model = PipelineParallel(model, config).to(device)
    
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallel(model, config).to(device)

    model.train()
    
    data_loader = MicroBatchDataLoader(GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LEN, dataset_name, model_name, num_samples=NUM_SAMPLES)
    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, config.hidden_size)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    trained_tokens, step = 0, 0
    tokens_per_step = data_loader.num_global_micro_batches * data_loader.micro_batch_size * SEQ_LEN

    dist.barrier()
    
    #TODO: Add Context Parallelism
    #TODO: Double-check consumed tokens after each steps (for example, MICRO_BATCH_SIZE=2 and using only dp_size=4, num_local_micro_batches=0 => division by 0)
    #TODO: Check convergence
    #TODO: Try multi-nodes
    #TODO: Add activation checkpointing
    #TODO: add gradient accumulation
    
    while trained_tokens < MAX_TOKENS:        
        data_loader.set_epoch(step)

        optimizer.zero_grad()
        
        if pgm.process_group_manager.pp_world_size > 1:
            loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device)
            # loss = train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device)
        else:
            loss = train_step(model, data_loader, device)

        if pgm.process_group_manager.dp_world_size > 1 or pgm.process_group_manager.cp_world_size > 1:
            all_reduce_gradients_across_dp_cp_ranks(model)

        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        
        if pgm.process_group_manager.global_rank == 0:
            print(f"[rank {pgm.process_group_manager.global_rank}] Step: {step}, Loss: {loss:.4f}, Tokens: {trained_tokens}/{MAX_TOKENS}")
        
        if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
            wandb.log({"loss": loss, "trained_tokens": trained_tokens})
    
    if pgm.process_group_manager.global_rank == 0 and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
