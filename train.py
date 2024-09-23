#VERBOSE=0 torchrun --nproc_per_node 3 train.py --pp_size 3 
import os
import torch, torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

import parallel_context as pc
from utils import set_all_seed
from parallel_context import setup_parallel_context
from pipeline_parallel import pipeline_parallel_1f1b, pipeline_parallel_afab, PipelineParallel

class MicroBatchDataLoader(DataLoader):
    def __init__(self, global_batch_size, micro_batch_size, data_parallel_size, seq_length, dataset_name, tokenizer_name, split="train", num_samples=None):
        self.global_batch_size, self.micro_batch_size, self.data_parallel_size, self.seq_length = global_batch_size, micro_batch_size, data_parallel_size, seq_length
        self.local_batch_size = self.global_batch_size // self.data_parallel_size
        self.num_local_micro_batches = self.local_batch_size // self.micro_batch_size
        self.num_global_micro_batches = self.global_batch_size // self.micro_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)
        if num_samples: self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        dist.barrier()
        self.dataset = self.dataset.map(lambda examples: self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.seq_length + 1, return_special_tokens_mask=False), batched=True, remove_columns=self.dataset.column_names).with_format("torch", columns=["input_ids"])
        super().__init__(self.dataset, batch_size=micro_batch_size, collate_fn=self.collate_batch, pin_memory=True, num_workers=3, sampler=DistributedSampler(self.dataset, num_replicas=data_parallel_size, rank=0, shuffle=False), shuffle=False)

    def collate_batch(self, batch_data):
        batch_input_ids = torch.stack([item['input_ids'] for item in batch_data])
        batch_size, seq_len = batch_input_ids.shape
        return {"input_ids": batch_input_ids[:, :-1].T.contiguous(), "target_ids": batch_input_ids[:, 1:].T.contiguous(), "position_index": torch.arange(seq_len-1, dtype=torch.long).unsqueeze(1).expand(-1, batch_size).contiguous(), "attn_mask": torch.tril(torch.ones((seq_len-1, seq_len-1), dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1).contiguous(), "hidden_states": None}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    
    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank, world_size  = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

    SEQ_LEN, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, NUM_SAMPLES, MAX_TOKENS = 10, 6, 2, 1e-4, 20, 1800
        
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    setup_parallel_context(tp_size=args.tp_size, pp_size=args.pp_size, dp_size=args.dp_size)

    if pc.parallel_context.global_rank == 0:
        pc.parallel_context.display_parallelism_grid()

    set_all_seed(seed=42)
    model = PipelineParallel("HuggingFaceTB/SmolLM-360M-Instruct").to(device)
    data_loader = MicroBatchDataLoader(GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, 1, SEQ_LEN, "roneneldan/TinyStories", "HuggingFaceTB/SmolLM-360M-Instruct", num_samples=NUM_SAMPLES)
    tensor_shapes = (SEQ_LEN, data_loader.micro_batch_size, model.config.hidden_size)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    trained_tokens, step = 0, 0
    tokens_per_step = data_loader.num_global_micro_batches * data_loader.micro_batch_size * SEQ_LEN
    while trained_tokens < MAX_TOKENS:
        optimizer.zero_grad()
        loss = pipeline_parallel_1f1b(model, data_loader, tensor_shapes, device) #loss = pipeline_parallel_afab(model, data_loader, tensor_shapes, device)
        optimizer.step()
        trained_tokens += tokens_per_step
        step += 1
        if pc.parallel_context.pp_is_last_stage:
            print(f"Step: {step}, Loss: {loss:.4f}, Tokens: {trained_tokens}/{MAX_TOKENS}")
