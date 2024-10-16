import src.distributed.process_group_manager as pgm
from src.distributed.distributed_primtives import pipeline_communicate, bidirectional_pipeline_communicate, all_reduce_loss_across_dp_ranks
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

from parallel.base_parallel import BaseParallel

class PipelineParallel(BaseParallel):
    def __init__(self, model, config):
        super().__init__(model, config)
        #TODO(fmom): find a better model to distributed layers without instantiating a base_model first
        layer_distribution = self.distribute_layers(config.num_hidden_layers)
        self.embedding = model.embedding if pgm.process_group_manager.pp_is_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): model.decoder_layers[i] for i in layer_distribution})
        self.final_norm = model.final_norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        self.final_proj = model.final_proj if pgm.process_group_manager.pp_is_last_stage else nn.Identity()

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // pgm.process_group_manager.pp_world_size + (1 if i < num_layers % pgm.process_group_manager.pp_world_size else 0) for i in range(pgm.process_group_manager.pp_world_size)]
        start_layer = sum(layers_per_gpu[:pgm.process_group_manager.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[pgm.process_group_manager.pp_rank]))

    def forward(self, batch, device):
        x = batch["hidden_states"].to(device) if batch["hidden_states"] is not None else batch["input_ids"].to(device)
        x = self.embedding(x)
        for layer in self.decoder_layers.values():
            x = layer(x)
        x = self.final_norm(x)
        return self.final_proj(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

def train_step_pipeline_afab(model, data_loader, tensor_shapes, device):
    logging_loss: torch.float32 = 0.0
    input_tensors, output_tensors = [], []
    
    for _ in range(data_loader.num_local_micro_batches): # All forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=torch.float32)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor
        output_tensor = model.forward(batch, device)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=torch.float32)
        
        # Don't need to keep track of the loss on every rank. Just choosing a single rank (TP rank 0 in the last PP stage) is enough
        if pgm.process_group_manager.pp_is_last_stage and pgm.process_group_manager.global_rank == pgm.process_group_manager.tp_first_rank:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item()

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    for _ in range(data_loader.num_local_micro_batches): # All backward passes
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=torch.float32)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=torch.float32)

    logging_loss = all_reduce_loss_across_dp_ranks(logging_loss, device)
    return logging_loss

def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device):
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.num_local_micro_batches)
    num_microbatches_remaining = data_loader.num_local_micro_batches - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    
    def _forward_step(input_tensor):
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor
        output_tensor = model.forward(batch, device)
        # Don't need to keep track of the loss on every rank. Just choosing a single rank (TP rank 0 in the last PP stage) is enough
        if pgm.process_group_manager.pp_is_last_stage and pgm.process_group_manager.global_rank == pgm.process_group_manager.tp_first_rank:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item()
        return output_tensor

    for _ in range(num_warmup_microbatches): # Warmup forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=torch.float32)
        output_tensor = _forward_step(input_tensor)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=torch.float32)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=torch.float32)
    
    for i in range(num_microbatches_remaining):  # 1F1B steady state
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=torch.float32)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        if i == num_microbatches_remaining - 1: # last iteration
            input_tensor = None
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=torch.float32)
        else:
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=torch.float32)

    for _ in range(num_warmup_microbatches): # Cooldown backward passes
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=torch.float32)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=torch.float32)

    logging_loss = all_reduce_loss_across_dp_ranks(logging_loss, device)
    return logging_loss