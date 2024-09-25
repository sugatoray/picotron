import parallel_context as pc
from distributed_primtives import communicate, bidirectional_communicate
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

def reduce_loss_across_dp_ranks(loss, device):
    # Reduce the loss across DP workers.
    reduced_loss = torch.tensor([loss if loss is not None else 0.0], dtype=torch.float32, device=device)
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG, group=pc.parallel_context.dp_group)
    return reduced_loss.item()

class PipelineParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        layer_distribution = self.distribute_layers(config.num_hidden_layers)
        self.embed_tokens = model.model.embed_tokens if pc.parallel_context.pp_is_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): model.model.layers[i] for i in layer_distribution})
        self.norm = model.model.norm if pc.parallel_context.pp_is_last_stage else nn.Identity()
        self.lm_head = model.lm_head if pc.parallel_context.pp_is_last_stage else nn.Identity()

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // pc.parallel_context.pp_world_size + (1 if i < num_layers % pc.parallel_context.pp_world_size else 0) for i in range(pc.parallel_context.pp_world_size)]
        start_layer = sum(layers_per_gpu[:pc.parallel_context.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[pc.parallel_context.pp_rank]))

    def forward(self, batch, device):
        x = batch["hidden_states"].to(device) if batch["hidden_states"] is not None else batch["input_ids"].to(device)
        x = self.embed_tokens(x)
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=batch["position_index"].to(device))[0]
        x = self.norm(x)
        return self.lm_head(x)

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
        input_tensor = communicate(operation='recv_forward', shapes=tensor_shapes, dtype=torch.float32)
        batch = next(iter(data_loader))
        batch["hidden_states"] = input_tensor
        output_tensor = model.forward(batch, device)
        communicate(operation='send_forward', tensor=output_tensor)
        
        # Don't need to keep track of the loss on every rank. Just choosing a single rank (TP rank 0 in the last PP stage) is enough
        if pc.parallel_context.pp_is_last_stage and pc.parallel_context.global_rank == pc.parallel_context.tp_first_rank:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item()

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    for _ in range(data_loader.num_local_micro_batches): # All backward passes
        output_tensor_grad = communicate(operation='recv_backward', shapes=tensor_shapes, dtype=torch.float32)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        communicate(operation='send_backward', tensor=input_tensor_grad)

    logging_loss = reduce_loss_across_dp_ranks(logging_loss, device)
    return logging_loss

def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device):
    num_warmup_microbatches = min(pc.parallel_context.pp_world_size - pc.parallel_context.pp_rank - 1, data_loader.num_local_micro_batches)
    num_microbatches_remaining = data_loader.num_local_micro_batches - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    
    def _forward_step(input_tensor):
        batch = next(iter(data_loader))
        batch["hidden_states"] = input_tensor
        output_tensor = model.forward(batch, device)
        # Don't need to keep track of the loss on every rank. Just choosing a single rank (TP rank 0 in the last PP stage) is enough
        if pc.parallel_context.pp_is_last_stage and pc.parallel_context.global_rank == pc.parallel_context.tp_first_rank:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item()
        return output_tensor

    for _ in range(num_warmup_microbatches): # Warmup forward passes
        input_tensor = communicate(operation='recv_forward', shapes=tensor_shapes, dtype=torch.float32)
        output_tensor = _forward_step(input_tensor)
        communicate(operation='send_forward', tensor=output_tensor)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = communicate(operation='recv_forward', shapes=tensor_shapes, dtype=torch.float32)
    
    for i in range(num_microbatches_remaining):  # 1F1B steady state
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, dtype=torch.float32, device=device)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        if i == num_microbatches_remaining - 1: # last iteration
            input_tensor = None
            communicate(operation='send_backward', tensor=input_tensor_grad)
        else:
            input_tensor = bidirectional_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, dtype=torch.float32, device=device)

    for _ in range(num_warmup_microbatches): # Cooldown backward passes
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = communicate(operation='recv_backward', shapes=tensor_shapes, dtype=torch.float32)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        communicate(operation='send_backward', tensor=input_tensor_grad)

    logging_loss = reduce_loss_across_dp_ranks(logging_loss, device)
    return logging_loss