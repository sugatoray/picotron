import torch.distributed as dist
import torch
import picotron.process_group_manager as pgm

def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension into num_partitions chunks."""
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)

class Reduce(torch.autograd.Function):
    """All-reduce in forward pass, identity in backward pass."""
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Gather(torch.autograd.Function):
    """Gather in forward pass, split in backward pass."""
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        last_dim = input.dim() - 1
        # Need contiguous tensors for collectives -> https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        input = input.contiguous()
        tensor_list = [torch.empty_like(input) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = input
        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

class Copy(torch.autograd.Function):
    """Identity in forward pass, all-reduce in backward pass."""
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
          return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output