"""
Inspired by Fair Scale/Megatron's Tensor Parallelism implementation
Ref: https://github.com/facebookresearch/fairscale/tree/main/fairscale
"""
from picotron.tensor_parallel.tp_utils import merge_first_two_dims, split_tensor_along_last_dim
import torch.distributed as dist
import torch
import picotron.process_group_manager as pgm
import torch.nn.functional as F

def _reduce(input_):
    """All-reduce the input tensor across model parallel(Tensor Parallel) group."""    
    # Bypass the function if we are using only 1 GPU.
    if pgm.process_group_manager.tp_size == 1:
        return input_

    # All-reduce across the tensor parallel group
    dist.all_reduce(input_, group=pgm.process_group_manager.tp_group)

    return input_

class _CopyToModelParallelRegion(torch.autograd.Function):
    """copy(identity) in forward pass, all reduce in backward pass"""
    @staticmethod
    def forward(ctx, input_):
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """all reduce in forward pass, copy(identity) in backward pass"""
    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output

# This is the `f` function in the paper: https://arxiv.org/abs/1909.08053
def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)

# This is the `g` function in the paper, which is the conjugate of `f`
def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)

def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the corresponding slice."""
    tp_rank = pgm.process_group_manager.tp_rank
    tp_world_size = pgm.process_group_manager.tp_size

    # Bypass the function if we are using only 1 GPU
    if tp_world_size == 1:
        return input_

    # Split along last dimension and keep the corresponding slice
    input_list = split_tensor_along_last_dim(input_, tp_world_size)
    output = input_list[tp_rank].contiguous()

    return output

def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    tp_rank = pgm.process_group_manager.tp_rank
    tp_world_size = pgm.process_group_manager.tp_size

    # Bypass the function if we are using only 1 GPU.
    if tp_world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1

    tensor_list = [torch.empty_like(input_) for _ in range(tp_world_size)]
    tensor_list[tp_rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=pgm.process_group_manager.tp_group)

    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gathher in the forward pass, split in the backward pass."""

    @staticmethod
    def forward(ctx, input_): 
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  
        return _split(grad_output)

def gather_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)

def linear_with_all_reduce(input_, weight, bias):
    input_parallel = copy_to_model_parallel_region(input_)
    output = F.linear(input_parallel, weight, bias) # XW_i^T + b, output is Y_i
    return output

class _LinearWithAsyncAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        output = input_ @ weight.t() + bias if bias is not None else input_ @ weight.t()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        The key difference with "linear_with_all_reduce" is that the all reduce of input_ gradeint is before 
        the calculation of the gradient of weights and bias, instead of after. So we can overlap the computation and communication
        This is only applicable to Column Parallel Linear

        Before: grad_output -> grad_input, grad_weight, grad_bias  -> grad_input all reduce
        Now:    grad_output -> grad_input -> grad_input all reduce -> grad_weight, grad_bias
        """
        input_, weight = ctx.saved_tensors
        grad_input = grad_output @ weight # (b, s, out_size) @ (out_size, input_size) = (b, s, input_size)
        # all-reduce input gradient. 
        input_gradient_all_reduce_handle = dist.all_reduce(grad_input, group=pgm.process_group_manager.tp_group, async_op=True)
        # merge first two dims to allow matrix multiplication
        grad_output, input_ = merge_first_two_dims(grad_output, input_)     # grad_output, input_: (b, s, out_size), (b, s, input_size) -> (b*s, out_size), (b*s, input_size)
        grad_weight = grad_output.t() @ input_                              # (out_size, b*s) @ (b*s, input_size) -> (out_size, input_size)
        grad_bias = grad_output.sum(0) if ctx.use_bias else None
        input_gradient_all_reduce_handle.wait()
        return grad_input, grad_weight, grad_bias

def linear_with_async_all_reduce(input_, weight, bias):
    return _LinearWithAsyncAllReduce.apply(input_, weight, bias)
