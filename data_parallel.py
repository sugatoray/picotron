import torch.distributed as dist
import torch.nn as nn
import parallel_context as pc

class DataParallel(nn.Module):
    def __init__(self, model, config):
        #TODO: Add Zero1
        #TODO: Interleave all_reduce
        super().__init__()
        self.model = model
        self.dp_world_size = pc.parallel_context.dp_world_size
        self.dp_rank = pc.parallel_context.dp_rank

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.model.backward(input_tensor, output_tensor, output_tensor_grad)

    def all_reduce_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=pc.parallel_context.dp_group)
                