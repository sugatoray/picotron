import torch.distributed as dist
import torch.nn as nn
import src.distributed.process_group_manager as pgm

class DataParallel(nn.Module):
    def __init__(self, model, config):
        #TODO: Add Zero1w
        #TODO: Interleave all_reduce
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.model.backward(input_tensor, output_tensor, output_tensor_grad)