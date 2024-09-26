import torch.distributed as dist
import torch.nn as nn
import process_group_manager as pgm


class ContextParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cp_world_size = pgm.process_group_manager.cp_world_size
        self.cp_rank = pgm.process_group_manager.cp_rank
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.model.backward(input_tensor, output_tensor, output_tensor_grad)