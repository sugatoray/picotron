import torch.nn as nn

class BaseParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward(self, *args, **kwargs):
        return self.model.backward(*args, **kwargs)