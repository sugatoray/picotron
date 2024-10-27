from functools import partial
from src.parallel.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, initialize_weight_tensor
import torch.nn.init as init
import torch.nn as nn

class TensorParallel():
    def __init__(self, model, init_method = initialize_weight_tensor):
        super().__init__()

        module_linear_name_stype_mapping_list = [
            ("attention", "q_proj", "column"),
            ("attention", "k_proj", "column"),
            ("attention", "v_proj", "column"),
            ("attention", "out_proj", "row"),
            ("mlp", "up_proj", "column"),
            ("mlp", "gate_proj", "column"),
            ("mlp", "down_proj", "row"),
        ]
        
        self.init_method = init_method

        for layer in model.decoder_layers:
            for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
                self.replace_module(getattr(layer, module_name), linear_proj_name, style)
        self.replace_module(model, "embedding", "vocab")   
        self.replace_module(model, "final_proj", "column", args={"gather_output": True})    

    def replace_module(self,module, linear_proj_name, style, args = {}):
        assert style in ["column", "row", 'vocab']
        linear_layer = getattr(module, linear_proj_name)
        if style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                init_method=self.init_method,
                gather_output=args.get("gather_output", False)
            )
        elif style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                init_method=self.init_method
            )
        else:
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
                init_method=partial(self.init_method, vocab_embedding=True)
            )
        setattr(module, linear_proj_name, new_linear_layer)