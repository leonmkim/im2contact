import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
# adapted from https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb#scrollTo=lAqzcW9XREvu
    def __init__(self, input_dim: int, hidden_layer_dim_list: list, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer_dim_list = hidden_layer_dim_list
        self.output_dim = output_dim

        self.hidden_layer_list = nn.ModuleList()
        curr_input_dim = input_dim
        for layer_dim in hidden_layer_dim_list:
            curr_output_dim = layer_dim 
            self.hidden_layer_list.append(nn.Linear(curr_input_dim, curr_output_dim))
            curr_input_dim = curr_output_dim
        self.last_layer = nn.Linear(curr_input_dim, output_dim)
    
    def get_init_params_dict(self):
        return {
            'input_dim': self.input_dim,
            'hidden_layer_dim_list': self.hidden_layer_dim_list,
            'output_dim': self.output_dim
        }
    def forward(self, x):
        curr_input = x
        for layer in self.hidden_layer_list:
            curr_output = F.silu(layer(curr_input)) 
            curr_input = curr_output
        return self.last_layer(curr_input)