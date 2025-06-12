import torch


def get_model(**kwargs) -> torch.nn.Module:
    
    
    model_type: str = kwargs["model_type"]
    input_dim = kwargs['input_dim']

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(model_type)

    if model_type == "BMLP":
        dropout = kwargs['dropout']
        hidden_layers = kwargs['hidden_layers']
        print("\nParams:")
        print(input_dim)
        print(dropout)
        print(hidden_layers)
        model = BMLP(input_dim = input_dim, dropout = dropout, hidden_layers = hidden_layers)

        return model.to(device)

    elif model_type =="LogisticRegression":
        from src.models import BLogisticRegression
        model = BLogisticRegression(input_dim = input_dim)

        return model.to(device)
    
    else:
        
        raise ValueError(f"{model_type} is not a valide model type!")

    
from torch import nn
import torch.nn.functional as F

class BMLP(nn.Module):
    def __init__(self, **kwargs):
        super(BMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_dropout = kwargs["dropout"] > 0.0
        
        # Create the first layer from the input dimension to the first hidden layer size
        current_dim = kwargs["input_dim"]
        for hidden_dim in kwargs["hidden_layers"]:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_dropout:
                self.layers.append(nn.Dropout(kwargs["dropout"]))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, 1))

    def forward(self, x: torch.Tensor):
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = F.relu(x)

        # Output layer
        x = self.layers[-1](x).squeeze(1)
        
        # Apply softmax if required
        if self.apply_softmax:
            x = F.softmax(x, dim=-1)
        
        return x
    
class BLogisticRegression(nn.Module): 
    def __init__(self, **kwargs):
        super(BLogisticRegression, self).__init__()
        self.linear = nn.Linear(kwargs["input_dim"], 1) # binary classification

    def forward(self, x):
        return self.linear(x).squeeze(1)
    