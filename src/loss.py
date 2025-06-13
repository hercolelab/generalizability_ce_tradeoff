import torch
from torch.nn import Module

def get_loss():
    return CrossEntropy()

class CrossEntropy(Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_loss = torch.nn.functional.binary_cross_entropy_with_logits


    def forward(self, **kwargs):
        """ input : model's predictions
            target: true classes
        """
        input : torch.Tensor = kwargs['input']
        target : torch.Tensor = kwargs['target']
        return self.train_loss(input, target)