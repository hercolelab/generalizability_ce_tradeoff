
from src.geometric import Sphere
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.decomposition import PCA
from typing import Tuple, TypeAlias
Tensor: TypeAlias = torch.Tensor


def get_estimator(n_samples = 1000, radius = 100, distribution = "uniform", function = None, train_set = None):
    return MontecarloEstimator(n_samples=n_samples, radius=radius, distribution=distribution, function= function, train_set= train_set)

class MontecarloEstimator():

    def __init__(self, 
                 function: torch.nn.Module, 
                 train_set: TensorDataset, 
                 n_samples: int = 1000, 
                 radius: float = 1.0,
                 fraction: float = 0.8,
                 distribution: str = "uniform",
                 **kwargs) -> None:
        """
        Parameters:
        - function (torch.nn.Module): The neural network model or function to be used.
        - train_set (TensorDataset): The training dataset to be used for training or other operations.
        - shape (Tuple[int, ...]): The shape of the perturbation tensor, if just a dimension use (n, ) otherwise (n, m, ...).
        - n_samples (int, optional): The number of samples to generate. Default is 1000.
        - radius (float, optional): The radius within which to generate the perturbations. Default is 1.0.
        - fraction (float, optional): Fraction used to estimate the counterfactual fraction over the training set. Default is 0.8.
        - **kwargs: Additional keyword arguments that might be required for other operations or configurations.

        Returns:
        - None
        """
        self.sphere: Sphere = Sphere()
        self.radius = radius
        self.function = function.train()
        self.n_samples = n_samples
        self.random_index = torch.randint(low=0, high=len(train_set), size=(int(len(train_set)*fraction),))
        self.X, _ = train_set[self.random_index]  
        self.shape = self.X[0].shape
        self.volume = self.sphere.hypersphere_volume(dimensions=np.prod(self.shape), radius=radius)
        self.include_volume = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.distribution: str = distribution
        self.random_function = self.sphere.random_normal_points_in_sphere if distribution == "normal" else self.sphere.random_uniform_points_in_sphere
        self.perturbation = self.random_function(num_points=self.n_samples, shape=self.shape, radius=self.radius)

    def get_counterfactual(self, 
                       data: Tensor, 
                       target: Tensor,
                       grad: bool) -> Tuple[Tensor, Tensor]:
        
        """
        Generate counterfactual samples by perturbing the input tensor `data` and computing the model's output.
        
        The perturbation tensor must have the dimensions (P, S, F), where:
        - P is the number of perturbations.
        - S is the number of samples.
        - F is the number of features.
        
        The input tensor `data` must have dimensions (S, F), where:
        - S is the number of samples.
        - F is the number of features.
        
        Parameters:
        - data (Tensor): The input tensor of shape (batch_size, num_features).
        - target (Tensor): The target tensor, which will be used to generate the counterfactual targets.
        
        Returns:
        - Tuple[Tensor, Tensor]: A tuple containing:
        - out (Tensor): The output tensor from the model after perturbation, reshaped as required.
        - target (Tensor): The repeated and reshaped target tensor to match the perturbation structure.
        """
        torch.set_grad_enabled(mode=grad)
        batch_size: int = data.shape[0]
        unit_dims: Tuple[int, ...] = (1, ) 
        new_shape: Tuple[int, ...] = (self.n_samples, *unit_dims, *self.shape)
        perturbation: Tensor = self.perturbation.view(new_shape)
        #print("perturbation.shape: ", perturbation.shape)
        repeat_dims: Tuple[int, ...] = (1, batch_size, *((1, )*len(new_shape[2:])))
        perturbation: Tensor = perturbation.repeat(repeat_dims)      
        #print("perturbation.shape: ", perturbation.shape) 
        data: Tensor = data.to(device=self.device) 
        sample_perturbed: Tensor = data + perturbation 
        #print("sample_perturbed.shape: ", sample_perturbed.shape)
        batch_dims: Tuple[int, ...] = (-1, *new_shape[2:])
        sample_perturbed: Tensor = sample_perturbed.reshape(batch_dims)
        #print("sample_perturbed.shape: ", sample_perturbed.shape)
        out: Tensor = self.function(sample_perturbed)
        #print("out.shape: ", out.shape)
        out = out.view(self.n_samples, batch_size).transpose(0, 1)
        #print("out.shape: ", out.shape)
        #print("target.shape: ", target.shape, len(target.shape))
        if len(target.shape) == 2:
            target: Tensor = torch.argmax(target, dim=-1)
        else: 
            target: Tensor = (target > 0).int()
        #print("target.shape: ", target.shape)
        target: Tensor = target.unsqueeze(1)
        #print("target.shape: ", target.shape)
        target: Tensor = target.repeat(1, self.n_samples)
        #print("target.shape: ", target.shape)
        #target: Tensor = target.reshape(batch_size * self.n_samples)
        #print("target.shape: ", target.shape)
        target: Tensor = target.to(self.device)
        #print("target.shape: ", target.shape)
        return out, target #[batch_size, n_sample]
    
    def get_estimate(self, data: Tensor, output: Tensor) -> Tensor:
        out, target_cf = self.get_counterfactual(data, output, grad=False)
        return self._get_estimate(out=out, target=target_cf)

    def _get_estimate(self, out: Tensor, target: Tensor) -> Tensor:
        """
        Returns:
            torch.Tensor
        """
        if len(out.shape) == 3:
            predicted_class_cf = torch.argmax(out, dim=1)
        elif len(out.shape) == 2:
            predicted_class_cf = (out > 0).int()

        cf_fraction = (target != predicted_class_cf).sum(dim=1) / predicted_class_cf.shape[1]
 
        return torch.tensor(cf_fraction, dtype=torch.float32)
    
    def get_estimate_name(self) -> str:
        """
        Implementation of get_estimate_name abstract method.
        """
        return "p_x"
    
    def build_log(self, values, stage):
        import numpy as np

        # Calculate the required statistics
        max_value = max(values)
        mean_value = np.mean(values)
        first_quartile = np.percentile(values, 25)
        third_quartile = np.percentile(values, 75)
        median_value = np.median(values)
        min_value = min(values)

        # Construct the dictionary with keys based on `stage` and metrics
        log_data = {
            f"{stage}/max p_x$": max_value,
            f"{stage}/mean p_x$": mean_value,
            f"{stage}/first_quartile p_x$": first_quartile,
            f"{stage}/third_quartile p_x$": third_quartile,
            f"{stage}/median p_x$": median_value,
            f"{stage}/min p_x$": min_value,
        }

        return log_data
      