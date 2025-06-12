
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
        if distribution == "normal" : 
            self.random_function = self.sphere.random_normal_points_in_sphere 
        elif distribution == "shell":
            self.random_function = self.sphere.random_uniform_points_in_shell 
        else: 
            self.sphere.random_uniform_points_in_sphere
        

    def get_counterfactual(
            self,
            data: Tensor,
            target: Tensor,
            grad: bool
        ) -> Tuple[Tensor, Tensor]:
        """
        Generate counterfactual samples by adding vectorized perturbations.

        data:   (batch_size,     F…)  
        target: (batch_size, …)  
        r1 (margin) is now (batch_size,)  
        self.perturbation: (batch_size, n_samples, *shape)
        """
        torch.set_grad_enabled(mode=grad)

        # 1) Compute margin per sample
        w = self.model.linear.weight.cpu()
        f_x = self.model.forward(data).cpu()
        #print("f_x.shape:", f_x.shape)
        #print("w.shape:", w.shape)
        margin = np.abs(f_x/np.linalg.norm(w)) #(batch_size, )

        # 2) Vectorized perturbation: now returns (batch_size, n_samples, *shape)
        self.perturbation = self.random_function(
            num_points=self.n_samples,
            shape=self.shape,
            r1=margin.numpy(),    # or margin.cpu().numpy()
            r2=self.radius
        )

        # 3) Move data & perturbation to device
        data = data.to(self.device)                      # (B, F…)
        pert = self.perturbation.to(self.device)         # (B, N, F…)

        # 4) Broadcast-add: for each sample b, add its N perturbations
        #    → sample_perturbed: (B, N, F…)
        sample_perturbed = data.unsqueeze(1) + pert

        # 5) Flatten into a big batch for forward()
        B, N = sample_perturbed.shape[:2]
        flat = sample_perturbed.reshape(B * N, *self.shape)

        # 6) Model outputs on all perturbed samples
        out_flat = self.function(flat)                   # (B*N, …)
        # reshape back to (B, N) if scalar outputs per sample
        out = out_flat.view(B, N).to(self.device)

        # 7) Prepare target: repeat each target N times
        if target.dim() == 2:
            # e.g. one-hot: take argmax
            tgt = torch.argmax(target, dim=1)
        else:
            # binary or scalar
            tgt = (target > 0).long().squeeze(-1)
        # (B,) → (B, N)
        target_expanded = tgt.unsqueeze(1).repeat(1, N).to(self.device)

        return out, target_expanded

    
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
      