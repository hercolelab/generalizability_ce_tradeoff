import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, TypeAlias
npArray: TypeAlias = np.array
Tensor: TypeAlias = torch.Tensor

class Sphere:
    
    
    
    @staticmethod
    def hypersphere_volume(dimensions: int, radius: float = 1.0) -> float:
        from scipy.special import gamma
        
        pi_pow = np.pi**(dimensions/2)
        euler_argument = (dimensions/2) + 1
        euler_value = gamma(euler_argument)
        result = (pi_pow/euler_value)*(radius**dimensions)

        return result

    @staticmethod
    def random_normal_points_in_sphere(num_points: int, 
                                  shape: Tuple[int, ...], 
                                  radius: float = 1.0, 
                                  device: str = "cuda") -> Tensor:
        """
        Generate random points inside an n-dimensional sphere of given radius with uniform distribution.

        Parameters:
        - num_points (int): Number of points to generate.
        - shape (Tuple[int, ...]): Shape of each point (e.g., (28, 28) for a 28x28 matrix).
        - radius (float): Radius of the sphere.
        - device (str): Device used (cuda or cpu)

        Returns:
        - Tensor: A tensor of shape (num_points, *shape) containing the generated points.
        """
        total_dim: npArray = np.prod(shape)
        points: npArray = 2 * np.random.rand(num_points, total_dim) - 1
        norms: npArray = np.linalg.norm(points, axis=1, keepdims=True)
        points: npArray = points / norms
        scales: npArray = 2 * np.random.rand(num_points, 1) - 1
        points: npArray = points * scales
        points *= radius
        points: npArray = points.reshape(num_points, *shape)
        
        return torch.tensor(points, device=device, dtype=torch.float32)

    @staticmethod
    def random_uniform_points_in_shell(
        num_points: int,
        shape: Tuple[int, ...],
        r1: np.ndarray,
        r2: float,
        device: str = "cuda"
    ) -> Tensor:
        """
        Generate random points uniformly in multiple spherical shells without overflow,
        fully vectorized over S shells.
        Returns tensor of shape (S, num_points, *shape).
        If r1[i] >= r2, that shell’s points are all zero.
        """
        # — prepare parameters —
        r1_arr = np.asarray(r1, dtype=float).ravel()               # (S,)
        S = r1_arr.size
        d = int(np.prod(shape))                                   # dimension

        # — 1) sample unit directions —
        points = np.random.randn(S, num_points, d)
        norms = np.linalg.norm(points, axis=2, keepdims=True)
        points /= norms                                           # now on unit spheres

        # — 2) sample radii in log‐space, vectorized —
        U = np.random.rand(S, num_points, 1)                      # (S, N, 1)

        # expand r1 over (S, N, 1)
        r1_b = r1_arr[:, None, None]                              # (S,1,1)
        log_r1 = np.log(r1_b, where=(r1_b>0), out=np.full_like(r1_b, -np.inf))
        log_r2 = np.log(r2)

        # masks
        invalid_mask = (r1_b >= r2)                               # (S,1,1)
        zero_inner  = (r1_b == 0) & ~invalid_mask                 # (S,1,1)
        normal_mask = (~invalid_mask) & (~zero_inner)             # (S,1,1)

        # placeholder for radii
        radii = np.empty((S, num_points, 1), dtype=float)

        # Case A: invalid shells → radii = 0
        radii[invalid_mask[:,0,0], ...] = 0.0

        # Case B: zero inner radius → r2 * U^(1/d)
        radii[zero_inner[:,0,0], ...] = r2 * (U[zero_inner[:,0,0]] ** (1.0 / d))

        # Case C: normal shells, use log‐domain mix
        if np.any(normal_mask):
            # compute δ = d*(ln r2 − ln r1) for each shell
            delta = d * (log_r2 - log_r1)                        # (S,1,1)
            # clamp delta to avoid overflow of exp; e.g. exp(700) ~1e304
            delta_clamped = np.minimum(delta, 700)

            # exp(delta) − 1 safely
            expm1_delta = np.expm1(delta_clamped)                # (S,1,1)

            # for those with true delta > 700, expm1_delta ~ huge → we treat like zero‐inner
            big_delta = (delta > 700)

            # compute log‐term: ln[1 + U * (e^δ − 1)]
            log_term = np.log1p(U * expm1_delta)                 # (S,N,1)

            # ℓ = d·ln r1 + log_term
            ell = d * log_r1 + log_term                          # broadcast to (S,N,1)

            # radii = exp(ℓ / d)
            radii_norm = np.exp(ell / d)                         # (S,N,1)

            # assign:
            #  - for shells where δ>700, fall back to r2 * U^(1/d)
            radii[np.logical_and(normal_mask[:,0,0], big_delta[:,0,0]), ...] = (
                r2 * (U[np.logical_and(normal_mask, big_delta)] ** (1.0 / d))
            )
            #  - for the rest
            mask_rest = np.logical_and(normal_mask[:,0,0], ~big_delta[:,0,0])
            radii[mask_rest, ...] = radii_norm[mask_rest, ...]

        # — 3) scale and reshape —
        points *= radii                                          # broadcast over last dim
        points = points.reshape((S, num_points) + shape)

        # — 4) to torch —
        return torch.tensor(points, device=device, dtype=torch.float32)
    
    @staticmethod
    def random_uniform_points_in_shell(
        num_points: int,
        shape: Tuple[int, ...],
        radius: float,
        radius2: np.ndarray,
        device: str = "cuda"
    ) -> Tensor:
        """
        Generate random points uniformly in the spherical shell with inner radius "radius2"
        and outer radius "radius" in d dimensions.

        Uses: if U ~ Uniform(0,1), then
        r = (radius2^d + (radius^d - radius2^d) * U)^(1/d)
        to get the correct radial distribution.
        """
        total_dim = int(np.prod(shape))
        # 1. Sample Gaussian vectors, project to unit-sphere
        points = np.random.randn(num_points, total_dim)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # 2. Sample radii in [r1, r2] with correct density
        U = np.random.rand(num_points, 1)
        radial_term = (radius2**total_dim + (radius**total_dim - radius2**total_dim) * U)
        radii = radial_term ** (1.0 / total_dim)
        
        # 3. Scale unit vectors by sampled radii
        points = points * radii
        
        # 4. Reshape and convert to Torch
        points = points.reshape(num_points, *shape)
        return torch.tensor(points, device=device, dtype=torch.float32)

if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    sphere = Sphere()
    from scipy.stats import gaussian_kde

    
    # Generate normal distributed points
    points_normal = sphere.random_normal_points_in_sphere(num_points=5000, shape=(2,), radius=0.1).cpu()

    # Generate uniform distributed points
    points_uniform = sphere.random_uniform_points_in_sphere(num_points=5000, shape=(2,), radius=0.1).cpu()

    # Compute KDE for normal points
    kde_normal = gaussian_kde(points_normal.T)

    # Compute KDE for uniform points
    kde_uniform = gaussian_kde(points_uniform.T)

    # Create a grid of points where the density will be evaluated
    x_min, x_max = min(points_normal[:, 0].min(), points_uniform[:, 0].min()) - 1, max(points_normal[:, 0].max(), points_uniform[:, 0].max()) + 1
    y_min, y_max = min(points_normal[:, 1].min(), points_uniform[:, 1].min()) - 1, max(points_normal[:, 1].max(), points_uniform[:, 1].max()) + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))

    # Evaluate the density on the grid
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density_normal = kde_normal(grid_coords).reshape(x_grid.shape)
    density_uniform = kde_uniform(grid_coords).reshape(x_grid.shape)

    # Plot the points and the density for normal distribution
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(points_normal[:, 0], points_normal[:, 1], s=5, label='Normal Points', alpha=0.5)
    plt.contourf(x_grid, y_grid, density_normal, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Density')
    plt.title('Density of Normal Distributed Points in 2D Sphere')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()

    # Plot the points and the density for uniform distribution
    plt.subplot(1, 2, 2)
    plt.scatter(points_uniform[:, 0], points_uniform[:, 1], s=5, label='Uniform Points', alpha=0.5)
    plt.contourf(x_grid, y_grid, density_uniform, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Density')
    plt.title('Density of Uniform Distributed Points in 2D Sphere')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()

    plt.show()