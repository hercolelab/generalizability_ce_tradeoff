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
    def random_uniform_points_in_sphere(num_points: int, 
                                shape: Tuple[int, ...], 
                                radius: float = 1.0, 
                                device: str = "cuda") -> Tensor:
        """
        Generate random points inside an n-dimensional sphere of given radius with a normal distribution.

        Parameters:
        - num_points (int): Number of points to generate.
        - shape (Tuple[int, ...]): Shape of each point (e.g., (28, 28) for a 28x28 matrix).
        - radius (float): Radius of the sphere.
        - device (str): Device used (cuda or cpu)

        Returns:
        - Tensor: A tensor of shape (num_points, *shape) containing the generated points.
        """
        total_dim: npArray = np.prod(shape)
        
        # Generate random points from a normal distribution
        points: npArray = np.random.randn(num_points, total_dim)
        
        # Normalize points to lie on the surface of a unit sphere
        norms: npArray = np.linalg.norm(points, axis=1, keepdims=True)
        points: npArray = points / norms
        
        # Generate random radii to scale points within the sphere
        random_radii: npArray = np.random.rand(num_points, 1)**((1/total_dim)) * radius
        
        # Scale points to lie within the sphere of given radius
        points: npArray = points * random_radii
        
        # Reshape the points to the desired shape
        points: npArray = points.reshape(num_points, *shape)
        
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