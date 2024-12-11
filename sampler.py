import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import List

class LaplaceSampler:
    def __init__(self, scale: float = 0.1):
        self.scale = scale
    
    def sample(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, 
               num_samples: int = 10) -> List[torch.Tensor]:
        """Generate samples using Laplace approximation."""
        # Create diagonal covariance matrix (simplified Laplace)
        cov = torch.eye(len(S)) * self.scale
        
        # Create distribution
        dist = MultivariateNormal(
            loc=torch.zeros(len(S)),
            covariance_matrix=cov
        )
        
        samples = []
        for _ in range(num_samples):
            # Sample perturbations for singular values
            perturbed_S = S + dist.sample()
            
            # Reconstruct full matrix
            sample = U @ torch.diag(perturbed_S) @ V
            samples.append(sample)
        
        return samples
    
    def save_samples(self, samples: List[torch.Tensor], path: str) -> None:
        """Save samples to disk."""
        torch.save(samples, path)
    
    @staticmethod
    def load_samples(path: str) -> List[torch.Tensor]:
        """Load samples from disk."""
        return torch.load(path)