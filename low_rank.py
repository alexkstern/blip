import torch
from torch.linalg import svd
from typing import Tuple, Dict

class LowRankApproximation:
    def __init__(self, rank: int):
        self.rank = rank
    
    def decompose(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute low-rank approximation using SVD."""
        U, S, Vh = svd(W)
        
        # Keep only top-k singular values/vectors
        U_k = U[:, :self.rank]
        S_k = S[:self.rank]
        V_k = Vh[:self.rank, :]
        
        return U_k, S_k, V_k
    
    def reconstruct(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Reconstruct full matrix from low-rank factors."""
        return U @ torch.diag(S) @ V
    
    def save(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, 
            original_shape: Tuple[int, ...], path: str) -> None:
        """Save low-rank approximation to disk."""
        torch.save({
            'U': U,
            'S': S,
            'V': V,
            'rank': self.rank,
            'original_shape': original_shape
        }, path)
    
    @classmethod
    def load(cls, path: str) -> Dict[str, torch.Tensor]:
        """Load low-rank approximation from disk."""
        return torch.load(path)
