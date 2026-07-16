import torch
import torch.nn as nn

class DeepSynBaLoss(nn.Module):
    """Loss module representing Mean Squared Error over dose viability matrices."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes the MSE Loss.

        Args:
            y_pred: Predicted viability of shape (B, M, N).
            y_true: Ground truth viability of shape (B, M, N).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.mse(y_pred, y_true)
