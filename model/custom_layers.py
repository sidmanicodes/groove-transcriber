import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqBN(nn.Module):
    """Custom Frequency Batch Normalization class"""
    def __init__(self, num_freq_bins: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_freq_bins))
        self.beta = nn.Parameter(torch.zeros(num_freq_bins))

        # Non-learnable parameters / moving statistics
        self.moving_avgs = torch.zeros(num_freq_bins)
        self.moving_stds = torch.ones(num_freq_bins)

    def forward(self, x) -> torch.Tensor:
        # While training, compute means and standard deviations per batch
        if self.training:
            # Compute the mean and standard deviations for all activations
            # across the frequency band for the current batch
            means = torch.mean(input=x, dim=[0, 1, 3], keepdim=True)
            stds = torch.std(input=x, dim=[0, 1, 3], keepdim=True)

            # Update running stats
            self.moving_avgs = (self.momentum * self.moving_avgs) + ((1 - self.momentum) * means.view(-1))
            self.moving_stds = (self.momentum * self.moving_stds) + ((1 - self.momentum) * stds.view(-1))

        # During inference, use the moving stats
        else:
            means = self.moving_means
            stds = self.moving_stds

        # Make sure that the means and standard deviations have the right shapes for broadcasting
        means = means.view(1, 1, -1, 1)
        stds = stds.view(1, 1, -1, 1)

        # Normalize data 
        x_norm = (x - means) / (stds + self.eps)

        # Apply scaling and shifting (make sure to fix the shapes of gamma and beta)
        x_norm = (self.gamma.view(1, 1, -1, 1) * x_norm) + self.beta.view(1, 1, -1, 1)

        return x_norm

class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        
        # Convolution for the gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels=F_g, out_channels=F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=F_int)
        )

        # Convolution for the skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels=F_l, out_channels=F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=F_int)
        )

        # Activation and convolution for attention
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=F_int, out_channels=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

        # ReLU for non-linearity
        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Apply gating convolution and skip convolution
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # print(f"g dimensions: {g.shape} | x dimensions: {x.shape}")

        # g1 may have smaller dimensions that x1, so we will interpolate g1
        # to match x1's dimensions
        g1 = F.interpolate(input=g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        # Generate attention map
        psi = self.psi(self.relu(g1 + x1))
        # print(f"psi dimensions: {psi.shape}")

        # Apply the attention map to the original
        # feature map from the skip connection
        return x * psi

# if __name__ == "__main__":