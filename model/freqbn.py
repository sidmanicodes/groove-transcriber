import torch
import torch.nn as nn

class FreqBN(nn.Module):
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

# if __name__ == "__main__":
#     freqbn = FreqBN(num_freq_bins=513)
#     batch_ex = torch.randn((1, 1, 513, 5168))
#     normalized: torch.Tensor = freqbn(batch_ex)
#     print(normalized.shape == batch_ex)

