import torch
import pytest
from ..freqbn import FreqBN

@pytest.fixture
def sample_input() -> torch.Tensor:
    """Setup fixture for example data"""
    batch_size = 16
    num_channels = 1
    num_freq_bins = 513
    num_frames = 5168
    return torch.randn((batch_size, num_channels, num_freq_bins, num_frames))

def test_freqbn_initialization(sample_input: torch.Tensor) -> None:
    """Test that all parameters are the correct size"""
    num_freq_bins = sample_input.shape[2]
    freqbn = FreqBN(num_freq_bins)
    assert freqbn.gamma.shape == torch.Size([num_freq_bins])
    assert freqbn.beta.shape == torch.Size([num_freq_bins])
    assert freqbn.moving_avgs.shape == torch.Size([num_freq_bins])
    assert freqbn.moving_stds.shape == torch.Size([num_freq_bins])

def test_freqbn_forward_pass(sample_input: torch.Tensor) -> None:
    """Test that the normalized tensor has the same shape as the unnormalized one"""
    num_freq_bins = sample_input.shape[2]
    freqbn = FreqBN(num_freq_bins)
    output = freqbn(sample_input)
    assert output.shape == sample_input.shape, f"Shape mismatch: output shape is {output.shape} but input shape was {sample_input.shape}"

def test_gamma_and_beta(sample_input: torch.Tensor) -> None:
    """Test that gamma and beta were applied correctly"""
    num_freq_bins = sample_input.shape[2]
    freqbn = FreqBN(num_freq_bins)
    output = freqbn(sample_input)

    assert torch.allclose(output.mean(dim=[0, 1, 3]), freqbn.beta, atol=1e-5), "Beta incorrectly applied"
    assert torch.allclose(output.std(dim=[0, 1, 3]), freqbn.gamma, atol=1e-5), "Gamma incorrectly applied"