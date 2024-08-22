# Imports
import torch
import torch.nn as nn
import torchaudio

class JojoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()