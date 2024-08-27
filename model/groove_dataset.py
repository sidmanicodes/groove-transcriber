import os
from typing import Tuple
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from new_midi_utils import midi_to_tensor, tensor_to_midi, convert_sec_to_tick

class GrooveDataset(Dataset):
    def __init__(
            self,
            metadata_path: str,
            data_path: str,
            split: str,
            sr: int,
            max_samples: int,
            transform: torchaudio.transforms.Spectrogram,
            device: torch.DeviceObjType,
            data_amt: float = 1,
            ) -> None:
        # Error handling
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"'{split}' is not a valid split; must be 'train', 'validation', or 'test'")
        
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split] # Filter by split
        
        # Drop any rows where there is no MIDI or .wav file
        self.metadata.dropna(subset=['audio_filename', 'midi_filename'])
        
        # Filter out rows with empty strings
        self.metadata = self.metadata[self.metadata['audio_filename'] != ''] 
        self.metadata = self.metadata[self.metadata['midi_filename'] != ''] 

        # Filter out rows with floats in audio_filename and midi_filename (edge case)
        self.metadata = self.metadata[~self.metadata['audio_filename'].apply(lambda x: isinstance(x, float))] 
        self.metadata = self.metadata[~self.metadata['midi_filename'].apply(lambda x: isinstance(x, float))] 

        # Only get a certain percentage of the entire data set (mainly for testing purposes)
        self.metadata = self.metadata[:int(len(self.metadata) * data_amt)]
        
        self.data_path = data_path
        self.sr = sr
        self.max_samples = max_samples
        self.device = device
        self.transform = transform.to(device)

    def __len__(self) -> None:
        return len(self.metadata)

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            # Get the path to the audio sample
            signal_path, midi_path = self._get_signal_and_velocity_path(index)

            # Load in the sample as a torch.Tensor
            signal, signal_sr = torchaudio.load(signal_path)

            # Load in the velocities matrix
            velocities = midi_to_tensor(midi_path=midi_path, max_samples=self.max_samples, sr=self.sr)

            # Load the sample onto the device to accelerate training
            signal = signal.to(self.device)

            # Resample if necessary
            signal = self._resample(signal, signal_sr)

            # Pad sample if necessary
            signal = self._pad(signal)

            # Truncate sample if necessary
            signal = self._truncate(signal)

            # Mix signal down to one channel
            signal = self._mix_down(signal)

            # Apply transform
            signal = self.transform(signal)

            return signal, velocities
        except IndexError:
            print(f"Index {index} is out of bounds")

    def _get_signal_and_velocity_path(self, index: int) -> Tuple[str, str]:
        # Metadata file has path with a file separator "/" so we have to
        # split it apart and reconnect it for cross-os functionality
        # print(f"File with id {self.metadata.iloc[index]['id']} is of type {type(self.metadata.iloc[index]['audio_filename'])}")
        signal_path = self.metadata.iloc[index]['audio_filename'].split(sep="/")
        midi_path = self.metadata.iloc[index]['midi_filename'].split(sep="/")

        full_signal_path = os.path.join(self.data_path, *signal_path)
        full_midi_path = os.path.join(self.data_path, *midi_path)

        return full_signal_path, full_midi_path
    
    def _resample(self, signal: torch.Tensor, signal_sr: int) -> torch.Tensor:
        # If the audio was sampled at a different rate, resample it
        # at the standard rate and return it
        if signal_sr != self.sr:
            resampler =  torchaudio.transforms.Resample(orig_freq=signal_sr, new_freq=self.sr).to(self.device)
            return resampler(signal)
        return signal
    
    def _pad(self, signal: torch.Tensor) -> torch.Tensor:
        # If the signal has fewer samples than the max number of samples, right-pad it
        # Otherwise, return the original sample
        if signal.shape[1] < self.max_samples:
            pad_amt = self.max_samples - signal.shape[1]
            return torch.nn.functional.pad(input=signal, pad=(0, pad_amt))
        return signal
    
    def _truncate(self, signal: torch.Tensor) -> torch.Tensor:
        # If the signal has more samples than the max number of samples, truncate it
        # Otherwise, return the original signal
        if signal.shape[1] > self.max_samples:
            return signal[:, :self.max_samples]
        return signal
    
    def _mix_down(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[0] > 1:
            return torch.mean(input=signal, dim=0, keepdim=True)
        return signal
        
if __name__ == "__main__":
    # Constants
    METADATA_PATH = "../data/info.csv"
    DATA_PATH = "../data"
    STANDARD_SR = 22_500
    MAX_SAMPLES = STANDARD_SR * 60 # Max sequence length will be 1 minute worth of samples
    FRAME_SIZE = 1_024
    HOP_SIZE = 512
    
    device = 'cpu'

    if torch.backends.mps.is_available():
        device = 'mps'

    print(f"{device} selected\nFetching data...")

    transform = torchaudio.transforms.MelSpectrogram(n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=62)

    # Create GrooveDataset object
    groove_data = GrooveDataset(
        metadata_path=METADATA_PATH,
        data_path=DATA_PATH,
        split='train',
        sr=STANDARD_SR,
        transform=transform,
        max_samples=MAX_SAMPLES,
        data_amt=0.01,
        device=device
    )

    # Inspect dataset length
    print(f"Dataset has {len(groove_data)} rows")

    audio_tensor, midi_tensor = groove_data[6]
    print(f"Audio tensor has shape {audio_tensor.shape}")
    print(f"Midi tensor has shape {midi_tensor.shape}")