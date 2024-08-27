from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
import torchaudio
from jojonet import JojoNet
from groove_dataset import GrooveDataset
from tqdm import tqdm
import json

# Constants
METADATA_PATH = "../data/info.csv"
DATA_PATH = "../data"
STANDARD_SR = 15_000
MAX_SAMPLES = STANDARD_SR * 60 # Max sequence length will be 1 minute worth of samples
FRAME_SIZE = 1_024
HOP_SIZE = 512
NUM_MEL_BANDS = 62
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

history = []

def train_one_epoch(
        model: JojoNet,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.modules.loss.MSELoss,
        optimizer: torch.optim.Optimizer,
        device: torch.DeviceObjType
        ) -> None:
    def adjust_output_length(outputs: torch.Tensor, expected_size: int) -> torch.Tensor:
        # Truncate if necessary
        if outputs.shape[2] > expected_size:
            return outputs[:, :, expected_size]

        # Pad if necessary
        elif outputs.shape[2] < expected_size:
            pad_amt = expected_size - outputs.shape[2]
            return F.pad(outputs, pad=(0, pad_amt), mode="constant", value=0)
        
        # Return the original output if no adjustment is necessary
        return outputs
            
    # Set the model to train mode
    model.train()

    for i, (signals, midi_tensors) in enumerate(data_loader):
        # Load signals and midi_tensors onto device
        signals: torch.Tensor =  signals.to(device)
        midi_tensors: torch.Tensor = midi_tensors.to(device)

        # Calculate loss
        outputs: torch.Tensor = model(signals)
        # adjusted_outputs = adjust_output_length(outputs, midi_tensors.shape[2])
        print(f"Outputs: {outputs.shape} | Labels: {midi_tensors.shape}")
        adjusted_outputs = F.interpolate(input=outputs, size=midi_tensors.shape[2], mode='linear', align_corners=False)
        loss = loss_fn(adjusted_outputs, midi_tensors)

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")
    history.append(loss.item())

def train(
        model: JojoNet,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.modules.loss.MSELoss,
        optimizer: torch.optim.Optimizer,
        device: torch.DeviceObjType,
        epochs: int
    ) -> None:
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model=model, data_loader=data_loader, loss_fn=loss_fn, optimizer=optimizer, device=device)
    print("Training complete!")

def custom_collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    signals, midi_tensors = zip(*batch)

    # Stack signals
    signals = torch.stack(signals, dim=0)

    # Pad midi_tensors
    midi_tensors = nn.utils.rnn.pad_sequence(midi_tensors, batch_first=True, padding_value=0)

    return signals, midi_tensors

if __name__ == "__main__":
    # Train model on cpu, mps, or cuda, whatever is available
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'

    print(f"{device} selected\nFetching data...")

    transform = torchaudio.transforms.MelSpectrogram(n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=NUM_MEL_BANDS)

    # Create GrooveDataset object
    groove_data_train = GrooveDataset(
        metadata_path=METADATA_PATH,
        data_path=DATA_PATH,
        split='train',
        sr=STANDARD_SR,
        transform=transform,
        max_samples=MAX_SAMPLES,
        data_amt=0.02,
        device=device
    )

    # Create the data loader
    train_set = DataLoader(
        dataset=groove_data_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # collate_fn=custom_collate_fn
    )

    print("Loaded data")

    # Create the model
    jojonet = JojoNet().to(device)

    print("Created model")

    # Instantiate loss function and optimizer
    mse_loss = nn.MSELoss()
    adam_optimizer = torch.optim.Adam(params=jojonet.parameters(), lr=LEARNING_RATE)

    print("Beginning training -- here we go!")

    # Train the model
    train(
        model=jojonet, 
        data_loader=train_set, 
        loss_fn=mse_loss, 
        optimizer=adam_optimizer, 
        device=device, 
        epochs=NUM_EPOCHS
    )

    # Save the model
    torch.save(jojonet.state_dict(), "jojonet.pth")

    # Write out the history to a JSON to visualize elsewhere
    with open('training_history.json', 'w') as f:
        json.dump(history, f)