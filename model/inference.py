from tqdm import tqdm
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
from jojonet import JojoNet
from groove_dataset import GrooveDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train import FRAME_SIZE, HOP_SIZE, NUM_MEL_BANDS, METADATA_PATH, DATA_PATH, STANDARD_SR, MAX_SAMPLES
from new_midi_utils import tensor_to_midi

def compute_eval_metrics(prediction: torch.Tensor, label: torch.Tensor) -> tuple[float, float]:
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    mse = mean_squared_error(prediction, label)
    mae = mean_absolute_error(prediction, label)
    return mse, mae


def predict(model: JojoNet, input: torch.Tensor, label: torch.Tensor, device: torch.DeviceObjType) -> torch.Tensor:
    # Switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        prediction: torch.Tensor = model(input)
        label = label.to(device)
        adjusted_prediction: torch.Tensor = F.interpolate(input=prediction, size=label.shape[2], mode='linear', align_corners=False)
        adjusted_prediction.squeeze_(0)
        return adjusted_prediction


if __name__ == "__main__":
    
    # Set device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    # Load model
    jojonet = JojoNet().to(device)
    state_dict = torch.load("jojonet.pth", weights_only=True)
    jojonet.load_state_dict(state_dict)

    # Create spectrogram transform
    transform = torchaudio.transforms.MelSpectrogram(n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=NUM_MEL_BANDS)

    # Create GrooveDataset object
    groove_data_val = GrooveDataset(
        metadata_path=METADATA_PATH,
        data_path=DATA_PATH,
        split='validation',
        sr=STANDARD_SR,
        transform=transform,
        max_samples=MAX_SAMPLES,
        data_amt=0.1,
        device=device
    )

    groove_data_test = GrooveDataset(
        metadata_path=METADATA_PATH,
        data_path=DATA_PATH,
        split='test',
        sr=STANDARD_SR,
        transform=transform,
        max_samples=MAX_SAMPLES,
        data_amt=0.1,
        device=device
    )

    val_loader = DataLoader(dataset=groove_data_val, batch_size=1)
    test_loader = DataLoader(dataset=groove_data_test, batch_size=1)

    total_mse_val = 0
    total_mae_val = 0

    for input, label in tqdm(val_loader):
        prediction = predict(model=jojonet, input=input, label=label, device=device)
        label.squeeze_(0)
        mse, mae = compute_eval_metrics(prediction, label)
        total_mse_val += mse
        total_mae_val += mae

    total_mse_val /= len(groove_data_test)
    total_mae_val /= len(groove_data_test)

    total_mse_test = 0
    total_mae_test = 0

    for input, label in tqdm(test_loader):
        prediction = predict(model=jojonet, input=input, label=label, device=device)
        label.squeeze_(0)
        mse, mae = compute_eval_metrics(prediction, label)
        total_mse_test += mse
        total_mae_test += mae

    total_mse_test /= len(groove_data_test)
    total_mae_test /= len(groove_data_test)

    print(f"Validation MSE: {total_mse_val} | Validation MAE: {total_mae_val}")
    print(f"Testing MSE: {total_mse_test} | Testing MAE: {total_mae_test}")

    # Get data for inference
    input, label = groove_data_val[10][0], groove_data_test[10][1]

    # Make prediction
    prediction = predict(model=jojonet, input=input.unsqueeze(0), label=label.unsqueeze(0), device=device)
    torch.save(prediction, 'prediction.pt')
    torch.save(label, 'label.pt')
    tensor_to_midi(prediction*127, out_file="prediction.mid")
    tensor_to_midi(label, out_file="ground_truth.mid")