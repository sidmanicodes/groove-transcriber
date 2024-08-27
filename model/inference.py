import torch
import torch.nn.functional as F
import torchaudio
from jojonet import JojoNet
from groove_dataset import GrooveDataset
from train import FRAME_SIZE, HOP_SIZE, NUM_MEL_BANDS, METADATA_PATH, DATA_PATH, STANDARD_SR, MAX_SAMPLES
from new_midi_utils import tensor_to_midi

def predict(model: JojoNet, input: torch.Tensor, label: torch.Tensor, device: torch.DeviceObjType) -> None:
    # Switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        prediction: torch.Tensor = model(input)
        label = label.to(device)
        adjusted_prediction: torch.Tensor = F.interpolate(input=prediction, size=label.shape[1], mode='linear', align_corners=False)
        adjusted_prediction.squeeze_(0)
        print(adjusted_prediction[torch.where(adjusted_prediction > 1)])
        # print(label[torch.where(label > 1)])
        # print(label.shape)

        # Write out to midi
        tensor_to_midi(adjusted_prediction, out_file="prediction.mid")

        # Write label out to midi too (for testing purposes)
        # tensor_to_midi(label, out_file="ground_truth.mid")


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
    groove_data_test = GrooveDataset(
        metadata_path=METADATA_PATH,
        data_path=DATA_PATH,
        split='test',
        sr=STANDARD_SR,
        transform=transform,
        max_samples=MAX_SAMPLES,
        data_amt=0.01,
        device=device
    )

    # Get data for inference
    input, label = groove_data_test[0][0], groove_data_test[0][1]
    input.unsqueeze_(0)

    # Make prediction
    predict(model=jojonet, input=input, label=label, device=device)