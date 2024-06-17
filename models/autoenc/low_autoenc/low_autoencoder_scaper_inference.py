import torch
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
import ast
import librosa
import matplotlib.pyplot as plt

import sys

sys.path.append("/work/aistwal/MMDS_MasterThesis/models/low_autoenc")
from low_autoencoder_scaper import *


def extract_label(audio_filename):
    class_label = re.search("[a-zA-Z]+", audio_filename).group(0)
    return class_label


def main():
    # Load the model
    model = ConvAutoencoder()
    model_path = Path("models/checkpoints/low_conv_autoencoder_200_50.pth")
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_dir = Path(
        "/work/aistwal/MasterThesis/data/jointSoundScene_data/syntheticSoundscenes/train_log_mel_features/features"
    )
    encoded_feature_dir = Path(
        "/work/aistwal/MMDS_MasterThesis/data/jointSoundScene_data/syntheticSoundscenes/low_autoencoder/encoded_log_mel_features/features"
    )
    decoded_feature_dir = Path(
        "/work/aistwal/MMDS_MasterThesis/data/jointSoundScene_data/syntheticSoundscenes/low_autoencoder/decoded_log_mel_features/features"
    )

    encoder_outputs = []
    labels = []

    input_tensors = []
    encoded_tensors = []

    for file in tqdm(train_dir.glob("*.pt")):
        try:
            audio_scene_label = extract_label(file.stem)
            input_data = torch.load(file).to(device)

            with torch.no_grad():
                encoder_output, decoder_output = model(input_data)

            encoder_filename = encoded_feature_dir / f"{file.stem}_encoded.pt"
            decoder_filename = decoded_feature_dir / f"{file.stem}_decoded.pt"
            torch.save(encoder_output, encoder_filename)
            torch.save(decoder_output, decoder_filename)

            # Update Lists
            flattened_tensor = torch.flatten(encoder_output.cpu())

            encoder_outputs.append(flattened_tensor.numpy())
            labels.append(audio_scene_label)

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Create DataFrame
    encoder_df = pd.DataFrame({"Label": labels, "Encoder Output": encoder_outputs})

    return encoder_df


if __name__ == "__main__":
    print("---------RUNNING AUTOENCODER INFERENCE-------")
    encoder_df = main()
    df_save_path = "data/encoder_data/low_conv_autoencoder_200_50.pkl"
    encoder_df.to_pickle(df_save_path)
    print(f"--------SAVED DATAFRAME TO {df_save_path}-------")

    # READ PICKLE FILE
    unpickled_encoder_df = pd.read_pickle(
        "data/encoder_data/low_conv_autoencoder_200_50.pkl"
    )
    print(unpickled_encoder_df[:3])
