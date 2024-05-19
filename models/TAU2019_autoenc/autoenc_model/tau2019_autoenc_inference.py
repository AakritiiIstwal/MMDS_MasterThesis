import torch
from pathlib import Path
from tau2019_autoenc import *
import re
import pandas as pd
from tqdm import tqdm
import ast
import librosa
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def extract_label(audio_filename):
    class_label = audio_filename.split("-")[0]
    return class_label


def main():
    # Load the model
    parser = ArgumentParser()
    parser.add_argument(
        "checkpoint", help="The checkpoint from where the model has to be loaded"
    )
    parser.add_argument("train_dir", help="The path where training features are saved")
    parser.add_argument(
        "encoded_feature_dir", help="The path where encoded features need to be saved"
    )
    parser.add_argument(
        "decoded_feature_dir", help="The path where decoded features need to be saved"
    )
    args = parser.parse_args()

    model = ConvAutoencoder()
    model_path = Path(args.checkpoint)
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_dir = Path(args.train_dir)

    encoded_feature_path = Path(args.encoded_feature_dir)
    print("Saving encoded features in : ", encoded_feature_path)
    encoded_feature_path.mkdir(parents=True, exist_ok=True)

    decoded_feature_path = Path(args.decoded_feature_dir)
    print("Saving decoded features in : ", decoded_feature_path)
    decoded_feature_path.mkdir(parents=True, exist_ok=True)

    encoder_outputs = []
    labels = []

    for file in tqdm(train_dir.glob("*.pt")):
        try:
            audio_scene_label = extract_label(file.stem)

            input_data = torch.load(file).to(device)
            input_data = input_data.unsqueeze(0)

            with torch.no_grad():
                encoder_output, decoder_output = model(input_data)

            encoder_filename = encoded_feature_path / f"{file.stem}_encoded.pt"
            decoder_filename = decoded_feature_path / f"{file.stem}_decoded.pt"
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
    save_dir_path = input("Enter the path to save the pickle file \n")
    df_save_path = save_dir_path + "/tau_2019_conv_autoencoder_50_10.pkl"
    encoder_df.to_pickle(df_save_path)
    print(f"--------SAVED DATAFRAME TO {df_save_path}-------")

    # READ PICKLE FILE
    unpickled_encoder_df = pd.read_pickle(
        save_dir_path + "/tau_2019_conv_autoencoder_200_25.pkl"
    )
    print(unpickled_encoder_df.shape)
