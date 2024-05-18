import os
import glob
from pathlib import Path
import torch

def tensorStacker(feature_dir):
    #Creating a list of tensors to stack them up
    tensors = []

    for log_mel_tensor in feature_dir.glob('*.pt'):
        tensor_data = torch.load(log_mel_tensor)
        tensors.append(tensor_data)

    if tensors:
        training_data = torch.stack(tensors)
        print(f"Stacked Tensor Shape: {training_data.shape}")
        return training_data

    else:
        print("No tensors found.")
        return torch.empty((0, 0))