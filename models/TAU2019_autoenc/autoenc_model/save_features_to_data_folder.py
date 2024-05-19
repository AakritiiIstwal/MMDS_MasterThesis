import h5py
import torch
from tqdm import tqdm
import os

filename = "/work/aistwal/dcase2019_task1/features/logmel_64frames_64melbins/TAU-urban-acoustic-scenes-2019-development.h5"

with h5py.File(filename, "r") as f:
    print(
        "Keys: %s" % f.keys()
    )  # ['audio_name', 'feature', 'identifier', 'scene_label', 'source_label']

    audio_name = list(f.keys())[0]  # returns audio_name
    feature = list(f.keys())[1]  # returns feature

    ds_audio_name = f[audio_name]  # Audio names are stored as byte strings
    # Decode each byte string to a regular string
    audio_names = [name.decode("utf-8") for name in ds_audio_name]

    ds_obj = f[
        feature
    ]  # returns as a h5py dataset object which contains all 14,400 audio features
    print("Total number of audio files : ", ds_obj.shape[0])
    print("Feature shapes for audio files : ", ds_obj.shape[1], ds_obj.shape[2])

    for idx, np_array in tqdm(enumerate(ds_obj)):
        tensor = torch.from_numpy(np_array)
        # Save this tensor as .pt file in /work/aistwal/MMDS_MasterThesis/data/tau2019/train_log_mel_features/features
        root_dir = "/work/aistwal/MMDS_MasterThesis/data/tau2019/train_log_mel_features/features"
        aname = audio_names[idx].split(".wav")[0] + ".pt"
        full_path = os.path.join(root_dir, aname)
        torch.save(tensor, full_path)
