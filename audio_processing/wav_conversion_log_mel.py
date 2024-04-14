import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import re

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = int(SAMPLE_RATE * 0.04)
HOP_LEN = int(SAMPLE_RATE * 0.02)


mel_spectrogram = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LEN,
    n_mels=N_MELS)


def extract_label(audio_filename):
    class_label = re.search("[a-zA-Z]+",audio_filename).group(0)
    return class_label

def load_audio(aud_filename):
    wav, sr = torchaudio.load(aud_filename, normalize=True)
    try:
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)
        return wav
    except:
        print("Error while loading audio")

    
def get_log_melSpectrogram(audio):
    mel_feats = mel_spectrogram(audio)
    log_mel_feats = T.AmplitudeToDB()(mel_feats)
    return log_mel_feats


def save_features(directory, save_dir):

    audio_dir = Path(directory)
    # Directory to save features
    feature_dir = Path(save_dir)
    features_dir =  feature_dir / 'features'
    features_dir.mkdir(exist_ok=True)

    audio_process_count = 0

    for audio_file in tqdm(audio_dir.glob('*.wav')):
        class_label = extract_label(audio_file.stem)
        audio_wav = load_audio(audio_file)
        log_mel_feats = get_log_melSpectrogram(audio_wav)

        feature_path = features_dir / f'{audio_file.stem}.pt'
        torch.save(log_mel_feats, feature_path)
        audio_process_count+=1

    return audio_process_count

def main():
    audio_process_count = save_features("/work/aistwal/MasterThesis/data/jointSoundScene_data/syntheticSoundscenes/train", "/work/aistwal/MasterThesis/data/jointSoundScene_data/syntheticSoundscenes/train_log_mel_features")
    print("Total audio processed: ", audio_process_count)


if __name__ == "__main__":
    main()