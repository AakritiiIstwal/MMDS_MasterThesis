import torch.utils.data as data
import librosa
import torch

'''
A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
The __len__ function returns the number of samples in our dataset.
The __getitem__ function loads and returns a sample from the dataset at the given index idx
'''



class AudioDataset(data.Dataset):
    def __init__(self, audio_files, labels, sr=32000, n_mels=64, hop_length=500):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Convert to log mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=self.hop_length, n_mels=self.n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Normalize the spectrogram
        log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / log_mel_spectrogram.std()
        
        return torch.tensor(log_mel_spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
def main():
    print("THIS FILE IS A CUSTOM DATASET CREATOR")

if __name__ == '__main__':
    main()