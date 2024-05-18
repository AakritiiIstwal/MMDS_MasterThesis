import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import numpy as np


audio_path = '/work/aistwal/panns_inference/resources/R9_ZSCveAHg_7s.wav'
(audio, sr) = librosa.core.load(audio_path, sr=32000, mono=True)
print("Loaded audio shape: " , audio.shape, " Sampling rate: ", sr)
audio = audio[None, :]  # (batch_size, segment_samples)

def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))



print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding) = at.inference(audio)
"""clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""
print_audio_tagging_result(clipwise_output[0])


# print('------ Sound event detection ------')
# sed = SoundEventDetection(checkpoint_path=None, device='cuda')
# framewise_output = sed.inference(audio)


#FETCH THE AUDIO TAGGING RESULT FOR ALL THE TAU2019 DEV AUDIOS AND THEN WRITE THEM UP IN "DATA FOLDER" IN A CSV/PICKLE. 
