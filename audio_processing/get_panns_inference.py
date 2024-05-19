import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import numpy as np
import os
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import csv
import emoji

"""
Checkpoint path: /home/aistwal/panns_data/Cnn14_mAP=0.431.pth

RUN=>   CUDA_VISIBLE_DEVICES=1 python audio_processing/get_panns_inference.py \
         --dir /work/aistwal/dataset_tau2019/extracted-files/TAU-urban-acoustic-scenes-2019-development/audio \
         --out /work/aistwal/MMDS_MasterThesis/data/tau2019
"""

def save_audio_tagging_result(clipwise_output, audio_dict, fieldnames):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        # print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
        #     clipwise_output[sorted_indexes[k]]))
        event_prob_tuple = (
            np.array(labels)[sorted_indexes[k]],
            "{:.3f}".format(clipwise_output[sorted_indexes[k]]),
        )
        audio_dict[fieldnames[k]] = event_prob_tuple

    return audio_dict


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dir", help="Input the audio directory which holds your .wav files"
    )
    parser.add_argument("--out", help="Output csv of audio tagged tau2019 dataset")
    args = parser.parse_args()
    print(
        "This is a script to get panns inference for all .wav files w.r.t the task of audio tagging"
    )

    # INITIALIZE THE AUDIO TAGGER
    at = AudioTagging(checkpoint_path=None, device="cuda")

    audio_dir = Path(args.dir)
    if args.out:
        output_csv = args.out + "/tau2019_audiotagged.csv"
    else:
        output_csv = "tau2019_audiotagged.csv"

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "Audio_Filename",
            "top_1_event",
            "top_2_event",
            "top_3_event",
            "top_4_event",
            "top_5_event",
            "top_6_event",
            "top_7_event",
            "top_8_event",
            "top_9_event",
            "top_10_event",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for audio in tqdm(audio_dir.glob("*.wav")):
            audio_dict = {}
            audio_dict["Audio_Filename"] = audio.stem
            audio_path = args.dir / audio
            (audio, sr) = librosa.core.load(audio_path, sr=32000, mono=True)
            audio = audio[None, :]  # (batch_size, segment_samples)
            (clipwise_output, embedding) = at.inference(audio)
            # print(audio.shape) # (1, 320000) total samples in each audio is 320000
            audio_dict = save_audio_tagging_result(
                clipwise_output[0], audio_dict, fieldnames[1:]
            )
            writer.writerow(audio_dict)
            break

    print(emoji.emojize(":grinning_face_with_big_eyes:"), "Output csv stored in: ", Path(output_csv))


if __name__ == "__main__":
    main()
