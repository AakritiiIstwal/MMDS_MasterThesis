import pandas as pd
import torch
import numpy as np
from collections import Counter
from scene_cls_model import Cnn_9layers_AvgPooling
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

def predict(model, inputs):
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    return predicted

def main():
    # Read evaluate CSV
    eval_df = pd.read_csv('/work/aistwal/dataset_tau2019/extracted-files/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold1_evaluate.csv', sep='\t')
    
    labels = [
        "airport", "shopping_mall", "metro_station", "street_pedestrian",
        "public_square", "street_traffic", "tram", "bus", "metro", "park"]
    lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
    idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
    
    audio_path_root = "/work/aistwal/dataset_tau2019/extracted-files/TAU-urban-acoustic-scenes-2019-development/"
    eval_df['filepath'] = eval_df["filename"].map(lambda x: audio_path_root + x)
    audio_files = eval_df['filepath'].tolist()
    scene_labels = eval_df["scene_label"].map(lb_to_idx).tolist()

    label_counts = Counter(scene_labels)
    print(f"Label distribution: {label_counts}")

    # Instantiate and load the model
    model = Cnn_9layers_AvgPooling(classes_num=10, activation="logsoftmax")
    model_path = '/work/aistwal/MMDS_MasterThesis/models/checkpoints/tau_2019_conv_10_16.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Run inference through the evaluation DataLoader
    predictions = []
    all_targets = []
    filenames = []

    for audio, label in tqdm(zip(audio_files, scene_labels), total=len(audio_files)):
        # Load the audio file
        y, sr = librosa.load(audio, sr=32000)
        
        # Convert to log mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=500, n_mels=64)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Normalize the spectrogram
        log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / log_mel_spectrogram.std()
        
        # Convert to torch tensor
        log_mel_spectrogram = torch.tensor(log_mel_spectrogram, dtype=torch.float32).unsqueeze(0) # gives torch.Size([1, 64, 641])

        # Perform prediction
        preds = predict(model, log_mel_spectrogram)
        predictions.extend(preds.cpu().numpy())
        all_targets.append(label)
        filenames.append(audio)

    # Convert indices to labels
    predicted_labels = [idx_to_lb[pred] for pred in predictions]
    actual_labels = [idx_to_lb[target] for target in all_targets]

    # Save results to a CSV file
    results_df = pd.DataFrame({
        'Filename': filenames,
        'Predicted Label': predicted_labels,
        'Actual Label': actual_labels
    })
    results_df.to_csv('predictions.csv', index=False)
    print('Predictions saved to predictions.csv')

    # Calculate accuracy if targets are known
    correct_predictions = sum(np.array(predictions) == np.array(all_targets))
    total_predictions = len(all_targets)
    accuracy = (correct_predictions / total_predictions) * 100

    print(f'Inference Accuracy: {accuracy:.2f}%')

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, predictions)

    # Plot and save the confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('TAU2019 Fold1 Evaluate Confusion Matrix')
    plt.savefig("fold1_evaluate_confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    main()
