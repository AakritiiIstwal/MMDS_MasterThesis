import pandas as pd
from dataloader import AudioDataset
from collections import Counter
import torch
from torch.utils.data import DataLoader
from scene_cls_model import Cnn_9layers_AvgPooling
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    audio_files = eval_df["filename"].map(lambda x: audio_path_root + x).tolist()
    scene_labels = eval_df["scene_label"].map(lb_to_idx).tolist()

    label_counts = Counter(scene_labels)
    print(f"Label distribution: {label_counts}")

    dataset = AudioDataset(audio_files, scene_labels)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print("Evaluation Data Size: ", len(data_loader))

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

    for inputs, targets in tqdm(data_loader):
        preds = predict(model, inputs)
        predictions.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        filenames.extend([dataset.audio_files[i] for i in range(len(inputs))])
        
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('TAU2019 Fold1 Evaluate Confusion Matrix')
    plt.savefig("fold1_evaluate_confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    main()
