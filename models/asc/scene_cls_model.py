import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from dataloader import AudioDataset
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from collections import Counter
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_layer(layer, nonlinearity="leaky_relu"):
    """Initialize a Linear or Convolutional layer."""
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.running_mean.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_var.data.fill_(1.0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu(self.bn1(self.conv1(x))) #changed from F.relu_ which is the inplace version of F.relu
        x = F.relu(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception("Incorrect argument!")
        return x

class Cnn_9layers_AvgPooling(nn.Module):
    def __init__(self, classes_num, activation):
        super(Cnn_9layers_AvgPooling, self).__init__()
        self.activation = activation
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc = nn.Linear(512, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        x = input[:, None, :, :]
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block4(x, pool_size=(1, 1), pool_type="avg")
        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        x = self.fc(x)
        if self.activation == "logsoftmax":
            output = F.log_softmax(x, dim=-1)
        elif self.activation == "sigmoid":
            output = torch.sigmoid(x)
        return output

def load_dataset():

    # LOAD DATASET
    train_df = pd.read_csv(
        "/work/aistwal/dataset_tau2019/extracted-files/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold1_train.csv",
        sep="\t",
    )
    labels = [
        "airport", "shopping_mall", "metro_station", "street_pedestrian",
        "public_square", "street_traffic", "tram", "bus", "metro", "park"
    ]
    lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}

    audio_path_root = "/work/aistwal/dataset_tau2019/extracted-files/TAU-urban-acoustic-scenes-2019-development/"
    audio_files = train_df["filename"].map(lambda x: audio_path_root + x).tolist()
    scene_labels = train_df["scene_label"].map(lb_to_idx).tolist()

    label_counts = Counter(scene_labels)
    logger.info(f"Label distribution: {label_counts}")

    dataset = AudioDataset(audio_files, scene_labels)

    train_indices, val_indices = train_test_split(
        np.arange(len(scene_labels)),
        test_size=0.2,
        stratify=scene_labels
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    train_features, train_labels = next(iter(train_data_loader))
    logger.info(f"Feature batch shape: {train_features.size()}")
    logger.info(f"Labels batch shape: {train_labels.size()}")

    val_features, val_labels = next(iter(val_data_loader))
    logger.info(f"Feature batch shape: {val_features.size()}")
    logger.info(f"Labels batch shape: {val_labels.size()}")

    return train_data_loader, val_data_loader


def train_model(train_data_loader, val_data_loader, model, criterion, optimizer, num_epochs=10):

    writer = SummaryWriter()
    logger.info("Training start!!")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        # TRAIN CHECK
        for inputs, targets in train_data_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Get Predictions
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add Loss on tensorboard
            writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            optimizer.step()

            # Calculate Loss and Accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

        epoch_loss = running_loss / len(train_data_loader)
        epoch_accuracy = (correct_predictions / total_predictions) * 100
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')

        # VALIDATION CHECK
        valid_loss = 0.0
        correct_val_predictions = 0
        total_val_predictions = 0
        model.eval()
        for inputs, targets in val_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            val_outputs = model(inputs)
            loss = criterion(val_outputs, targets)

            # Calculate Loss and Accuracy
            valid_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(val_outputs, 1)
            correct_val_predictions += (predicted == targets).sum().item()
            total_val_predictions += targets.size(0)

        val_epoch_loss = valid_loss / len(val_data_loader)
        val_epoch_accuracy = (correct_val_predictions / total_val_predictions) * 100
        writer.add_scalar("Loss/valid", loss, epoch)
        logger.info(f'\tValidation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%')

    torch.save(model.state_dict(), f"/work/aistwal/MMDS_MasterThesis/models/checkpoints/tau_2019_conv_new_{num_epochs}_{batch_size}.pth")
    writer.flush()
    logger.info(f"Model saved to tau_2019_conv_new_{num_epochs}_{batch_size}.pth")
    logger.info("Training complete")

def main():
    train_data_loader, val_data_loader = load_dataset()
    model = Cnn_9layers_AvgPooling(classes_num=10, activation="logsoftmax")
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-08, weight_decay=0.0, amsgrad=True)
    train_model(train_data_loader, val_data_loader, model, criterion, optimizer, num_epochs=10)

if __name__ == "__main__":
    main()
