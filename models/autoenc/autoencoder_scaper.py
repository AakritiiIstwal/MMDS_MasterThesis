import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.insert(1, "/work/aistwal/MMDS_MasterThesis/models/dataset_prep")

from torch.utils.data import DataLoader, TensorDataset
from scaper_log_mel import *
# from dataset_prep.scaper_log_mel import *

import logging
from tqdm import tqdm

log_path = '/work/aistwal/MMDS_MasterThesis/models/autoenc/logs/'

# Configure logging
logging.basicConfig(filename=log_path+'training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1
            ),  # Output: (16, 20, 751)
            nn.ReLU(True),
            nn.Conv2d(
                16, 32, kernel_size=3, stride=2, padding=1
            ),  # Output: (32, 10, 376)
            nn.ReLU(True),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # Output: (64, 5, 188)
            nn.ReLU(True),
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
            ),  # Output: (32, 10, 376)
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)
            ),  # Output: (16, 20, 751)
            nn.ReLU(True),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)
            ),  # Adjusted padding
        )

    def forward(self, x):
        encoder_x = self.encoder(x)
        decoder_x = self.decoder(encoder_x)
        return (encoder_x, decoder_x)


def main():
    logger.info("Training logs for autoencoder on scapper dataset")
    # Hyperparameters
    batch_size = 50
    num_epochs = 150
    learning_rate = 1e-3
    logger.info(f"Batch Size:{batch_size}\n Epochs:{num_epochs}")

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare and Fetch training data
    feature_dir = Path(
        "/work/aistwal/MMDS_MasterThesis/data/jointSoundScene_data/syntheticSoundscenes/train_log_mel_features/features"
    )

    x_train = tensorStacker(feature_dir).to(device)
    y_train = x_train.clone().to(
        device
    )  # For autoencoders, input and output are the same

    # Creating datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    writer = SummaryWriter()

    # Training the model with 30 batches in each epoch
    for epoch in tqdm(range(num_epochs)):
        for data in train_loader:
            inputs, labels = data

            # Forward pass
            encoder_output, decoder_output = model(inputs)
            loss = criterion(decoder_output, labels)
            writer.add_scalar("Loss/train", loss, epoch)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # check for val loss

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}"
        )

    # Save the model
    torch.save(
        model.state_dict(),
        f"/work/aistwal/backup/MMDS_MasterThesis/models/checkpoints/conv_autoencoder_{num_epochs}_{batch_size}.pth",
    )
    writer.flush()
    logger.info(f"Model saved to conv_autoencoder_epoch_{num_epochs}_{batch_size}.pth")


if __name__ == "__main__":
    main()
