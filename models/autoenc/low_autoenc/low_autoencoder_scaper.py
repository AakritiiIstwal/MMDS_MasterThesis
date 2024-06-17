import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


import sys

sys.path.append("/work/aistwal/MMDS_MasterThesis/models/dataset_prep")
from scaper_log_mel import tensorStacker


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [50, 32, 10, 376]
            nn.LeakyReLU(),
            nn.Conv2d(32, 5, kernel_size=3, stride=2, padding=1),  # [50, 5, 5, 188]
            nn.LeakyReLU(),
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                5, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        encoder_x = self.encoder(x)
        # print(encoder_x.size())
        # exit(0)
        decoder_x = self.decoder(encoder_x)
        return (encoder_x, decoder_x)


def main():

    # Hyperparameters
    batch_size = 50
    num_epochs = 500
    learning_rate = 1e-3

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare and Fetch training data
    feature_dir = Path(
        "/work/aistwal/MasterThesis/data/jointSoundScene_data/syntheticSoundscenes/train_log_mel_features/features"
    )

    x_train = tensorStacker(feature_dir).to(device)
    y_train = x_train.clone().to(device)

    # Creating datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, labels = data

            # Forward pass
            encoder_output, decoder_output = model(inputs)
            # print(encoder_output.size())
            # print(decoder_output.size())

            loss = criterion(decoder_output, labels)
            writer.add_scalar("Loss/train", loss, epoch)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}"
        )  # TODO Add loss to another log directory

    # Save the model
    torch.save(
        model.state_dict(),
        f"/work/aistwal/MMDS_MasterThesis/models/checkpoints/low_conv_autoencoder_3_{num_epochs}_{batch_size}.pth",
    )
    writer.flush()
    print(f"Model saved to low_conv_autoencoder_3_epoch_{num_epochs}_{batch_size}.pth")


if __name__ == "__main__":
    main()
