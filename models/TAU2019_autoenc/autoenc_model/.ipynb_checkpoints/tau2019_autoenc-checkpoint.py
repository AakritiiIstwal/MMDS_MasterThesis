import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

def tensorStacker(feature_dir):
    #Creating a list of tensors to stack them up
    tensors = []

    for log_mel_tensor in feature_dir.glob('*.pt'):
        tensor_data = torch.load(log_mel_tensor)
        tensors.append(tensor_data)

    if tensors:
        training_data = torch.stack(tensors)
        print(f"Stacked Tensor Shape: {training_data.shape}")
        return training_data

    else:
        print("No tensors found.")
        return torch.empty((0, 0))
    
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 20, 751)
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: (32, 10, 376)
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: [64, 80, 8]
            nn.LeakyReLU(),
            nn.Conv2d(64, 10, kernel_size=3, stride=2, padding=1), # Output: 
            nn.LeakyReLU()
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # Output: (32, 10, 376)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # Output: (32, 10, 376)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # Output: (16, 20, 751)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),    # Adjusted padding
            nn.LeakyReLU()
        )

    def forward(self, x):
        encoder_x = self.encoder(x)
        decoder_x = self.decoder(encoder_x)
        return (encoder_x, decoder_x)


def main():

    # Hyperparameters
    batch_size = 20
    num_epochs = 200
    learning_rate = 1e-3

    print("-----------Autoencoder Training for tau2019----------")
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Prepare and Fetch training data
    feature_path = '/work/aistwal/MMDS_MasterThesis/data/tau2019/train_log_mel_features/features'
    print("Feature directory exists: " , os.path.isdir(feature_path))
    feature_dir = Path(feature_path)

    x_train =  tensorStacker(feature_dir).to(device)
    y_train = x_train.clone().to(device)  # For autoencoders, input and output are the same

    # Creating datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.unsqueeze(1)  # Adds a channel dimension (1) after the batch dimension
            # This was needed since the dimensions of inputs for tau2019 was 50*640*64 but the model needs 50*1*640*64
            labels = labels.unsqueeze(1)

            # Forward pass
            encoder_output, decoder_output = model(inputs)
            loss = criterion(decoder_output, labels)
            writer.add_scalar("Loss/train", loss, epoch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #check for val loss
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') #todo Add loss to another log directory

    # Save the model
    torch.save(model.state_dict(), f'/work/aistwal/MMDS_MasterThesis/models/checkpoints/tau_2019_conv_autoencoder_{num_epochs}_{batch_size}.pth')
    writer.flush()
    print(f'Model saved to tau_2019_conv_autoencoder_{num_epochs}_{batch_size}.pth')


if __name__ == "__main__":
    main()