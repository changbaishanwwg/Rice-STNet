import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import logging
from PIL import Image
from torch.utils.data import Dataset
import json

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, predictions, targets):
        loss = (predictions - targets) ** 2
        weighted_loss = self.weight * loss
        return weighted_loss.mean()

class HeightDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)  # Load JSON data
        self.transform = transform
        self.samples = []

        for year, points in self.data.items():
            for point_id, records in points.items():
                for record in records:
                    image_path = record["rgb_path"]
                    height = record["height_cm"]

                    # Normalize path separators (important on Windows)
                    image_path = os.path.normpath(image_path)

                    self.samples.append({
                        "image_path": image_path,
                        "height": height,
                        "year": int(year),
                        "date": record["date"],
                        "point_id": int(point_id)
                    })

        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image_path"]
        height = sample["height"]
        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(height, dtype=torch.float32)

def create_result_subfolders(savetype, base_path="result"):
    # Define paths for train and test subdirectories
    save_path = os.path.join(base_path, savetype)

    # Ensure the base and subdirectories exist
    os.makedirs(save_path, exist_ok=True)

    # Get the next folder number for train and test
    def get_next_folder_number(path):
        # List all subdirectories in the path
        subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # Extract numerical folder names and find the highest number
        numerical_folders = [int(folder) for folder in subfolders if folder.isdigit()]
        next_number = max(numerical_folders, default=0) + 1
        return next_number

    next_number = get_next_folder_number(save_path)

    # Create new folders for this run
    new_folder = os.path.join(save_path, str(next_number))
    os.makedirs(new_folder)

    return new_folder

# Function to load the best model and continue training
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def plot_loss(epochs, epoch_losses, new_folder):
    plt.plot(range(1, epochs + 1), epoch_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    # Save the plot as an image file
    output_path = os.path.join(new_folder, "training_loss.png")  # Specify the output file name and path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure with high resolution
    plt.show()

# Training Loop
def train_net(model, dataloader, criterion, optimizer, epochs=10,
              pretrained_model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained model if a path is provided
    if pretrained_model_path:
        model = load_model(model, pretrained_model_path)
        print(f"Model loaded from {pretrained_model_path}")

    epoch_losses = []
    best_epoch = 0
    best_loss = float('inf')  # Initialize with a large value to track the smallest loss
    new_folder = create_result_subfolders("Train")

    # Set up logging
    log_file = os.path.join(new_folder, "training_log.txt")
    logging.basicConfig(
        filename=log_file,  # Save the log to a file
        level=logging.INFO,  # Set the logging level
        format="%(message)s"  # Add timestamps to the logs
    )
    # Add console handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console_handler)

    for epoch in range(epochs):
        total_loss = 0
        num_samples_in_epoch = 0
        model.train()  # Ensure the model is in training mode
        for images, height_values in dataloader:
            images, height_values = images.to(device), height_values.to(device)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions.squeeze(), height_values)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss and sample count
            total_loss += loss.item() * images.size(0)  # Multiply by the batch size
            num_samples_in_epoch += images.size(0)

        # Store loss for this epoch
        avg_loss = total_loss / num_samples_in_epoch
        epoch_losses.append(avg_loss)
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        logging.info(f"Epoch {epoch + 1}/{epochs}, AverageLoss: {avg_loss :.4f}")
        # Save the model if the current loss is lower than the best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path1 = os.path.join(new_folder, "height_cnn_best_model.pth")
            torch.save(model.state_dict(), save_path1)
            best_epoch = epoch

    logging.info(f"Model saved with loss {best_loss:.4f} at epoch {best_epoch + 1}")

    # Plot the loss over epochs
    plot_loss(epochs, epoch_losses, new_folder)

    save_path2 = os.path.join(new_folder, "height_cnn_model.pth")
    torch.save(model.state_dict(), save_path2)
    print(f"train results have be saved in: {new_folder}")

