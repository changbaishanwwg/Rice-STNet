import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import logging
from torchvision import transforms
from MSRCP import msrcp
import numpy as np
import cv2
from copy import deepcopy
from datetime import datetime
from sklearn.metrics import r2_score

plt.rcParams.update({
    "font.family": "Times New Roman",  # or "Arial"
    "font.size": 18,
    # "axes.titlesize": 16,
    # "axes.labelsize": 16,
    # "xtick.labelsize": 16,
    # "ytick.labelsize": 16,
})

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, predictions, targets):
        # print(f"Predictions shape: {predictions.shape}")
        # print(f"Targets shape: {targets.shape}")
        loss = (predictions - targets) ** 2
        weighted_loss = self.weight * loss
        return weighted_loss.mean()

def days_since_sowing_from_yyyymmdd(date_str: str) -> int:
    """
    date_str expected like "20240711" (YYYYMMDD) or "2024-07-11".
    Returns integer days since sowing for that year:
      - 2024 sowing date = 2024-04-27
      - 2025 sowing date = 2025-05-02
    """
    # normalize format
    if "-" in date_str:
        fmt = "%Y-%m-%d"
    else:
        fmt = "%Y%m%d"
    d = datetime.strptime(date_str, fmt)
    if d.year == 2024:
        sow = datetime(2024, 4, 27)
    elif d.year == 2025:
        sow = datetime(2025, 5, 2)
    else:
        raise ValueError(f"Unsupported year found in date '{date_str}'")
    return (d - sow).days

# Add msrcp as a transform in augmentation
class MSRCPTransform:
    def __init__(self, sigma_scales, low_per=2, high_per=0):
        self.sigma_scales = sigma_scales
        self.low_per = low_per
        self.high_per = high_per
    def __call__(self, img):
        if isinstance(img, Image.Image):  # Convert PIL to NumPy
            img = np.array(img)  # Converts to HxWxC (RGB)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        img = msrcp(img, sigma_scales=self.sigma_scales, low_per=self.low_per, high_per=self.high_per)  # Apply MSRCP
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB
        return Image.fromarray(img)  # Convert back to PIL for compatibility

class New_HeightDataset(Dataset):
    def __init__(self, json_file, window_size=2, transform=None, augmentations_per_image=8):
        """
        Initialize the dataset with a JSON file.

        :param json_file: Path to the JSON file containing marker data
        :param window_size: The sliding window size for the sequences
        :param transform: Transformation to apply to the images
        :param augmentations_per_image: Number of augmented versions per image
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.window_size = window_size
        self.transform = transform
        self.augmentations_per_image = augmentations_per_image

        self.augment = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=60),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=0.4),
            transforms.ColorJitter(saturation=0.4),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2)
            ),
            MSRCPTransform([15, 100, 200]),
        ]

        self.sequences = self._prepare_sequences()

    def _prepare_sequences(self):
        sequences = []
        all_point_records = []

        for year_str, points in self.data.items():
            for point_id, records in points.items():

                entries = deepcopy(records)

                for rec in entries:
                    missing = [
                        k for k in ("date", "rgb_path", "height_cm")
                        if k not in rec
                    ]

                    if missing:
                        raise KeyError(
                            f"Missing keys {missing} in record "
                            f"for point {point_id} year {year_str}"
                        )

                all_point_records.append(
                    (int(year_str), str(point_id), entries)
                )

        for year, pid, recs in all_point_records:

            for rec in recs:
                rec["time_enc"] = days_since_sowing_from_yyyymmdd(rec["date"])

            recs.sort(key=lambda r: r["time_enc"])

            for i in range(len(recs) - self.window_size + 1):
                seq = recs[i:i + self.window_size]
                sequences.append((year, pid, seq))

        return sequences

    def __len__(self):
        return len(self.sequences) * (1 + self.augmentations_per_image)

    def __getitem__(self, idx):
        original_idx = idx // (1 + self.augmentations_per_image)
        augmentation_idx = idx % (1 + self.augmentations_per_image)

        year, point_id, sequence = self.sequences[original_idx]

        images = []
        height_values = []
        time_encodings = []

        if augmentation_idx > 0:
            augmentation = self.augment[augmentation_idx - 1]
        else:
            augmentation = None

        for item in sequence:
            image_path = item["rgb_path"]
            image = Image.open(image_path).convert("RGB")

            height = item["height_cm"]
            time_encoding = item["time_enc"]

            if augmentation is not None:
                image = augmentation(image)

            if self.transform:
                image = self.transform(image)

            images.append(image)
            height_values.append(height)
            time_encodings.append(time_encoding)

        images = torch.stack(images)

        height_values = torch.tensor(
            height_values,
            dtype=torch.float32
        )

        time_encodings = torch.tensor(
            time_encodings,
            dtype=torch.float32
        )

        # Target is the last record in the sequence
        target_height = height_values[-1]

        # New metadata for CSV saving
        target_date = sequence[-1]["date"]
        target_point_id = int(point_id)
        target_year = int(year)

        return (
            images,
            time_encodings,
            target_height,
            target_point_id,
            target_date,
            target_year
        )

class HeightDataset(Dataset):
    def __init__(self, json_file, window_size=2, transform=None, augmentations_per_image=8):
        """
        Initialize the dataset with a JSON file.
        :param json_file: Path to the JSON file containing marker data
        :param window_size: The sliding window size for the sequences
        :param transform: Transformation to apply to the images
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)  # Load JSON data
        self.window_size = window_size
        self.transform = transform
        self.augmentations_per_image = augmentations_per_image  # Number of augmented versions per image
        # Define augmentations (excluding ToTensor, applied separately)
        self.augment = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=60),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=0.4),
            transforms.ColorJitter(saturation=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            MSRCPTransform([15, 100, 200]),
            # MSRCRTransform([15, 80, 250])
        ]

        # Prepare the data for sliding window sequences
        self.sequences = self._prepare_sequences()

    def _prepare_sequences(self):
        sequences = []
        all_point_records = []

        for year_str, points in self.data.items():
            for point_id, records in points.items():
                # deepcopy to avoid modifying original structure
                entries = deepcopy(records)
                # validate that each record contains required paths
                for rec in entries:
                    # required keys: date, rgb_path, ndvi_path, lci_path, gndvi_path, ndre_path, height_cm, spad
                    # raise informative error if missing
                    missing = [k for k in
                               ("date", "rgb_path", "height_cm") if k not in rec]
                    if missing:
                        raise KeyError(f"Missing keys {missing} in record for point {point_id} year {year_str}")
                all_point_records.append((int(year_str), str(point_id), entries))

        # prepare sequence index list (generate sequences per year per point)
        # each sequence is a tuple (year, point_id, [record1, record2, ... window_size])
        for (year, pid, recs) in all_point_records:
            # compute time enc and sort by date within this year's records
            for rec in recs:
                rec["time_enc"] = days_since_sowing_from_yyyymmdd(rec["date"])
            recs.sort(key=lambda r: r["time_enc"])
            # sliding windows
            for i in range(len(recs) - self.window_size + 1):
                seq = recs[i:i + self.window_size]
                sequences.append((year, pid, seq))

        return sequences

    def __len__(self):
        """Total dataset size is original size * (1 + augmentations_per_image)"""
        return len(self.sequences)* (1 + self.augmentations_per_image)

    def __getitem__(self, idx):
        original_idx = idx // (1 + self.augmentations_per_image)  # Index of original image
        augmentation_idx = idx % (1 + self.augmentations_per_image)  # Which augmentation to apply
        year, point_id, sequence = self.sequences[original_idx]

        images = []
        height_values = []
        time_encodings = []
        # Apply the same augmentation to all images in the sequence
        if augmentation_idx > 0:
            augmentation = self.augment[augmentation_idx - 1]
        else:
            augmentation = None  # No augmentation

        for item in sequence:
            # Load image and height value
            image_path = item["rgb_path"]
            image = Image.open(image_path).convert("RGB")
            height = item["height_cm"]
            time_encoding = item["time_enc"]  # Time encoding (e.g., days since April 1st)

            if augmentation is not None:
                image = augmentation(image)  # Apply the same augmentation to all images
            # Apply transformations to the image
            if self.transform:
                image = self.transform(image)

            images.append(image)
            height_values.append(height)
            time_encodings.append(time_encoding)

        # Convert to tensors
        images = torch.stack(images)  # Stack images into a tensor of shape [window_size, C, H, W]
        height_values = torch.tensor(height_values, dtype=torch.float32)  # Shape [window_size]
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)  # Shape [window_size]
        # Target is the last height value in the sequence (predict this)
        target_height = height_values[-1]  # Last height value is the target

        return images, time_encodings, target_height

# Function to create result subfolders
def create_result_subfolders(savetype, base_path="result"):
    save_path = os.path.join(base_path, savetype)
    os.makedirs(save_path, exist_ok=True)

    def get_next_folder_number(path):
        subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        numerical_folders = [int(folder) for folder in subfolders if folder.isdigit()]
        return max(numerical_folders, default=0) + 1

    next_number = get_next_folder_number(save_path)
    new_folder = os.path.join(save_path, str(next_number))
    os.makedirs(new_folder)

    return new_folder

# Function to plot loss curve
def plot_loss(epochs, epoch_train_losses, new_folder, epoch_val_losses=None):
    x = range(1, epochs + 1)
    plt.plot(x, epoch_train_losses, label="Training Loss")
    if epoch_val_losses is not None:
        plt.plot(x, epoch_val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    output_path = os.path.join(new_folder, "loss_epoch.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy(epochs, r2_train, new_folder, ):
    x = range(1, epochs + 1)
    plt.plot(x, r2_train, label="Training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    output_path = os.path.join(new_folder, "r2_epoch.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def train_net(model, train_loader, criterion, optimizer, epochs=1000,
              pretrained_model_path=None, val_loader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Get model device
    # Load pre-trained model if specified
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path,
                                         map_location=device,
                                         weights_only=True))
        print(f"Loaded pretrained model from {pretrained_model_path}")

    new_folder = create_result_subfolders("Train")
    # Set up logging
    log_file = os.path.join(new_folder, "training_log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console_handler)

    epoch_train_losses = []
    best_train_loss = float('inf')
    best_train_epoch = 0
    epoch_val_losses = []
    best_val_loss = float('inf')
    best_val_epoch = 0
    r2_train = []

    for epoch in range(epochs):
        total_train_loss = 0
        num_train_samples = 0
        epoch_train_accs = []
        epoch_val_accs = []
        measured_values_accs = []

        # Training loop
        model.train()  # Set model to training mode
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, time_encodings, targets in progress_bar:
            images, time_encodings, targets = images.to(device), time_encodings.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(images, time_encodings)
            train_loss = criterion(predictions.squeeze(), targets)
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item() * images.size(0)
            num_train_samples += images.size(0)

            predictions_ac = predictions.detach().cpu().numpy().flatten()  # Convert to NumPy
            # print(predictions)
            measured_ac = targets.detach().cpu().numpy()  # Convert to NumPy
            epoch_train_accs.extend(predictions_ac)
            measured_values_accs.extend(measured_ac)

            progress_bar.set_postfix(loss=train_loss.item())

        avg_train_loss = total_train_loss / num_train_samples
        epoch_train_losses.append(avg_train_loss)
        r2 = r2_score(measured_values_accs, epoch_train_accs)
        r2_train.append(r2)

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_train_epoch = epoch
            save_path_best = os.path.join(new_folder, "best_train_model.pth")
            torch.save(model.state_dict(), save_path_best)

        # Validation loop (only if val_loader is provided)
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            num_val_samples = 0

            with torch.no_grad():
                for images, time_encodings, targets in val_loader:
                    images, time_encodings, targets = images.to(device), time_encodings.to(device), targets.to(device)
                    predictions = model(images, time_encodings)
                    val_loss = criterion(predictions.squeeze(), targets)
                    total_val_loss += val_loss.item() * images.size(0)
                    num_val_samples += images.size(0)

                avg_val_loss = total_val_loss / num_val_samples
                epoch_val_losses.append(avg_val_loss)
                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_epoch = epoch
                    save_path = os.path.join(new_folder, "best_val_model.pth")
                    torch.save(model.state_dict(), save_path)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Average Train Loss: {avg_train_loss:.6f}, "
                         f"Average Val Loss: {avg_val_loss:.6f}")
        else:
            logging.info(f"Epoch {epoch + 1}/{epochs}, "
                         f"Average Train Loss: {avg_train_loss:.6f}, "
                         f"Train Accuracy: {r2:.6f}")

    logging.info(f"Training completed. Best Train loss: {best_train_loss:.6f} at epoch {best_train_epoch + 1}")
    if val_loader is not None:
        logging.info(f"Best Val loss: {best_val_loss:.6f} at epoch {best_val_epoch + 1}")

    # Save final model and plot loss curve
    final_model_path = os.path.join(new_folder, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    if val_loader is not None:
        plot_loss(epochs, epoch_train_losses, new_folder, epoch_val_losses=epoch_val_losses)
    else:
        plot_loss(epochs, epoch_train_losses, new_folder)
        plot_accuracy(epochs, r2_train, new_folder)

    print(f"results saved in: {new_folder}")
