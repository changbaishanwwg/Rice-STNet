import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from RiceSTNet import PredictionModel
from commonFunction import HeightDataset, WeightedMSELoss, train_net
import torch.optim as optim
import torch.nn as nn

def init_weights(m):
    # CNN / Linear
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    # BatchNorm
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # GRU
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

# Transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Ensure images are 160x160
    transforms.ToTensor(),          # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

# Hyperparameters
hidden_size = 64
gruoutput_size = 32
batch_size = 16
epochs = 1000
learning_rate = 0.0004277703862798587
weight_decay = 1.7703537394600929e-06

# Load datasets
json_file = "./dataset/train.json" # Path to your JSON file
dataset = HeightDataset(json_file, transform=transform, augmentations_per_image=8)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PredictionModel(hidden_size, gruoutput_size).to(device)
model.apply(init_weights)

criterion = WeightedMSELoss(weight=1)  # Loss function for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the Model
train_net(model, dataloader, criterion, optimizer,
          # pretrained_model_path="osavi_cnn_best_model.pth",
          epochs=epochs)

