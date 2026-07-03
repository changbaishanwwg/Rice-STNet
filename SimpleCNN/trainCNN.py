import torch
from torchvision import transforms
from simpleCNN import HeightCNN
import torch.optim as optim
from torch.utils.data import DataLoader
from commonfuction import train_net, WeightedMSELoss, HeightDataset
import torch.nn as nn

# Transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Ensure images are 160x160
    transforms.ToTensor(),          # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

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

# Hyperparameters
learning_rate = 0.0005
batch_size = 8
epochs = 1000
weight_decay = 5e-7

# Dataset and DataLoader
json_file = "./dataset/train.json"  # Path to your JSON file
dataset = HeightDataset(json_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# # Check the number of samples
# print(f"Total samples in the dataset: {len(dataset)}")
# # Access a single sample
# sample_idx = 0
# image, height = dataset[sample_idx]
# print(f"Image shape: {image.shape}")  # Expected: [3, 160, 160] (after ToTensor)
# print(f"height value: {height}")          # height value for the sample

# Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeightCNN().to(device)
model.apply(init_weights)
criterion = WeightedMSELoss(weight=1)  # Loss function for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the Model
train_net(model, dataloader, criterion, optimizer,
          # pretrained_model_path="height_cnn_best_model.pth",
          epochs=epochs)


# # Save the Model
# # Create a dummy input tensor with the same dimensions as the training data
# dummy_input = torch.randn(1, 3, 160, 160).to(device)  # Batch size 1, 3 channels, 160x160 image
# # Define the output ONNX file path
# onnx_file = "height_cnn_model.onnx"
#
# # Export the model
# torch.onnx.export(
#     model,                          # The model to be exported
#     dummy_input,                    # Dummy input tensor for tracing
#     onnx_file,                      # Output file path
#     export_params=True,             # Store trained parameters in the model
#     opset_version=11,               # ONNX version to use (11 is widely supported)
#     input_names=['input'],          # Name of the input nodes
#     output_names=['output'],        # Name of the output nodes
#     dynamic_axes={                  # Specify dynamic axes if needed
#         'input': {0: 'batch_size'},  # Dynamic batch size
#         'output': {0: 'batch_size'}
#     }
# )
#
# print(f"Model has been exported to {onnx_file}")

