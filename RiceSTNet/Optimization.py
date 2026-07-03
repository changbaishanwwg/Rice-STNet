from torchvision import transforms
from commonFunction import HeightDataset, WeightedMSELoss
import optuna
import torch
import logging
from sklearn.model_selection import KFold
from RiceSTNet import PredictionModel
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import os
from tqdm import tqdm
import optuna.visualization as vis
import torch.nn as nn

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
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

new_folder = create_result_subfolders("BayesianHyperband")
# Set up logging
# 创建文件处理器
log_file = os.path.join(new_folder, "training_log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s"
)
# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console_handler)

# Load dataset
json_file = "./dataset/train.json"
dataset = HeightDataset(json_file, transform=transform, augmentations_per_image=0)
train_dataset = HeightDataset(json_file, transform=transform)

# K-Fold Cross Validation setup
k_folds = 5
batch_size = 16
epochs = 70

# Function to train and evaluate the model
def train_and_evaluate_model(trial):
    # Sample hyperparameters
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)

    logging.info(f"Trying: LR={learning_rate:.8f}, WD={weight_decay:.8f}")

    fold_val_loss = 0.0  # Track validation loss across folds

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        new_train_idx = []
        for idx in train_idx:
            new_train_idx.extend([idx * 9 + i for i in range(9)])
        train_subset = Subset(train_dataset, new_train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PredictionModel(hidden_size=64, gruoutput_size=32).to(device)
        model.apply(init_weights)
        criterion = WeightedMSELoss(weight=1)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for images, time_encodings, targets in progress_bar:
                images, time_encodings, targets = images.to(device), time_encodings.to(device), targets.to(device)

                optimizer.zero_grad()
                predictions = model(images, time_encodings)
                train_loss = criterion(predictions.squeeze(), targets)
                train_loss.backward()
                optimizer.step()

            # Validation
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

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

        fold_val_loss += best_val_loss
        logging.info(f"🔹 Fold {fold + 1}/{k_folds} Training completed. "
                     f"Best Val loss: {best_val_loss:.6f}")

        # Report intermediate results to Optuna for pruning
        # print(f"Reporting: Fold {fold}, Loss: {best_val_loss:.6f}")
        trial.report(best_val_loss, fold)

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()  # Free GPU memory

        # Hyperband Pruner: Stop training if the trial is not promising
        if trial.should_prune():
            logging.info(f"⏳ Pruned at Fold {fold + 1}")
            raise optuna.exceptions.TrialPruned()

    # Average validation loss across folds
    avg_fold_val_loss = fold_val_loss / k_folds
    logging.info(f"Training completed: Average Validation loss: {avg_fold_val_loss:.6f}")
    return avg_fold_val_loss

# Use Hyperband Pruner for faster optimization
pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=k_folds, reduction_factor=3)
sampler = optuna.samplers.TPESampler()
# Optimize hyperparameters using Bayesian Optimization
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)  # We want to minimize validation loss
study.optimize(train_and_evaluate_model, n_trials=50)  # Run 50 trials

# Print the best hyperparameters
logging.info(f"Best Hyperparameters: {study.best_params}")

# Save visualization plots after optimization
# Save optimization history
fig1 = vis.plot_optimization_history(study)
fig1_path = os.path.join(new_folder, "optimization_history.png")
fig1.write_image(fig1_path)
# Save hyperparameter importance
fig2 = vis.plot_param_importances(study)
fig2_path = os.path.join(new_folder, "hyperparameter_importance.png")
fig2.write_image(fig2_path)
# Save slice plot
fig3 = vis.plot_slice(study)
fig3_path = os.path.join(new_folder, "hyperparameter_slice.png")
fig3.write_image(fig3_path)
# Save parallel coordinate plot
fig4 = vis.plot_parallel_coordinate(study)
fig4_path = os.path.join(new_folder, "parallel_coordinate.png")
fig4.write_image(fig4_path)
# Save contour plot
fig5 = vis.plot_contour(study)
fig5_path = os.path.join(new_folder, "contour_plot.png")
fig5.write_image(fig5_path)
# Save pruning analysis
fig6 = vis.plot_intermediate_values(study)
fig6_path = os.path.join(new_folder, "pruning_analysis.png")
fig6.write_image(fig6_path)
