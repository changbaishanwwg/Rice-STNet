import torch
import os
from torchvision import transforms
from commonFuction import HeightDataset, create_result_subfolders
from torch.utils.data import DataLoader
from RiceSTNet import PredictionModel
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Ensure images are 160x160
    transforms.ToTensor(),          # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

hidden_size = 64
gruoutput_size = 32
batch_size = 16

plt.rcParams.update({
    "font.family": "Times New Roman",  # or "Arial"
    "font.size": 24,
    # "axes.titlesize": 16,
    # "axes.labelsize": 16,
    # "xtick.labelsize": 16,
    # "ytick.labelsize": 16,
})

# Dataset and DataLoader
json_file = "./dataset/test.json"  # Path to your JSON file
dataset = HeightDataset(json_file, transform=transform, augmentations_per_image=0)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PredictionModel(hidden_size, gruoutput_size).to(device)
# Load the saved model weights
model.load_state_dict(torch.load("./result/Train/6/best_train_model.pth", weights_only=True))
# Set the model to evaluation mode
model.to(device)
model.eval()

new_folder = create_result_subfolders("Test")
# File path to save predictions and metrics
output_txt_path = os.path.join(new_folder, "predictions_with_metrics.txt")

measured_values = []
predicted_values = []

with torch.no_grad():  # Disable gradient calculation
    for images, time_encodings, targets in dataloader:
        images, time_encodings, targets = images.to(device), time_encodings.to(device), targets.to(device)

        # Forward pass
        predictions = model(images, time_encodings)
        predictions = predictions.cpu().numpy().flatten()  # Convert to NumPy
        # print(predictions)
        measured = targets.cpu().numpy()  # Convert to NumPy
        # print(measured)

        # Collect predictions and ground truth values
        predicted_values.extend(predictions)
        measured_values.extend(measured)

print(predicted_values)
print(measured_values)

# Calculate performance metrics
r2 = r2_score(measured_values, predicted_values)
rmse = root_mean_squared_error(measured_values, predicted_values)
mae = mean_absolute_error(measured_values, predicted_values)

# Convert to numpy arrays
measured_values = np.array(measured_values)
predicted_values = np.array(predicted_values)

# Linear regression: predicted = slope * measured + intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(measured_values, predicted_values)

# Generate smooth x values for regression line
x_fit = np.linspace(measured_values.min(), measured_values.max(), 200)
y_fit = slope * x_fit + intercept

# Calculate residual standard error
n = len(measured_values)
y_pred_reg = slope * measured_values + intercept
residuals = predicted_values - y_pred_reg
s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))

# t value for 95% confidence interval
t_value = stats.t.ppf(0.975, df=n - 2)

# Calculate 95% confidence interval
x_mean = np.mean(measured_values)
Sxx = np.sum((measured_values - x_mean) ** 2)

ci = t_value * s_err * np.sqrt(
    1 / n + (x_fit - x_mean) ** 2 / Sxx
)

# If you want 95% prediction interval instead, use this:
pi = t_value * s_err * np.sqrt(
    1 + 1 / n + (x_fit - x_mean) ** 2 / Sxx
)

# Open the text file for writing
with open(output_txt_path, "w", encoding="utf-8") as file:
    # Write metrics to the file
    file.write("Performance Metrics:\n")
    file.write(f"R²: {r2:.10f}\n")
    file.write(f"RMSE: {rmse:.10f}\n")
    file.write(f"MAE: {mae:.10f}\n")

plt.figure(figsize=(8, 8))
ax = plt.gca()
plt.scatter(measured_values, predicted_values, alpha=0.7, label="Predictions", s=100)
plt.plot([min(measured_values)-0.0001, max(measured_values)+0.0001],
         [min(measured_values)-0.0001, max(measured_values)+0.0001],
         linestyle="--", label="y = x (Diagonal)", color="black")  # Dashed diagonal line
# text = f"$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}"
# plt.text(0.03, 0.75, text, fontsize=24,
#          transform=plt.gca().transAxes, bbox=dict(facecolor="white", alpha=0.7, pad=2))

# Regression line
plt.plot(
    x_fit,
    y_fit,
    color="red",
    linewidth=2.5,
    label="Regression line"
)
# # 95% CI band
plt.fill_between(
    x_fit,
    y_fit - ci,
    y_fit + ci,
    color="red",
    alpha=0.15,
    label="95% CI"
)

# If you want PI instead of CI, comment out the CI part above and use this:
# plt.fill_between(
#     x_fit,
#     y_fit - pi,
#     y_fit + pi,
#     color="red",
#     alpha=0.12,
#     label="95% PI"
# )

# Regression equation text
if intercept >= 0:
    equation = f"y = {slope:.2f}x + {intercept:.2f}"
else:
    equation = f"y = {slope:.2f}x - {abs(intercept):.2f}"

text = (
    f"{equation}\n"
    f"R² = {r2:.4f}\n"
    f"RMSE = {rmse:.4f}\n"
    f"MAE = {mae:.4f}"
)

# Text box
plt.text(
    0.05,
    0.95,
    text,
    fontsize=22,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.85,
        pad=6
    )
)

plt.xlabel("Measured Height")
plt.ylabel("Predicted Height")
output_path = os.path.join(new_folder, "Measured vs. Predicted Height.png")  # Specify the output file name and path
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure with high resolution
plt.show()
