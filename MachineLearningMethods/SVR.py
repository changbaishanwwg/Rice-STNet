import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from commonFunction import create_result_subfolders
from scipy import stats

# ================= CONFIG =================
data_file = r".\dataset\plant_height_ml_dataset.csv"
save_dir = create_result_subfolders("SVR")

test_size = 0.25
random_state = 8

model_out = os.path.join(save_dir, "svr_height.joblib")
importance_csv = os.path.join(save_dir, "feature_importance.csv")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 24,
})

# ================= FEATURES =================
feature_cols = [
    "time_code",
    "VARI",
    "GLI",
    "NGRDI",
    "ExG",
    "ExR",
    "ExGR",
    "R_mean",
    "G_mean",
    "B_mean",
    "R_std",
    "G_std",
    "B_std"
]

target_col = "height_cm"

# ================= LOAD DATA =================
df = pd.read_csv(data_file)
df.columns = df.columns.astype(str)

X = df[feature_cols].values
y = df[[target_col]].values  # shape (N, 1)

print(f"X shape = {X.shape}, y shape = {y.shape}")

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

# ================= SCALE FEATURES =================
scaler_X = StandardScaler().fit(X_train)
X_train_s = scaler_X.transform(X_train)
X_test_s = scaler_X.transform(X_test)

# ================= SCALE TARGET (Recommended for SVR) =================
scaler_y = StandardScaler().fit(y_train)
y_train_s = scaler_y.transform(y_train).ravel()
y_test_s = scaler_y.transform(y_test).ravel()

# ================= MODEL =================
svr = SVR()

param_grid = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10, 50, 100],
    "epsilon": [0.01, 0.05, 0.1, 0.2],
    "gamma": ["scale", "auto", 0.01, 0.1, 1]
}

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=random_state
)

search = GridSearchCV(
    svr,
    param_grid=param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_s, y_train_s)

best_model = search.best_estimator_

print("Best params:", search.best_params_)
print("Best CV R2:", search.best_score_)

# ================= EVALUATION =================
y_pred_s = best_model.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# Convert to numpy arrays
measured_values = np.array(y_test).ravel()
predicted_values = np.array(y_pred).ravel()

# Linear regression: predicted = slope * measured + intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(
    measured_values,
    predicted_values
)
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

print(f"\nTest RMSE = {rmse:.4f}")
print(f"Test R2   = {r2:.4f}")
print(f"Test MAE  = {mae:.4f}")

# ================= PLOT =================
plt.figure(figsize=(8, 8))
ax = plt.gca()
plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions", s=100)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="black",
    label="y = x"
)

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
# plt.legend()

output_path_1 = os.path.join(save_dir, "Measured_vs_Predicted_Height.png")
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
plt.show()

# ================= PERMUTATION FEATURE IMPORTANCE =================
result = permutation_importance(
    best_model,
    X_test_s,
    y_test_s,
    n_repeats=10,
    random_state=random_state,
    n_jobs=-1
)

importances = result.importances_mean

df_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False)

df_imp.to_csv(importance_csv, index=False)
print(f"Saved feature importance to {importance_csv}")

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(
    df_imp["Feature"][::-1],
    df_imp["Importance"][::-1],
    color="black"
)

plt.xlabel("Permutation importance")
plt.title("SVR Feature Importance")
plt.tight_layout()

output_path_2 = os.path.join(save_dir, "Feature_Importance.png")
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
plt.show()

# ================= SAVE MODEL =================
joblib.dump(
    {
        "model": best_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "best_params": search.best_params_,
        "random_state": random_state
    },
    model_out
)

print(f"Saved model to {model_out}")
