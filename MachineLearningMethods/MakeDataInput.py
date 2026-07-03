import os
import yaml
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ================= UTILITIES =================
def read_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_and_resize_rgb(img_path, img_size=(160, 160)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img.astype(np.float32)

def compute_indices(rgb):
    """
    rgb: H x W x 3 (float32)
    """
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    eps = 1e-6

    VARI = np.mean((G - R) / (G + R - B + eps))
    GLI = np.mean((2 * G - R - B) / (2 * G + R + B + eps))
    NGRDI = np.mean((G - R) / (G + R + eps))
    ExG = np.mean(2 * G - R - B)
    ExR = np.mean(1.4 * R - G)
    ExGR = ExG - ExR

    return {
        "VARI": VARI,
        "GLI": GLI,
        "NGRDI": NGRDI,
        "ExG": ExG,
        "ExR": ExR,
        "ExGR": ExGR,
        "R_mean": np.mean(R),
        "G_mean": np.mean(G),
        "B_mean": np.mean(B),
        "R_std": np.std(R),
        "G_std": np.std(G),
        "B_std": np.std(B)
    }

def compute_time_code(date_str, year):
    """
    date_str: YYYYMMDD
    """
    obs_date = datetime.strptime(date_str, "%Y%m%d")
    sow_date = SOWING_DATES[year]
    return (obs_date - sow_date).days

# ================= CONFIG =================
BASE_DIRS = {
    2024: r"D:/Vebots/Iwamizawa/2024",
    2025: r"D:/Vebots/Iwamizawa/2025"
}

SOWING_DATES = {
    2024: datetime(2024, 4, 27),
    2025: datetime(2025, 5, 2)
}

IMAGE_SIZE = (160, 160)   # resize for consistency
OUTPUT_CSV = "./dataset/plant_height_ml_dataset.csv"

# ================= MAIN PROCESS =================
records = []

for year, base_dir in BASE_DIRS.items():
    print(f"\nProcessing year {year}")
    yaml_dir = os.path.join(base_dir, "data")

    for yaml_file in tqdm(os.listdir(yaml_dir)):
        if not yaml_file.endswith(".yaml"):
            continue

        date = yaml_file.replace(".yaml", "")
        yaml_path = os.path.join(yaml_dir, yaml_file)
        ydata = read_yaml(yaml_path)

        # usually first element contains points
        for p in ydata[0]["points"]:
            point_id = p["Point_id"]
            height = p["crop_height_cm"]

            rgb_path = os.path.join(base_dir, "RGB", date, f"{point_id}.tif")
            if not os.path.exists(rgb_path):
                continue

            try:
                rgb = read_and_resize_rgb(rgb_path)
                feats = compute_indices(rgb)
            except Exception as e:
                print(f"Skip {rgb_path}: {e}")
                continue

            time_code = compute_time_code(date, year)

            record = {
                "year": year,
                "date": date,
                "point_id": point_id,
                "time_code": time_code,
                "height_cm": height
            }

            record.update(feats)
            records.append(record)

# ================= SAVE =================
df = pd.DataFrame(records)
df.sort_values(["year", "point_id", "date"], inplace=True)
df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved dataset to:", OUTPUT_CSV)
print(df.head())
