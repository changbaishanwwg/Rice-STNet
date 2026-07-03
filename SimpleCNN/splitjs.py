import json
import os
import random

# ======= USER SETTINGS =======
INPUT_JSON = "./dataset/dataset.json"
TRAIN_JSON = "./dataset/train.json"
TEST_JSON  = "./dataset/test.json"

train_ratio = 0.75
# =============================

def clearfile(path):
    if os.path.exists(path):
        os.remove(path)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Load the main dataset
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# Clean output files
clearfile(TRAIN_JSON)
clearfile(TEST_JSON)

train_output = {}
test_output = {}

# SPLIT LOGIC: split by Point_id sequence (keeps full timeline)
for year, year_data in data.items():
    point_ids = list(year_data.keys())
    random.shuffle(point_ids)

    n_train = int(len(point_ids) * train_ratio)
    train_ids = point_ids[:n_train]
    test_ids  = point_ids[n_train:]

    train_output[year] = {}
    test_output[year] = {}

    for pid in train_ids:
        train_output[year][pid] = year_data[pid]

    for pid in test_ids:
        test_output[year][pid] = year_data[pid]

# Save split datasets
save_json(TRAIN_JSON, train_output)
save_json(TEST_JSON, test_output)

print(f" Train sets saved to {TRAIN_JSON}")
print(f" Test  sets saved to {TEST_JSON}")
for year in data:
    print(f"Year {year}: train={len(train_output[year])}, test={len(test_output[year])}")
