import os
import yaml
import json

# FUNCTION TO PROCESS ONE YEAR
def build_data_for_year(base_dir):
    result = {}

    yaml_dir = os.path.join(base_dir, "data")

    # Loop YAML files (each is one observation date)
    for filename in os.listdir(yaml_dir):
        if not filename.endswith(".yaml"):
            continue

        # Extract date: "20250708.yaml" → "20250708"
        date = filename.replace(".yaml", "")
        yaml_path = os.path.join(yaml_dir, filename)

        # Load YAML file (crop height)
        with open(yaml_path, "r", encoding="utf-8") as f:
            ydata = yaml.safe_load(f)
            first_entry = ydata[0]

        # Loop through points in the YAML
        for p in first_entry["points"]:
            point_id = p["Point_id"]
            height = p["crop_height_cm"]

            # RGB path
            rgb_path = os.path.join(base_dir, "RGB", date, f"{point_id}.tif")

            # ---------- Create record ----------
            record = {
                "date": date,
                "rgb_path": rgb_path,
                "height_cm": height,
            }

            # ---------- Append to Point_id ----------
            if point_id not in result:
                result[point_id] = []

            result[point_id].append(record)

    # ========== Sort the sequence for each point by date ==========
    for pid in result.keys():
        result[pid].sort(key=lambda x: x["date"])

    return result

# CONFIGURATION
base_dirs = {
    2024: r"D:/Vebots/Iwamizawa/2024",
    2025: r"D:/Vebots/Iwamizawa/2025"
}

# Output file
json_output = "./dataset/dataset.json"
if os.path.exists(json_output):
    os.remove(json_output)

# MAIN SCRIPT
full_data = {}

for year, base_dir in base_dirs.items():
    print(f"Processing year {year}...")
    full_data[str(year)] = build_data_for_year(base_dir)

# Save JSON
with open(json_output, "w", encoding="utf-8") as f:
    json.dump(full_data, f, indent=2, ensure_ascii=False)

print(f"JSON file saved: {json_output}")
