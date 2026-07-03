import os
from datetime import datetime

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
