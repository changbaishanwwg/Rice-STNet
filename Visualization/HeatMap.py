import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from torchvision import transforms
from RiceSTNet import PredictionModel
from commonFuction import days_since_sowing_from_yyyymmdd, create_result_subfolders
from tqdm import tqdm
from scipy.stats import gaussian_kde

# ================= CONFIG =================
MODEL_PATH = "./weights/best_train_model.pth"

# Trimmed farmland images for prediction
RGB_TEST_ROOT = r"D:\Vebots\Iwamizawa\2024\RGBTest"

# Original full orthomosaic images for overlay background
RGB_ORIGINAL_ROOT = r"D:\Vebots\Iwamizawa\2024\RGB"

OUTPUT_ROOT = create_result_subfolders("FieldHeightMap")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 64
gruoutput_size = 32

BATCH_SIZE = 128
BLOCK_TILE_SIZE = 128
square_side_length = 1.0

HEIGHT_MIN = 45
HEIGHT_MAX = 90
OVERLAY_ALPHA = 0.65

print("Using device:", DEVICE)

# ================= Style =================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 32,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ================= Load Model =================
model = PredictionModel(hidden_size, gruoutput_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully.")

# ================= Get Dates =================
date_folders = sorted([
    f for f in os.listdir(RGB_TEST_ROOT)
    if os.path.isdir(os.path.join(RGB_TEST_ROOT, f))
])

if len(date_folders) < 2:
    raise ValueError("Need at least 2 dates.")

print("Available dates:", date_folders)

# ================= Image Transform =================
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= Colormap =================
cmap = LinearSegmentedColormap.from_list(
    "green_yellow_red",
    ["green", "yellow", "red"]
)
cmap.set_bad(color="white", alpha=0)

norm = Normalize(vmin=HEIGHT_MIN, vmax=HEIGHT_MAX)


# ================= Helper: Read RGB for display =================
def read_rgb_for_display(src):
    rgb = src.read([1, 2, 3]).astype(np.float32)
    rgb = np.transpose(rgb, (1, 2, 0))

    # Robust stretch for visualization
    out = np.zeros_like(rgb, dtype=np.float32)
    for b in range(3):
        band = rgb[:, :, b]
        valid = band[band > 0]

        if valid.size > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            out[:, :, b] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
        else:
            out[:, :, b] = 0

    return out


# ================= Helper: Save KDE =================
def save_kde_curve(height_values, target_date, save_path):
    valid = height_values[~np.isnan(height_values)]
    valid = valid[(valid >= HEIGHT_MIN) & (valid <= HEIGHT_MAX)]

    if len(valid) < 5:
        print(f"Not enough valid pixels for KDE: {target_date}")
        return

    x_grid = np.linspace(HEIGHT_MIN, HEIGHT_MAX, 500)
    kde = gaussian_kde(valid)
    density = kde(x_grid)
    density = density / density.max()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Gradient fill
    for j in range(len(x_grid) - 1):
        x1, x2 = x_grid[j], x_grid[j + 1]
        y1, y2 = density[j], density[j + 1]
        color = cmap(norm((x1 + x2) / 2))

        ax.add_patch(
            Polygon(
                [[x1, 0], [x1, y1], [x2, y2], [x2, 0]],
                facecolor=color,
                edgecolor=None,
                linewidth=0,
                alpha=0.75
            )
        )

    # Colored KDE line
    points = np.array([x_grid, density]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=3.0
    )
    lc.set_array(x_grid)
    ax.add_collection(lc)

    mean_height = np.mean(valid)

    ax.axvline(
        mean_height,
        color="black",
        linestyle="--",
        linewidth=1.5
    )

    ax.text(
        mean_height + 0.5,
        density.max() * 0.85,
        f"Mean = {mean_height:.1f} cm",
        fontsize=plt.rcParams["font.size"],
        color="black"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("Plant Height (cm)", labelpad=14)

    ax.set_xlim(HEIGHT_MIN, HEIGHT_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(50, HEIGHT_MAX + 1, 10))
    ax.set_yticks(np.arange(0, 1.01, 0.25))

    ax.set_xlabel("Plant Height (cm)")
    ax.set_ylabel("Normalized Density")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print("KDE curve saved:", save_path)


# ================= Helper: Overlay height map on original RGB =================
def save_overlay_on_original(
    height_map,
    height_transform,
    height_crs,
    original_rgb_path,
    target_date,
    save_path
):
    MAX_DISPLAY_WIDTH = 6000
    MAX_DISPLAY_HEIGHT = 5000

    with rasterio.open(original_rgb_path) as src_bg:

        scale = max(
            src_bg.width / MAX_DISPLAY_WIDTH,
            src_bg.height / MAX_DISPLAY_HEIGHT,
            1
        )

        out_width = int(src_bg.width / scale)
        out_height = int(src_bg.height / scale)

        print(f"Overlay display size: {out_height} x {out_width}")

        bg = src_bg.read(
            [1, 2, 3],
            out_shape=(3, out_height, out_width),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        bg = np.transpose(bg, (1, 2, 0))

        display_transform = src_bg.transform * Affine.scale(
            src_bg.width / out_width,
            src_bg.height / out_height
        )

        height_on_bg = np.full(
            (out_height, out_width),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=height_map.astype(np.float32),
            destination=height_on_bg,
            src_transform=height_transform,
            src_crs=height_crs,
            dst_transform=display_transform,
            dst_crs=src_bg.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest
        )

        valid_mask = ~np.isnan(height_on_bg)

        if not np.any(valid_mask):
            print(f"No valid overlay pixels for {target_date}")
            return

        # ================= RGB stretch with white background =================
        bg_rgb = np.ones_like(bg, dtype=np.float32)  # white background

        for b in range(3):
            band = bg[:, :, b]
            valid_band = band[band > 0]

            if valid_band.size > 0:
                p2, p98 = np.percentile(valid_band, [2, 98])
                stretched = np.clip(
                    (band - p2) / (p98 - p2 + 1e-6),
                    0,
                    1
                )

                # Only keep non-zero original pixels; zero background remains white
                bg_rgb[:, :, b] = np.where(band > 0, stretched, 1.0)

        # ================= Height map color =================
        clipped_height = np.clip(
            height_on_bg,
            HEIGHT_MIN,
            HEIGHT_MAX
        ).astype(np.float32)

        height_rgb = cmap(norm(clipped_height))[:, :, :3].astype(np.float32)

        # ================= Hollow-out replacement =================
        # Use height map color directly, without mixing original RGB underneath.
        overlay_rgb = bg_rgb.copy()
        overlay_rgb[valid_mask] = height_rgb[valid_mask]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(overlay_rgb)
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # cbar = plt.colorbar(
        #     sm,
        #     ax=ax,
        #     fraction=0.046,
        #     pad=0.04,
        #     shrink=0.75
        # )
        # cbar.set_label("Plant Height (cm)", labelpad=18)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Overlay saved:", save_path)


# ================= MAIN LOOP =================
for i in range(1, len(date_folders)):

    prev_date = date_folders[i - 1]
    target_date = date_folders[i]

    print(f"\nProcessing: {prev_date} → {target_date}")

    t1 = days_since_sowing_from_yyyymmdd(prev_date)
    t2 = days_since_sowing_from_yyyymmdd(target_date)

    time_tensor = torch.tensor(
        [t1, t2],
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    tif1_path = os.path.join(RGB_TEST_ROOT, prev_date, "result.tif")
    tif2_path = os.path.join(RGB_TEST_ROOT, target_date, "result.tif")

    original_rgb_path = os.path.join(
        RGB_ORIGINAL_ROOT,
        target_date,
        "result.tif"
    )

    if not os.path.exists(original_rgb_path):
        raise FileNotFoundError(f"Original RGB image not found: {original_rgb_path}")

    with rasterio.open(tif1_path) as src1, rasterio.open(tif2_path) as src2:

        # ================= Spatial alignment =================
        bounds1 = src1.bounds
        bounds2 = src2.bounds

        left = max(bounds1.left, bounds2.left)
        right = min(bounds1.right, bounds2.right)
        bottom = max(bounds1.bottom, bounds2.bottom)
        top = min(bounds1.top, bounds2.top)

        if left >= right or bottom >= top:
            raise ValueError("No overlapping area between images.")

        window1 = rasterio.windows.from_bounds(
            left, bottom, right, top,
            transform=src1.transform
        )

        window2 = rasterio.windows.from_bounds(
            left, bottom, right, top,
            transform=src2.transform
        )

        window1 = window1.round_offsets().round_lengths()
        window2 = window2.round_offsets().round_lengths()

        width = int(window1.width)
        height = int(window1.height)

        pixel_width = abs(src1.transform[0])
        pixel_height = abs(src1.transform[4])

        pixels_in_width = int(round(square_side_length / pixel_width))
        pixels_in_height = int(round(square_side_length / pixel_height))

        n_rows = height // pixels_in_height
        n_cols = width // pixels_in_width

        print("Tiles:", n_rows, "x", n_cols)

        height_map = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

        total_tiles = n_rows * n_cols

        # ================= Prediction =================
        with torch.no_grad():
            with tqdm(total=total_tiles, desc="Predicting", unit="tile") as pbar:

                for tile_block_r in range(0, n_rows, BLOCK_TILE_SIZE):
                    for tile_block_c in range(0, n_cols, BLOCK_TILE_SIZE):

                        block_tile_rows = min(BLOCK_TILE_SIZE, n_rows - tile_block_r)
                        block_tile_cols = min(BLOCK_TILE_SIZE, n_cols - tile_block_c)

                        pixel_row = tile_block_r * pixels_in_height
                        pixel_col = tile_block_c * pixels_in_width

                        block_height = block_tile_rows * pixels_in_height
                        block_width = block_tile_cols * pixels_in_width

                        block_window1 = Window(
                            col_off=window1.col_off + pixel_col,
                            row_off=window1.row_off + pixel_row,
                            width=block_width,
                            height=block_height
                        )

                        block_window2 = Window(
                            col_off=window2.col_off + pixel_col,
                            row_off=window2.row_off + pixel_row,
                            width=block_width,
                            height=block_height
                        )

                        img1_block = src1.read(window=block_window1)[:3]
                        img2_block = src2.read(window=block_window2)[:3]

                        img1_block = np.transpose(img1_block, (1, 2, 0))
                        img2_block = np.transpose(img2_block, (1, 2, 0))

                        batch_sequences = []
                        batch_indices = []

                        for r in range(block_tile_rows):
                            for c in range(block_tile_cols):

                                global_r = tile_block_r + r
                                global_c = tile_block_c + c

                                row_start = r * pixels_in_height
                                col_start = c * pixels_in_width

                                row_end = row_start + pixels_in_height
                                col_end = col_start + pixels_in_width

                                tile1 = img1_block[row_start:row_end, col_start:col_end]
                                tile2 = img2_block[row_start:row_end, col_start:col_end]

                                if np.all(tile1 == 0) or np.all(tile2 == 0):
                                    pbar.update(1)
                                    continue

                                tile1 = img_transform(tile1)
                                tile2 = img_transform(tile2)

                                sequence = torch.stack([tile1, tile2])

                                batch_sequences.append(sequence)
                                batch_indices.append((global_r, global_c))

                                if len(batch_sequences) == BATCH_SIZE:

                                    batch_tensor = torch.stack(batch_sequences).to(DEVICE)
                                    batch_time = time_tensor.repeat(len(batch_sequences), 1)

                                    preds = model(batch_tensor, batch_time)

                                    for idx, (rr, cc) in enumerate(batch_indices):
                                        height_map[rr, cc] = preds[idx].item()

                                    pbar.update(len(batch_sequences))

                                    batch_sequences = []
                                    batch_indices = []

                        if len(batch_sequences) > 0:

                            batch_tensor = torch.stack(batch_sequences).to(DEVICE)
                            batch_time = time_tensor.repeat(len(batch_sequences), 1)

                            preds = model(batch_tensor, batch_time)

                            for idx, (rr, cc) in enumerate(batch_indices):
                                height_map[rr, cc] = preds[idx].item()

                            pbar.update(len(batch_sequences))

        print("Prediction completed.")

        # ================= Save GeoTIFF =================
        cropped_transform = src1.window_transform(window1)

        out_transform = cropped_transform * Affine.scale(
            pixels_in_width,
            pixels_in_height
        )

        output_tif = os.path.join(
            OUTPUT_ROOT,
            f"HeightMap_{target_date}.tif"
        )

        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=n_rows,
            width=n_cols,
            count=1,
            dtype=rasterio.float32,
            crs=src1.crs,
            transform=out_transform,
            nodata=np.nan
        ) as dst:
            dst.write(height_map, 1)

        print("GeoTIFF saved:", output_tif)

        # ================= Draw Height Map Only =================
        masked_map = np.ma.masked_invalid(height_map)
        clipped_map = np.ma.clip(masked_map, HEIGHT_MIN, HEIGHT_MAX)

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(
            clipped_map,
            cmap=cmap,
            norm=norm
        )

        # cbar = plt.colorbar(
        #     im,
        #     ax=ax,
        #     fraction=0.046,
        #     pad=0.04,
        #     shrink=0.75
        # )
        # cbar.set_label("Plant Height (cm)", labelpad=18)

        ax.axis("off")

        plt.tight_layout()

        output_png = os.path.join(
            OUTPUT_ROOT,
            f"HeightMap_{target_date}.png"
        )

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()

        print("Heatmap saved:", output_png)

        # ================= Overlay on Original RGB =================
        overlay_png = os.path.join(
            OUTPUT_ROOT,
            f"HeightMap_Overlay_{target_date}.png"
        )

        save_overlay_on_original(
            height_map=height_map,
            height_transform=out_transform,
            height_crs=src1.crs,
            original_rgb_path=original_rgb_path,
            target_date=target_date,
            save_path=overlay_png
        )

        # ================= Draw KDE Curve =================
        kde_png = os.path.join(
            OUTPUT_ROOT,
            f"HeightMap_KDE_{target_date}.png"
        )

        save_kde_curve(
            height_values=height_map,
            target_date=target_date,
            save_path=kde_png
        )

print("\nAll predictions complete.")