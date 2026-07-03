import os
import math
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import ScalarFormatter, MultipleLocator, FuncFormatter
from torchvision import transforms
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from RiceSTNet import PredictionModel
from commonFuction import days_since_sowing_from_yyyymmdd, create_result_subfolders

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./weights/best_train_model.pth"
RGB_ROOT = r"D:\Vebots\Iwamizawa\2024\RGBTest"
OUTPUT_ROOT = create_result_subfolders("FieldHeightMap3D")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 64
gruoutput_size = 32
BATCH_SIZE = 128
BLOCK_TILE_SIZE = 128
square_side_length = 1.0   # one tile = 1.0 m

# ===== Colorbar range (cm) =====
COLOR_MIN = 45
COLOR_MAX = 90

# ===== Z-axis display range (cm) =====
Z_AXIS_MIN = 40
Z_AXIS_MAX = 80

SAVE_GEOTIFF = False

# ===== Fixed crop size around reference center =====
CENTER_CROP_SIZE_M = 40.0

# ===== Smoothing =====
SMOOTH_SIGMA = 0.9
SMOOTH_KERNEL_SIZE = 5  # must be odd

# ===== Plot settings =====
FIGSIZE_3D = (12, 10)
DOWNSAMPLE_SCATTER = 1
DOWNSAMPLE_SURFACE = 1
SCATTER_SIZE = 20
SCATTER_ALPHA = 0.95
SURFACE_ALPHA = 0.98

# ===== View =====
ELEV_ANGLE = 38
AZIM_ANGLE = -128

# ===== Axis label spacing =====
X_LABELPAD = 38
Y_LABELPAD = 50
Z_LABELPAD = 34

# ===== Tick spacing =====
XY_TICK_STEP = 15      # x/y spacing = 10
Z_TICK_STEP_CM = 10    # z spacing = 10 cm

# ===== Fonts =====
BASE_FONT_SIZE = 38
TICK_FONT_SIZE = 38
TITLE_FONT_SIZE = 38
LABEL_FONT_SIZE = 38
CBAR_FONT_SIZE = 38

# ===== Display =====
SHOW_GRID = True
DISABLE_AXIS_OFFSET = True

print("Using device:", DEVICE)

# ============================================================
# LOAD MODEL
# ============================================================
model = PredictionModel(hidden_size, gruoutput_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully.")

# ============================================================
# GET DATE FOLDERS
# ============================================================
date_folders = sorted([
    f for f in os.listdir(RGB_ROOT)
    if os.path.isdir(os.path.join(RGB_ROOT, f))
])

if len(date_folders) < 2:
    raise ValueError("Need at least 2 dates.")

print("Available dates:", date_folders)

# ============================================================
# IMAGE TRANSFORM
# ============================================================
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# COLORMAP
# ============================================================
cmap = LinearSegmentedColormap.from_list(
    "green_yellow_red",
    ["green", "yellow", "red"]
)
cmap.set_bad(color="white", alpha=0)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": BASE_FONT_SIZE,
})

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def gaussian_kernel_1d(kernel_size=9, sigma=2.0):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def convolve1d_reflect(arr, kernel, axis):
    pad = len(kernel) // 2

    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode='reflect')
        out = np.empty_like(arr, dtype=np.float32)
        for r in range(arr.shape[0]):
            out[r, :] = np.sum(
                padded[r:r + len(kernel), :] * kernel[:, None],
                axis=0
            )
    elif axis == 1:
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode='reflect')
        out = np.empty_like(arr, dtype=np.float32)
        for c in range(arr.shape[1]):
            out[:, c] = np.sum(
                padded[:, c:c + len(kernel)] * kernel[None, :],
                axis=1
            )
    else:
        raise ValueError("axis must be 0 or 1")

    return out


def nan_gaussian_smooth(arr, kernel_size=9, sigma=2.0):
    kernel = gaussian_kernel_1d(kernel_size, sigma)

    data = np.array(arr, dtype=np.float32)
    valid = np.isfinite(data).astype(np.float32)
    data_filled = np.nan_to_num(data, nan=0.0)

    num = convolve1d_reflect(data_filled, kernel, axis=0)
    num = convolve1d_reflect(num, kernel, axis=1)

    den = convolve1d_reflect(valid, kernel, axis=0)
    den = convolve1d_reflect(den, kernel, axis=1)

    out = np.full_like(data, np.nan, dtype=np.float32)
    mask = den > 1e-6
    out[mask] = num[mask] / den[mask]

    return out


def maybe_smooth_height_map(arr, use_smoothing=True, kernel_size=9, sigma=2.0):
    if not use_smoothing:
        return np.array(arr, dtype=np.float32)
    return nan_gaussian_smooth(arr, kernel_size=kernel_size, sigma=sigma)


def create_geospatial_meshgrid(shape, transform):
    """
    Create real map coordinates at pixel centers using the affine transform.
    """
    n_rows, n_cols = shape
    cols = np.arange(n_cols)
    rows = np.arange(n_rows)
    C, R = np.meshgrid(cols, rows)

    X = transform.c + (C + 0.5) * transform.a + (R + 0.5) * transform.b
    Y = transform.f + (C + 0.5) * transform.d + (R + 0.5) * transform.e

    return X, Y


def axis_unit_from_crs(crs):
    if crs is None:
        return "map unit"

    try:
        if crs.is_geographic:
            return "degree"
    except Exception:
        pass

    try:
        if crs.is_projected:
            return "m"
    except Exception:
        pass

    return "map unit"


def apply_full_number_formatter(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)

    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    try:
        ax.ticklabel_format(style='plain', useOffset=False, axis='x')
        ax.ticklabel_format(style='plain', useOffset=False, axis='y')
    except Exception:
        pass

    try:
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
        ax.zaxis.get_offset_text().set_visible(False)
    except Exception:
        pass


def z_cm_formatter(x, pos):
    return f"{int(round(x * 100.0))}"


def world_to_fractional_rc(transform, x, y):
    """
    Convert world coordinates to fractional row/col in the tile-based grid.
    """
    inv = ~transform
    col_f, row_f = inv * (x, y)
    return row_f, col_f


def get_center_world_coordinate(transform, shape):
    """
    Center of the first generated tile map, in world coordinates.
    """
    n_rows, n_cols = shape
    center_row = n_rows / 2.0
    center_col = n_cols / 2.0
    x, y = transform * (center_col, center_row)
    return x, y


def crop_by_reference_world_center(height_map, transform, ref_center_x, ref_center_y, crop_size_m):
    """
    Crop every date using the same world-coordinate center.
    """
    n_rows, n_cols = height_map.shape

    pixel_width_m = abs(transform.a)
    pixel_height_m = abs(transform.e)

    crop_cols = max(1, int(round(crop_size_m / pixel_width_m)))
    crop_rows = max(1, int(round(crop_size_m / pixel_height_m)))

    center_row_f, center_col_f = world_to_fractional_rc(transform, ref_center_x, ref_center_y)

    center_row = int(round(center_row_f))
    center_col = int(round(center_col_f))

    row_start = center_row - crop_rows // 2
    row_end = row_start + crop_rows
    col_start = center_col - crop_cols // 2
    col_end = col_start + crop_cols

    # clip to valid range while preserving requested crop size as much as possible
    if row_start < 0:
        row_end -= row_start
        row_start = 0
    if col_start < 0:
        col_end -= col_start
        col_start = 0
    if row_end > n_rows:
        row_start -= (row_end - n_rows)
        row_end = n_rows
    if col_end > n_cols:
        col_start -= (col_end - n_cols)
        col_end = n_cols

    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_end = min(n_rows, row_end)
    col_end = min(n_cols, col_end)

    cropped_map = height_map[row_start:row_end, col_start:col_end]
    cropped_transform = transform * Affine.translation(col_start, row_start)

    return cropped_map, cropped_transform, row_start, row_end, col_start, col_end


def prepare_plot_arrays(height_map, transform, downsample=1,
                        use_smoothing=True, smooth_kernel_size=9, smooth_sigma=2.0,
                        color_min=45, color_max=90):
    """
    Returns X, Y, Z_cm, C_cm for plotting.
    Z is kept in cm for color logic, then converted to m for geometry only.
    """
    z_map_cm = maybe_smooth_height_map(
        height_map,
        use_smoothing=use_smoothing,
        kernel_size=smooth_kernel_size,
        sigma=smooth_sigma
    )

    if downsample < 1:
        downsample = 1

    z_map_cm = z_map_cm[::downsample, ::downsample]
    ds_transform = transform * Affine.scale(downsample, downsample)

    X, Y = create_geospatial_meshgrid(z_map_cm.shape, ds_transform)
    C = np.where(np.isfinite(z_map_cm), np.clip(z_map_cm, color_min, color_max), np.nan)

    return X, Y, z_map_cm, C


def set_common_3d_style(ax, x, y, axis_unit="m"):
    """
    Common axis, ticks, aspect, grid formatting.
    Z is plotted in meters, but labeled in cm using a formatter.
    """
    # Labels
    # ax.set_xlabel(f"X ({axis_unit})", labelpad=X_LABELPAD, fontsize=LABEL_FONT_SIZE)
    # ax.set_ylabel(f"Y ({axis_unit})", labelpad=Y_LABELPAD, fontsize=LABEL_FONT_SIZE)
    ax.set_zlabel("Plant Height (cm)", labelpad=Z_LABELPAD, fontsize=LABEL_FONT_SIZE)

    # Limits
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))
    ax.set_zlim(Z_AXIS_MIN / 100.0, Z_AXIS_MAX / 100.0)  # geometry uses meters

    # Tick spacing
    ax.xaxis.set_major_locator(MultipleLocator(XY_TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(XY_TICK_STEP))
    ax.zaxis.set_major_locator(MultipleLocator(Z_TICK_STEP_CM / 100.0))

    # Format z-axis labels in cm
    ax.zaxis.set_major_formatter(FuncFormatter(z_cm_formatter))

    # Font size
    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE, pad=12)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE, pad=13)
    ax.tick_params(axis='z', labelsize=TICK_FONT_SIZE, pad=12)

    # View
    ax.view_init(elev=ELEV_ANGLE, azim=AZIM_ANGLE)

    # Make grid cells square by matching the number of major intervals
    x_intervals = max((np.nanmax(x) - np.nanmin(x)) / XY_TICK_STEP, 1e-6)
    y_intervals = max((np.nanmax(y) - np.nanmin(y)) / XY_TICK_STEP, 1e-6)
    z_intervals = max((Z_AXIS_MAX - Z_AXIS_MIN) / Z_TICK_STEP_CM, 1e-6)
    try:
        ax.set_box_aspect((x_intervals, y_intervals, z_intervals))
    except Exception:
        pass

    ax.grid(SHOW_GRID)

    if DISABLE_AXIS_OFFSET:
        apply_full_number_formatter(ax)


def add_colorbar(fig, ax, mappable):
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.08)
    cbar.set_label("Plant Height (cm)", labelpad=14, fontsize=CBAR_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    return cbar


def save_scatter_plot(date_str, height_map, transform, output_path, axis_unit, use_smoothing):
    X, Y, Z_cm, C = prepare_plot_arrays(
        height_map=height_map,
        transform=transform,
        downsample=DOWNSAMPLE_SCATTER,
        use_smoothing=use_smoothing,
        smooth_kernel_size=SMOOTH_KERNEL_SIZE,
        smooth_sigma=SMOOTH_SIGMA,
        color_min=COLOR_MIN,
        color_max=COLOR_MAX
    )

    valid = np.isfinite(Z_cm)
    if not np.any(valid):
        print(f"Warning: no valid scatter points for date {date_str}")
        return

    x = X[valid]
    y = Y[valid]
    z_m = Z_cm[valid] / 100.0
    c = C[valid]

    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX)
    sc = ax.scatter(
        x, y, z_m,
        c=c,
        cmap=cmap,
        norm=norm,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        marker='o',
        linewidths=0
    )

    set_common_3d_style(ax, x, y, axis_unit=axis_unit)
    # add_colorbar(fig, ax, sc)

    title_tag = "Smoothed" if use_smoothing else "Raw"
    # ax.set_title(f"3D Scatter - {date_str} ({title_tag})", pad=22, fontsize=TITLE_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN LOOP
# ============================================================
reference_center_x = None
reference_center_y = None

for i in range(1, len(date_folders)):
    prev_date = date_folders[i - 1]
    target_date = date_folders[i]

    print(f"\nProcessing: {prev_date} -> {target_date}")

    t1 = days_since_sowing_from_yyyymmdd(prev_date)
    t2 = days_since_sowing_from_yyyymmdd(target_date)
    time_tensor = torch.tensor([t1, t2], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    tif1_path = os.path.join(RGB_ROOT, prev_date, "result.tif")
    tif2_path = os.path.join(RGB_ROOT, target_date, "result.tif")

    with rasterio.open(tif1_path) as src1, rasterio.open(tif2_path) as src2:
        print("CRS:", src1.crs)
        print("Transform:", src1.transform)

        axis_unit = axis_unit_from_crs(src1.crs)
        print("Detected XY axis unit:", axis_unit)

        # ------------------------------------------------------------
        # Spatial alignment
        # ------------------------------------------------------------
        bounds1 = src1.bounds
        bounds2 = src2.bounds

        left = max(bounds1.left, bounds2.left)
        right = min(bounds1.right, bounds2.right)
        bottom = max(bounds1.bottom, bounds2.bottom)
        top = min(bounds1.top, bounds2.top)

        if left >= right or bottom >= top:
            raise ValueError("No overlapping area between images.")

        window1 = rasterio.windows.from_bounds(left, bottom, right, top, transform=src1.transform)
        window2 = rasterio.windows.from_bounds(left, bottom, right, top, transform=src2.transform)

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

        # ------------------------------------------------------------
        # Tile-based inference
        # ------------------------------------------------------------
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

                                    for idx_pred, (rr, cc) in enumerate(batch_indices):
                                        height_map[rr, cc] = preds[idx_pred].item()

                                    pbar.update(len(batch_sequences))
                                    batch_sequences = []
                                    batch_indices = []

                        if len(batch_sequences) > 0:
                            batch_tensor = torch.stack(batch_sequences).to(DEVICE)
                            batch_time = time_tensor.repeat(len(batch_sequences), 1)
                            preds = model(batch_tensor, batch_time)

                            for idx_pred, (rr, cc) in enumerate(batch_indices):
                                height_map[rr, cc] = preds[idx_pred].item()

                            pbar.update(len(batch_sequences))

        print("Prediction completed.")

        # ------------------------------------------------------------
        # Tile-based geospatial transform
        # ------------------------------------------------------------
        cropped_transform = src1.window_transform(window1)
        out_transform = cropped_transform * Affine.scale(
            pixels_in_width,
            pixels_in_height
        )

        # ------------------------------------------------------------
        # Set reference center from the first generated image only
        # ------------------------------------------------------------
        if reference_center_x is None or reference_center_y is None:
            reference_center_x, reference_center_y = get_center_world_coordinate(
                out_transform,
                height_map.shape
            )
            print(f"Reference center set from first image: X={reference_center_x}, Y={reference_center_y}")

        # ------------------------------------------------------------
        # Optional GeoTIFF
        # ------------------------------------------------------------
        if SAVE_GEOTIFF:
            output_tif = os.path.join(OUTPUT_ROOT, f"HeightMap_{target_date}.tif")
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
            ) as dst:
                dst.write(height_map, 1)
            print("GeoTIFF saved:", output_tif)

        # ------------------------------------------------------------
        # Crop using fixed reference center
        # ------------------------------------------------------------
        center_crop_map, center_crop_transform, rs, re, cs, ce = crop_by_reference_world_center(
            height_map=height_map,
            transform=out_transform,
            ref_center_x=reference_center_x,
            ref_center_y=reference_center_y,
            crop_size_m=CENTER_CROP_SIZE_M
        )

        print(f"Fixed-center crop rows: {rs}:{re}, cols: {cs}:{ce}")
        print("Fixed-center crop shape:", center_crop_map.shape)

        crop_tag = f"{int(CENTER_CROP_SIZE_M)}m"

        # ------------------------------------------------------------
        # Save 4 images for the same date
        # ------------------------------------------------------------
        save_path = os.path.join(
            OUTPUT_ROOT,
            f"Scatter_{crop_tag}_Smooth_{target_date}.png"
        )

        save_scatter_plot(
            date_str=target_date,
            height_map=center_crop_map,
            transform=center_crop_transform,
            output_path=save_path,
            axis_unit=axis_unit,
            use_smoothing=True
        )

        print("Saved:")

print("\nAll predictions complete.")