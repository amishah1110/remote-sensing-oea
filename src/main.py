import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PyCharm
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import pandas as pd
import os
import numpy.ma as ma
from matplotlib.colors import ListedColormap

# ------------------- Configuration -------------------
DATA_DIR = "C:\\Users\\amisa\\PycharmProjects\\RemoteSensing\\data"  # Replace with your path
BAND_FILES = {
    "B2": "LC08_L1TP_137045_20250328_20250401_02_T1_B2.TIF",  # Blue
    "B3": "LC08_L1TP_137045_20250328_20250401_02_T1_B3.TIF",  # Green
    "B5": "LC08_L1TP_137045_20250328_20250401_02_T1_B5.TIF",  # NIR
}

# ------------------- Band Reader -------------------
def read_band(path, scale=4):
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        if scale > 1:
            band = band[::scale, ::scale]
        return band

# ------------------- Index Calculators -------------------
def compute_ndwi(green, nir):
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir)
        return np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)

def compute_fui(blue, green):
    with np.errstate(divide='ignore', invalid='ignore'):
        fui = (blue - green) / (blue + green)
        return np.nan_to_num(fui, nan=0.0, posinf=0.0, neginf=0.0)

# ------------------- Water Mask -------------------
def apply_water_mask(fui, ndwi, threshold=0.1):
    mask = ndwi > threshold
    return np.where(mask, fui, 0)

# ------------------- FUI Classification -------------------
def classify_fui(fui):
    classified = np.zeros_like(fui, dtype=np.uint8)
    classified[(fui > 0.001) & (fui <= 0.35)] = 1  # Low
    classified[(fui > 0.35) & (fui <= 0.45)] = 2  # Medium
    classified[fui > 0.45] = 3                    # High
    return classified

# ------------------- Visualizer -------------------
def visualize_classified_fui(classified_array, filename):
    colors = ['black', '#56b1f7', '#f7c842', '#e73030']  # 0: background, 1: low, 2: medium, 3: high
    cmap = ListedColormap(colors)
    masked_array = ma.masked_where(classified_array == 0, classified_array)

    plt.figure(figsize=(8, 6))
    plt.imshow(masked_array, cmap=cmap)
    cbar = plt.colorbar(ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
    plt.title("FUI - Microplastic Risk Levels")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def visualize_index(index_array, title, filename, cmap="viridis"):
    masked_array = ma.masked_where(index_array == 0, index_array)
    plt.figure(figsize=(8, 6))
    plt.imshow(masked_array, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ------------------- Main Script -------------------
def main():
    # Load bands
    B2 = read_band(os.path.join(DATA_DIR, BAND_FILES["B2"]), scale=2)
    B3 = read_band(os.path.join(DATA_DIR, BAND_FILES["B3"]), scale=2)
    B5 = read_band(os.path.join(DATA_DIR, BAND_FILES["B5"]), scale=2)

    # Compute indices
    NDWI = compute_ndwi(B3, B5)
    FUI = compute_fui(B2, B3)

    # Apply water mask
    FUI_masked = apply_water_mask(FUI, NDWI)

    # Visualize NDWI and FUI
    visualize_index(NDWI, "NDWI - Water Detection", "ndwi.png", cmap="Blues")
    visualize_index(FUI_masked, "Masked FUI - Microplastic Indicator", "fui_masked.png", cmap="inferno")

    # Classify microplastic levels
    classified_fui = classify_fui(FUI_masked)
    visualize_classified_fui(classified_fui, "fui_classified_map.png")

    # Prepare DataFrame
    data = pd.DataFrame({
        "NDWI": NDWI.flatten(),
        "FUI": FUI_masked.flatten(),
    })
    data["Microplastic_Level"] = (data["FUI"] > 0.25).astype(int)
    data = data.astype({"NDWI": "float32", "FUI": "float32"})
    data = data.head(100_000)  # Optional limit

    # Save features
    data.to_csv("microplastic_features.csv", index=False)
    print("Sample Data:")
    print(data.head())
    print("Features saved to microplastic_features.csv")

if __name__ == "__main__":
    main()
