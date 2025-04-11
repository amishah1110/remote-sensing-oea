import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy.ma as ma
import glob

# ==================== CONFIGURATION ====================
PATHS = {
    "pre_kumbh": r"C:\Users\amisa\PycharmProjects\RemoteSensing\pre_kumbh",
    "post_kumbh": r"C:\Users\amisa\PycharmProjects\RemoteSensing\post_kumbh",
    "output": r"C:\Users\amisa\PycharmProjects\RemoteSensing\results"
}

BAND_PATTERNS = {
    "blue": "*B2.TIF",
    "green": "*B3.TIF",
    "red": "*B4.TIF",
    "nir": "*B5.TIF",
    "swir1": "*B6.TIF"
}

PARAMS = {
    "target_resolution": 30,  # meters
    "water_threshold": 0.2,
    "risk_thresholds": {
        "low": (0.001, 0.05),
        "medium": (0.051, 0.25),
        "high": (0.25, 1.0)
    }
}

def load_and_resize_band(band_key, scene_dir, reference_meta=None):
    matches = glob.glob(os.path.join(scene_dir, BAND_PATTERNS[band_key]))
    if not matches:
        available = [f for f in os.listdir(scene_dir) if f.endswith('.TIF')]
        raise FileNotFoundError(f"No {band_key} band found. Available files:\n - " + "\n - ".join(available))

    file_path = matches[0]
    print(f"Loading {band_key} band from: {os.path.basename(file_path)}")

    with rasterio.open(file_path) as src:
        if reference_meta is None:
            # First image - return as is
            return src.read(1).astype(np.float32), src.meta
        else:
            # Resize to match reference image
            data = np.empty((reference_meta['height'], reference_meta['width']), dtype=np.float32)
            reproject(
                rasterio.band(src, 1),
                data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_meta['transform'],
                dst_crs=reference_meta['crs'],
                resampling=Resampling.bilinear
            )
            return data, reference_meta


def compute_indices(green, nir, blue, swir1):
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir)
        fmpi = (swir1 - (blue + green)) / (swir1 + (blue + green))
    return {
        "ndwi": np.nan_to_num(ndwi, nan=0.0),
        "fmpi": np.nan_to_num(fmpi, nan=0.0)
    }


def classify_pollution(fmpi, ndwi):
    water_mask = ndwi > PARAMS["water_threshold"]
    masked_fmpi = np.where(water_mask, fmpi, 0)

    classified = np.zeros_like(masked_fmpi, dtype=np.uint8)
    thresholds = PARAMS["risk_thresholds"]

    classified[(masked_fmpi > thresholds["low"][0]) &
               (masked_fmpi <= thresholds["low"][1])] = 1
    classified[(masked_fmpi > thresholds["medium"][0]) &
               (masked_fmpi <= thresholds["medium"][1])] = 2
    classified[masked_fmpi > thresholds["high"][0]] = 3

    return classified


# VISUALIZATION
def create_plot(data, title, filename, cmap="viridis", colorbar=True):
    """Generate and save visualization plots with improved settings"""
    plt.figure(figsize=(12, 10), dpi=300)

    # Debug information
    print(f"\nCreating {filename}:")
    print(f"- Data range: {np.nanmin(data)} to {np.nanmax(data)}")
    print(f"- Unique values: {np.unique(data)}")

    if isinstance(cmap, ListedColormap):
        # For classification plots
        masked = ma.masked_where(data == 0, data)
        img = plt.imshow(masked, cmap=cmap, interpolation='nearest',
                         vmin=1, vmax=3)  # Ensure correct value range

        if colorbar:
            cbar = plt.colorbar(img, ticks=[1, 2, 3], fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
            cbar.set_label('Pollution Risk Level', rotation=270, labelpad=15)
    else:
        # For continuous data plots
        img = plt.imshow(data, cmap=cmap)
        if colorbar:
            cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
            cbar.set_label('Value', rotation=270, labelpad=15)

    plt.title(title, fontsize=14, pad=20)
    plt.axis("off")

    # Save with tight layout
    output_path = os.path.join(PATHS["output"], filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Saved visualization to: {output_path}")


# ANALYSIS PIPELINE
def analyze_scene(scene_dir, scene_name, reference_meta=None):
    print(f"\n{'=' * 50}")
    print(f"PROCESSING: {scene_name.upper()} SCENE")
    print(f"Directory: {scene_dir}")

    # Load all bands with consistent sizing
    print("\nLoading and resizing bands...")
    blue, meta = load_and_resize_band("blue", scene_dir, reference_meta)
    green, _ = load_and_resize_band("green", scene_dir, meta)
    red, _ = load_and_resize_band("red", scene_dir, meta)
    nir, _ = load_and_resize_band("nir", scene_dir, meta)
    swir1, _ = load_and_resize_band("swir1", scene_dir, meta)

    # Compute indices
    print("Calculating indices...")
    indices = compute_indices(green, nir, blue, swir1)

    # Classify pollution
    print("Classifying pollution levels...")
    classified = classify_pollution(indices["fmpi"], indices["ndwi"])

    # Generate visualizations
    print("Creating visualizations...")
    colors = ['black', '#56b1f7', '#f7c842', '#e73030']  # Black for non-water, then blue->yellow->red
    cmap_custom = ListedColormap(colors)

    create_plot(indices["ndwi"], f"NDWI - {scene_name}", f"ndwi_{scene_name}.png", "Blues")
    create_plot(indices["fmpi"], f"FMPI - {scene_name}", f"fmpi_{scene_name}.png", "inferno")
    create_plot(classified, f"Microplastic Risk - {scene_name}",
                f"risk_{scene_name}.png", cmap_custom)

    # Calculate statistics
    stats = {
        "low_risk": np.sum(classified == 1),
        "medium_risk": np.sum(classified == 2),
        "high_risk": np.sum(classified == 3),
        "total_water": np.sum(indices["ndwi"] > PARAMS["water_threshold"])
    }

    print("\nAnalysis complete!")
    return {"classified": classified, "stats": stats, "indices": indices, "meta": meta}


def compare_results(pre, post):
    """Compare pre and post Kumbh results with size validation"""
    print("\nCOMPARING RESULTS...")

    # Verify shapes match
    if pre["classified"].shape != post["classified"].shape:
        raise ValueError(f"Shape mismatch: pre {pre['classified'].shape} vs post {post['classified'].shape}")

    # Calculate change detection with enhanced method
    change = post["classified"] - pre["classified"]
    increased = (change > 0).astype(float)  # Use float for better visualization

    # Debug information
    print(f"Change detection stats:")
    print(f"- Pixels with increased risk: {np.sum(increased > 0)}")
    print(f"- Change values distribution: {np.unique(change, return_counts=True)}")

    # Create comparison plot with improved settings
    create_plot(increased, "Areas of Increased Microplastic Pollution",
                "pollution_increase.png", "RdYlGn_r")  # Reversed Red-Yellow-Green colormap

    # Generate statistics
    comparison_stats = {
        "new_high_risk": np.sum((pre["classified"] < 3) & (post["classified"] == 3)),
        "total_increase": np.sum(increased),
        "percent_change": round((post["stats"]["high_risk"] - pre["stats"]["high_risk"]) /
                                max(1, pre["stats"]["total_water"]) * 100, 2)  # Avoid division by zero
    }

    return comparison_stats


# MAIN METHOD
if __name__ == "__main__":
    os.makedirs(PATHS["output"], exist_ok=True)

    try:
        # Process pre-Kumbh scene first (will set reference size)
        print("\nStarting Pre-Kumbh analysis...")
        pre = analyze_scene(PATHS["pre_kumbh"], "pre_kumbh")

        # Process post-Kumbh scene using pre-Kumbh as reference
        print("\nStarting Post-Kumbh analysis...")
        post = analyze_scene(PATHS["post_kumbh"], "post_kumbh", pre["meta"])

        # Compare results
        print("\nComparing results...")
        comparison = compare_results(pre, post)

        # Generate final report
        print("\n\nFINAL RESULTS:")
        print(f"Pre-Kumbh High Risk Areas: {pre['stats']['high_risk']} pixels")
        print(f"Post-Kumbh High Risk Areas: {post['stats']['high_risk']} pixels")
        print(f"New High Risk Areas Identified: {comparison['new_high_risk']} pixels")
        print(f"Percentage Increase: {comparison['percent_change']}%")

        print(f"\nAll results saved to: {PATHS['output']}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Processing failed. Please check:")
        print("1. All required band files exist in both directories")
        print("2. Files follow the expected naming pattern")
        print("3. Images cover the same geographic area")
        print("4. Output directory is accessible")