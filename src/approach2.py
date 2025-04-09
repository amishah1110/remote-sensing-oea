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
    "pre_kumbh": r"D:\gdrive\RS OEA\remote-sensing-oea\pre_kumbh",
    "post_kumbh": r"D:\gdrive\RS OEA\remote-sensing-oea\post_kumbh",
    "output": r"D:\gdrive\RS OEA\remote-sensing-oea\results"
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
        "low": (0.05, 0.15),
        "medium": (0.15, 0.25),
        "high": (0.25, 1.0)
    }
}


# ==================== ENHANCED CORE FUNCTIONS ====================
def load_and_resize_band(band_key, scene_dir, reference_meta=None):
    matches = glob.glob(os.path.join(scene_dir, BAND_PATTERNS[band_key]))
    if not matches:
        available = [f for f in os.listdir(scene_dir) if f.endswith('.TIF')]
        raise FileNotFoundError(f"No {band_key} band found. Available files:\n - " + "\n - ".join(available))

    file_path = matches[0]
    print(f"Loading {band_key} band from: {os.path.basename(file_path)}")

    with rasterio.open(file_path) as src:
        if reference_meta is None:
            return src.read(1).astype(np.float32), src.meta
        else:
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


def classify_pollution(ndwi, fmpi):
    water_mask = ndwi > -0.1

    low = (fmpi > 0.05) & (fmpi <= 0.1)
    medium = (fmpi > 0.1) & (fmpi <= 0.2)
    high = fmpi > 0.2

    classified = np.zeros_like(fmpi, dtype=np.uint8)
    classified[water_mask & low] = 1
    classified[water_mask & medium] = 2
    classified[water_mask & high] = 3

    return classified


# ==================== VISUALIZATION ====================
def create_plot(data, title, filename, cmap='viridis'):
    plt.figure(figsize=(10, 8), dpi=120)
    plt.title(title, fontsize=14)

    if isinstance(cmap, ListedColormap):
        masked = ma.masked_where(data == 0, data)
        plt.imshow(masked, cmap=cmap)
        cbar = plt.colorbar(shrink=0.7)
        cbar.set_ticks([1, 2, 3])
        cbar.set_ticklabels(['Low', 'Medium', 'High'])
    else:
        # Contrast enhancement for continuous data
        p2, p98 = np.percentile(data[data > 0], (2, 98))
        plt.imshow(np.clip(data, p2, p98), cmap=cmap)
        plt.colorbar(shrink=0.7)

    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(PATHS['output'], filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved plot: {filename}")



# ==================== ANALYSIS PIPELINE ====================
def analyze_scene(scene_dir, scene_name, reference_meta=None):
    print(f"\n{'=' * 50}")
    print(f"PROCESSING: {scene_name.upper()} SCENE")
    print(f"Directory: {scene_dir}")

    print("\nLoading and resizing bands...")
    blue, meta = load_and_resize_band("blue", scene_dir, reference_meta)
    green, _ = load_and_resize_band("green", scene_dir, meta)
    red, _ = load_and_resize_band("red", scene_dir, meta)
    nir, _ = load_and_resize_band("nir", scene_dir, meta)
    swir1, _ = load_and_resize_band("swir1", scene_dir, meta)

    print("Calculating indices...")
    indices = compute_indices(green, nir, blue, swir1)

    print("Classifying pollution levels...")
    classified = classify_pollution(indices["ndwi"], indices["fmpi"])

    print("Creating visualizations...")
    colors = ['black', '#56b1f7', '#f7c842', '#e73030']
    cmap_custom = ListedColormap(colors)

    create_plot(indices["ndwi"], f"NDWI - {scene_name}", f"ndwi_{scene_name}.png", "Blues")
    create_plot(indices["fmpi"], f"FMPI - {scene_name}", f"fmpi_{scene_name}.png", "plasma")
    create_plot(classified, f"Microplastic Risk - {scene_name}", f"risk_{scene_name}.png", cmap_custom)

    stats = {
        "low_risk": np.sum(classified == 1),
        "medium_risk": np.sum(classified == 2),
        "high_risk": np.sum(classified == 3),
        "total_water": np.sum(indices["ndwi"] > PARAMS["water_threshold"])
    }

    print("\nAnalysis complete!")
    return {"classified": classified, "stats": stats, "indices": indices, "meta": meta}


def compare_results(pre, post):
    print("\nCOMPARING RESULTS...")

    if pre["classified"].shape != post["classified"].shape:
        raise ValueError(f"Shape mismatch: pre {pre['classified'].shape} vs post {post['classified'].shape}")

    change = post["classified"] - pre["classified"]
    increased = (change > 0).astype(int)

    create_plot(increased, "Areas of Increased Microplastic Pollution", "pollution_increase.png", "coolwarm")

    comparison_stats = {
        "new_high_risk": np.sum((pre["classified"] < 3) & (post["classified"] == 3)),
        "total_increase": np.sum(increased),
        "percent_change": round((post["stats"]["high_risk"] - pre["stats"]["high_risk"]) /
                                pre["stats"]["total_water"] * 100, 2)
    }

    return comparison_stats


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    os.makedirs(PATHS["output"], exist_ok=True)

    try:
        pre = analyze_scene(PATHS["pre_kumbh"], "pre_kumbh")
        post = analyze_scene(PATHS["post_kumbh"], "post_kumbh", pre["meta"])
        comparison = compare_results(pre, post)
        # ==================== RANDOM FOREST CLASSIFICATION ====================
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        print("\nTRAINING RANDOM FOREST ON PRE-KUMBH DATA...")

        # Prepare training data (flattened)
        X_train = np.stack((
            pre["indices"]["ndwi"].flatten(),
            pre["indices"]["fmpi"].flatten()
        ), axis=1)
        y_train = pre["classified"].flatten()

        # Filter out non-water or non-risk areas (label 0)
        mask = y_train > 0
        X_train = X_train[mask]
        y_train = y_train[mask]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("Predicting on POST-KUMBH data...")
        X_test = np.stack((
            post["indices"]["ndwi"].flatten(),
            post["indices"]["fmpi"].flatten()
        ), axis=1)
        y_test = post["classified"].flatten()
        mask_test = y_test > 0
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]

        y_pred = model.predict(X_test)

        print("\n=== RANDOM FOREST EVALUATION ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))


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
