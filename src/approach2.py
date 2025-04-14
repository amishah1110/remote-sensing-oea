import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy.ma as ma
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    "swir1": "*B6.TIF",
    "thermal": "*B10.TIF"  # Landsat thermal band
}

PARAMS = {
    "target_resolution": 30,  # meters
    "water_threshold": 0.2,
    "risk_thresholds": {
        "low": (0.05, 0.15),
        "medium": (0.15, 0.25),
        "high": (0.25, 1.0)
    },
    "weights": {
        "fmpi": 0.6,
        "turbidity": 0.3,
        "temperature": 0.1
    }
}

def create_turbidity_colormap():
    """Create a custom colormap for turbidity visualization"""
    colors = ["#8B4513", "#D2B48C", "#F5DEB3",  # Land colors (browns)
              "#0066CC", "#0000FF", "#FF0000"]   # Water colors (blue to red)
    return LinearSegmentedColormap.from_list("turbidity", colors)

def load_and_resize_band(band_key, scene_dir, reference_meta=None):
    matches = glob.glob(os.path.join(scene_dir, BAND_PATTERNS[band_key]))
    if not matches:
        if band_key != "thermal":  # Thermal is optional
            available = [f for f in os.listdir(scene_dir) if f.endswith('.TIF')]
            raise FileNotFoundError(f"No {band_key} band found. Available files:\n - " + "\n - ".join(available))
        return None, None

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

def compute_indices(green, nir, blue, swir1, red=None, thermal=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        # Water and microplastic indices
        ndwi = (green - nir) / (green + nir)
        fmpi = (swir1 - (blue + green)) / (swir1 + (blue + green))

        # Additional environmental parameters
        results = {
            "ndwi": np.nan_to_num(ndwi, nan=0.0),
            "fmpi": np.nan_to_num(fmpi, nan=0.0)
        }

        if red is not None:
            # Turbidity Index (Normalized Difference Turbidity Index)
            ndti = (red - green) / (red + green)
            results["turbidity"] = np.nan_to_num(ndti, nan=0.0)

        if thermal is not None:
            # Convert thermal band to temperature in °C (Landsat specific)
            temperature = thermal * 0.1  # Scale factor for Landsat
            results["temperature"] = temperature

    return results

def calculate_composite_risk(indices, weights):
    """Calculate integrated risk score using weighted parameters"""
    # Normalize all inputs to 0-1 range
    fmpi_norm = (indices["fmpi"] - np.min(indices["fmpi"])) / (np.max(indices["fmpi"]) - np.min(indices["fmpi"]))
    turb_norm = (indices["turbidity"] - np.min(indices["turbidity"])) / (
            np.max(indices["turbidity"]) - np.min(indices["turbidity"]))
    temp_norm = (indices["temperature"] - np.min(indices["temperature"])) / (
            np.max(indices["temperature"]) - np.min(indices["temperature"]))

    return (weights["fmpi"] * fmpi_norm +
            weights["turbidity"] * turb_norm +
            weights["temperature"] * temp_norm)

def classify_pollution(composite_risk, ndwi):
    water_mask = ndwi > -0.1  # Relaxed water mask

    # Classify based on composite risk
    low = (composite_risk > 0.1) & (composite_risk <= 0.3)
    medium = (composite_risk > 0.3) & (composite_risk <= 0.6)
    high = composite_risk > 0.6

    classified = np.zeros_like(composite_risk, dtype=np.uint8)
    classified[water_mask & low] = 1
    classified[water_mask & medium] = 2
    classified[water_mask & high] = 3

    return classified

def plot_risk_distribution_percent_change(pre_stats, post_stats, output_dir):
    """Plot percentage change in risk levels between pre and post scenes"""
    labels = ['Low', 'Medium', 'High']
    pre = np.array([pre_stats['low_risk'], pre_stats['medium_risk'], pre_stats['high_risk']])
    post = np.array([post_stats['low_risk'], post_stats['medium_risk'], post_stats['high_risk']])
    percent_change = ((post - pre) / (pre + 1e-6)) * 100

    plt.figure(figsize=(8, 5), dpi=120)
    bars = plt.bar(labels, percent_change, color=['#56b1f7', '#f7c842', '#e73030'])
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.ylabel('Percentage Change (%)')
    plt.title('Percentage Change in Microplastic Risk Levels (Post vs Pre)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_percentage_change.png"))
    plt.close()


def analyze_band_correlations_ndwi(band_data, ndwi, output_dir):
    """Analyze correlation between bands and NDWI"""
    from scipy.stats import pearsonr

    band_names = []
    correlation_values = []

    for band_key, band_array in band_data.items():
        flat_band = band_array.flatten()
        flat_ndwi = ndwi.flatten()
        valid_mask = ~np.isnan(flat_band) & ~np.isnan(flat_ndwi)
        corr, _ = pearsonr(flat_band[valid_mask], flat_ndwi[valid_mask])
        band_names.append(band_key.upper())
        correlation_values.append(corr)

    plt.figure(figsize=(8, 5), dpi=120)
    bars = plt.bar(band_names, correlation_values, color='teal')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylabel('Pearson Correlation')
    plt.title('Correlation of Bands with NDWI')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "band_ndwi_correlation.png"))
    plt.close()


def analyze_band_correlations_fmpi(band_data, fmpi, output_dir):
    """Analyze correlation between bands and FMPI"""
    from scipy.stats import pearsonr

    band_names = []
    correlation_values = []

    for band_key, band_array in band_data.items():
        flat_band = band_array.flatten()
        flat_fmpi = fmpi.flatten()
        valid_mask = ~np.isnan(flat_band) & ~np.isnan(flat_fmpi)
        corr, _ = pearsonr(flat_band[valid_mask], flat_fmpi[valid_mask])
        band_names.append(band_key.upper())
        correlation_values.append(corr)

    plt.figure(figsize=(8, 5), dpi=120)
    bars = plt.bar(band_names, correlation_values, color='darkorange')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylabel('Pearson Correlation')
    plt.title('Correlation of Bands with FMPI')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "band_fmpi_correlation.png"))
    plt.close()



def save_visualization(data, title, filename, cmap="viridis", is_classified=False, vmin=None, vmax=None):
    plt.figure(figsize=(12, 10), dpi=300)

    if isinstance(cmap, str) and cmap == "turbidity":
        cmap = create_turbidity_colormap()
        # Special scaling for turbidity
        water_mask = data > -0.5  # Simple water/land separation
        water_values = data[water_mask]
        if len(water_values) > 0:
            vmin = np.percentile(water_values, 2)
            vmax = np.percentile(water_values, 98)

    if is_classified:
        masked = ma.masked_where(data == 0, data)
        img = plt.imshow(masked, cmap=cmap, interpolation='nearest', vmin=1, vmax=3)
        cbar = plt.colorbar(img, ticks=[1, 2, 3], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
    else:
        img = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
        if cmap == "turbidity":
            cbar.set_label('Turbidity Index (Land-Water)', rotation=270, labelpad=15)

    plt.title(title, fontsize=14, pad=20)
    plt.axis("off")

    png_path = os.path.join(PATHS["output"], filename)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Saved visualization: {png_path}")

def analyze_scene(scene_dir, scene_name, reference_meta=None):
    print(f"\n{'=' * 50}")
    print(f"PROCESSING: {scene_name.upper()} SCENE")

    # Load all bands
    blue, meta = load_and_resize_band("blue", scene_dir, reference_meta)
    green, _ = load_and_resize_band("green", scene_dir, meta)
    red, _ = load_and_resize_band("red", scene_dir, meta)
    nir, _ = load_and_resize_band("nir", scene_dir, meta)
    swir1, _ = load_and_resize_band("swir1", scene_dir, meta)
    thermal, _ = load_and_resize_band("thermal", scene_dir, meta)  # May be None

    # Compute indices
    indices = compute_indices(green, nir, blue, swir1, red, thermal)

    # Calculate composite risk if we have all parameters
    if "turbidity" in indices and "temperature" in indices:
        composite_risk = calculate_composite_risk(indices, PARAMS["weights"])
        indices["composite_risk"] = composite_risk
        classified = classify_pollution(composite_risk, indices["ndwi"])
    else:
        classified = classify_pollution(indices["fmpi"], indices["ndwi"])

    # Analyze band correlations
    print("Analyzing band correlations with NDWI...")
    band_data = {"blue": blue, "green": green, "red": red, "nir": nir, "swir1": swir1}
    analyze_band_correlations_ndwi(band_data, indices["ndwi"], PATHS["output"])

    print("Analyzing band correlations with FMPI...")
    analyze_band_correlations_fmpi(band_data, indices["fmpi"], PATHS["output"])

    # Generate visualizations
    colors = ['black', '#56b1f7', '#f7c842', '#e73030']
    cmap_custom = ListedColormap(colors)

    save_visualization(indices["ndwi"], f"NDWI - {scene_name}", f"ndwi_{scene_name}.png", "Blues")
    save_visualization(indices["fmpi"], f"FMPI - {scene_name}", f"fmpi_{scene_name}.png", "inferno")

    if "turbidity" in indices:
        save_visualization(indices["turbidity"], f"Turbidity - {scene_name}",
                          f"turbidity_{scene_name}.png", "turbidity")
    if "temperature" in indices:
        save_visualization(indices["temperature"], f"Temperature - {scene_name}",
                          f"temperature_{scene_name}.png", "hot")
    if "composite_risk" in indices:
        save_visualization(composite_risk, f"Composite Risk - {scene_name}",
                          f"composite_risk_{scene_name}.png", "RdYlGn_r")

    save_visualization(classified, f"Microplastic Risk - {scene_name}",
                      f"risk_{scene_name}.png", cmap_custom, is_classified=True)

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
    """Compare results and save visualization"""
    print("\nCOMPARING RESULTS...")
    if pre["classified"].shape != post["classified"].shape:
        raise ValueError(f"Shape mismatch: pre {pre['classified'].shape} vs post {post['classified'].shape}")

    change = post["classified"] - pre["classified"]
    increased = (change > 0).astype(float)

    # Save visualization
    save_visualization(increased, "Areas of Increased Microplastic Pollution",
                      "pollution_increase.png", "RdYlGn_r")

    return {
        "new_high_risk": np.sum((pre["classified"] < 3) & (post["classified"] == 3)),
        "total_increase": np.sum(increased),
        "percent_change": round((post["stats"]["high_risk"] - pre["stats"]["high_risk"]) /
                              max(1, pre["stats"]["total_water"]) * 100, 2)
    }

if __name__ == "__main__":
    os.makedirs(PATHS["output"], exist_ok=True)

    try:
        print("\nStarting Pre-Kumbh analysis...")
        pre = analyze_scene(PATHS["pre_kumbh"], "pre_kumbh")

        print("\nStarting Post-Kumbh analysis...")
        post = analyze_scene(PATHS["post_kumbh"], "post_kumbh", pre["meta"])

        print("\nComparing results...")
        comparison = compare_results(pre, post)

        # Machine Learning Evaluation
        print("\nTRAINING RANDOM FOREST MODEL...")
        X_train = np.stack((
            pre["indices"]["ndwi"].flatten(),
            pre["indices"]["fmpi"].flatten(),
            pre["indices"].get("turbidity", np.zeros_like(pre["indices"]["ndwi"])).flatten(),
            pre["indices"].get("temperature", np.zeros_like(pre["indices"]["ndwi"])).flatten()
        ), axis=1)
        y_train = pre["classified"].flatten()
        mask = y_train > 0
        X_train, y_train = X_train[mask], y_train[mask]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.stack((
            post["indices"]["ndwi"].flatten(),
            post["indices"]["fmpi"].flatten(),
            post["indices"].get("turbidity", np.zeros_like(post["indices"]["ndwi"])).flatten(),
            post["indices"].get("temperature", np.zeros_like(post["indices"]["ndwi"])).flatten()
        ), axis=1)
        y_test = post["classified"].flatten()
        mask_test = y_test > 0
        X_test, y_test = X_test[mask_test], y_test[mask_test]

        y_pred = model.predict(X_test)
        print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Final report
        print("\nFINAL RESULTS:")
        print(f"Pre-Kumbh High Risk Areas: {pre['stats']['high_risk']} pixels")
        print(f"Post-Kumbh High Risk Areas: {post['stats']['high_risk']} pixels")
        print(f"New High Risk Areas: {comparison['new_high_risk']} pixels")
        print(f"Percentage Increase: {comparison['percent_change']}%")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
