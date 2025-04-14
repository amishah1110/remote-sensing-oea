import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import gc
import glob

# Configuration
PATHS = {
    "pre_kumbh": r"D:\gdrive\RS OEA\remote-sensing-oea\pre_kumbh",
    "output": r"D:\gdrive\RS OEA\remote-sensing-oea\band_analysis"
}

BAND_INFO = {
    "B2": {"name": "Blue", "wavelength": "0.45-0.51μm", "color": "blue"},
    "B3": {"name": "Green", "wavelength": "0.53-0.59μm", "color": "green"},
    "B4": {"name": "Red", "wavelength": "0.64-0.67μm", "color": "red"},
    "B5": {"name": "NIR", "wavelength": "0.85-0.88μm", "color": "darkred"},
    "B6": {"name": "SWIR1", "wavelength": "1.57-1.65μm", "color": "gray"},
    "B10": {"name": "Thermal", "wavelength": "10.6-11.2μm", "color": "magenta"}
}

def get_band_stats(data, band_key):
    band_info = BAND_INFO[band_key]
    flat_data = data.flatten()
    sample_size = min(100000, len(flat_data))
    idx = np.random.choice(len(flat_data), sample_size, replace=False)
    sample = flat_data[idx]

    return {
        'Band': band_info['name'],
        'Mean': np.mean(sample),
        'Std': np.std(sample),
        'Min': np.min(sample),
        '25%': np.percentile(sample, 25),
        '50%': np.percentile(sample, 50),
        '75%': np.percentile(sample, 75),
        'Max': np.max(sample)
    }

def plot_band_histograms(bands, output_dir):
    plt.figure(figsize=(15, 8))
    for band_key, data in bands.items():
        if data is None:
            continue
        band_info = BAND_INFO[band_key]
        sample = data[::10, ::10]
        plt.hist(sample.flatten(), bins=100, alpha=0.5,
                 label=f"{band_info['name']} ({band_key})",
                 color=band_info['color'])

    plt.title('Pixel Value Distribution', fontsize=14)
    plt.xlabel('Digital Number (DN)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'band_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()

def process_band(band_key, scene_dir):
    try:
        band_file = glob.glob(os.path.join(scene_dir, f"*{band_key}.TIF"))[0]
        with rasterio.open(band_file) as src:
            if src.width * src.height > 10000000:
                data = np.zeros((src.height, src.width), dtype=np.float32)
                for ji, window in src.block_windows(1):
                    data[window.row_off:window.row_off + window.height,
                    window.col_off:window.col_off + window.width] = src.read(1, window=window)
            else:
                data = src.read(1).astype(np.float32)
        return data
    except IndexError:
        if band_key == "B10":
            return None
        raise

def analyze_correlations(bands, output_dir):
    valid_bands = {k: v for k, v in bands.items() if v is not None}
    sample_size = min(10000, valid_bands['B2'].size)
    idx = np.random.choice(valid_bands['B2'].size, sample_size, replace=False)

    sampled_data = {b: valid_bands[b].flatten()[idx] for b in valid_bands}
    band_names = [BAND_INFO[b]['name'] for b in valid_bands]

    corr_matrix = np.zeros((len(valid_bands), len(valid_bands)))
    for i, b1 in enumerate(valid_bands):
        for j, b2 in enumerate(valid_bands):
            corr_matrix[i, j] = pearsonr(sampled_data[b1], sampled_data[b2])[0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=band_names, yticklabels=band_names,
                vmin=-1, vmax=1, square=True)
    plt.title('Inter-band Correlation Matrix', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'band_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return corr_matrix

def create_band_images(bands, output_dir):
    for band_key in bands:
        if bands[band_key] is None:
            continue

        band_info = BAND_INFO[band_key]
        data = bands[band_key]

        plt.figure(figsize=(10, 8))
        plt.imshow(data[::4, ::4], cmap='gray')
        plt.title(f"{band_info['name']} Band ({band_key})\n{band_info['wavelength']}", fontsize=14)

        if band_key == "B10":
            plt.colorbar(label='Temperature (°C)')
            vmin, vmax = np.percentile(data, [5, 95])
        else:
            plt.colorbar(label='Reflectance (DN)')
            vmin, vmax = np.percentile(data, [2, 98])

        plt.clim(vmin, vmax)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'band_{band_key}_image.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    os.makedirs(PATHS["output"], exist_ok=True)

    try:
        print("Loading and processing bands...")
        bands = {}
        stats = []

        for band_key in tqdm(BAND_INFO.keys(), desc="Processing bands"):
            data = process_band(band_key, PATHS["pre_kumbh"])
            bands[band_key] = data

            if data is not None:
                stats.append(get_band_stats(data, band_key))

            if len(bands) > 1:
                del bands[list(bands.keys())[0]]
                gc.collect()

        bands = {b: process_band(b, PATHS["pre_kumbh"]) for b in BAND_INFO.keys()}

        print("\nAnalyzing band distributions...")
        plot_band_histograms({k: v for k, v in bands.items() if v is not None}, PATHS["output"])

        stats_df = pd.DataFrame(stats).set_index('Band')
        stats_df.to_csv(os.path.join(PATHS["output"], 'band_statistics.csv'))

        print("Creating band visualizations...")
        create_band_images(bands, PATHS["output"])

        print("Calculating band correlations...")
        analyze_correlations(bands, PATHS["output"])

        print("\nAnalysis complete! Results saved to:", PATHS["output"])

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()