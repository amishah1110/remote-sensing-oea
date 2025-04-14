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
    "B6": {"name": "SWIR1", "wavelength": "1.57-1.65μm", "color": "gray"}
}


def get_band_stats(data, band_key):
    """Calculate statistics for a band using chunking"""
    band_info = BAND_INFO[band_key]
    flat_data = data.flatten()
    sample_size = min(100000, len(flat_data))  # Use subset for statistics
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
    """Plot histograms for all bands using downsampling"""
    plt.figure(figsize=(15, 8))
    for band_key, data in bands.items():
        band_info = BAND_INFO[band_key]
        sample = data[::10, ::10]  # Downsample by factor of 10
        plt.hist(sample.flatten(), bins=100, alpha=0.5,
                 label=f"{band_info['name']} ({band_key})",
                 color=band_info['color'])

    plt.title('Pixel Value Distribution (Downsampled)', fontsize=14)
    plt.xlabel('Reflectance Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'band_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()


def process_band(band_key, scene_dir):
    """Process a single band with memory efficiency"""
    band_file = glob.glob(os.path.join(scene_dir, f"*{band_key}.TIF"))[0]
    with rasterio.open(band_file) as src:
        # Read in chunks if the image is large
        if src.width * src.height > 10000000:  # 10 million pixels
            data = np.zeros((src.height, src.width), dtype=np.float32)
            for ji, window in src.block_windows(1):
                data[window.row_off:window.row_off + window.height,
                window.col_off:window.col_off + window.width] = src.read(1, window=window)
        else:
            data = src.read(1).astype(np.float32)
    return data


def analyze_correlations(bands, output_dir):
    """Calculate and plot band correlations using sampling"""
    # Prepare sampled data
    sample_size = min(10000, bands['B2'].size)  # Use smaller sample size
    idx = np.random.choice(bands['B2'].size, sample_size, replace=False)

    sampled_data = {b: bands[b].flatten()[idx] for b in bands}
    band_names = [BAND_INFO[b]['name'] for b in bands]

    # Calculate correlation matrix
    corr_matrix = np.zeros((len(bands), len(bands)))
    for i, b1 in enumerate(bands):
        for j, b2 in enumerate(bands):
            corr_matrix[i, j] = pearsonr(sampled_data[b1], sampled_data[b2])[0]

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=band_names, yticklabels=band_names,
                vmin=-1, vmax=1, square=True)
    plt.title('Inter-band Correlation Matrix (Sampled)', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'band_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return corr_matrix


def main():
    # Create output directory
    os.makedirs(PATHS["output"], exist_ok=True)

    try:
        # Load and process bands one at a time
        print("Loading and processing bands...")
        bands = {}
        stats = []

        for band_key in tqdm(BAND_INFO.keys(), desc="Processing bands"):
            # Process band
            data = process_band(band_key, PATHS["pre_kumbh"])

            # Calculate statistics
            stats.append(get_band_stats(data, band_key))

            # Store band data (clear previous band if memory is tight)
            bands[band_key] = data
            if len(bands) > 1:  # Keep only current and previous band in memory
                del bands[list(bands.keys())[0]]
                gc.collect()

        # Combine all bands for final analysis
        bands = {b: process_band(b, PATHS["pre_kumbh"]) for b in BAND_INFO.keys()}

        # Perform analyses
        print("\nAnalyzing band distributions...")
        plot_band_histograms(bands, PATHS["output"])

        stats_df = pd.DataFrame(stats).set_index('Band')
        stats_df.to_csv(os.path.join(PATHS["output"], 'band_statistics.csv'))

        print("Creating band visualizations...")
        # Plot only a subset of each band for visualization
        for band_key in bands:
            band_info = BAND_INFO[band_key]
            data = bands[band_key]

            plt.figure(figsize=(10, 8))
            plt.imshow(data[::4, ::4], cmap='gray')  # Downsample by 4
            plt.title(f"{band_info['name']} Band ({band_key})\n{band_info['wavelength']}", fontsize=14)
            plt.colorbar(label='Reflectance Value')
            plt.axis('off')

            vmin, vmax = np.percentile(data, [2, 98])
            plt.clim(vmin, vmax)

            plt.savefig(os.path.join(PATHS["output"], f'band_{band_key}_image.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        print("Calculating band correlations...")
        corr_matrix = analyze_correlations(bands, PATHS["output"])

        print("\nAnalysis complete! Results saved to:", PATHS["output"])

        # Print key findings
        print("\nKEY FINDINGS:")
        print("1. Band Statistics:")
        print(stats_df)

        print("\n2. Recommended Band Combinations:")
        print("NDWI: Green (B3) + NIR (B5) - Highest water/land contrast")
        print("FMPI: SWIR1 (B6) + (Blue (B2) + Green (B3)) - Best for floating material detection")

    except Exception as e:
        print(f"Error: {str(e)}")


if _name_ == "_main_":
    import glob

    main()