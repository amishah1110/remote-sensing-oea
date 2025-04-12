import streamlit as st
from approach2 import analyze_scene, compare_results, PATHS
import os
from PIL import Image
import pandas as pd
import numpy as np
import rasterio
import glob
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import gc

# Set up the Streamlit interface
st.set_page_config(page_title="Microplastic Pollution Detection", layout="centered")

st.title("🛰️ Microplastic Pollution Detection from Remote Sensing Data")
st.markdown("Analyze and compare pre- and post-Kumbh satellite imagery to detect microplastic risks.")

# Band information
BAND_INFO = {
    "B2": {"name": "Blue", "wavelength": "0.45-0.51μm", "color": "blue"},
    "B3": {"name": "Green", "wavelength": "0.53-0.59μm", "color": "green"},
    "B4": {"name": "Red", "wavelength": "0.64-0.67μm", "color": "red"},
    "B5": {"name": "NIR", "wavelength": "0.85-0.88μm", "color": "darkred"},
    "B6": {"name": "SWIR1", "wavelength": "1.57-1.65μm", "color": "gray"},
    "B10": {"name": "Thermal", "wavelength": "10.6-11.2μm", "color": "magenta"}
}


def load_band(band_key, scene_dir):
    """Load a single band with memory efficiency"""
    band_file = glob.glob(os.path.join(scene_dir, f"*{band_key}.TIF"))[0]
    with rasterio.open(band_file) as src:
        if src.width * src.height > 10000000:  # Large image - read in chunks
            data = np.zeros((src.height, src.width), dtype=np.float32)
            for ji, window in src.block_windows(1):
                data[window.row_off:window.row_off + window.height,
                window.col_off:window.col_off + window.width] = src.read(1, window=window)
        else:
            data = src.read(1).astype(np.float32)
    return data


def run_band_analysis(scene_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load and process bands
    bands = {}
    stats = []

    for band_key in BAND_INFO.keys():
        data = load_band(band_key, scene_dir)
        bands[band_key] = data

        # Calculate statistics
        sample = data[::10, ::10].flatten()  # Downsample for stats
        stats.append({
            'Band': BAND_INFO[band_key]['name'],
            'Mean': np.mean(sample),
            'Std': np.std(sample),
            'Min': np.min(sample),
            '25%': np.percentile(sample, 25),
            '50%': np.percentile(sample, 50),
            '75%': np.percentile(sample, 75),
            'Max': np.max(sample)
        })

    # Save statistics
    stats_df = pd.DataFrame(stats).set_index('Band')
    stats_df.to_csv(os.path.join(output_dir, 'band_statistics.csv'))

    # Create visualizations
    create_band_visualizations(bands, output_dir)
    return bands, stats_df


def create_band_visualizations(bands, output_dir):
    """Generate all band analysis plots"""
    # Histograms
    plt.figure(figsize=(15, 8))
    for band_key, data in bands.items():
        sample = data[::10, ::10].flatten()  # Downsample
        plt.hist(sample, bins=100, alpha=0.5,
                 label=f"{BAND_INFO[band_key]['name']} ({band_key})",
                 color=BAND_INFO[band_key]['color'])
    plt.title('Pixel Value Distribution Across Bands', fontsize=14)
    plt.xlabel('Reflectance Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'band_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Band images
    for band_key, data in bands.items():
        plt.figure(figsize=(10, 8))
        plt.imshow(data[::4, ::4], cmap='gray')  # Downsample
        plt.title(f"{BAND_INFO[band_key]['name']} Band ({band_key})", fontsize=14)
        plt.colorbar(label='Reflectance Value')
        plt.axis('off')
        vmin, vmax = np.percentile(data, [2, 98])
        plt.clim(vmin, vmax)
        plt.savefig(os.path.join(output_dir, f'band_{band_key}_image.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Correlation matrix
    sample_size = min(10000, bands['B2'].size)
    idx = np.random.choice(bands['B2'].size, sample_size, replace=False)
    sampled_data = {b: bands[b].flatten()[idx] for b in bands}

    corr_matrix = np.zeros((len(bands), len(bands)))
    for i, b1 in enumerate(bands):
        for j, b2 in enumerate(bands):
            corr_matrix[i, j] = pearsonr(sampled_data[b1], sampled_data[b2])[0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=[BAND_INFO[b]['name'] for b in bands],
                yticklabels=[BAND_INFO[b]['name'] for b in bands],
                vmin=-1, vmax=1, square=True)
    plt.title('Inter-band Correlation Matrix', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'band_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Tab layout
tab1, tab2 = st.tabs(["🏞️ Band Analysis", "📊 Pollution Detection"])

with tab1:
    st.header("Spectral Band Analysis")
    st.markdown("""
    ### Understanding the Input Bands
    Before calculating indices, we analyze each band's characteristics and suitability for water/microplastic detection.
    """)

    if st.button("🔍 Run Band Analysis", key="band_analysis"):
        with st.spinner("Analyzing spectral bands..."):
            try:
                band_analysis_dir = os.path.join(PATHS["output"], "band_analysis")
                bands, stats_df = run_band_analysis(PATHS["pre_kumbh"], band_analysis_dir)
                st.success("✅ Band analysis complete!")

                # Display results
                st.subheader("📈 Band Statistics")
                st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

                st.markdown("""
                **Conclusions from Statistics:**
                - NIR and SWIR1 show lower mean reflectance (water absorption)
                - Green band shows highest variability (turbidity/sediment influence)
                - Blue band has the narrowest range (consistent water penetration)
                """)

                # Band Visualizations
                st.subheader("🌌 Band Visualizations")
                cols = st.columns(2)
                for i, band_key in enumerate(BAND_INFO.keys()):
                    with cols[i % 2]:
                        img_path = os.path.join(band_analysis_dir, f'band_{band_key}_image.png')
                        st.image(Image.open(img_path),
                                 caption=f"{BAND_INFO[band_key]['name']} Band ({band_key}) - {BAND_INFO[band_key]['wavelength']}")

                # Correlation matrix
                st.subheader("🔄 Band Correlations")
                st.image(Image.open(os.path.join(band_analysis_dir, 'band_correlations.png')))

                st.markdown("""
                **Correlation Insights:**
                - Green and NIR have low correlation → Good for NDWI
                - SWIR1 is least correlated with visible bands → Best for FMPI
                - Red and NIR are highly correlated → Avoid combining in indices
                """)

                # Histograms
                st.subheader("📊 Band Value Distributions")
                st.image(Image.open(os.path.join(band_analysis_dir, 'band_histograms.png')))

                st.markdown("""
                **Histogram Observations:**
                - NIR shows bimodal distribution (clear water/land separation)
                - SWIR1 has long tail (potential floating materials)
                - Visible bands show normal distributions (typical reflectance)
                """)

            except Exception as e:
                st.error(f"❌ Band analysis failed: {str(e)}")

with tab2:
    st.header("Pollution Detection Results")
    st.markdown("""
    ### Microplastic Risk Assessment
    Using the optimal band combinations:
    - **NDWI** = (Green - NIR)/(Green + NIR) for water detection
    - **FMPI** = (SWIR1 - (Blue + Green))/(SWIR1 + (Blue + Green)) for microplastic detection
    """)

    if st.button("🚀 Run Pollution Analysis", key="pollution_analysis"):
        with st.spinner("Processing scenes..."):
            try:
                pre = analyze_scene(PATHS["pre_kumbh"], "pre_kumbh")
                post = analyze_scene(PATHS["post_kumbh"], "post_kumbh", pre["meta"])
                comparison = compare_results(pre, post)

                st.success("✅ Analysis complete!")

                # Show statistics
                st.subheader("📊 Summary Results")
                st.markdown(f"- Pre-Kumbh High Risk Areas: **{pre['stats']['high_risk']} pixels**")
                st.markdown(f"- Post-Kumbh High Risk Areas: **{post['stats']['high_risk']} pixels**")
                st.markdown(f"- New High Risk Areas: **{comparison['new_high_risk']} pixels**")
                st.markdown(f"- Percentage Increase: **{comparison['percent_change']}%**")

                # Display result images
                st.subheader("🖼️ Results Visualization")
                col1, col2 = st.columns(2)

                result_images = [
                    ("NDWI (Pre-Kumbh)", "ndwi_pre_kumbh.png"),
                    ("NDWI (Post-Kumbh)", "ndwi_post_kumbh.png"),
                    ("FMPI (Pre-Kumbh)", "fmpi_pre_kumbh.png"),
                    ("FMPI (Post-Kumbh)", "fmpi_post_kumbh.png"),
                    ("Risk Map (Pre-Kumbh)", "risk_pre_kumbh.png"),
                    ("Risk Map (Post-Kumbh)", "risk_post_kumbh.png"),
                    ("Turbidity Map (Pre-Kumbh", "turbidity_pre_kumbh.png"),
                    ("Turbidity Map (Post-Kumbh", "turbidity_post_kumbh.png"),
                    ("Pollution Increase", "pollution_increase.png")
                ]

                for i, (title, filename) in enumerate(result_images):
                    image_path = os.path.join(PATHS["output"], filename)
                    if os.path.exists(image_path):
                        with (col1 if i % 2 == 0 else col2):
                            st.image(Image.open(image_path),
                                     caption=title,
                                     use_container_width=True)
                    else:
                        st.warning(f"Missing: {filename}")

            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")

st.markdown("---")
