import streamlit as st
from approach2 import analyze_scene, compare_results, PATHS
import os
from PIL import Image

# Set up the Streamlit interface
st.set_page_config(
    page_title="Microplastic Pollution Detection",
    layout="centered",
    page_icon="🛰️"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .header-style {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subheader-style {
        font-size: 20px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<p class="header-style">🛰️ Microplastic Pollution Detection from Remote Sensing Data</p>',
            unsafe_allow_html=True)
st.markdown("""
<p style="font-size: 16px; color: #555;">
    Analyze and compare pre- and post-Kumbh satellite imagery to detect microplastic pollution risks.
    Click the button below to run the analysis pipeline.
</p>
""", unsafe_allow_html=True)

# Button to run the analysis pipeline
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Run Analysis", key="run_analysis"):
        with st.spinner("Processing scenes... This may take a few moments."):
            try:
                pre = analyze_scene(PATHS["pre_kumbh"], "pre_kumbh")
                post = analyze_scene(PATHS["post_kumbh"], "post_kumbh", pre["meta"])
                comparison = compare_results(pre, post)

                st.markdown('<div class="success-box">✅ Analysis complete! Results are shown below.</div>',
                            unsafe_allow_html=True)

                # Show statistics in a card layout
                st.markdown('<p class="subheader-style">📊 Summary Results</p>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Pre-Kumbh</h3>
                        <p style="font-size: 24px; font-weight: bold;">{pre['stats']['high_risk']}</p>
                        <p>High Risk Pixels</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Post-Kumbh</h3>
                        <p style="font-size: 24px; font-weight: bold;">{post['stats']['high_risk']}</p>
                        <p>High Risk Pixels</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>New Areas</h3>
                        <p style="font-size: 24px; font-weight: bold;">{comparison['new_high_risk']}</p>
                        <p>High Risk Pixels</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Change</h3>
                        <p style="font-size: 24px; font-weight: bold;">{comparison['percent_change']}%</p>
                        <p>Increase</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Display result images with tabs for better organization
                st.markdown('<p class="subheader-style">🖼️ Visual Analysis</p>', unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs(["Pre-Kumbh", "Post-Kumbh", "Comparison"])

                with tab1:
                    cols = st.columns(3)
                    for i, filename in enumerate(["ndwi_pre_kumbh.png", "fmpi_pre_kumbh.png", "risk_pre_kumbh.png"]):
                        image_path = os.path.join(PATHS["output"], filename)
                        if os.path.exists(image_path):
                            with cols[i]:
                                st.image(
                                    Image.open(image_path),
                                    caption=filename.replace("_", " ").replace(".png", "").title(),
                                    use_column_width=True
                                )
                        else:
                            st.warning(f"Missing: {filename}")

                with tab2:
                    cols = st.columns(3)
                    for i, filename in enumerate(["ndwi_post_kumbh.png", "fmpi_post_kumbh.png", "risk_post_kumbh.png"]):
                        image_path = os.path.join(PATHS["output"], filename)
                        if os.path.exists(image_path):
                            with cols[i]:
                                st.image(
                                    Image.open(image_path),
                                    caption=filename.replace("_", " ").replace(".png", "").title(),
                                    use_column_width=True
                                )
                        else:
                            st.warning(f"Missing: {filename}")

                with tab3:
                    image_path = os.path.join(PATHS["output"], "pollution_increase.png")
                    if os.path.exists(image_path):
                        st.image(
                            Image.open(image_path),
                            caption="Pollution Increase Areas",
                            use_column_width=True
                        )
                    else:
                        st.warning("Missing: pollution_increase.png")

            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #777; font-size: 14px;">
    Microplastic Pollution Detection System • Powered by Remote Sensing Data Analysis
</p>
""", unsafe_allow_html=True)