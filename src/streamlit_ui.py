import streamlit as st
from approach2 import analyze_scene, compare_results, PATHS
import os
from PIL import Image

# Set up the Streamlit interface
st.set_page_config(page_title="Microplastic Pollution Detection", layout="centered")

st.title("🛰️ Microplastic Pollution Detection from Remote Sensing Data")
st.markdown("Analyze and compare pre- and post-Kumbh satellite imagery to detect microplastic risks.")

# Button to run the analysis pipeline
if st.button("🚀 Run Analysis"):
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
            st.subheader("🖼️ Visualizations")
            for filename in ["ndwi_pre_kumbh.png", "fmpi_pre_kumbh.png", "risk_pre_kumbh.png",
                             "ndwi_post_kumbh.png", "fmpi_post_kumbh.png", "risk_post_kumbh.png",
                             "pollution_increase.png"]:
                image_path = os.path.join(PATHS["output"], filename)
                if os.path.exists(image_path):
                    st.image(Image.open(image_path), caption=filename.replace("_", " ").replace(".png", "").title(),
                             use_column_width=True)
                else:
                    st.warning(f"Missing: {filename}")

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
