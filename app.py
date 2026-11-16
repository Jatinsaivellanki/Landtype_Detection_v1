"""
🛰️ LandScope AI — Next-Generation Unsupervised Satellite Land-Type Discovery
Designed for non-technical users & remote sensing researchers alike.
EuroSAT Sentinel-2 RGB Imagery • PCA Dimensionality Reduction • K-Means Clustering
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Suppress minor library deprecation/version warnings for clean output
warnings.filterwarnings("ignore")

from features import extract, FEATURE_NAMES

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LandScope AI | Satellite Land-Type Discovery",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CACHED MODEL & ARTIFACT LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Loads the pre-trained unsupervised PCA + K-Means model."""
    model_path = "artifacts/model.joblib"
    if not os.path.exists(model_path):
        st.error(f"Model artifact not found at {model_path}. Please run train.py first.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_study_data():
    """Loads supplementary study metrics, cluster profiles, and validation data."""
    results = {}
    if os.path.exists("artifacts/results.json"):
        with open("artifacts/results.json", "r") as f:
            results = json.load(f)
            
    profiles = pd.read_csv("artifacts/cluster_profiles.csv").set_index("cluster") if os.path.exists("artifacts/cluster_profiles.csv") else pd.DataFrame()
    k_sel = pd.read_csv("artifacts/k_selection.csv") if os.path.exists("artifacts/k_selection.csv") else pd.DataFrame()
    crosstab = pd.read_csv("artifacts/cluster_vs_hidden_labels.csv", index_col=0) if os.path.exists("artifacts/cluster_vs_hidden_labels.csv") else pd.DataFrame()
    
    return results, profiles, k_sel, crosstab

M = load_model()
results_meta, cluster_profiles_df, k_selection_df, crosstab_df = load_study_data()

# -----------------------------------------------------------------------------
# 3. DOMAIN METADATA: HUMAN-FRIENDLY CLUSTER DEFINITIONS
# -----------------------------------------------------------------------------
CLUSTER_METADATA = {
    0: {
        "title": "Grassland & Open Green Terrain",
        "short_name": "Open Greenery",
        "icon": "🌾",
        "color": "#10B981",          # Emerald
        "accent": "rgba(16, 185, 129, 0.2)",
        "summary": "Expansive meadows, grazing pastures, and open vegetated terrain with gentle natural chlorophyll and moderate ground texture.",
        "dominant_traits": ["High natural greenness (ExG)", "Low structural concrete edges", "Gentle surface texture", "Open horizon reflection"],
        "real_world_matches": "Herbaceous Vegetation (Pastures, meadows, open fields, river corridors)",
        "landscape_badge": "Organic • Open • Vegetated"
    },
    1: {
        "title": "Industrial & Urban Built-Up Complex",
        "short_name": "Built-Up / Industrial",
        "icon": "🏭",
        "color": "#F59E0B",          # Amber/Orange
        "accent": "rgba(245, 158, 11, 0.2)",
        "summary": "Human-made infrastructure, commercial warehouses, concrete highways, and residential clusters with high edge sharpness.",
        "dominant_traits": ["High edge sharpness & geometry", "Elevated Laplacian texture roughness", "Strong optical reflectance", "Negligible living vegetation"],
        "real_world_matches": "Industrial facilities, commercial complexes, paved highways & dense buildings",
        "landscape_badge": "Constructed • High Texture • Geometric"
    },
    2: {
        "title": "Dense Forest Canopy & Deep Waters",
        "short_name": "Forest & Deep Canopy",
        "icon": "🌲",
        "color": "#06B6D4",          # Cyan/Teal
        "accent": "rgba(6, 182, 212, 0.2)",
        "summary": "Dense woodland, coniferous/deciduous forests, or deep water bodies exhibiting high optical absorption and uniform dark tones.",
        "dominant_traits": ["Low surface reflectance (dark absorbs light)", "High color purity & saturation", "Extremely low roughness variation", "Dense organic canopy"],
        "real_world_matches": "Natural Forests, closed tree canopies, deep lakes and ocean surfaces",
        "landscape_badge": "Natural • Dense Foliage • Low Reflectance"
    },
    3: {
        "title": "Cultivated Farmland & Agricultural Crops",
        "short_name": "Farmland & Crops",
        "icon": "🚜",
        "color": "#A855F7",          # Purple
        "accent": "rgba(168, 85, 247, 0.2)",
        "summary": "Organized agricultural plots, seasonal crop fields, orchards, and vineyards with distinct furrow patterns and soil-plant contrast.",
        "dominant_traits": ["Strong red-green optical variance", "Repetitive field pattern textures", "High seasonal soil brightness", "Moderate vegetative index"],
        "real_world_matches": "Permanent crops, orchards, vineyards & managed agricultural fields",
        "landscape_badge": "Cultivated • Patterned • High Contrast"
    }
}

SAMPLE_PRESETS = [
    {"name": "Dense Pine Forest", "path": "samples/Forest_Sample.jpg", "category": "Forest", "icon": "🌲", "desc": "Lush evergreen tree canopy"},
    {"name": "Industrial Complex", "path": "samples/Industrial_Sample.jpg", "category": "Industrial", "icon": "🏭", "desc": "Warehouses & transport hubs"},
    {"name": "Agricultural Crop", "path": "samples/Agriculture_Sample.jpg", "category": "Crops", "icon": "🚜", "desc": "Cultivated farmland plots"},
    {"name": "Green Pasture", "path": "samples/Pasture_Meadow_Sample.jpg", "category": "Pasture", "icon": "🌾", "desc": "Open grass and grazing fields"},
    {"name": "Highway Corridor", "path": "samples/Highway_Sample.jpg", "category": "Highway", "icon": "🛣️", "desc": "Asphalt transit network"},
    {"name": "Suburban Homes", "path": "samples/Residential_Sample.jpg", "category": "Residential", "icon": "🏡", "desc": "Neighborhood residential block"},
    {"name": "Winding River", "path": "samples/River_Sample.jpg", "category": "River", "icon": "🏞️", "desc": "Natural waterway corridor"},
    {"name": "Deep Sea / Lake", "path": "samples/Water_SeaLake_Sample.jpg", "category": "Water", "icon": "🌊", "desc": "Deep waterbody surface"}
]

# -----------------------------------------------------------------------------
# 4. CUTTING-EDGE CUSTOM CSS & ANIMATIONS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sleek Dark Cyber-Aesthetic Background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #030712 50%, #050b14 100%);
        color: #f1f5f9;
    }

    /* Keyframe Animations */
    @keyframes laserScan {
        0% { top: 0%; opacity: 0.8; }
        50% { opacity: 1; filter: drop-shadow(0 0 10px #38bdf8); }
        100% { top: 96%; opacity: 0.8; }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.25); border-color: rgba(56, 189, 248, 0.4); }
        50% { box-shadow: 0 0 35px rgba(56, 189, 248, 0.65); border-color: rgba(56, 189, 248, 0.9); }
        100% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.25); border-color: rgba(56, 189, 248, 0.4); }
    }

    @keyframes floatAnimation {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    @keyframes textGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes radarSweep {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #38bdf8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textGradient 8s ease infinite;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
    }

    /* Satellite Telemetry Badge */
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    .beacon-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        display: inline-block;
        animation: pulseGlow 2s infinite;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45);
    }

    /* Interactive Verdict Banner */
    .verdict-box {
        border-radius: 20px;
        padding: 26px 28px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        border: 1.5px solid;
    }

    .verdict-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 14px;
    }

    .verdict-tag {
        font-size: 0.8rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        opacity: 0.85;
    }

    .verdict-name {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .verdict-desc {
        font-size: 1.05rem;
        color: #e2e8f0;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    /* HUD Reticle Wrapper for Satellite Image */
    .hud-scanner-wrapper {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(56, 189, 248, 0.5);
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
        background: #030712;
        margin-bottom: 14px;
        max-width: 360px;
        aspect-ratio: 1 / 1;
    }

    .hud-scanline {
        position: absolute;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent 0%, rgba(56, 189, 248, 0.4) 15%, #38bdf8 50%, rgba(56, 189, 248, 0.4) 85%, transparent 100%);
        box-shadow: 0 0 16px #38bdf8, 0 0 32px #0284c7;
        animation: laserScan 2.4s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
        z-index: 10;
        pointer-events: none;
    }

    .hud-reticle-tl, .hud-reticle-tr, .hud-reticle-bl, .hud-reticle-br {
        position: absolute;
        width: 22px;
        height: 22px;
        border-color: #38bdf8;
        border-style: solid;
        z-index: 12;
        pointer-events: none;
        filter: drop-shadow(0 0 4px #0284c7);
    }

    .hud-reticle-tl { top: 10px; left: 10px; border-width: 3px 0 0 3px; }
    .hud-reticle-tr { top: 10px; right: 10px; border-width: 3px 3px 0 0; }
    .hud-reticle-bl { bottom: 10px; left: 10px; border-width: 0 0 3px 3px; }
    .hud-reticle-br { bottom: 10px; right: 10px; border-width: 0 3px 3px 0; }

    .hud-badge {
        position: absolute;
        bottom: 10px;
        right: 10px;
        background: rgba(3, 7, 18, 0.88);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        border: 1px solid rgba(56, 189, 248, 0.4);
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        z-index: 14;
        letter-spacing: 0.5px;
    }

    /* Metric Cards */
    .metric-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }

    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Preset Quick-Buttons */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(30, 41, 59, 0.8);
        color: #f1f5f9;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background: rgba(56, 189, 248, 0.2);
        border-color: #38bdf8;
        color: #38bdf8;
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);
    }

    /* Tab bar styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        color: #94a3b8;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
    }

    /* Clarification Note Callout */
    .explainer-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #38bdf8;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin: 14px 0;
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. HEADER & TOP BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
            <div class="hero-title">🛰️ LandScope AI</div>
            <div class="hero-subtitle">
                Autonomous satellite land-cover discovery engine powered by <strong>Unsupervised Computer Vision (PCA + K-Means)</strong>.
                Analyzes optical color, texture roughness, and vegetative spectral signatures from Earth observation imagery without predefined labels.
            </div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
            <div class="telemetry-pill">
                <span class="beacon-dot"></span> ORBITAL SENSOR ONLINE
            </div>
            <div style="font-size: 0.78rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                EUROPEAN SPACE AGENCY • SENTINEL-2
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. SIDEBAR: IMAGE INPUT & ONE-CLICK PRESET GALLERY
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Satellite Sensor Input")
    st.caption("Upload your own patch or test with curated satellite samples below.")

    # State management for selected sample vs uploaded image
    if "selected_sample" not in st.session_state:
        st.session_state["selected_sample"] = SAMPLE_PRESETS[0]["path"]

    uploaded_file = st.file_uploader(
        "Upload Satellite Tile (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an optical satellite patch (ideally 64x64px or square crop)."
    )

    st.markdown("---")
    st.markdown("#### ⚡ Quick-Select Demo Samples")
    st.caption("Click any terrain tile below to immediately run real-time analysis:")

    # Render interactive preset gallery buttons in a 2-column layout
    col_a, col_b = st.columns(2)
    for i, preset in enumerate(SAMPLE_PRESETS):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            if st.button(
                f"{preset['icon']} {preset['name']}",
                key=f"preset_btn_{i}",
                help=preset["desc"],
                use_container_width=True
            ):
                st.session_state["selected_sample"] = preset["path"]

    st.markdown("---")
    st.markdown("#### ℹ️ About Unsupervised AI")
    st.info(
        "**No Cheat Sheets Used!**\n\n"
        "Traditional models are trained with human labels. LandScope discovers natural clusters purely by "
        "grouping mathematical patterns of color, light, and texture from space."
    )

# -----------------------------------------------------------------------------
# 7. LOAD AND PROCESS THE ACTIVE IMAGE
# -----------------------------------------------------------------------------
active_image = None
image_source_label = ""

if uploaded_file is not None:
    active_image = Image.open(uploaded_file)
    image_source_label = f"Uploaded File: {uploaded_file.name}"
elif st.session_state.get("selected_sample") and os.path.exists(st.session_state["selected_sample"]):
    active_image = Image.open(st.session_state["selected_sample"])
    sample_name = os.path.basename(st.session_state["selected_sample"]).replace(".jpg", "").replace("_Sample", "")
    image_source_label = f"Curated Preset: {sample_name} Landscape"
else:
    # Fallback to the first available sample
    default_path = SAMPLE_PRESETS[0]["path"]
    if os.path.exists(default_path):
        active_image = Image.open(default_path)
        image_source_label = "Default Curated Sample: Forest"

if active_image is None:
    st.warning("Please upload an image or choose a demo sample from the sidebar to begin.")
    st.stop()

# Extract 19-dimensional hand-crafted visual features
raw_features = extract(active_image).reshape(1, -1)
feature_df = pd.DataFrame(raw_features, columns=M["features"])

# Transform through pipeline: Median Imputer -> Standard Scaler -> PCA Projection
xs = M["sc"].transform(M["imp"].transform(feature_df))
z = M["pca"].transform(xs)

# Calculate distances to all 4 K-Means cluster centroids in PCA space
distances = M["km"].transform(z)[0]
assigned_cluster = int(distances.argmin())
sorted_distances = np.sort(distances)

# Separation score: (2nd_min - 1st_min) / 2nd_min  (0 = ambiguous boundary, 1 = razor-sharp cluster)
d_nearest = sorted_distances[0]
d_second = sorted_distances[1]
separation_score = float((d_second - d_nearest) / max(d_second, 1e-6))
clarity_percent = min(100, max(5, int((1.0 - (d_nearest / (d_nearest + d_second))) * 200)))
# Normalized softmax-like similarity score across all clusters for intuitive bar charts
inv_distances = 1.0 / (distances + 1e-5)
similarity_percentages = (inv_distances / inv_distances.sum()) * 100

meta = CLUSTER_METADATA.get(assigned_cluster, CLUSTER_METADATA[0])
posthoc_groundtruth = M["label_map"][assigned_cluster]

# -----------------------------------------------------------------------------
# 8. TOP-LEVEL HERO SCANNER & VERDICT (SUPER EASY FOR NON-TECH USERS)
# -----------------------------------------------------------------------------
st.markdown("### 🔍 Live Satellite Terrain Scan")

# Convert active image to base64 data URI so it stays strictly inside the HUD container
import io
import base64

buffered = io.BytesIO()
# Maintain high visual quality
active_image.convert("RGB").resize((360, 360), Image.Resampling.LANCZOS).save(buffered, format="PNG")
img_b64 = base64.b64encode(buffered.getvalue()).decode()
img_data_uri = f"data:image/png;base64,{img_b64}"

col_img, col_verdict = st.columns([1, 2.3], gap="large")

with col_img:
    hud_html = f"""
    <div class="hud-scanner-wrapper">
        <div class="hud-scanline"></div>
        <div class="hud-reticle-tl"></div>
        <div class="hud-reticle-tr"></div>
        <div class="hud-reticle-bl"></div>
        <div class="hud-reticle-br"></div>
        <img src="{img_data_uri}" alt="Satellite Patch" style="width: 100%; aspect-ratio: 1/1; display: block; object-fit: cover; border-radius: 14px;" />
        <div class="hud-badge">SENTINEL-2 • 64x64 TILE</div>
    </div>
    """
    st.markdown(hud_html, unsafe_allow_html=True)
    st.caption(f"📍 **Input Source:** {image_source_label}")

with col_verdict:
    # Plain English Verdict Card with Dynamic Accent Border
    verdict_html = f"""
    <div class="verdict-box" style="background: {meta['accent']}; border-color: {meta['color']}; box-shadow: 0 0 30px {meta['accent']};">
        <div class="verdict-header">
            <div>
                <span class="verdict-tag" style="color: {meta['color']};">DETECTED TERRAIN FAMILY</span>
                <div class="verdict-name">{meta['icon']} {meta['title']}</div>
            </div>
            <div style="background: rgba(0,0,0,0.4); padding: 8px 16px; border-radius: 9999px; border: 1px solid {meta['color']};">
                <span style="font-size: 0.85rem; font-family: 'JetBrains Mono'; font-weight: 700; color: {meta['color']};">
                    CLUSTER #{assigned_cluster}
                </span>
            </div>
        </div>
        <div class="verdict-desc">{meta['summary']}</div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600;">
                🏷️ Archetype: {meta['landscape_badge']}
            </span>
            <span style="background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600;">
                🛰️ EuroSAT Sanity Check: {posthoc_groundtruth['class_']} ({posthoc_groundtruth['purity']:.0%} purity)
            </span>
        </div>
    </div>
    """
    st.markdown(verdict_html, unsafe_allow_html=True)

    # 4 Human-Friendly Telemetry Badges
    m1, m2, m3 = st.columns(3)
    with m1:
        match_level = "🟢 Very High" if separation_score >= 0.25 else ("🟡 Moderate" if separation_score >= 0.12 else "🟠 Borderline")
        st.markdown(f"""
        <div class="metric-pill">
            <span class="metric-label">Match Clarity</span>
            <span class="metric-val">{match_level}</span>
            <span style="font-size: 0.75rem; color: #94a3b8;">Clarity Score: {separation_score:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        # Greenness indicator
        exg_val = raw_features[0, 11]  # ExG_mean
        green_text = "🌿 Lush / High" if exg_val > 0.03 else ("🌱 Moderate" if exg_val > 0.0 else "🍂 Sparse / None")
        st.markdown(f"""
        <div class="metric-pill">
            <span class="metric-label">Vegetation Foliage</span>
            <span class="metric-val">{green_text}</span>
            <span style="font-size: 0.75rem; color: #94a3b8;">Excess Green: {exg_val:+.3f}</span>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        # Texture roughness indicator
        lap_val = raw_features[0, 17]  # Laplacian_var
        texture_text = "🏢 High Texture" if lap_val > 0.015 else ("🌾 Medium Grain" if lap_val > 0.004 else "🌊 Smooth Canopy")
        st.markdown(f"""
        <div class="metric-pill">
            <span class="metric-label">Surface Complexity</span>
            <span class="metric-val">{texture_text}</span>
            <span style="font-size: 0.75rem; color: #94a3b8;">Texture Var: {lap_val:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

if separation_score < 0.12:
    st.warning(
        f"⚠️ **Borderline / Mixed Terrain Detection:** This satellite tile lies near the boundary between "
        f"**{meta['short_name']}** and another cluster (2nd nearest is {sorted_distances[1]:.2f} units away). "
        "It likely represents a transitional landscape, such as a suburban greenway, riverbank, or mixed farmland."
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. INTERACTIVE TAB NAVIGATION
# -----------------------------------------------------------------------------
tab_overview, tab_features, tab_galaxy, tab_profiles, tab_science, tab_guide = st.tabs([
    "📊 Terrain Similarity & Radar",
    "🔬 Visual Feature Fingerprint",
    "🌌 2D Landscape Galaxy (PCA & t-SNE)",
    "📑 Discovered Cluster Profiles",
    "📐 Machine Learning Diagnostics",
    "💡 How It Works (For Beginners)"
])

# -----------------------------------------------------------------------------
# TAB 1: TERRAIN SIMILARITY & RADAR FINGERPRINT
# -----------------------------------------------------------------------------
with tab_overview:
    col_sim, col_radar = st.columns([1.1, 1], gap="large")

    with col_sim:
        st.markdown("#### 🎯 Where Does This Patch Belong?")
        st.caption("Normalized similarity affinity across the 4 discovered terrain families:")

        # Horizontal Bar Chart for similarity
        cluster_names = [f"{CLUSTER_METADATA[i]['icon']} {CLUSTER_METADATA[i]['short_name']}" for i in range(4)]
        bar_colors = [CLUSTER_METADATA[i]['color'] for i in range(4)]

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            y=cluster_names,
            x=similarity_percentages,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(255,255,255,0.2)", width=1.5)
            ),
            text=[f"<b>{val:.1f}%</b> similarity" for val in similarity_percentages],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Affinity: %{x:.2f}%<br>Distance: %{customdata:.2f} units<extra></extra>",
            customdata=distances
        ))

        fig_sim.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            margin=dict(l=10, r=20, t=10, b=10),
            height=280,
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                title="Similarity Affinity (%)",
                range=[0, max(similarity_percentages) * 1.15]
            ),
            yaxis=dict(autorange="reversed"),
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown(f"""
        <div class="explainer-box">
            💡 <strong>How to read this:</strong> The algorithm measures Euclidean distance in reduced 6-dimensional PCA space.
            The closest centroid is <strong>Cluster {assigned_cluster}</strong> ({distances[assigned_cluster]:.2f} units away).
            A higher percentage indicates that your satellite patch's color, brightness, and texture patterns closely mirror that landscape family.
        </div>
        """, unsafe_allow_html=True)

    with col_radar:
        st.markdown("#### 🕸️ 5-Dimension Terrain Radar")
        st.caption("Comparing your patch's signature against the standard archetype:")

        # Compute 5 intuitive normalized dimensions from the 19 features:
        # 1. Vegetation Greenery (ExG, GR index)
        # 2. Structural Density (Edges, Texture variance)
        # 3. Brightness / Glare (RGB means, Value mean)
        # 4. Color Saturation (HSV saturation mean)
        # 5. Visual Complexity (Entropy, Color standard deviations)
        
        # Normalize roughly between 0 and 100 for intuitive radar display
        patch_green = float(np.clip((raw_features[0, 11] + 0.1) * 350, 5, 100))
        patch_struct = float(np.clip(raw_features[0, 17] * 2500, 5, 100))
        patch_bright = float(np.clip(raw_features[0, 9] * 120, 10, 100))
        patch_sat = float(np.clip(raw_features[0, 7] * 150, 5, 100))
        patch_complex = float(np.clip((raw_features[0, 18] - 0.5) * 35, 10, 100))

        # Cluster archetype values from cluster_profiles_df
        if not cluster_profiles_df.empty and assigned_cluster in cluster_profiles_df.index:
            c_prof = cluster_profiles_df.loc[assigned_cluster]
            arch_green = float(np.clip((c_prof["ExG_mean (greenness)"] + 0.1) * 350, 5, 100))
            arch_struct = float(np.clip(c_prof["Laplacian_var (texture)"] * 2500, 5, 100))
            arch_bright = float(np.clip(c_prof["Val_mean"] * 120, 10, 100))
            arch_sat = float(np.clip(c_prof["Sat_mean"] * 150, 5, 100))
            arch_complex = float(np.clip((c_prof["Gray_entropy"] - 0.5) * 35, 10, 100))
        else:
            arch_green, arch_struct, arch_bright, arch_sat, arch_complex = 50, 50, 50, 50, 50

        radar_categories = [
            "🌿 Vegetation Vitality",
            "🏢 Structural Edges",
            "☀️ Optical Brightness",
            "🎨 Color Purity",
            "🔍 Visual Texture"
        ]

        fig_radar = go.Figure()

        # Archetype Polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=[arch_green, arch_struct, arch_bright, arch_sat, arch_complex, arch_green],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            name=f"Cluster {assigned_cluster} Typical",
            fillcolor=f"{meta['accent']}",
            line=dict(color=meta['color'], width=2, dash="dash"),
            hoverinfo="r+theta"
        ))

        # Uploaded Image Polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=[patch_green, patch_struct, patch_bright, patch_sat, patch_complex, patch_green],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            name="Your Uploaded Patch",
            fillcolor="rgba(56, 189, 248, 0.25)",
            line=dict(color="#38bdf8", width=3),
            hoverinfo="r+theta"
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 105],
                    showticklabels=False,
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    linecolor="rgba(255,255,255,0.15)"
                ),
                bgcolor="rgba(15,23,42,0.5)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=10, b=10),
            height=280,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#cbd5e1")
            ),
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: VISUAL FEATURE FINGERPRINT
# -----------------------------------------------------------------------------
with tab_features:
    st.markdown("#### 🔬 Detailed Optical & Texture Feature Fingerprint")
    st.caption("How your satellite tile compares against the cluster's expected standard deviations (Z-Scores):")

    df_feat = pd.DataFrame({
        "Feature Name": M["features"],
        "Raw Measurement": raw_features[0],
        "Standardized Z-Score": xs[0],
        "Cluster Typical Z": M["cm_scaled"][assigned_cluster]
    })

    df_feat["Deviation Gap"] = (df_feat["Standardized Z-Score"] - df_feat["Cluster Typical Z"]).abs()
    
    # Human-friendly feature mapping
    def feature_descriptor(name):
        desc_map = {
            "R_mean": "Red optical reflection (Iron, soils, dry roofs)",
            "G_mean": "Green optical reflection (Chlorophyll & foliage)",
            "B_mean": "Blue optical reflection (Water, sky, bright concrete)",
            "R_std": "Red channel color contrast",
            "G_std": "Green channel color contrast",
            "B_std": "Blue channel color contrast",
            "Hue_mean": "Dominant color wavelength / hue angle",
            "Sat_mean": "Color intensity (Vibrant foliage vs muted concrete)",
            "Sat_std": "Color vibrancy variation across tile",
            "Val_mean": "Overall surface brightness / illumination",
            "Val_std": "Light-to-shadow contrast variation",
            "ExG_mean (greenness)": "Excess Green index (Pure plant vegetative health)",
            "ExG_std": "Canopy greenness clumpiness",
            "GR_index (veg)": "Green-to-Red vegetation ratio",
            "GB_index (water)": "Green-to-Blue moisture / water ratio",
            "Edge_mean": "Sobel edge sharpness (Roads, roof borders, boundaries)",
            "Edge_std": "Edge magnitude variation",
            "Laplacian_var (texture)": "Laplacian surface roughness (Smooth lake vs rough city)",
            "Gray_entropy": "Grayscale information complexity / visual entropy"
        }
        return desc_map.get(name, "Optical satellite measurement")

    df_feat["What It Measures"] = df_feat["Feature Name"].apply(feature_descriptor)
    df_feat["Conformity"] = np.where(
        df_feat["Deviation Gap"] < 0.75,
        "🟢 Perfectly Typical",
        np.where(df_feat["Deviation Gap"] < 1.5, "🟡 Slight Variation", "🟠 Distinctive / Unusual")
    )

    col_f1, col_f2 = st.columns([1.8, 1], gap="large")

    with col_f1:
        st.dataframe(
            df_feat[["Feature Name", "What It Measures", "Standardized Z-Score", "Cluster Typical Z", "Conformity"]]
            .sort_values(by="Standardized Z-Score", ascending=False)
            .style.format({
                "Standardized Z-Score": "{:+.2f}",
                "Cluster Typical Z": "{:+.2f}"
            }),
            use_container_width=True,
            height=400
        )

    with col_f2:
        st.markdown("##### 📌 Key Driver Features")
        st.caption("The features that most strongly anchored this tile to its cluster:")

        # Top 3 most aligned features (lowest gap)
        top_aligned = df_feat.sort_values("Deviation Gap").head(3)
        for _, row in top_aligned.iterrows():
            st.markdown(f"""
            <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;">
                <div style="font-weight: 700; color: #34d399; font-size: 0.9rem;">✅ {row['Feature Name']}</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">{row['What It Measures']}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px; font-family: 'JetBrains Mono';">
                    Tile Z: {row['Standardized Z-Score']:+.2f} | Cluster Mean: {row['Cluster Typical Z']:+.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 🔍 Most Unusual Feature")
        most_unusual = df_feat.sort_values("Deviation Gap", ascending=False).iloc[0]
        st.markdown(f"""
        <div style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 10px; padding: 10px 14px;">
            <div style="font-weight: 700; color: #fbbf24; font-size: 0.9rem;">⚠️ {most_unusual['Feature Name']}</div>
            <div style="font-size: 0.8rem; color: #cbd5e1;">{most_unusual['What It Measures']}</div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px; font-family: 'JetBrains Mono';">
                Gap: {most_unusual['Deviation Gap']:.2f} standard deviations away from cluster norm.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: 2D LANDSCAPE GALAXY (PCA & t-SNE)
# -----------------------------------------------------------------------------
with tab_galaxy:
    st.markdown("#### 🌌 The 2D Satellite Landscape Galaxy")
    st.markdown(
        "Every dot in this chart represents a real EuroSAT Sentinel-2 satellite image. "
        "The AI compressed 19 complex optical dimensions into 2D coordinates. "
        "The **pulsing gold star** (`★`) shows where your uploaded satellite tile lands!"
    )

    space_mode = st.radio(
        "Choose Space Projection:",
        ["Global PCA Coordinate Space (Preserves Global Variance)", "Local t-SNE Manifold (Preserves Local Neighborhood Clusters)"],
        horizontal=True
    )

    cluster_color_map = {
        0: "#10B981",  # Green
        1: "#F59E0B",  # Orange
        2: "#06B6D4",  # Cyan
        3: "#A855F7"   # Purple
    }

    cluster_label_names = {
        0: "Cluster 0: Grassland / Open",
        1: "Cluster 1: Industrial / Built-up",
        2: "Cluster 2: Forest & Deep Canopy",
        3: "Cluster 3: Farmland & Crops"
    }

    fig_galaxy = go.Figure()

    if "PCA" in space_mode:
        # 4,000 reference points in PCA 2D
        pca_coords = M["Z2"]
        labels = M["labels"]

        for c_id in range(4):
            mask = (labels == c_id)
            fig_galaxy.add_trace(go.Scattergl(
                x=pca_coords[mask, 0],
                y=pca_coords[mask, 1],
                mode="markers",
                name=cluster_label_names[c_id],
                marker=dict(
                    color=cluster_color_map[c_id],
                    size=5,
                    opacity=0.45
                ),
                hovertemplate=f"<b>{cluster_label_names[c_id]}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
            ))

        # Add User Point in PCA Space
        fig_galaxy.add_trace(go.Scatter(
            x=[z[0, 0]],
            y=[z[0, 1]],
            mode="markers+text",
            name="📍 YOUR SATELLITE IMAGE",
            text=["📍 YOUR SATELLITE IMAGE"],
            textposition="top center",
            textfont=dict(family="JetBrains Mono", size=13, color="#fbbf24"),
            marker=dict(
                symbol="star",
                size=22,
                color="#fbbf24",
                line=dict(color="#ffffff", width=2)
            ),
            hovertemplate="<b>📍 YOUR SATELLITE IMAGE</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Cluster: " + str(assigned_cluster) + "<extra></extra>"
        ))

        fig_galaxy.update_layout(
            title="<b>Principal Component Analysis (PC1 vs PC2)</b> — Explains the largest variance in satellite imagery",
            xaxis_title="PC1: Overall Surface Brightness & Edge Complexity →",
            yaxis_title="PC2: Chlorophyll Greenness & Foliage Saturation →"
        )

    else:
        # t-SNE 2D Space
        tsne_coords = M["tsne_emb"]
        tsne_labels = M["labels"][M["tsne_idx"]]

        for c_id in range(4):
            mask = (tsne_labels == c_id)
            fig_galaxy.add_trace(go.Scattergl(
                x=tsne_coords[mask, 0],
                y=tsne_coords[mask, 1],
                mode="markers",
                name=cluster_label_names[c_id],
                marker=dict(
                    color=cluster_color_map[c_id],
                    size=5,
                    opacity=0.45
                ),
                hovertemplate=f"<b>{cluster_label_names[c_id]}</b><br>t-SNE 1: %{{x:.2f}}<br>t-SNE 2: %{{y:.2f}}<extra></extra>"
            ))

        # Project user image into t-SNE using 10 nearest neighbors approximation in PCA space
        nn_idx = np.argsort(((M["tsne_Z"] - z) ** 2).sum(axis=1))[:10]
        user_tsne_approx = M["tsne_emb"][nn_idx].mean(axis=0)

        fig_galaxy.add_trace(go.Scatter(
            x=[user_tsne_approx[0]],
            y=[user_tsne_approx[1]],
            mode="markers+text",
            name="📍 YOUR SATELLITE IMAGE (Approx.)",
            text=["📍 YOUR SATELLITE IMAGE"],
            textposition="top center",
            textfont=dict(family="JetBrains Mono", size=13, color="#fbbf24"),
            marker=dict(
                symbol="star",
                size=22,
                color="#fbbf24",
                line=dict(color="#ffffff", width=2)
            ),
            hovertemplate="<b>📍 YOUR SATELLITE IMAGE (Approx via 10-NN)</b><br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>"
        ))

        fig_galaxy.update_layout(
            title="<b>t-SNE Manifold Projection (Perplexity=30)</b> — Nonlinear neighborhood relationships",
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2"
        )


    fig_galaxy.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        height=540,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#cbd5e1")
        ),
        font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")
    )

    st.plotly_chart(fig_galaxy, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: DISCOVERED CLUSTER PROFILES
# -----------------------------------------------------------------------------
with tab_profiles:
    st.markdown("#### 📑 The 4 Discovered Terrain Archetypes")
    st.caption("How 4,000 unlabeled satellite images organized themselves naturally:")

    c_cols = st.columns(4)
    for c_id in range(4):
        c_meta = CLUSTER_METADATA[c_id]
        c_purity = M["label_map"][c_id]
        is_active = (c_id == assigned_cluster)

        with c_cols[c_id]:
            border_style = f"2px solid {c_meta['color']}" if is_active else "1px solid rgba(255,255,255,0.08)"
            bg_glow = c_meta['accent'] if is_active else "rgba(15,23,42,0.6)"
            badge_html = " ⭐ MATCHED TO YOUR IMAGE" if is_active else ""

            st.markdown(f"""
            <div class="glass-card" style="border: {border_style}; background: {bg_glow}; padding: 16px;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">{c_meta['icon']}</div>
                <div style="font-weight: 800; font-size: 1.05rem; color: {c_meta['color']}; margin-bottom: 4px;">
                    Cluster {c_id}: {c_meta['short_name']}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono'; margin-bottom: 10px;">
                    {c_meta['landscape_badge']}{badge_html}
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4; margin-bottom: 12px;">
                    {c_meta['summary']}
                </div>
                <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Dominant EuroSAT Match</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">
                        {c_purity['class_']} ({c_purity['purity']:.0%} purity)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🔍 EuroSAT Cross-Tabulation Ground Truth (Post-Hoc Check)")
    st.caption("Distribution of human-labeled EuroSAT classes inside each unsupervised cluster:")
    if not crosstab_df.empty:
        st.dataframe(crosstab_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: ML DIAGNOSTICS & EXPERIMENTAL METRICS
# -----------------------------------------------------------------------------
with tab_science:
    st.markdown("#### 📐 Engineering & Mathematical Diagnostics")
    st.caption("Quantitative evaluations recorded during training on 4,000 EuroSAT images:")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Images Sampled", f"{results_meta.get('n_images', 4000):,}")
    with col_m2:
        st.metric("Interpretable Features", f"{results_meta.get('n_features', 19)}")
    with col_m3:
        st.metric("PCs Explaining 95% Var", f"{results_meta.get('n_pc_95', 6)}")
    with col_m4:
        st.metric("Chosen K (Clusters)", f"{results_meta.get('chosen_k', 4)}")

    st.markdown("##### 📈 Model Validation Figures")
    fig_col1, fig_col2, fig_col3 = st.columns(3)
    with fig_col1:
        if os.path.exists("figures/02_pca_variance.png"):
            st.image("figures/02_pca_variance.png", caption="Scree Plot & Cumulative Variance")
    with fig_col2:
        if os.path.exists("figures/03_k_selection.png"):
            st.image("figures/03_k_selection.png", caption="K-Selection Curves (Silhouette, DB, CH, ARI)")
    with fig_col3:
        if os.path.exists("figures/05_tsne_perplexity.png"):
            st.image("figures/05_tsne_perplexity.png", caption="t-SNE Perplexity Exploration (10, 30, 50)")

    with st.expander("🔬 Full Results JSON Dump"):
        st.json(results_meta)

    with st.expander("📊 K-Selection Evaluation Table"):
        if not k_selection_df.empty:
            st.dataframe(k_selection_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 6: BEGINNER'S GUIDE FOR NON-TECH USERS
# -----------------------------------------------------------------------------
with tab_guide:
    st.markdown("### 💡 How Does This Work Without Human Supervision?")
    st.markdown("""
    Imagine giving someone 4,000 satellite photos without telling them what country, city, or forest they show.
    You simply ask: **"Sort these into 4 neat piles based on how they look."**
    
    That is exactly what this artificial intelligence does! Here is the step-by-step process:
    """)

    g1, g2 = st.columns(2, gap="large")

    with g1:
        st.markdown("""
        #### 1. The Satellite's Eye (Computer Vision)
        - Each satellite photo is a small 64 × 64 pixel patch taken by the Sentinel-2 constellation.
        - Instead of looking at raw pixels, our algorithm measures **19 optical properties**:
            - **Color Spectrum**: How much Red, Green, and Blue light reflects off the ground.
            - **Vegetation Indices**: Special ratios like *Excess Green (ExG)* that tell whether living plants are photosynthesizing.
            - **Texture & Edges**: Using mathematical edge filters (Sobel & Laplacian) to measure whether the terrain is smooth like a lake or jagged with straight lines like roads and buildings.
        
        #### 2. Condensing Dimensions (PCA)
        - Having 19 features is like describing a house using 19 different measurements.
        - **Principal Component Analysis (PCA)** combines these into the 6 most important master coordinates, capturing 95% of all visual variety without losing essential information.
        """)

    with g2:
        st.markdown("""
        #### 3. Discovering Natural Families (K-Means)
        - The algorithm places all 4,000 photos in a multi-dimensional space.
        - It lets **K-Means clustering** naturally group similar landscapes together into 4 clusters.
        - Notice: **We never told the AI what "Forest" or "Industrial" means!** It discovered that dark, lush green patches naturally group together, while bright concrete roofs group into another corner.
        
        #### 4. Post-Hoc Sanity Verification
        - After clustering was finished, we peeked at the official EuroSAT labels only to verify whether the AI's clusters made sense in the real world.
        - As expected, Cluster 1 caught mostly Industrial structures, Cluster 2 caught Forests, and Cluster 3 caught Agricultural crops!
        """)

    st.markdown("""
    <div class="explainer-box">
        <strong>⚠️ Responsible Remote Sensing Notes:</strong>
        <ul style="margin-top: 8px; margin-bottom: 0;">
            <li><strong>RGB Spectral Bands Only:</strong> This model operates on standard visual RGB channels without Near-Infrared (NIR) or Short-Wave Infrared (SWIR) bands.</li>
            <li><strong>Exploratory Categories:</strong> The discovered clusters are statistical similarity groups, not legal land-use categories.</li>
            <li><strong>Geographic Boundary:</strong> The EuroSAT dataset is centered on European biomes; tropical or desert terrains may produce transitional assignments.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. EXPORT INTELLIGENCE BRIEF
# -----------------------------------------------------------------------------
st.markdown("---")
col_down_text, col_down_btn = st.columns([3, 1])
with col_down_text:
    st.markdown(f"**Export Scan Summary:** Download a structured markdown report for **Cluster {assigned_cluster}: {meta['short_name']}**.")

with col_down_btn:
    report_content = f"""# LandScope AI — Satellite Terrain Intelligence Brief
Date: 2026-11-16
Image Source: {image_source_label}

## 1. Classification Verdict
- Assigned Cluster: Cluster {assigned_cluster} ({meta['title']})
- Archetype: {meta['landscape_badge']}
- Summary: {meta['summary']}
- Closest EuroSAT Class (Post-Hoc): {posthoc_groundtruth['class_']} (Purity: {posthoc_groundtruth['purity']:.1%})

## 2. Match Confidence & Distances
- Distance to Centroid: {distances[assigned_cluster]:.2f} units
- 2nd Nearest Cluster: {sorted_distances[1]:.2f} units
- Separation Score: {separation_score:.2f} (0=Ambiguous, 1=Clear)
- Match Quality: {match_level}

## 3. Cluster Distance Matrix (PCA Space)
{pd.DataFrame({"Cluster": [CLUSTER_METADATA[i]['title'] for i in range(4)], "Distance": distances, "Similarity": similarity_percentages}).to_string(index=False)}

## 4. Key Optical Indicators
- Excess Green (Vegetation): {raw_features[0, 11]:+.4f}
- Laplacian Variance (Texture Roughness): {raw_features[0, 17]:.6f}
- Overall Surface Brightness (Value Mean): {raw_features[0, 9]:.4f}

---
Generated by LandScope AI Unsupervised Satellite Engine (EuroSAT Sentinel-2)
"""
    st.download_button(
        label="📥 Download Intelligence Brief",
        data=report_content,
        file_name=f"terrain_report_cluster_{assigned_cluster}.md",
        mime="text/markdown",
        use_container_width=True
    )
