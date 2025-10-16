import sys
sys.path.append("../")
from dataclasses import dataclass
from model import SAEConfig

@dataclass
class TrainConfig:
    model: SAEConfig
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    compile: bool

# Standard library imports
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_CSV, TOP_K, SLIDER_MIN, SLIDER_MAX, 
    SLIDER_STEP, PRIORITY_GROUPS
)
from data_loader import (
    load_csv, get_emotion_columns, get_unique_features,
    group_features_by_type, sort_groups
)
from feature_utils import (
    load_emov5_module, build_weighted_vector, l2_normalize
)
from api_client import search_images


st.set_page_config(page_title="SAEmotion", layout="wide")
st.title("SAEmotion")

# Load CSV
try:
    df = load_csv(DEFAULT_CSV)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

# Validate emotion columns
emotion_cols = get_emotion_columns(df)
if not emotion_cols:
    st.error("No emotion columns detected.")
    st.stop()

# Load emov5 module
try:
    import importlib
    if 'emov5' in sys.modules:
        importlib.reload(sys.modules['emov5'])
    emov5 = load_emov5_module()
    if not hasattr(emov5, 'sae') or not hasattr(emov5, 'feature_vector'):
        raise RuntimeError("emov5 missing required attributes")
except Exception as e:
    st.error(f"Failed to load emov5: {e}")
    emov5 = None

top_k_input = TOP_K
slider_min = SLIDER_MIN
slider_max = SLIDER_MAX
slider_step = SLIDER_STEP

# Initialize slider state
all_features = get_unique_features(df)
if 'slider_values' not in st.session_state:
    st.session_state.slider_values = {fid: 0.0 for fid in all_features}

# All options for dropdown (emotions + types)
all_dropdown_options = list(emotion_cols)
if 'type' in df.columns:
    unique_types = [str(t).title() for t in df['type'].dropna().unique() if t and str(t).lower() not in ['nan', 'none']]
    all_dropdown_options = sorted(list(set(all_dropdown_options + unique_types)))

# =====================================================================
# Set Emotion
# =====================================================================

st.header("Set Emotion")

with st.form(key="global_controls_form"):
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        set_emotion = st.selectbox("Set all sliders from:", options=all_dropdown_options)
    
    with col2:
        apply_all = st.form_submit_button("✅ Apply to all sliders", use_container_width=True)
    
    with col3:
        reset_all = st.form_submit_button("🔄 Reset", use_container_width=True)
    
    with col4:
        normalize = st.form_submit_button("📊 L2 Norm", use_container_width=True)
    
    # Handle button clicks
    if apply_all:
        if set_emotion in emotion_cols:
            emo_series = df.groupby("feature_id")[set_emotion].mean()
            for fid in all_features:
                v = emo_series.get(fid, 0.0)
                if pd.isna(v):
                    v = 0.0
                st.session_state.slider_values[int(fid)] = max(slider_min, min(slider_max, float(v)))
        else:
            type_features = set()
            if 'type' in df.columns:
                matching_rows = df[df['type'].str.lower() == set_emotion.lower()]
                type_features = set(matching_rows['feature_id'].unique())
            for fid in all_features:
                st.session_state.slider_values[int(fid)] = min(slider_max, 1.0) if fid in type_features else 0.0
    
    if reset_all:
        for fid in all_features:
            st.session_state.slider_values[int(fid)] = 0.0
    
    if normalize:
        vals = np.array([st.session_state.slider_values[int(fid)] for fid in all_features], dtype=np.float32)
        norm = np.linalg.norm(vals)
        if norm > 1e-8:
            vals = (vals / norm).tolist()
            for i, fid in enumerate(all_features):
                st.session_state.slider_values[int(fid)] = float(vals[i])

# =====================================================================
# ADD/SUBTRACT EMOTION
# =====================================================================

st.markdown("### Add/Subtract Emotion")

with st.form(key="modify_emotion_form"):
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        modify_emotion = st.selectbox("Emotion to modify:", options=all_dropdown_options)
    
    with col2:
        operation = st.selectbox("Operation:", options=["Add", "Subtract"])
    
    with col3:
        multiplier = st.number_input("Multiplier:", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    
    with col4:
        apply_modify = st.form_submit_button("▶️ Apply", use_container_width=True)
    
    if apply_modify:
        if modify_emotion in emotion_cols:
            emo_series = df.groupby("feature_id")[modify_emotion].mean()
            for fid in all_features:
                v = emo_series.get(fid, 0.0)
                if pd.isna(v):
                    v = 0.0
                current = st.session_state.slider_values[int(fid)]
                if operation == "Add":
                    new_val = current + (v * multiplier)
                else:
                    new_val = current - (v * multiplier)
                st.session_state.slider_values[int(fid)] = max(slider_min, min(slider_max, float(new_val)))
        else:
            type_features = set()
            if 'type' in df.columns:
                matching_rows = df[df['type'].str.lower() == modify_emotion.lower()]
                type_features = set(matching_rows['feature_id'].unique())
            for fid in all_features:
                if fid in type_features:
                    current = st.session_state.slider_values[int(fid)]
                    if operation == "Add":
                        new_val = current + multiplier
                    else:
                        new_val = current - multiplier
                    st.session_state.slider_values[int(fid)] = max(slider_min, min(slider_max, float(new_val)))


# =====================================================================
# MANUAL FEATURE INPUT
# =====================================================================

st.header("Manual Feature Input")
with st.expander("Add specific feature IDs", expanded=False):
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        manual_fids = st.text_input(
            "Feature IDs (comma-separated)",
            placeholder="e.g., 82295, 206810, 15659",
            help="Enter feature IDs to add to Custom group"
        )
    
    with col2:
        manual_weight = st.number_input(
            "Weight",
            value=1.0,
            step=0.1,
            format="%.2f"
        )
    
    with col3:
        st.write("")  # Spacing
        if st.button("Add Features"):
            if manual_fids:
                try:
                    fids = [int(f.strip()) for f in manual_fids.split(",")]
                    for fid in fids:
                        if fid in st.session_state.slider_values:
                            st.session_state.slider_values[fid] = float(manual_weight)
                    st.success(f"Added {len(fids)} features")
                except ValueError:
                    st.error("Invalid feature IDs. Use comma-separated integers.")

st.markdown("---")

# =====================================================================
# FEATURE SLIDERS
# =====================================================================

st.header("Feature Weights")

# Group and sort features
groups = group_features_by_type(df)
sorted_groups = sort_groups(groups, PRIORITY_GROUPS)

# Use form to batch slider updates
with st.form(key="slider_form", clear_on_submit=False):
    # Track which features we've already rendered to avoid duplicates
    rendered_features = set()
    
    for group_name, group_df in sorted_groups:
        # Determine if this is a priority group
        is_priority = group_name in PRIORITY_GROUPS
        expanded_default = (group_name == 'Custom')
        
        # Format display name
        if group_name == 'Custom':
            display_name = f"{group_name}"
        elif group_name == 'Shared':
            display_name = f"{group_name}"
        elif is_priority:
            display_name = f"{group_name}"
        else:
            display_name = f"{group_name}"
        
        # Filter out features already rendered in previous groups
        group_features = []
        for _, row in group_df.iterrows():
            feature_id = int(row['feature_id'])
            if feature_id not in rendered_features:
                group_features.append(row)
                rendered_features.add(feature_id)
        
        # Skip group if all features already rendered
        if not group_features:
            continue
        
        with st.expander(f"{display_name}  •  {len(group_features)} features", expanded=expanded_default):
            # Sort by rank if available
            if "rank" in group_df.columns:
                group_features_sorted = sorted(group_features, key=lambda x: x['rank'] if pd.notna(x.get('rank')) else float('inf'))
            else:
                group_features_sorted = sorted(group_features, key=lambda x: int(x['feature_id']))
            
            # 3-column layout
            cols = st.columns(3)
            
            for idx, row in enumerate(group_features_sorted):
                feature_id = int(row['feature_id'])
                
                label = f"Feature {feature_id}"
                
                # Slider key for session state
                slider_key = f"slider_{feature_id}"
                
                # Place in column
                with cols[idx % 3]:
                    # Get value from session state, default to 0.0
                    default_value = st.session_state.slider_values.get(feature_id, 0.0)
                    st.slider(
                        label=label,
                        min_value=float(slider_min),
                        max_value=float(slider_max),
                        value=float(default_value),
                        step=float(slider_step),
                        key=slider_key,
                    )
    
    st.markdown("---")
    
    # Submit button
    submitted = st.form_submit_button(
        "Run Search",
        type="primary",
        use_container_width=True
    )


# =====================================================================
# SEARCH EXECUTION
# =====================================================================

st.markdown("---")

if submitted:
    
    # Collect non-zero feature weights
    nonzero_items = []
    for group_name, group_df in sorted_groups:
        for _, row in group_df.iterrows():
            feature_id = int(row['feature_id'])
            slider_key = f"slider_{feature_id}"
            
            if slider_key in st.session_state:
                weight = st.session_state[slider_key]
                if abs(weight) > 1e-9:
                    nonzero_items.append((feature_id, weight))
                    # Update stored value
                    st.session_state.slider_values[feature_id] = weight

    if not nonzero_items:
        st.warning("No features selected (all sliders at zero). Please adjust sliders.")
        st.stop()
    
    # Build and search
    if emov5 is None:
        st.error("Cannot search: emov5 module not loaded.")
    else:
        try:
            # Build weighted feature vector
            with st.spinner("Building feature vector..."):
                vec_sum = build_weighted_vector(emov5.sae, nonzero_items)
            
            # Normalize
            vec_norm = l2_normalize(vec_sum)
            st.success(
                f"Built feature vector of dim {vec_norm.shape[0]} "
                f"(L2 norm={np.linalg.norm(vec_norm):.6f})"
            )
            
            # Search
            with st.spinner("Querying image backend..."):
                result = search_images(
                    embedding=vec_norm,
                )
            
            # Display results
            matches = result.get("matches", [])
            if not matches:
                st.warning("No results returned.")
            else:
                st.subheader("Results")
                
                # Grid layout
                cols = st.columns(5)
                for i, match in enumerate(matches):
                    score = match[0] if len(match) > 0 else None
                    url = match[1] if len(match) > 1 else None
                    size = match[4] if len(match) > 4 else None
                    
                    with cols[i % 5]:
                        if url:
                            st.image(url, caption=f"{score:.4f} • {size}", use_column_width=True)
                        else:
                            st.write(match)
     
        except Exception as e:
            st.exception(e)
