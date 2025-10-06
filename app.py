# streamlit_app_emotion_search.py
# Streamlit app: sliders grouped by emotion type + search via feature-vector sum
# Usage:
#   pip install streamlit requests pandas numpy
#   streamlit run streamlit_app_emotion_search.py --server.runOnSave=true
#
# The app expects a CSV like topk_shared_and_distinct.csv
# with columns:
#   ['type','rank','feature_id','score', <emotion columns...>, 'feat' (optional duplicate)]
# It uses your local 'emov5' module that defines:
#   fv = emov5.feature_vector(emov5.sae, feature_id)
#
import os
import math
import json
import importlib
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =============================
# SAE import
# =============================
from dataclasses import dataclass
import sys
sys.path.append("../")
from model import SAEConfig, SAE  # noqa: E402
@dataclass
class TrainConfig:
    model: SAEConfig
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    compile: bool


# ------------------------ Config ------------------------
DEFAULT_CSV = os.getenv("EMOTION_FREQ_CSV", "topk_shared_and_distinct.csv")
ENDPOINT = os.getenv("NOOSCOPE_ENDPOINT", "https://nooscope.osmarks.net/backend")
TOP_K = int(os.getenv("TOP_K", "10"))
SLIDER_MIN = float(os.getenv("SLIDER_MIN", "-1.0"))
SLIDER_MAX = float(os.getenv("SLIDER_MAX", "1.0"))
SLIDER_STEP = float(os.getenv("SLIDER_STEP", "0.001"))

# ------------------------ Helpers ------------------------
RESERVED_COLS = {"type", "rank", "feature_id", "score", "feat"}

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    # If 'feat' exists and feature_id missing, use it
    if "feature_id" not in df.columns and "feat" in df.columns:
        df = df.rename(columns={"feat": "feature_id"})
    # Enforce dtypes
    if "feature_id" in df.columns:
        df["feature_id"] = df["feature_id"].astype(int)
    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
    return df

def emotion_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in RESERVED_COLS]

def get_unique_features(df: pd.DataFrame) -> List[int]:
    return sorted(df["feature_id"].unique().tolist())

def group_by_type(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return dict(tuple(df.groupby("type", sort=False)))

def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n

def load_emov5():
    try:
        return importlib.import_module("emov5")
    except Exception as e:
        raise RuntimeError(
            "Failed to import 'emov5'. Ensure it's on PYTHONPATH and provides "
            "feature_vector(sae, feature_id) and a global 'sae' object."
        ) from e

def fv_for_feature(emov5_mod, feature_id: int) -> np.ndarray:
    fv = emov5_mod.feature_vector(emov5_mod.sae, int(feature_id))
    fv = np.asarray(fv, dtype=np.float32)
    return fv

def post_nooscope(payload):
    resp = requests.post(ENDPOINT, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ------------------------ UI ------------------------
st.set_page_config(page_title="SAEmotion", layout="wide")
st.title("SAEmotion")

with st.sidebar:
    st.header("Data")
    csv_path = st.text_input("CSV path", value=DEFAULT_CSV, help="Path to the feature-by-emotion frequency CSV.")
    uploaded = st.file_uploader("...or upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        try:
            df = load_df(csv_path)
        except Exception as e:
            st.error(f"Could not load CSV: {e}")
            st.stop()

    st.caption(f"{len(df)} rows • {df['feature_id'].nunique()} unique features")
    em_cols = emotion_columns(df)
    if not em_cols:
        st.error("No emotion columns detected. CSV must have columns besides 'type','rank','feature_id','score','feat'.")
        st.stop()

    st.header("emov5 module")
    emov5_status = st.empty()
    try:
        emov5 = load_emov5()
        emov5_status.success("✅ Loaded emov5 module")
    except Exception as e:
        emov5 = None
        emov5_status.error(f"⚠️ {e}")

    st.header("Search Settings")
    TOP_K = st.number_input("Top-K results", min_value=1, max_value=100, value=TOP_K, step=1)
    SLIDER_MIN = st.number_input("Slider min", min_value=-100.0, max_value=0.0, value=SLIDER_MIN, step=0.1)
    SLIDER_MAX = st.number_input("Slider max", min_value=0.1, max_value=100.0, value=SLIDER_MAX, step=0.1)
    SLIDER_STEP = st.number_input("Slider step", min_value=0.0001, max_value=1.0, value=SLIDER_STEP, step=0.0001, format="%.4f")

# Group features by their 'type' column as they appear in the original CSV
def group_by_original_structure(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group features by their original 'type' column, preserving all entries as they appear in the CSV.
    Features can appear multiple times under different categories."""
    groups = {}
    
    # Group by 'type' column if it exists
    if 'type' in df.columns:
        for type_name, group_df in df.groupby('type', sort=False):
            if pd.notna(type_name) and str(type_name).strip():
                display_name = str(type_name).title()
                groups[display_name] = group_df.copy()
    
    # If no 'type' column or no valid types, group by emotion columns
    if not groups:
        em_cols = emotion_columns(df)
        if em_cols:
            # Create artificial groups based on highest emotion score per row
            for _, row in df.iterrows():
                best_score = -1
                best_emotion = None
                
                for em_col in em_cols:
                    if em_col in row and pd.notna(row[em_col]):
                        score = float(row[em_col])
                        if score > best_score:
                            best_score = score
                            best_emotion = em_col
                
                if best_emotion:
                    if best_emotion not in groups:
                        groups[best_emotion] = []
                    groups[best_emotion].append(row)
            
            # Convert lists to DataFrames
            for emotion in groups:
                groups[emotion] = pd.DataFrame(groups[emotion])
    
    return groups

groups = group_by_original_structure(df)
all_features = get_unique_features(df)

# Initialize session state for sliders early, before we try to access it
if "slider_values" not in st.session_state:
    st.session_state.slider_values = {int(fid): 0.0 for fid in all_features}

# Initialize manual features tracking
if "manual_features" not in st.session_state:
    st.session_state.manual_features = set()

# Add any custom features that exist in session state to all_features
for fid in st.session_state.manual_features:
    if fid not in all_features:
        all_features.append(fid)
    # Ensure manual features have entries in slider_values
    if fid not in st.session_state.slider_values:
        st.session_state.slider_values[fid] = 0.0

# Create combined options for dropdowns (emotions + special types)
def get_all_dropdown_options(df: pd.DataFrame, groups_dict: Dict) -> List[str]:
    """Get all available options for dropdowns: emotion columns + special types"""
    em_cols = emotion_columns(df)
    special_types = []
    
    # Get unique types from the data
    if 'type' in df.columns:
        unique_types = df['type'].dropna().unique()
        for t in unique_types:
            if t and str(t).lower() not in ['nan', 'none']:
                special_types.append(str(t).title())
    
    # Combine and deduplicate
    all_options = list(em_cols) + special_types
    return sorted(list(set(all_options)))

all_dropdown_options = get_all_dropdown_options(df, groups)

# Session state for sliders is already initialized above

# ---- Global controls row ----
colA, colB, colC, colD = st.columns([2,2,1,1])
with colA:
    chosen_emotion = st.selectbox("Set all sliders from emotion/type frequencies", options=all_dropdown_options, index=0)
with colB:
    with st.container():
        st.write("")  # spacing
        if st.button("Apply to all sliders"):
            # Handle both emotion columns and special types
            if chosen_emotion in emotion_columns(df):
                # It's an emotion column - use the frequency values
                emo_series = df.groupby("feature_id")[chosen_emotion].mean()
                for fid in all_features:
                    v = emo_series.get(fid, 0.0)
                    if pd.isna(v):
                        v = 0.0
                    st.session_state.slider_values[int(fid)] = max(float(SLIDER_MIN), min(float(SLIDER_MAX), float(v)))
            else:
                # It's a special type - set sliders to 1.0 for features of that type, 0.0 for others
                type_features = set()
                if 'type' in df.columns:
                    matching_rows = df[df['type'].str.lower() == chosen_emotion.lower()]
                    type_features = set(matching_rows['feature_id'].unique())
                
                for fid in all_features:
                    if fid in type_features:
                        st.session_state.slider_values[int(fid)] = min(float(SLIDER_MAX), 1.0)
                    else:
                        st.session_state.slider_values[int(fid)] = 0.0
with colC:
    st.write("")
    if st.button("Reset sliders"):
        for fid in all_features:
            st.session_state.slider_values[int(fid)] = 0.0
with colD:
    st.write("")
    if st.button("Normalize sliders (L2)"):
        vals = np.array([st.session_state.slider_values[int(fid)] for fid in all_features], dtype=np.float32)
        norm = np.linalg.norm(vals)
        if norm > 1e-8:
            vals = (vals / norm).tolist()
            for i, fid in enumerate(all_features):
                st.session_state.slider_values[int(fid)] = float(vals[i])

# ---- Add/Subtract emotion controls ----
st.markdown("### Add/Subtract Emotion")
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    modify_emotion = st.selectbox("Select emotion/type to add/subtract", options=all_dropdown_options, index=0, key="modify_emotion")
with col2:
    operation = st.selectbox("Operation", options=["Add", "Subtract"], index=0)
with col3:
    multiplier = st.number_input("Multiplier", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
with col4:
    st.write("")  # spacing
    if st.button("Apply modification"):
        # Handle both emotion columns and special types
        if modify_emotion in emotion_columns(df):
            # It's an emotion column - use the frequency values
            emo_series = df.groupby("feature_id")[modify_emotion].mean()
            
            # Apply add or subtract operation with multiplier
            for fid in all_features:
                v = emo_series.get(fid, 0.0)
                if pd.isna(v):
                    v = 0.0
                
                current_value = st.session_state.slider_values[int(fid)]
                
                if operation == "Add":
                    new_value = current_value + (float(v) * float(multiplier))
                else:  # Subtract
                    new_value = current_value - (float(v) * float(multiplier))
                
                # Clamp to valid range
                new_value = max(float(SLIDER_MIN), min(float(SLIDER_MAX), new_value))
                st.session_state.slider_values[int(fid)] = new_value
        else:
            # It's a special type - add/subtract multiplier for features of that type
            type_features = set()
            if 'type' in df.columns:
                matching_rows = df[df['type'].str.lower() == modify_emotion.lower()]
                type_features = set(matching_rows['feature_id'].unique())
            
            for fid in all_features:
                if fid in type_features:
                    current_value = st.session_state.slider_values[int(fid)]
                    
                    if operation == "Add":
                        new_value = current_value + float(multiplier)
                    else:  # Subtract
                        new_value = current_value - float(multiplier)
                    
                    # Clamp to valid range
                    new_value = max(float(SLIDER_MIN), min(float(SLIDER_MAX), new_value))
                    st.session_state.slider_values[int(fid)] = new_value
        
        st.success(f"Applied {operation.lower()} operation with {modify_emotion} (×{multiplier}) to all sliders")

st.markdown("---")

# ---- Manual Feature Input Section ----
st.subheader("🎯 Manual Feature Input")
with st.expander("Add specific features by ID", expanded=False):
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        manual_feature_input = st.text_input(
            "Feature ID(s)", 
            placeholder="Enter feature IDs separated by commas (e.g., 1754, 49950, 19495)",
            help="Add specific features by their ID, regardless of whether they appear in the CSV"
        )
    
    with col2:
        manual_weight = st.number_input(
            "Weight", 
            min_value=float(SLIDER_MIN), 
            max_value=float(SLIDER_MAX), 
            value=1.0, 
            step=float(SLIDER_STEP),
            format="%.3f"
        )
    
    with col3:
        st.write("")  # spacing
        if st.button("Add Features"):
            if manual_feature_input.strip():
                try:
                    # Parse the input
                    feature_ids = [int(fid.strip()) for fid in manual_feature_input.split(',') if fid.strip()]
                    
                    # Add to session state
                    for fid in feature_ids:
                        st.session_state.slider_values[fid] = float(manual_weight)
                        st.session_state.manual_features.add(fid)
                    
                    st.success(f"Added {len(feature_ids)} features with weight {manual_weight}")
                    st.rerun()
                except ValueError:
                    st.error("Please enter valid feature IDs (integers separated by commas)")

# Debug: Show current session state
st.write("Debug - Session state slider_values:", dict(st.session_state.slider_values))
st.write("Debug - All features list:", sorted(all_features))

# Add custom features to groups if any exist
# Use the explicitly tracked manual features
manual_features = list(st.session_state.manual_features)

st.write(f"Debug - Manual features detected: {manual_features}")
st.write(f"Debug - Manual features set: {st.session_state.manual_features}")
st.write(f"Debug - Total features in all_features: {len(all_features)}")

if manual_features:
    # Create a custom group DataFrame for manual features
    custom_rows = []
    for fid in manual_features:
        custom_rows.append({
            'type': 'custom',
            'rank': None,
            'feature_id': fid,
            'score': st.session_state.slider_values.get(fid, 0.0)
        })
    
    if custom_rows:
        custom_df = pd.DataFrame(custom_rows)
        groups['Custom'] = custom_df
        st.success(f"✅ Created Custom group with {len(custom_rows)} features")
    else:
        st.warning("⚠️ Manual features found but no custom_rows created")
else:
    st.info("ℹ️ No manual features detected")

st.markdown("---")

# ---- Sliders grouped by original CSV structure ----
st.subheader("Sliders grouped by original CSV categories")

# Sort groups to prioritize "Shared" and other special types
def sort_groups(groups_dict):
    """Sort groups to show important ones first"""
    priority_order = ['Custom', 'Shared', 'Distinct', 'Unique', 'Common']
    sorted_items = []
    
    # Add priority groups first
    for priority_group in priority_order:
        if priority_group in groups_dict:
            sorted_items.append((priority_group, groups_dict[priority_group]))
    
    # Add remaining groups alphabetically
    remaining_groups = [(k, v) for k, v in groups_dict.items() if k not in priority_order]
    remaining_groups.sort(key=lambda x: x[0])
    sorted_items.extend(remaining_groups)
    
    return sorted_items

sorted_groups = sort_groups(groups)

for gname, gdf in sorted_groups:
    # Special styling for important groups
    is_priority_group = gname in ['Custom', 'Shared', 'Distinct', 'Unique', 'Common']
    expanded_default = (gname == 'Custom' or gname == 'Shared')  # Expand Custom and Shared by default
    
    # Add emoji and special formatting for priority groups
    if gname == 'Custom':
        display_name = f"🎯 {gname} (Manual Features)"
    elif gname == 'Shared':
        display_name = f"🌟 {gname} (High Priority)"
    elif is_priority_group:
        display_name = f"⭐ {gname}"
    else:
        display_name = f"Emotion: {gname}"
    
    with st.expander(f"{display_name}  •  {len(gdf)} features", expanded=expanded_default):
        # Sort by rank if available, else feature_id
        if "rank" in gdf.columns:
            gdf_sorted = gdf.sort_values("rank")
        else:
            gdf_sorted = gdf.sort_values("feature_id")
        # 3 columns layout
        cols = st.columns(3)
        for i, row in enumerate(gdf_sorted.itertuples(index=False)):
            fid = int(getattr(row, "feature_id"))
            rank_val = getattr(row, "rank", None)
            rank = int(rank_val) if rank_val is not None else None
            score = getattr(row, "score", None)
            
            # Enhanced label with score for priority groups
            if is_priority_group and score is not None:
                label = f"feat {fid}" + (f"  (rank {rank}, score {score:.3f})" if rank is not None else f"  (score {score:.3f})")
            else:
                label = f"feat {fid}" + (f"  (rank {rank})" if rank is not None else "")
            
            col = cols[i % 3]
            with col:
                # Create unique key for this slider instance
                slider_key = f"slider_{gname}_{fid}_{i}"
                
                # Get the current value from the shared state
                current = st.session_state.slider_values.get(fid, 0.0)
                
                new_val = st.slider(
                    label=label,
                    min_value=float(SLIDER_MIN),
                    max_value=float(SLIDER_MAX),
                    value=float(current),
                    step=float(SLIDER_STEP),
                    key=slider_key,
                )
                
                # If this slider changed, update the global value for this feature
                if new_val != current:
                    st.session_state.slider_values[fid] = float(new_val)
                    # Force a rerun to sync all other sliders for this feature
                    st.rerun()

# Synchronization happens automatically through the shared slider_values dictionary

st.markdown("---")

# ---- Search action ----
st.subheader("🔍 Search")
left, right = st.columns([2,1])
with left:
    st.markdown("The weighted sum is computed over **all features** (non‑zero sliders).")
    if st.button("Run search", type="primary"):
        # Build weighted feature vector
        nonzero_items = [(fid, w) for fid, w in st.session_state.slider_values.items() if float(w) != 0.0]
        if not nonzero_items:
            st.warning("All sliders are zero. Set some weights first.")
        elif emov5 is None:
            st.error("emov5 module is not loaded. The search requires feature vectors from emov5.")
        else:
            try:
                # Accumulate sum
                vec_sum = None
                # Cache vectors to avoid duplicate loads across groups
                for fid, w in nonzero_items:
                    fv = fv_for_feature(emov5, fid)  # shape [D]
                    if vec_sum is None:
                        vec_sum = (fv * float(w)).astype(np.float32, copy=True)
                    else:
                        vec_sum += (fv * float(w)).astype(np.float32, copy=False)
                if vec_sum is None:
                    raise RuntimeError("No vector could be constructed.")
                vec_norm = l2norm(vec_sum)
                st.success(f"Built feature vector of dim {vec_norm.shape[0]} (L2 norm={np.linalg.norm(vec_norm):.6f})")

                # Show JSON payload preview
                payload = {
                    "terms": [{"embedding": [float(x) for x in vec_norm.tolist()], "weight": 1}],
                    "include_video": True,
                    "debug_enabled": False,
                    "k": int(TOP_K),
                }
                with st.expander("🔎 Request JSON preview"):
                    st.code(json.dumps(payload, indent=2)[:2000], language="json")

                # Make the request
                with st.spinner("Querying image backend..."):
                    result = post_nooscope(payload)

                # Render results
                matches = result.get("matches", [])
                if not matches:
                    st.warning("No results returned.")
                else:
                    st.subheader("Results")
                    # Show in a grid
                    cols = st.columns(5)
                    for i, m in enumerate(matches):
                        score = m[0] if len(m) > 0 else None
                        url = m[1] if len(m) > 1 else None
                        size = m[4] if len(m) > 4 else None
                        c = cols[i % 5]
                        with c:
                            if url:
                                st.image(url, caption=f"{score:.4f} • {size}", use_column_width=True)
                            else:
                                st.write(m)

                # Show raw response
                with st.expander("🧾 Raw response JSON"):
                    st.code(json.dumps(result, indent=2)[:4000], language="json")

            except Exception as e:
                st.exception(e)

with right:
    st.markdown("#### Quick tools")
    # Compute a pre-canned vector from selected single emotion (for reference)
    emo_for_quick = st.selectbox("Preview vector for emotion/type", options=all_dropdown_options, index=0, key="quick_emo")
    if st.button("Compute preview vector (no search)"):
        vec_sum = None
        
        if emo_for_quick in emotion_columns(df):
            # Handle emotion column - use frequency values
            emo_series = df.groupby("feature_id")[emo_for_quick].mean()
            for fid, v in emo_series.items():
                if pd.isna(v) or not math.isfinite(v) or v == 0.0:
                    continue
                if emov5 is None:
                    st.error("emov5 not loaded.")
                    break
                fv = fv_for_feature(emov5, int(fid))
                vec_sum = (fv * float(v)) if vec_sum is None else (vec_sum + fv * float(v))
        else:
            # Handle special type - use equal weight (1.0) for all features of that type
            type_features = set()
            if 'type' in df.columns:
                matching_rows = df[df['type'].str.lower() == emo_for_quick.lower()]
                type_features = set(matching_rows['feature_id'].unique())
            
            for fid in type_features:
                if emov5 is None:
                    st.error("emov5 not loaded.")
                    break
                fv = fv_for_feature(emov5, int(fid))
                vec_sum = fv if vec_sum is None else (vec_sum + fv)
        
        if vec_sum is not None:
            vec_norm = l2norm(vec_sum)
            st.info(f"Preview vector dim {vec_norm.shape[0]}, norm {np.linalg.norm(vec_norm):.6f}")
            with st.expander("Preview JSON"):
                st.code(json.dumps({"terms": [{"feature": [float(x) for x in vec_norm.tolist()]}], "k": int(TOP_K)}, indent=2)[:2000], language="json")
        else:
            st.warning(f"No entries found for {emo_for_quick}.")

st.markdown("---")
st.caption("Tip: Use the sidebar to change slider bounds/step. 'Apply to all sliders' copies frequencies from the chosen emotion into every feature's slider.")
