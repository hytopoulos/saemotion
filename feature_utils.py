"""
Feature vector computation and manipulation utilities.
"""
import importlib
import sys
import numpy as np
import streamlit as st
from dataclasses import dataclass

# SAE model imports
sys.path.append("../")
from model import SAEConfig, SAE  # noqa: E402


@dataclass
class TrainConfig:
    """Training configuration for SAE model loading."""
    model: SAEConfig
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    compile: bool


def load_emov5_module():
    """Load the emov5 module with proper error handling."""
    try:
        return importlib.import_module("emov5")
    except Exception as e:
        raise RuntimeError(
            "Failed to import 'emov5'. Ensure it's on PYTHONPATH and provides "
            "feature_vector(sae, feature_id) and a global 'sae' object."
        ) from e


@st.cache_data
def get_feature_vector_cached(_sae, feature_id: int) -> np.ndarray:
    """Get feature vector with caching for performance."""
    import emov5
    fv = emov5.feature_vector(_sae, int(feature_id))
    return np.asarray(fv, dtype=np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize a vector."""
    norm = float(np.linalg.norm(x))
    if norm < eps:
        return x
    return x / norm


def build_weighted_vector(emov5_sae, feature_weights: list) -> np.ndarray:
    """
    Build weighted sum of feature vectors.
    
    Args:
        emov5_sae: The SAE model object
        feature_weights: List of (feature_id, weight) tuples
    
    Returns:
        Weighted sum vector as numpy array
    """
    vec_sum = None
    
    for feature_id, weight in feature_weights:
        fv = get_feature_vector_cached(emov5_sae, feature_id)
        
        if vec_sum is None:
            vec_sum = (fv * float(weight)).astype(np.float32, copy=True)
        else:
            vec_sum += (fv * float(weight)).astype(np.float32, copy=False)
    
    if vec_sum is None:
        raise RuntimeError("No vector could be constructed from feature weights.")
    
    return vec_sum
