"""
Data loading and CSV parsing utilities.
"""
from typing import Dict, List
import pandas as pd
import streamlit as st
from config import RESERVED_COLS


@st.cache_data
def load_csv(csv_path: str) -> pd.DataFrame:
    """Load and normalize CSV data with caching."""
    df = pd.read_csv(csv_path)
    
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # Handle legacy 'feat' column
    if "feature_id" not in df.columns and "feat" in df.columns:
        df = df.rename(columns={"feat": "feature_id"})
    
    # Enforce dtypes
    if "feature_id" in df.columns:
        df["feature_id"] = df["feature_id"].astype(int)
    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
    
    return df


def get_emotion_columns(df: pd.DataFrame) -> List[str]:
    """Extract emotion column names (non-reserved columns)."""
    return [c for c in df.columns if c not in RESERVED_COLS]


def get_unique_features(df: pd.DataFrame) -> List[int]:
    """Get sorted list of unique feature IDs."""
    return sorted(df["feature_id"].unique().tolist())


def group_features_by_type(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group features by their 'type' column, preserving all entries.
    Features can appear multiple times under different categories.
    """
    groups = {}
    
    # Group by 'type' column if it exists
    if 'type' in df.columns:
        for type_name, group_df in df.groupby('type', sort=False):
            if pd.notna(type_name) and str(type_name).strip():
                display_name = str(type_name).title()
                groups[display_name] = group_df.copy()
    
    # Fallback: group by highest emotion score if no type column
    if not groups:
        em_cols = get_emotion_columns(df)
        if em_cols:
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


def sort_groups(groups: Dict[str, pd.DataFrame], priority_groups: List[str]) -> List[tuple]:
    """
    Sort groups with priority groups first, then alphabetically.
    Returns list of (group_name, dataframe) tuples.
    """
    priority_items = [(g, groups[g]) for g in priority_groups if g in groups]
    other_items = [(g, groups[g]) for g in sorted(groups.keys()) if g not in priority_groups]
    return priority_items + other_items
