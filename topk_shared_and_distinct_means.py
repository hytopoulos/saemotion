#!/usr/bin/env python3
"""
Compute:
1) Top-K features most *shared* across all emotions (based on means).
2) Top-K features most *distinct* for each emotion (based on means).

Inputs:
- CSV with rows = features, columns = emotion mean/variance pairs.
  Format: feat, Emotion1_mean, Emotion1_var, Emotion2_mean, Emotion2_var, ...
  
Outputs:
- One CSV combining:
  - K rows for shared features (type="shared")
  - K rows per emotion for distinct features (type=<emotion>)
  Columns include: [type, rank, feature_id, score, <all emotion mean/var columns>]

Usage:
  python topk_shared_and_distinct_means.py \
      --input feature_by_emotion_mean_var.csv \
      --output topk_shared_distinct_means.csv \
      --k 20 \
      [--id-col feat]

Notes:
- "Shared" score = min mean across all emotion columns (ties broken by sum across emotions).
- "Distinct" score for emotion e = mean[e] - mean(mean[others]) (ties broken by mean[e]).
- NaNs are treated as 0.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def infer_id_column(df: pd.DataFrame, override: str | None) -> str | None:
    """Infer the feature ID column name from the DataFrame.
    
    Args:
        df: Input DataFrame
        override: User-specified column name (takes precedence)
        
    Returns:
        Column name if found, None if ID should be synthesized from index
    """
    if override:
        if override not in df.columns:
            raise ValueError(f"--id-col '{override}' not found in columns: {list(df.columns)}")
        return override
    for cand in ["feat", "feature", "feature_id", "id", "Feature", "FEAT", "FEATURE_ID"]:
        if cand in df.columns:
            return cand
    return None


def get_emotion_mean_columns(df: pd.DataFrame, id_col: str | None) -> list[str]:
    """Extract all emotion mean columns (columns ending with '_mean').
    
    Args:
        df: Input DataFrame
        id_col: Feature ID column to exclude from search
        
    Returns:
        List of mean column names
    """
    reserved = set([id_col]) if id_col else set()
    mean_cols = [c for c in df.columns if c not in reserved and c.endswith('_mean')]
    if not mean_cols:
        raise ValueError("No emotion mean columns found. Expected columns ending with '_mean'.")
    return mean_cols


def get_emotion_names(mean_cols: list[str]) -> list[str]:
    """Extract emotion names from mean column names.
    
    Args:
        mean_cols: List of mean column names (e.g., 'Happiness_mean')
        
    Returns:
        List of emotion names (e.g., 'Happiness')
    """
    return [c.replace('_mean', '') for c in mean_cols]


def topk_shared(df_means: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compute top-K features most shared across all emotions.
    
    Shared score = minimum mean across emotions (encourages breadth).
    Ties are broken by sum across emotions.
    
    Args:
        df_means: DataFrame with emotion mean columns
        k: Number of top features to return
        
    Returns:
        DataFrame with score column, sorted by score descending
    """
    min_across = df_means.min(axis=1)
    sum_across = df_means.sum(axis=1)
    out = pd.DataFrame({
        "score": min_across,
        "_tie_sum": sum_across,
    })
    out = out.sort_values(by=["score", "_tie_sum"], ascending=[False, False]).head(k).drop(columns="_tie_sum")
    return out


def topk_distinct_per_emotion(df_means: pd.DataFrame, mean_cols: list[str], k: int) -> dict[str, pd.DataFrame]:
    """Compute top-K features most distinct for each emotion.
    
    Distinct score = mean[emotion] - mean(mean[other emotions]).
    Ties are broken by the emotion's mean value.
    
    Args:
        df_means: DataFrame with emotion mean columns
        mean_cols: List of mean column names
        k: Number of top features to return per emotion
        
    Returns:
        Dictionary mapping mean column names to DataFrames with scores
    """
    results = {}
    others_sum = df_means.sum(axis=1)
    for col in mean_cols:
        others_mean = ((others_sum - df_means[col]) / max(len(mean_cols) - 1, 1)).replace([np.inf, -np.inf], 0.0)
        margin = df_means[col] - others_mean
        out = pd.DataFrame({
            "score": margin,
            "_tie_mean_e": df_means[col],
        }).sort_values(by=["score", "_tie_mean_e"], ascending=[False, False]).head(k).drop(columns="_tie_mean_e")
        results[col] = out
    return results


def main():
    """Main entry point for computing shared and distinct features."""
    parser = argparse.ArgumentParser(description="Compute top-K shared and distinct SAE features across emotions.")
    parser.add_argument("--input", required=True, help="Path to input CSV (feature x emotion mean/var matrix).")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    parser.add_argument("--k", type=int, default=20, help="Top-K to select for shared and for each emotion.")
    parser.add_argument("--id-col", default=None, help="Optional name of the feature id column.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_path)
    df = df.fillna(0)

    id_col = infer_id_column(df, args.id_col)
    mean_cols = get_emotion_mean_columns(df, id_col)
    emotion_names = get_emotion_names(mean_cols)

    df_means = df[mean_cols].astype(float)

    # Build feature identifier series
    if id_col is not None:
        feature_id = df[id_col].astype(str)
    else:
        feature_id = pd.Series(df.index.astype(int).astype(str), name="feature_idx")

    # Compute shared Top-K
    shared_scores = topk_shared(df_means, args.k)
    shared_rows = df.loc[shared_scores.index].copy()
    shared_rows.insert(0, "feature_id", list(feature_id.iloc[shared_scores.index]))
    shared_rows.insert(0, "rank", range(1, len(shared_rows) + 1))
    shared_rows.insert(0, "type", "shared")
    shared_rows.insert(shared_rows.columns.get_loc("type") + 1, "score", list(shared_scores["score"].values))

    # Compute distinct Top-K per emotion
    distinct_dict = topk_distinct_per_emotion(df_means, mean_cols, args.k)
    distinct_blocks = []
    for mean_col, scores in distinct_dict.items():
        emotion_name = mean_col.replace('_mean', '')
        block = df.loc[scores.index].copy()
        block.insert(0, "feature_id", list(feature_id.iloc[scores.index]))
        block.insert(0, "rank", range(1, len(block) + 1))
        block.insert(0, "type", emotion_name)
        block.insert(block.columns.get_loc("type") + 1, "score", list(scores["score"].values))
        distinct_blocks.append(block)

    # Concatenate: shared first, then per-emotion blocks
    out_df = pd.concat([shared_rows] + distinct_blocks, axis=0, ignore_index=True)

    # Order columns: type, rank, feature_id, score, emotion columns, remaining
    base_cols = ["type", "rank", "feature_id", "score"]
    emotion_cols = [c for c in df.columns if any(c.startswith(e + '_') for e in emotion_names)]
    remaining = [c for c in out_df.columns if c not in base_cols + emotion_cols]
    ordered = base_cols + emotion_cols + remaining
    out_df = out_df[ordered]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"✓ Wrote {len(out_df)} rows to {args.output}")
    print(f"  - {args.k} shared features")
    print(f"  - {args.k} distinct features × {len(emotion_names)} emotions")


if __name__ == "__main__":
    main()
