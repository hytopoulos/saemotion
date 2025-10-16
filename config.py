"""
Configuration and constants for SAEmotion app.
"""
import os

# Environment-based configuration
DEFAULT_CSV = os.getenv("EMOTION_FREQ_CSV", "topk_shared_and_distinct.csv")
ENDPOINT = os.getenv("NOOSCOPE_ENDPOINT", "https://nooscope.osmarks.net/backend")
TOP_K = int(os.getenv("TOP_K", "10"))
SLIDER_MIN = float(os.getenv("SLIDER_MIN", "-1.0"))
SLIDER_MAX = float(os.getenv("SLIDER_MAX", "1.0"))
SLIDER_STEP = float(os.getenv("SLIDER_STEP", "0.001"))

# CSV column names that are reserved (not emotion columns)
RESERVED_COLS = {"type", "rank", "feature_id", "score", "feat"}

# Priority groups for UI display
PRIORITY_GROUPS = ['Custom', 'Shared', 'Distinct', 'Unique', 'Common']
