# Environment-based configuration
DEFAULT_CSV = "topk_shared_and_distinct.csv"
ENDPOINT = "https://nooscope.osmarks.net/backend"
TOP_K = 10
SLIDER_MIN = -1.0
SLIDER_MAX = 1.0
SLIDER_STEP = 0.001

# CSV column names that are reserved (not emotion columns)
RESERVED_COLS = {"type", "rank", "feature_id", "score", "feat"}

# Priority groups for UI display
PRIORITY_GROUPS = ['Custom', 'Shared', 'Distinct', 'Unique', 'Common']

# sparse autoencoder
BATCH_SIZE = 16
CKPT_STEPS = 111885

# file paths
GLOBAL_EMBS_NPY = "all_clip_embs_fp16.npy"
GLOBAL_INDEX_CSV = "all_clip_index.csv"
ANNOTS_CSV = "archive/annots_arrs/annot_arrs_train_exploded.csv"
IMG_ROOT = "archive/img_arrs"
