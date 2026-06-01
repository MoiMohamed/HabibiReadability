from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
LENGTH_MATCHED_DIR = DATA_DIR / "length_matched"
MIN_L_DIR = DATA_DIR / "minL"
SPLITS_DIR = DATA_DIR / "splits"
DEV_PREDICTIONS_DIR = DATA_DIR / "dev_predictions"
LENGTH_MATCHED_DEV_PREDICTIONS_CSV = (
    LENGTH_MATCHED_DIR / "dev_bert-base-arabertv02_local_19levels_length_matched.csv"
)
MIN_L8_DEV_PREDICTIONS_CSV = MIN_L_DIR / "dev_bert-base-arabertv02_local_19levels_minL8.csv"
LENGTH_MATCHED_TRAINING_LOG = "caps10_length_matching-15852637.out"
MIN_L8_TRAINING_LOG = "barec_tarab_2x_minL8-15943060.out"
LENGTH_MATCHED_BAREC_CHARS_BY_LEVEL_PNG = (
    LENGTH_MATCHED_DIR / "barec_length_matching_train_chars_by_level.png"
)
LENGTH_MATCHED_TARAB_ORIGINAL_CHARS_BY_LEVEL_PNG = (
    LENGTH_MATCHED_DIR / "tarab_full.sentence_aldi_ags_readability_chars_by_level.png"
)
LENGTH_MATCHED_TARAB_AGGREGATED_CHARS_BY_LEVEL_PNG = (
    LENGTH_MATCHED_DIR / "tarab_aggregated_pseudo_corpus_caps10_chars_by_level.png"
)
BAREC_PARQUET_DIR = DATA_DIR / "barec"
EMBEDDED_DIR = DATA_DIR / "embedded"
TARAB_FULL_CORPUS_EMBED = EMBEDDED_DIR / "tarab_full_corpus_overlap.json"

# #ff7f0e at 80% on white — dist_155K bars in training / experiment charts.
DIST_ORANGE = "#ff993e"
# Tarab-share panel only (composition %, not dist volume).
TARAB_SHARE_COLOR = "#6d9fa3"
