from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
SPLITS_DIR = DATA_DIR / "splits"
DEV_PREDICTIONS_DIR = DATA_DIR / "dev_predictions"
BAREC_PARQUET_DIR = DATA_DIR / "barec"
EMBEDDED_DIR = DATA_DIR / "embedded"
TARAB_FULL_CORPUS_EMBED = EMBEDDED_DIR / "tarab_full_corpus_overlap.json"

# #ff7f0e at 80% on white — dist_155K bars in training / experiment charts.
DIST_ORANGE = "#ff993e"
# Tarab-share panel only (composition %, not dist volume).
TARAB_SHARE_COLOR = "#6d9fa3"
