#!/usr/bin/env python3
"""One-off: precompute full Tarab×BAREC overlap stats for deploy (no 399MB CSV at runtime)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tarab_model_experimentation.constants import DATA_DIR  # noqa: E402
from tarab_model_experimentation.pseudo_label_oov import (  # noqa: E402
    TARAB_FULL_CSV,
    load_full_barec_gold_vocab,
    load_full_tarab_vocab,
    vocab_overlap_stats,
)

OUT = DATA_DIR / "embedded" / "tarab_full_corpus_overlap.json"


def main() -> None:
    if not TARAB_FULL_CSV.exists():
        raise SystemExit(f"Missing corpus CSV: {TARAB_FULL_CSV}")

    barec_vocab, barec_meta, splits_loaded = load_full_barec_gold_vocab()
    if barec_vocab is None or barec_meta is None or not splits_loaded:
        raise SystemExit("Could not load BAREC parquets")

    tarab_vocab, tarab_sentences = load_full_tarab_vocab()
    if tarab_vocab is None:
        raise SystemExit("Could not build Tarab vocab from CSV")

    stats = vocab_overlap_stats(barec_vocab, tarab_vocab)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tarab_sentences": tarab_sentences,
        "corpus_stats": stats,
        "barec_meta": {
            "total_records": barec_meta["total_records"],
            "unique_sentence_texts": barec_meta["unique_sentence_texts"],
            "per_split_records": barec_meta["per_split_records"],
        },
        "splits_loaded": splits_loaded,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
