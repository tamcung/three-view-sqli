#!/usr/bin/env python3
"""Pre-build the three-view feature cache for all splits.

Without this step, train.py / evaluate.py do the preprocessing lazily on
first load. Running this once up-front is convenient when iterating over
multiple training runs.

Output:
  data/cache/train.pkl
  data/cache/val.pkl
  data/cache/test.pkl
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SamplePreprocessor
from dataset import SQLDataset


def main():
    logging.basicConfig(level=logging.INFO,
                          format="%(asctime)s  %(message)s",
                          datefmt="%H:%M:%S")
    log = logging.getLogger("preprocess")

    pre = SamplePreprocessor()
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        jsonl = ROOT / "data" / "splits" / f"{split}.jsonl"
        cache = cache_dir / f"{split}.pkl"
        if cache.exists():
            log.info(f"{split}: cache already exists at {cache} — skipping")
            continue
        log.info(f"{split}: preprocessing {jsonl}")
        ds = SQLDataset(jsonl, cache, pre)
        log.info(f"{split}: cached {len(ds)} samples → {cache}")


if __name__ == "__main__":
    main()
