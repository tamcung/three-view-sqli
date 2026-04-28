#!/usr/bin/env python3
"""
WAF-A-MoLE three-view dataset with on-disk caching.

Usage:
    pre = SamplePreprocessor()
    train_set = WafamoleDataset("train", limit_per_class=5000, preprocessor=pre)
    loader = DataLoader(train_set, batch_size=32, collate_fn=collate_three_view, shuffle=True)
"""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .preprocessing import SamplePreprocessor
except ImportError:
    from preprocessing import SamplePreprocessor

# Data and cache locations are configurable via env vars (RunPod-friendly).
WAFAMOLE_ROOT = Path(os.environ.get("WAFAMOLE_ROOT", "data/wafamole"))
CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", "data/cache"))

ATTACKS = WAFAMOLE_ROOT / "attacks.sql.statements.jsonl"
SANE = WAFAMOLE_ROOT / "sane.sql.statements.jsonl"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _stream_split(path: Path, n_train: int, n_val: int, n_test: int, label: int, seed: int = 42):
    """Yield (split_name, label, text) tuples.

    Assigns first n_train to train, next n_val to val, next n_test to test.
    Uses a deterministic shuffle of indices via seed.
    """
    # Streaming read into list (~393k items, manageable in memory)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line)["text"])
            except Exception:
                continue
    rng = random.Random(seed)
    rng.shuffle(items)
    splits = {
        "train": items[:n_train],
        "val":   items[n_train:n_train + n_val],
        "test":  items[n_train + n_val:n_train + n_val + n_test],
    }
    for split_name, texts in splits.items():
        for t in texts:
            yield split_name, label, t


def build_split_files(
    n_train_per_class: int,
    n_val_per_class: int,
    n_test_per_class: int,
    seed: int = 42,
    overwrite: bool = False,
):
    """Build train/val/test JSONL files at the given per-class sizes.

    Stored at: cache/split_{split}_seed{seed}_n{train}-{val}-{test}.jsonl
    """
    suffix = f"seed{seed}_n{n_train_per_class}-{n_val_per_class}-{n_test_per_class}"
    out_paths = {
        s: CACHE_ROOT / f"split_{s}_{suffix}.jsonl"
        for s in ("train", "val", "test")
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths

    print(f"Building splits ({suffix}) ...")
    files = {s: open(p, "w", encoding="utf-8") for s, p in out_paths.items()}
    try:
        for split, label, text in _stream_split(
            ATTACKS, n_train_per_class, n_val_per_class, n_test_per_class, label=1, seed=seed
        ):
            files[split].write(json.dumps({"label": label, "text": text}, ensure_ascii=False) + "\n")
        for split, label, text in _stream_split(
            SANE, n_train_per_class, n_val_per_class, n_test_per_class, label=0, seed=seed + 1
        ):
            files[split].write(json.dumps({"label": label, "text": text}, ensure_ascii=False) + "\n")
    finally:
        for f in files.values():
            f.close()

    for s, p in out_paths.items():
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {s}: {n} samples → {p}")
    return out_paths


def preprocess_split_file(jsonl_path: Path, preprocessor: SamplePreprocessor, overwrite: bool = False) -> Path:
    """Preprocess a JSONL split file into a .pt cache.

    Returns path to the .pt cache.
    """
    cache_path = jsonl_path.with_suffix(".pt")
    if cache_path.exists() and not overwrite:
        return cache_path

    print(f"Preprocessing {jsonl_path.name} ...")
    samples = []
    t0 = time.time()
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                rate = i / (time.time() - t0)
                print(f"  {i} ({rate:.0f}/s)...")
            obj = json.loads(line)
            try:
                feats = preprocessor(obj["text"])
            except Exception as e:
                # Fall back to all-CLS, label remains
                feats = {
                    "surface_ids": [preprocessor.surface_cls, preprocessor.surface_sep],
                    "surface_mask": [1, 1],
                    "lex_ids": [2],  # <CLS>
                    "lex_mask": [1],
                    "ast_ids": [2],
                    "ast_mask": [1],
                    "ast_valid": 0,
                }
            feats["label"] = obj["label"]
            samples.append(feats)
    print(f"  done {len(samples)} in {time.time()-t0:.0f}s")

    torch.save(samples, cache_path)
    print(f"  → {cache_path}")
    return cache_path


class WafamoleThreeViewDataset(Dataset):
    """In-memory three-view dataset loaded from a preprocessed .pt cache."""

    def __init__(self, cache_path: Path):
        self.samples = torch.load(cache_path, weights_only=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _pad(seq: list[int], max_len: int, pad_id: int = 0) -> list[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))


def collate_three_view(batch: list[dict], surface_pad_id: int = 1) -> dict:
    """Collate a list of preprocessed samples into a batch tensor dict.

    Pads each view independently to the max length in the batch.
    """
    max_S = max(len(b["surface_ids"]) for b in batch)
    max_L = max(len(b["lex_ids"]) for b in batch)
    max_A = max(len(b["ast_ids"]) for b in batch)

    out = {
        "surface_ids":  [],
        "surface_mask": [],
        "lex_ids":      [],
        "lex_mask":     [],
        "ast_ids":      [],
        "ast_mask":     [],
        "ast_valid":    [],
        "label":        [],
    }
    for b in batch:
        out["surface_ids"].append(_pad(b["surface_ids"], max_S, pad_id=surface_pad_id))
        out["surface_mask"].append(_pad(b["surface_mask"], max_S, pad_id=0))
        out["lex_ids"].append(_pad(b["lex_ids"], max_L, pad_id=0))
        out["lex_mask"].append(_pad(b["lex_mask"], max_L, pad_id=0))
        out["ast_ids"].append(_pad(b["ast_ids"], max_A, pad_id=0))
        out["ast_mask"].append(_pad(b["ast_mask"], max_A, pad_id=0))
        out["ast_valid"].append(b["ast_valid"])
        out["label"].append(b["label"])

    return {
        "surface_ids":  torch.tensor(out["surface_ids"], dtype=torch.long),
        "surface_mask": torch.tensor(out["surface_mask"], dtype=torch.bool),
        "lex_ids":      torch.tensor(out["lex_ids"], dtype=torch.long),
        "lex_mask":     torch.tensor(out["lex_mask"], dtype=torch.bool),
        "ast_ids":      torch.tensor(out["ast_ids"], dtype=torch.long),
        "ast_mask":     torch.tensor(out["ast_mask"], dtype=torch.bool),
        "ast_valid":    torch.tensor(out["ast_valid"], dtype=torch.long),
        "label":        torch.tensor(out["label"], dtype=torch.long),
    }


if __name__ == "__main__":
    pre = SamplePreprocessor()
    paths = build_split_files(n_train_per_class=2500, n_val_per_class=500, n_test_per_class=500)
    cache_paths = {s: preprocess_split_file(p, pre) for s, p in paths.items()}

    ds_train = WafamoleThreeViewDataset(cache_paths["train"])
    ds_val = WafamoleThreeViewDataset(cache_paths["val"])
    ds_test = WafamoleThreeViewDataset(cache_paths["test"])
    print(f"\nDataset sizes:")
    print(f"  train: {len(ds_train)}")
    print(f"  val:   {len(ds_val)}")
    print(f"  test:  {len(ds_test)}")

    loader = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=collate_three_view)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k:15s} {tuple(v.shape)}  dtype={v.dtype}")
