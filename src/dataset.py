#!/usr/bin/env python3
"""Three-view SQL injection dataset.

Loads JSONL splits produced by `scripts/split_dataset.py` and converts each
SQL string to surface / lexical / AST token-id arrays via SamplePreprocessor.

Caches the preprocessed tensors so subsequent epochs are I/O-free.
"""
from __future__ import annotations
import json
import logging
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from .preprocessing import SamplePreprocessor
except ImportError:
    from preprocessing import SamplePreprocessor


META_KEYS = (
    "label",
    "source",      # httpparams / sqliv3 / sqlmap / httpparams_norm / sqliv3_valid / llm
    "subtype",     # for benigns: real_param / llm_keyword_in_text / llm_sql_mimicking
    "technique",   # for sqlmap attacks: time_blind / boolean_blind / etc.
    "id",
    "ast_sig",
)


class SQLDataset(Dataset):
    """Wraps a JSONL split. Pre-computes three-view token ids on first load.

    Args:
        jsonl_path: path to one of train/val/test.jsonl
        cache_path: where to read/write the preprocessed pickle. None disables.
        preprocessor: SamplePreprocessor instance (only used on cache miss).
        max_samples: optional cap (debug).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        cache_path: str | Path | None = None,
        preprocessor: SamplePreprocessor | None = None,
        max_samples: int | None = None,
        verbose: bool = True,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.cache_path = Path(cache_path) if cache_path else None
        self._log = logging.getLogger(self.__class__.__name__)

        if self.cache_path and self.cache_path.exists():
            if verbose:
                self._log.info(f"Loading cached features: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.records = pickle.load(f)
            if max_samples:
                self.records = self.records[:max_samples]
            return

        # Cache miss — preprocess from scratch
        if preprocessor is None:
            raise ValueError(
                f"Cache miss for {self.jsonl_path}; pass preprocessor to build."
            )

        if verbose:
            self._log.info(f"Preprocessing {self.jsonl_path} (cache miss)...")

        with open(self.jsonl_path, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f]
        if max_samples:
            raw = raw[:max_samples]

        self.records = []
        for i, rec in enumerate(raw):
            features = preprocessor(rec["user_input"])
            entry = {
                **features,
                "label_int": 1 if rec["label"] == "attack" else 0,
                "meta": {k: rec.get(k) for k in META_KEYS},
            }
            self.records.append(entry)
            if verbose and (i + 1) % 10_000 == 0:
                self._log.info(f"  preprocessed {i + 1}/{len(raw)}")

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.records, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                self._log.info(f"Wrote cache: {self.cache_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ============================================================
# Collate: pad each view independently to the batch's max length
# ============================================================
SURFACE_PAD_ID = 1   # CodeBERT/RoBERTa pad
LEX_PAD_ID = 0
AST_PAD_ID = 0
CHAR_PAD_ID = 0


def _pad_field(batch: list[dict], key: str, pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(b[key]) for b in batch)
    ids = torch.full((len(batch), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b[key])
        ids[i, :n] = torch.tensor(b[key], dtype=torch.long)
        mask[i, :n] = 1
    return ids, mask


def collate_three_view(batch: list[dict]) -> dict:
    """Collate function — pads per-view to batch max.

    For Tree-LSTM, also passes through the variable-length tree structure
    (`ast_node_ids` and `ast_parent`) as Python lists. Most models ignore
    these.
    """
    surface_ids, surface_mask = _pad_field(batch, "surface_ids", SURFACE_PAD_ID)
    lex_ids, lex_mask = _pad_field(batch, "lex_ids", LEX_PAD_ID)
    ast_ids, ast_mask = _pad_field(batch, "ast_ids", AST_PAD_ID)
    # char_ids may not exist on records cached before CharCNN was added — fallback
    if "char_ids" in batch[0]:
        char_ids, char_mask = _pad_field(batch, "char_ids", CHAR_PAD_ID)
    else:
        B = len(batch)
        char_ids = torch.zeros((B, 1), dtype=torch.long)
        char_mask = torch.zeros((B, 1), dtype=torch.long)

    ast_valid = torch.tensor([b["ast_valid"] for b in batch], dtype=torch.float)
    labels = torch.tensor([b["label_int"] for b in batch], dtype=torch.long)

    return {
        "surface_ids": surface_ids,
        "surface_mask": surface_mask,
        "lex_ids": lex_ids,
        "lex_mask": lex_mask,
        "ast_ids": ast_ids,
        "ast_mask": ast_mask,
        "ast_valid": ast_valid,
        "ast_node_ids": [b.get("ast_node_ids", []) for b in batch],
        "ast_parent":   [b.get("ast_parent", [])   for b in batch],
        "char_ids": char_ids,
        "char_mask": char_mask,
        "labels": labels,
        "meta": [b["meta"] for b in batch],
    }


def move_batch_to(batch: dict, device: torch.device) -> dict:
    """Move tensor fields onto device; keep `meta` (list of dicts) intact."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    pre = SamplePreprocessor()
    ds = SQLDataset(
        jsonl_path=ROOT / "data" / "splits" / f"{args.split}.jsonl",
        cache_path=ROOT / "data" / "cache" / f"{args.split}.pkl",
        preprocessor=pre,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(ds)} samples from {args.split}")
    print(f"First record:")
    r = ds[0]
    print(f"  surface_ids[:10]: {r['surface_ids'][:10]} ... (len={len(r['surface_ids'])})")
    print(f"  lex_ids:          {r['lex_ids'][:20]} ... (len={len(r['lex_ids'])})")
    print(f"  ast_ids[:10]:     {r['ast_ids'][:10]} ... (len={len(r['ast_ids'])})")
    print(f"  ast_valid:        {r['ast_valid']}")
    print(f"  label_int:        {r['label_int']}")
    print(f"  meta:             {r['meta']}")

    # Collate test
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_three_view)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k in ("surface_ids", "lex_ids", "ast_ids"):
        print(f"  {k}: {batch[k].shape}")
    print(f"  labels: {batch['labels']}")
