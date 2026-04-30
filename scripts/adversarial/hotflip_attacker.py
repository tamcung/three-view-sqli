#!/usr/bin/env python3
"""HotFlip-style gradient-guided adversarial attacker (§4.3 gradient-based).

Implements a discrete optimization attack against the multi-view victim:
    1. Encode the seed payload to (surface_ids, char_ids, lex_ids).
    2. Forward through a re-implemented `BPECharLexStageModel` path that
       takes the char-level *embedding* as input (so we can backprop into
       it). Extract gradient w.r.t. the char embedding.
    3. Rank (position, candidate_char) pairs by the linearization
            score(t, v) = <- ∂L/∂e[t], W[v] - W[char_ids[t]]>
       i.e., how much swapping char at position t for byte v decreases the
       attack-prob, to first order.
    4. For the top-K candidate edits, do an honest full re-tokenized
       forward pass and pick the one that lowers the model's prob the
       most while keeping the payload functionally SQLi.
    5. Apply the chosen edit, repeat for at most `n_flips` iterations.

Compared to pure HotFlip, the validation step uses the real (re-tokenized)
forward pass — gradient is only used to *prune* the candidate space. This
controls for the approximation error introduced by the BPE / lexical
views, which depend on the raw string in a non-differentiable way.

Limitation: only supports BPECharLexStageModel today. The gradient hook
is model-specific. Other variants would need an analogous custom forward.

Usage:
    python -m scripts.adversarial.hotflip_attacker \
        --checkpoint results/tri_view_stage_aug/best_checkpoint.pt \
        --output data/adversarial/hotflip_v1.jsonl \
        --n-seeds 100 --n-flips 12 --top-k-per-iter 32
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adversarial.utils import (
    load_victim, batch_predict, is_functional_sqli,
    load_seed_attacks, save_adv_records, setup_logger,
    _MemDataset,
)

sys.path.insert(0, str(ROOT / "src"))
from dataset import collate_three_view, move_batch_to                  # noqa: E402


# ============================================================
# Model-specific custom forward
# ============================================================
def _stage_forward_with_char_emb(model, batch, char_emb):
    """BPECharLexStageModel forward, but with char_emb injected directly.

    char_emb: [1, T, D]  — pre-embedding tensor with requires_grad.
    """
    s_out = model.surface_enc(batch["surface_ids"], batch["surface_mask"])
    H_S = model.surface_proj(s_out["full"])

    # CharCNN with our injected embedding
    x = char_emb.transpose(1, 2)                                  # [B, D, T]
    per_kernel = [F.relu(conv(x)) for conv in model.char_convs]
    H = torch.cat(per_kernel, dim=1)                              # [B, F*n, T]
    z_C_raw = F.adaptive_max_pool1d(H, 1).squeeze(-1)             # [B, F*n]
    z_C = model.char_proj(z_C_raw)                                # [B, d]

    z_L = model.lex_enc(batch["lex_ids"], batch["lex_mask"])["pooled"]

    abstract_seq = torch.stack([z_L, z_C], dim=1)
    q1 = model.s1_norm1(abstract_seq)
    s1_out, _ = model.s1_self_attn(q1, q1, q1, need_weights=False)
    abstract_seq = abstract_seq + s1_out
    ffn1 = model.s1_ffn(model.s1_norm2(abstract_seq))
    abstract_seq = abstract_seq + ffn1

    q2 = model.s2_norm_q(abstract_seq)
    kv = model.s2_norm_kv(H_S)
    kv_pad = ~batch["surface_mask"].bool()
    attn_out, _ = model.s2_cross_attn(
        query=q2, key=kv, value=kv,
        key_padding_mask=kv_pad, need_weights=False,
    )
    attended_seq = abstract_seq + attn_out
    ffn2 = model.s2_ffn(model.s2_norm_ffn(attended_seq))
    attended_seq = attended_seq + ffn2

    z_LA = abstract_seq.mean(dim=1)
    z_final = attended_seq.mean(dim=1)
    cls_input = torch.cat([z_LA, z_final], dim=-1)
    return model.classifier(cls_input).squeeze(-1)


# ============================================================
# Encode a single string and build the model batch
# ============================================================
def _build_batch(pre, sql: str, device):
    rec = {**pre(sql), "label_int": 1, "meta": {}}
    batch = collate_three_view([rec])
    return move_batch_to(batch, device)


# ============================================================
# Gradient-guided ranking
# ============================================================
# Candidate replacement byte universe — printable ASCII + control bytes that
# show up in real WAF-bypass payloads. We exclude the PAD byte (id=0).
_CAND_BYTES_RAW = (
    list(range(0x20, 0x7F))   # printable
    + [0x09, 0x0A, 0x0B, 0x0C, 0x0D]  # whitespace
)
_CAND_IDS = [b + 1 for b in _CAND_BYTES_RAW]  # +1 because 0 is PAD


def hotflip_step(
    model, pre, device, sql: str, n_top: int = 32,
) -> list[tuple[int, int, float]]:
    """One HotFlip iteration: returns top-N (position, byte_id, score) tuples
    where higher score = better candidate (more likely to lower attack prob).

    Position is the byte offset into `sql.encode("utf-8")`.
    Returned byte_id is the model's char vocab id (= byte+1).
    """
    raw_bytes = sql.encode("utf-8")
    if not raw_bytes:
        return []

    batch = _build_batch(pre, sql, device)
    char_ids = batch["char_ids"]                     # [1, T]

    # First-token embedding lookup, mark for grad
    emb = model.char_embed(char_ids).detach()        # [1, T, D]
    emb.requires_grad_(True)

    # Forward + backward
    model.eval()
    p_main = _stage_forward_with_char_emb(model, batch, emb)   # [1] logit
    # Loss: maximize probability of being benign = minimize attack prob.
    # logit = log p / (1-p). Push logit DOWN.
    loss = p_main.sum()
    if model.char_embed.weight.grad is not None:
        model.char_embed.weight.grad.zero_()
    loss.backward()
    g = emb.grad.detach()[0]                         # [T, D]

    W = model.char_embed.weight.detach()             # [V, D]
    cur_ids = char_ids[0].detach()                   # [T]

    # score(t, v) = <-g[t], W[v] - W[cur_ids[t]]>     (we want logit DOWN)
    # We compute it in one matmul.
    T = g.size(0)
    cand_ids_t = torch.tensor(_CAND_IDS, device=device, dtype=torch.long)
    Wc = W[cand_ids_t]                                # [Vc, D]
    cur_W = W[cur_ids]                                # [T, D]

    # delta[t, v] = Wc[v] - cur_W[t]                  shape [T, Vc, D]
    # score[t, v] = <-g[t], delta[t, v]>             shape [T, Vc]
    # vectorize: score = -g @ Wc.T + (g * cur_W).sum(-1, keepdim=True)
    score = (-g @ Wc.T) + (g * cur_W).sum(dim=-1, keepdim=True)
    # score shape: [T, Vc]

    # We can only edit positions that correspond to actual bytes in `sql`
    # (not PAD). T may include CLS or just be the raw byte count.
    n_real = min(len(raw_bytes), T)
    score = score[:n_real]                            # [n_real, Vc]

    # Mask: don't replace a byte with itself
    cur_in_real = cur_ids[:n_real]                    # [n_real]
    self_mask = (cand_ids_t.unsqueeze(0) == cur_in_real.unsqueeze(1))  # [n_real, Vc]
    score = score.masked_fill(self_mask, float("-inf"))

    # Take top-N candidates globally
    flat = score.flatten()
    n_top = min(n_top, flat.numel())
    top_vals, top_idx = torch.topk(flat, n_top)
    Vc = score.size(1)
    pos = (top_idx // Vc).tolist()
    cand = (top_idx % Vc).tolist()
    out = []
    for p, c, s in zip(pos, cand, top_vals.tolist()):
        out.append((int(p), int(_CAND_IDS[c]), float(s)))
    return out


def apply_byte_edit(sql: str, byte_pos: int, new_byte_id: int,
                     op: str = "sub") -> str | None:
    """Apply a single-byte edit. op ∈ {sub, ins, del}.
    For 'del', new_byte_id is ignored.
    Returns new string or None if invalid."""
    raw = bytearray(sql.encode("utf-8"))
    if not (0 <= byte_pos < len(raw) + (1 if op == "ins" else 0)):
        return None
    if op == "sub":
        new_byte = new_byte_id - 1
        if not (0 <= new_byte < 256):
            return None
        if not (0 <= byte_pos < len(raw)):
            return None
        raw[byte_pos] = new_byte
    elif op == "ins":
        new_byte = new_byte_id - 1
        if not (0 <= new_byte < 256):
            return None
        raw.insert(byte_pos, new_byte)
    elif op == "del":
        if not (0 <= byte_pos < len(raw)):
            return None
        del raw[byte_pos]
    else:
        return None
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


# ============================================================
# Per-seed attack
# ============================================================
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def attack_one_seed(
    seed: str,
    model, pre, device, accepted: set[str],
    threshold: float = 0.5,
    n_flips: int = 12,
    n_top_per_iter: int = 32,
    require_functional: bool = True,
) -> tuple[str, float, list[dict]]:
    """Returns (best_payload, best_prob, edit_history).

    Operates in logit space: when sigmoid saturates near 1.0, prob comparisons
    become useless — but logits keep ranking information. We always pick the
    candidate that lowers the logit, then convert at the end.
    """
    cur = seed
    history = []

    base_logit = batch_predict(model, pre, device, accepted, [seed],
                                batch_size=1, return_logits=True)[0]
    best_payload, best_logit = seed, float(base_logit)

    for it in range(n_flips):
        cur_logit = batch_predict(model, pre, device, accepted, [cur],
                                    batch_size=1, return_logits=True)[0]
        cur_prob = _sigmoid(cur_logit)
        if cur_prob < threshold:
            if cur_logit < best_logit:
                best_logit = float(cur_logit)
                best_payload = cur
            if it >= 2:
                break

        try:
            cands = hotflip_step(model, pre, device, cur,
                                  n_top=n_top_per_iter)
        except Exception as e:
            logging.warning(f"  hotflip_step failed: {e}")
            break
        if not cands:
            break

        # Honest re-tokenized scoring. For each gradient-suggested (pos, byte),
        # try three operations: substitute, insert, delete (delete ignores
        # byte_id). This expands the candidate space at no extra gradient cost.
        candidate_payloads = []
        keep_meta = []
        for (bp, bid, sc) in cands:
            for op in ("sub", "ins", "del"):
                new_pay = apply_byte_edit(cur, bp, bid, op=op)
                if new_pay is None or new_pay == cur:
                    continue
                candidate_payloads.append(new_pay)
                keep_meta.append((bp, bid, sc, op))
        if not candidate_payloads:
            break

        logits = batch_predict(model, pre, device, accepted, candidate_payloads,
                               batch_size=64, return_logits=True)
        order = np.argsort(logits)   # ascending: smallest logit first
        chosen = None
        for j in order:
            new_pay = candidate_payloads[j]
            new_logit = float(logits[j])
            if new_logit >= cur_logit:
                continue
            if require_functional and not is_functional_sqli(new_pay):
                continue
            chosen = (new_pay, new_logit, j)
            break

        if chosen is None:
            history.append({"iter": it, "no_improvement": True,
                            "cur_prob": float(cur_prob),
                            "cur_logit": float(cur_logit)})
            break

        new_pay, new_logit, j = chosen
        bp, bid, sc, op = keep_meta[j]
        history.append({
            "iter": it, "byte_pos": bp, "byte_id": bid, "op": op,
            "linear_score": sc,
            "before_logit": float(cur_logit), "after_logit": new_logit,
            "before_prob": float(cur_prob),
            "after_prob": float(_sigmoid(new_logit)),
        })
        cur = new_pay
        if new_logit < best_logit:
            best_payload, best_logit = cur, new_logit

    best_prob = float(_sigmoid(best_logit))
    return best_payload, best_prob, history


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed-split", type=str,
                    default=str(ROOT / "data" / "splits" / "test.jsonl"))
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-flips", type=int, default=12)
    p.add_argument("--top-k-per-iter", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-seeds-already-broken", action="store_true")
    p.add_argument("--no-require-functional", action="store_true")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger("adv.hotflip", out_path.with_suffix(".log"))
    rng = random.Random(args.seed)

    # ---- victim ----
    log.info(f"Loading victim from {args.checkpoint}")
    model, pre, device, variant, accepted = load_victim(args.checkpoint)
    if variant != "bpe_char_lex_stage":
        log.warning(f"  HotFlip is only validated on bpe_char_lex_stage; got {variant}. "
                     f"You may need to extend _stage_forward_with_char_emb.")

    # ---- seeds ----
    seeds = load_seed_attacks(args.seed_split, n=args.n_seeds, seed=args.seed)
    base_probs = batch_predict(model, pre, device, accepted,
                                [s["user_input"] for s in seeds])
    n_already_broken = int((base_probs < args.threshold).sum())
    log.info(f"  loaded {len(seeds)} seeds  pre-broken={n_already_broken}")

    # ---- attack ----
    adv_records = []
    per_seed = []
    n_success, n_skip = 0, 0
    t0 = time.time()
    for idx, seed_row in enumerate(seeds):
        seed_text = seed_row["user_input"]
        if args.limit_seeds_already_broken and base_probs[idx] < args.threshold:
            n_skip += 1
            continue

        best_payload, best_prob, history = attack_one_seed(
            seed_text, model, pre, device, accepted,
            threshold=args.threshold, n_flips=args.n_flips,
            n_top_per_iter=args.top_k_per_iter,
            require_functional=not args.no_require_functional,
        )
        per_seed.append({
            "id": seed_row["id"],
            "base_prob": float(base_probs[idx]),
            "best_prob": float(best_prob),
            "n_iters": len(history),
            "history": history,
        })

        if best_prob < args.threshold and best_payload != seed_text:
            n_success += 1
            adv_records.append({
                "user_input": best_payload,
                "label": "attack",
                "source": "adv_hotflip",
                "seed_id": seed_row["id"],
                "seed_input": seed_text,
                "n_flips": len(history),
                "model_prob": float(best_prob),
                "technique": seed_row.get("technique"),
            })

        if (idx + 1) % 10 == 0 or idx == len(seeds) - 1:
            attempted = idx + 1 - n_skip
            asr = n_success / max(1, attempted)
            log.info(f"  [{idx+1:>4d}/{len(seeds)}]  succ={n_success}  "
                      f"ASR={asr:.2%}  elapsed={time.time()-t0:.0f}s")

    n_written = save_adv_records(out_path, adv_records)
    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "checkpoint": str(args.checkpoint),
                "n_seeds": len(seeds),
                "n_flips": args.n_flips,
                "top_k_per_iter": args.top_k_per_iter,
                "threshold": args.threshold,
            },
            "n_already_broken": n_already_broken,
            "n_skipped": n_skip,
            "n_attempted": len(seeds) - n_skip,
            "n_success": n_success,
            "asr": n_success / max(1, len(seeds) - n_skip),
            "per_seed": per_seed,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n  Wrote {n_written} adv records to {out_path}")
    log.info(f"  ASR: {n_success}/{len(seeds) - n_skip} = "
              f"{n_success / max(1, len(seeds) - n_skip):.2%}")


if __name__ == "__main__":
    main()
