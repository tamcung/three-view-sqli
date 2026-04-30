# Three-View SQL Injection Detection

A multi-view SQL injection detector that combines **surface (raw text)**, **lexical (libinjection-style type codes)**, and **AST (sqlglot syntax tree)** views with hierarchical fusion.

## Architecture

```
SQL string
   │
   ├─► Surface  ─►  4-layer Transformer (BPE)  ──► H_S [B,512,384]
   │
   ├─► Lexical  ─►  4-layer Transformer (24 vocab)  ──► z_L [B,256]
   │                                                       │
   └─► AST      ─►  4-layer Transformer (~85 vocab)  ──► z_A [B,256]
                                                          │
                                                  Stage 1: self-attention
                                                  over [z_L, z_A]
                                                          │
                                                  Stage 2: cross-attention
                                                  Q = abstract_seq, K,V = H_S
                                                          │
                                                  Stage 3: concat + classifier
                                                          │
                                                          ▼
                                                       p_main
```

Plus deep supervision: each view also has its own auxiliary classifier head, trained with combined loss `0.7·L_main + 0.1·L_S + 0.1·L_L + 0.1·L_A`.

## Dataset

Built from MyBatis Mapper.xml templates harvested from four open-source projects (mall, jeecg-boot, RuoYi-Vue, ruoyi-vue-pro). Pipeline:

1. **`extract_raw_sql.py`**: 528 raw `<select>/<update>/<insert>/<delete>` statements
2. **`expand_dynamic_sql.py`**: 1,200 deterministic variants (one variant per `<if>` activation pattern)
3. **`verify_sql_parses.py`** + **`classify_injection_points.py`**: 1,916 user-controllable injection slots, each annotated with `(quote_status, payload_type, sigil, clause_context)`
4. **`classify_attack_payloads.py`** → **`merge_payload_pool.py`** → **`validate_payloads.py`** → **`dedupe_payload_pool.py`** → **`programmatic_fillers.py`** → **`programmatic_diversity.py`** → **`build_unified_attack_pool.py`**: 378 MySQL-only attack payloads, all 8 main techniques with ≥15 unique structural variants (after pure-constant collapse)
5. **`build_benign_pools.py`** + **`build_probe_pool.py`**: 32k benign strings (Faker multilingual + special-chars + edge-case) and 4,980 attack-keyword-text probes (hard negatives)
6. **`synthesize_dataset.py`**: matched-pair synthesis — for each (template, victim slot) pair, generate K attack-benign twins sharing other-slot fillers; per-context budget rebalances numeric/identifier/string slots; cap `attack_breaks_parsing` ≤ 10%
7. **`split_dataset.py`**: AST-equivalence-class disjoint stratified train/val/test (70/15/15) — every (label × context × subtype/technique) stratum is split with zero AST overlap

Final dataset: `data/synthesized_dataset.jsonl` (100,194 unique samples, 49.5k attack / 50.7k benign), splits at `data/splits/{train,val,test}.jsonl`.

Re-run any pipeline stage with the corresponding `scripts/<name>.py`. The full chain takes ~30 minutes on a single multi-core machine.

## Quick start

```bash
# 1. Compile libinjection (native C lib used by lexical view)
bash scripts/build_libinjection.sh    # Linux / macOS
# or
scripts/build_libinjection.bat        # Windows MSVC

# 2. Python deps
pip install -r requirements.txt

# 3. Pre-build the feature cache (one-time, ~few minutes)
python scripts/preprocess_dataset.py

# 4. Smoke test (1 epoch, 2k samples — verifies the pipeline runs end-to-end)
python train.py --config configs/smoke.yaml --output results/smoke/

# 5. Main training (5 epochs, full data)
python train.py --config configs/main.yaml --output results/main/

# 6. Standalone evaluation against baselines (libinjection, TF-IDF + LR)
python evaluate.py --checkpoint results/main/best_checkpoint.pt --output results/main/eval/
```

### Ablations

Drop one or two views to test each view's marginal contribution:

```bash
python train.py --config configs/ablate_surface_only.yaml  --output results/abl_S/
python train.py --config configs/ablate_lexical_only.yaml  --output results/abl_L/
python train.py --config configs/ablate_ast_only.yaml      --output results/abl_A/
python train.py --config configs/ablate_no_surface.yaml    --output results/abl_LA/
python train.py --config configs/ablate_no_lexical.yaml    --output results/abl_SA/
python train.py --config configs/ablate_no_ast.yaml        --output results/abl_SL/
```

To regenerate the dataset from scratch (requires `external/PayloadsAllTheThings/` and the four MyBatis projects checked out as siblings of this repo):

```bash
python scripts/extract_raw_sql.py
python scripts/expand_dynamic_sql.py
python scripts/verify_sql_parses.py
python scripts/classify_injection_points.py
python scripts/classify_attack_payloads.py
python scripts/extract_paa_extra.py
python scripts/merge_payload_pool.py
python scripts/validate_payloads.py
python scripts/programmatic_fillers.py
python scripts/dedupe_payload_pool.py
python scripts/programmatic_diversity.py
python scripts/build_unified_attack_pool.py
python scripts/build_probe_pool.py
python scripts/build_benign_pools.py
python scripts/synthesize_dataset.py --workers 4
python scripts/split_dataset.py --workers 4
```

## Repository layout

```
three-view-sqli/
├── src/
│   ├── libinjection_wrapper.py   # ctypes wrapper around libinjection.{so,dll}
│   ├── preprocessing.py           # SQL → 3-view token ids
│   ├── dataset.py                 # SQLDataset + collate_three_view
│   ├── model.py                   # ThreeViewModel
│   └── ablation_models.py         # single-view + two-view variants
├── scripts/                       # dataset generation pipeline + utilities
│   ├── build_schema_map.py        # Phase 2.6: parse CREATE TABLE → column types
│   ├── extract_raw_sql.py         # Phase 1: extract raw SQL from Mapper.xml
│   ├── expand_dynamic_sql.py      # Phase 2: expand <if> variants
│   ├── verify_sql_parses.py       # Phase 2.5: parser audit
│   ├── classify_injection_points.py     # Phase 3: schema-driven slot typing
│   ├── classify_attack_payloads.py      # Phase 4a: classify by technique
│   ├── extract_paa_extra.py       # Phase 4b: pull PaA MySQL files
│   ├── merge_payload_pool.py      # Phase 4c: dedupe + merge
│   ├── validate_payloads.py       # Phase 4d: per-slot effectiveness
│   ├── programmatic_fillers.py    # Phase 4e: fill (technique × context) gaps
│   ├── dedupe_payload_pool.py     # Phase 4f: AST-equivalence-class collapse
│   ├── programmatic_diversity.py  # Phase 4g: ≥15 unique structures per technique
│   ├── build_unified_attack_pool.py     # Phase 4h: final attack pool
│   ├── build_benign_pools.py      # Phase 4i: benign Faker pools
│   ├── build_probe_pool.py        # Phase 4j: hard-negative probe pool
│   ├── synthesize_dataset.py      # Phase 5: matched-pair synthesis
│   ├── split_dataset.py           # Phase 6: AST-disjoint splits
│   ├── preprocess_dataset.py      # Pre-build feature cache (run once before training)
│   ├── build_libinjection.sh      # native libinjection build
│   └── build_libinjection.bat     # Windows libinjection build
├── configs/                       # training configs
│   ├── main.yaml                  # full three-view, 5 epochs
│   ├── smoke.yaml                 # 1 epoch / 2k subset for pipeline check
│   ├── ablate_surface_only.yaml
│   ├── ablate_lexical_only.yaml
│   ├── ablate_ast_only.yaml
│   ├── ablate_no_surface.yaml
│   ├── ablate_no_lexical.yaml
│   └── ablate_no_ast.yaml
├── data/                          # dataset artifacts (synthesized + splits + schema)
├── external/libinjection/         # vendored C source (BSD license)
├── lib/                           # compiled libinjection.{dll,so}
├── train.py                       # training entrypoint
├── evaluate.py                    # standalone eval + baselines (libinjection, TF-IDF+LR)
└── requirements.txt
```

## Outputs (per `results/<run_name>/`)

| File | Description |
|---|---|
| `train.log` | Full training log |
| `config.yaml` | Resolved config (for reproducibility) |
| `metrics.json` | Per-epoch train/val + final test metrics |
| `best_checkpoint.pt` | Best val-F1 checkpoint |
| `final_checkpoint.pt` | Final epoch checkpoint |
| `test_predictions.npz` | Test logits + labels (for offline analysis) |
| `eval/evaluation.json` | Stratified test metrics + baseline comparison |
| `eval/evaluation.log` | Eval run log |

## License

Code: MIT (see LICENSE in repo root).
Vendored libinjection: BSD-3-Clause (see `external/libinjection/LICENSE`).
