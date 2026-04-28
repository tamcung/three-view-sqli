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

## Quick start (RunPod or any Linux box with NVIDIA GPU)

```bash
# 1. Clone
git clone git@github.com:tamcung/three-view-sqli.git
cd three-view-sqli

# 2. One-shot setup (compiles libinjection, installs deps, runs sanity checks)
bash setup.sh

# 3. Place WAF-A-MoLE dataset (download from https://github.com/AvalZ/WAF-A-MoLE-dataset)
#    into data/wafamole/
#    Should contain:
#      data/wafamole/attacks.sql.statements.jsonl
#      data/wafamole/sane.sql.statements.jsonl

# 4. Train (medium config, ~5-10 min on RTX 4090 with BF16)
python train.py --config configs/medium.yaml --output results/run_001/

# 5. Train full (≈30-50 min on RTX 4090)
python train.py --config configs/full.yaml --output results/run_002/

# 6. Evaluate against baselines (libinjection, TF-IDF + LR)
python evaluate.py --checkpoint results/run_002/best_checkpoint.pt --output results/run_002/eval/
```

## Outputs (per `results/run_XXX/`)

| File | Description |
|---|---|
| `train.log` | Full training log |
| `metrics.json` | Per-epoch + final test set metrics |
| `config.yaml` | Resolved config (for reproducibility) |
| `best_checkpoint.pt` | Best val-F1 checkpoint |
| `final_checkpoint.pt` | Final epoch checkpoint |
| `eval/evaluation.json` | Test set + baseline comparison |

## Repository layout

```
three-view-sqli/
├── src/
│   ├── libinjection_wrapper.py  # ctypes wrapper around libinjection.{so,dll}
│   ├── preprocessing.py          # SQL → 3-view token ids
│   ├── dataset.py                # WAF-A-MoLE Dataset + DataLoader
│   └── model.py                  # ThreeViewModel
├── configs/
│   ├── medium.yaml               # 100k samples, 5 epoch (smoke test)
│   └── full.yaml                 # full data, 5 epoch (main run)
├── scripts/
│   ├── build_libinjection.sh     # Linux/macOS native build
│   └── build_libinjection.bat    # Windows MSVC build
├── external/libinjection/        # Vendored C source (BSD license)
├── train.py                      # Production training entrypoint
├── evaluate.py                   # Standalone evaluation + baselines
├── setup.sh                      # One-shot RunPod setup
└── requirements.txt
```

## Hardware notes

- **RTX 4090 (24GB)** — main target; BF16 mixed precision; full run ~30-50 min
- **RTX 3090 / A5000 (24GB)** — also works at similar speed
- **GTX 1070 Ti (8GB)** — works for medium config; full run takes ~9 hours

## Resuming a crashed run

```bash
python train.py --config configs/full.yaml --output results/run_002/ \
                --resume results/run_002/final_checkpoint.pt
```

## License

Code: MIT (see LICENSE in repo root).
Vendored libinjection: BSD-3-Clause (see `external/libinjection/LICENSE`).
