#!/usr/bin/env python3
"""Apply each sqlmap tamper script to the real exploit pool.

Reads `data/sqlmap_exploits.json` (built by build_sqlmap_exploits.py),
imports each `tamper/*.py` from the cloned sqlmap repository, applies
every tamper to every exploit, dedupes, and writes one JSONL OOD set per
tamper.

Output:
  data/tamper_oods/<tamper_name>.jsonl   each line: {payload, label, tamper, base_id, technique}
  data/tamper_oods/_summary.json         per-tamper statistics
"""
from __future__ import annotations
import importlib.util
import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
SQLMAP_REPO = ROOT.parent / "external" / "sqlmap"
TAMPER_DIR = SQLMAP_REPO / "tamper"

# tamper modules import lib.core.* — make sqlmap root importable
if str(SQLMAP_REPO) not in sys.path:
    sys.path.insert(0, str(SQLMAP_REPO))

EXPLOIT_POOL = ROOT / "data" / "sqlmap_exploits.json"
OUT_DIR = ROOT / "data" / "tamper_oods"

# A few tampers depend on sqlmap's runtime state (kb / conf) and don't work
# as standalone string transformers. Skip them.
SKIP_TAMPERS = {
    "__init__",
    # tampers that need sqlmap kb (knowledge base) state
    "binary",            # uses BIGDATA_MARKER / kb features
    "halfversionedmorekeywords",  # depends on dbms detection
    "informationschemacomment",   # similarly
    "schemasplit",
    "0eunion",           # needs PAYLOAD_DELIMITER state
    "dunion",
    # Tampers that don't transform the payload (only set hints / kb state)
    # but generate enormous side-effect data per call — useless to us:
    "luanginx",          # only sets hints, payload unchanged
    "luanginxmore",      # generates 4.2 million params in hints — hangs
}


def load_tamper_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"tamper_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"    [warn] {name} failed to load: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None
    return mod


def discover_tampers() -> list[tuple[str, callable]]:
    """Return list of (name, tamper_fn) for every usable tamper script."""
    if not TAMPER_DIR.exists():
        raise FileNotFoundError(f"sqlmap tamper directory not found: {TAMPER_DIR}")

    out = []
    skipped = []
    failed = []
    for pyfile in sorted(TAMPER_DIR.glob("*.py")):
        name = pyfile.stem
        if name in SKIP_TAMPERS:
            skipped.append(name)
            continue
        mod = load_tamper_module(name, pyfile)
        if mod is None:
            failed.append((name, "import error"))
            continue
        if not hasattr(mod, "tamper"):
            failed.append((name, "no tamper() function"))
            continue
        out.append((name, mod.tamper))
    if failed:
        print(f"  {len(failed)} tampers failed to load:", flush=True)
        for n, why in failed[:10]:
            print(f"    - {n}: {why}", flush=True)
    if skipped:
        print(f"  {len(skipped)} tampers explicitly skipped: {skipped}", flush=True)
    return out


def main():
    print(f"Loading exploit pool from {EXPLOIT_POOL}")
    with open(EXPLOIT_POOL, encoding="utf-8") as f:
        exploits = json.load(f)
    print(f"  {len(exploits)} base exploits")

    print(f"\nDiscovering tampers in {TAMPER_DIR}")
    tampers = discover_tampers()
    print(f"  loaded {len(tampers)} tampers")
    for name, _ in tampers[:5]:
        print(f"    {name}")
    print(f"    ...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    base_strs = set(e["payload"] for e in exploits)

    # Save the un-tampered exploit pool as "no_tamper" baseline
    base_jsonl = OUT_DIR / "_no_tamper.jsonl"
    with open(base_jsonl, "w", encoding="utf-8") as f:
        for e in exploits:
            f.write(json.dumps({
                "user_input": e["payload"],
                "label": "attack",
                "tamper": "none",
                "base_id": e["id"],
                "technique": e["technique"],
                "source": "sqlmap_exploit_no_tamper",
            }, ensure_ascii=False) + "\n")
    summary["_no_tamper"] = {
        "n_base": len(exploits),
        "n_unique_outputs": len(base_strs),
        "n_changed": 0,
    }
    print(f"  wrote {base_jsonl}: {len(exploits)} (no tamper)")

    # Save summary periodically so we can recover partial runs
    def write_summary():
        with open(OUT_DIR / "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Apply each tamper
    for tamper_idx, (name, tamper_fn) in enumerate(tampers):
        out_path = OUT_DIR / f"{name}.jsonl"
        unique = {}  # payload -> first record
        n_attempted = 0
        n_failed = 0
        n_unchanged = 0
        for e in exploits:
            n_attempted += 1
            try:
                tampered = tamper_fn(e["payload"])
            except Exception:
                n_failed += 1
                continue
            if not tampered:
                n_failed += 1
                continue
            if tampered == e["payload"]:
                n_unchanged += 1
                # still keep as a sample (some tampers no-op for some inputs)
            if tampered not in unique:
                unique[tampered] = {
                    "user_input": tampered,
                    "label": "attack",
                    "tamper": name,
                    "base_id": e["id"],
                    "technique": e["technique"],
                    "source": f"tamper_{name}",
                }

        truly_ood = [r for p, r in unique.items() if p not in base_strs]

        with open(out_path, "w", encoding="utf-8") as f:
            for r in truly_ood:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = {
            "n_base": n_attempted,
            "n_failed": n_failed,
            "n_unchanged": n_unchanged,
            "n_unique_outputs": len(unique),
            "n_truly_ood": len(truly_ood),
        }
        marker = "*" if len(truly_ood) > 4000 else " "
        print(f"  [{tamper_idx + 1:2d}/{len(tampers):2d}] {marker} {name:35s}  "
              f"ood={len(truly_ood):>5d}  failed={n_failed}  unchanged={n_unchanged}",
              flush=True)
        # Write summary after each tamper
        write_summary()

    # Final summary save
    write_summary()

    # Aggregate stats
    total_ood = sum(v.get("n_truly_ood", 0) for k, v in summary.items() if k != "_no_tamper")
    print(f"\n{'='*60}")
    print(f"Total tamper-OOD samples: {total_ood:,} across {len(tampers)} tampers")
    print(f"Wrote per-tamper JSONL files to {OUT_DIR}")


if __name__ == "__main__":
    main()
