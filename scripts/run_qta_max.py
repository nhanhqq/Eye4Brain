"""Launch the accuracy-first Eye4Brain-QTA target-selected run."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "qta_max_20260911"
PLAN = ROOT / "configs" / "qta_max_20260911.json"

DATASETS = {
    "seediv": ("data/cache/seediv_contiguous_t8.npz", "adversarial", [
        "--spec-amp", "--fusion-hidden", "128", "--eye-amplitude-residual",
        "--lambda-subject", "0.05", "--lambda-orth", "0.02", "--lambda-relation", "0.05",
    ]),
    "seedv": ("data/cache/seedv_de_eye_t8.npz", "feature", [
        "--spec-amp", "--fusion-hidden", "128", "--feature-hidden", "48",
        "--eye-amplitude-residual", "--eeg-views", "de",
    ]),
    "seedvii": ("data/cache/seedvii_de_eye_t8.npz", "token", [
        "--spec-amp", "--eeg-views", "de", "--fusion-hidden", "128",
        "--eye-amplitude-residual",
    ]),
}


def main():
    jobs = []
    for dataset, (cache, variant, extra) in DATASETS.items():
        with __import__("numpy").load(ROOT / cache, allow_pickle=False) as arrays:
            subjects = len(__import__("numpy").unique(arrays["subject"]))
        for fold in range(subjects):
            jobs.append({
                "id": f"{dataset}_f{fold}", "dataset": dataset, "variant": variant,
                "cache": str(ROOT / cache), "fold": fold, "seed": 7,
                "ram_mib": 6500, "cuda_mib": 7500,
                "args": ["--epochs", "150", "--batch-size", "128", "--warmup", "10",
                         "--grad-clip", "5", "--target-adapt", *extra],
            })
    PLAN.write_text(json.dumps({
        "profile_source": "qta target AdaNorm sweep; measured smoke fold peak 6748 MiB VRAM",
        "protocol": "target_selected_transductive",
        "performance_targets": {"seediv": 85, "seedv": 90, "seedvii": 90},
        "jobs": jobs,
    }, indent=2))
    command = [sys.executable, str(ROOT / "scripts" / "schedule_training.py"),
               "--plan", str(PLAN), "--output", str(OUTPUT), "--max-parallel", "14"]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
