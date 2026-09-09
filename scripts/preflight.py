"""Command-line script to run pre-training data preflight diagnostics."""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics.config import ExperimentConfig
from cancer_combo_brics.data.preflight import run_data_preflight


def main():
    parser = argparse.ArgumentParser(description="Run pre-training data preflight diagnostics.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--combination_file", type=str, default=None, help="Path to combination CSV")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else None
    comb_file = args.combination_file

    res = run_data_preflight(config=cfg, comb_file=comb_file)
    print("Preflight Diagnostic Summary:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
