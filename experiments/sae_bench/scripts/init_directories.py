#!/usr/bin/env python3
"""
Initialize directory structure for SAE Bench DVC pipeline.

This script ensures all required output directories exist before running DVC pipeline,
preventing errors when stages expect certain directories to be present.
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create all required directories for the DVC pipeline."""

    # Get the base directory (parent of scripts directory)
    base_dir = Path(__file__).parent.parent

    # Define all required directories
    directories = [
        # Results directories (used by core and sparse_probing evaluations)
        "results/core",
        "results/sparse_probing",
        "results/merged",
        "results/images",
        "results/downloaded_saes",

        # Eval_results directories (hardcoded by sae_bench library for optional evaluations)
        "eval_results/scr",
        "eval_results/tpp",
        "eval_results/absorption",
        "eval_results/unlearning",

        # Metrics directory for storing extracted metrics
        "metrics",

        # Artifacts directory (if needed)
        "artifacts",
    ]

    print("Initializing SAE Bench directory structure...")

    for dir_path in directories:
        full_path = base_dir / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {dir_path}")
        else:
            print(f"  • Exists:  {dir_path}")

    print("\nDirectory structure ready for DVC pipeline!")
    print("You can now run: dvc repro")

if __name__ == "__main__":
    create_directory_structure()