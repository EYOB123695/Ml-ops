"""
Emergency Rollback Script for Production GNN Model.

Provides zero-downtime model rollback:
  1. Inspects the previous archived champion (e.g. models/champion_previous).
  2. Restores it as the active champion (models/champion).
  3. Verifies that the restored model artifact has a valid state_dict and metadata.
  4. Triggers reload on the serving endpoint if running.

Usage:
    python -m src.pipeline.rollback --champion-dir models/champion
"""

import os
import sys
import shutil
import json
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.validation_gate import rollback_model, load_packaged_model


def verify_model_health(model_dir: str) -> bool:
    """Check that the restored model can be successfully loaded and instantiated."""
    try:
        device = torch.device("cpu")
        model, metadata = load_packaged_model(model_dir, device)
        print(f"Health verification SUCCESS: Model '{metadata['model_type']}' loaded properly.")
        return True
    except Exception as e:
        print(f"Health verification FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Emergency Model Rollback Tool")
    parser.add_argument(
        "--champion-dir",
        type=str,
        default="models/champion",
        help="Path to production champion model directory",
    )
    args = parser.parse_args()

    print(f"Initiating rollback for production directory: {args.champion_dir}")
    success = rollback_model(args.champion_dir)

    if not success:
        print("Rollback aborted: No prior champion backup available.")
        sys.exit(1)

    print("Verifying restored model health...")
    if verify_model_health(args.champion_dir):
        print("ROLLBACK COMPLETE: Active production model reverted successfully.")
        sys.exit(0)
    else:
        print("CRITICAL: Restored model failed integrity checks!")
        sys.exit(1)


if __name__ == "__main__":
    main()
