"""
Model Registry Validation Gate.

Compares a candidate model against the current champion model on the
same validation data. Implements a deterministic promotion policy:

  - Metric:               Validation Macro-F1
  - Promotion Threshold:  Candidate must exceed Champion by >= 0.015 (+1.5%)
  - Tie Behavior:         Retain Champion (favor stability)
  - Missing Metric:       Block promotion
  - First Model:          Auto-promote if no Champion exists
  - Blocked Regression:   If candidate is worse, Champion is retained

Usage:
    python -m src.pipeline.validation_gate --candidate-dir models/candidate --champion-dir models/champion
"""

import os
import sys
import json
import argparse
import shutil

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.cora_loader import load_cora
from src.models.gnn_models import build_model

# Promotion policy constants
PROMOTION_THRESHOLD = 0.015  # Candidate must beat champion by at least 1.5%
METRIC_NAME = "val_macro_f1"


def load_packaged_model(model_dir: str, device: torch.device) -> tuple:
    """
    Load a packaged model from a directory containing weights.pt and metadata.json.

    Args:
        model_dir: Path to the model package directory.
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model, metadata_dict).

    Raises:
        FileNotFoundError: If weights.pt or metadata.json is missing.
        RuntimeError: If model loading fails.
    """
    metadata_path = os.path.join(model_dir, "metadata.json")
    weights_path = os.path.join(model_dir, "weights.pt")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights.pt not found in {model_dir}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    arch = metadata["architecture"]
    model = build_model(
        model_type=arch["name"],
        in_dim=arch["in_dim"],
        hidden_dim=arch["hidden_dim"],
        out_dim=arch["out_dim"],
        num_layers=arch.get("num_layers", 2),
        dropout=arch.get("dropout", 0.5),
    )

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, metadata


def evaluate_model(model, data, device) -> dict:
    """
    Evaluate a model on the Cora validation set.

    Returns:
        Dictionary with val_macro_f1 and val_accuracy.
    """
    x = data["x"].to(device)
    edge_index = data["edge_index"].to(device)
    y = data["y"].to(device)
    val_mask = data["val_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        pred = logits[val_mask].argmax(dim=1).cpu().numpy()
        true = y[val_mask].cpu().numpy()

    macro_f1 = float(f1_score(true, pred, average="macro", zero_division=0))
    accuracy = float((pred == true).mean())

    return {
        "val_macro_f1": macro_f1,
        "val_accuracy": accuracy,
    }


def run_validation_gate(candidate_dir: str, champion_dir: str) -> dict:
    """
    Execute the validation gate comparison between candidate and champion.

    Args:
        candidate_dir: Path to the candidate model package.
        champion_dir: Path to the current champion model package.

    Returns:
        Dictionary with:
            - decision: "promote", "block", or "auto_promote"
            - reason: Human-readable explanation
            - candidate_f1: Candidate's validation Macro-F1
            - champion_f1: Champion's validation Macro-F1 (or None)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cora()

    result = {
        "decision": None,
        "reason": None,
        "candidate_f1": None,
        "champion_f1": None,
    }

    # ===== LOAD AND EVALUATE CANDIDATE =====
    try:
        candidate_model, candidate_meta = load_packaged_model(candidate_dir, device)
    except (FileNotFoundError, RuntimeError) as e:
        result["decision"] = "block"
        result["reason"] = f"Candidate model loading failed: {e}"
        return result

    try:
        candidate_metrics = evaluate_model(candidate_model, data, device)
        result["candidate_f1"] = candidate_metrics["val_macro_f1"]
    except Exception as e:
        result["decision"] = "block"
        result["reason"] = f"Candidate evaluation failed (missing metric): {e}"
        return result

    print(f"Candidate Val Macro-F1: {result['candidate_f1']:.4f}")

    # ===== CHECK IF CHAMPION EXISTS =====
    if not os.path.exists(champion_dir) or not os.path.exists(
        os.path.join(champion_dir, "weights.pt")
    ):
        # First model rule: auto-promote
        result["decision"] = "auto_promote"
        result["reason"] = "No existing champion found. Auto-promoting first model."
        print(f"Decision: AUTO-PROMOTE (first model)")
        return result

    # ===== LOAD AND EVALUATE CHAMPION =====
    try:
        champion_model, champion_meta = load_packaged_model(champion_dir, device)
        champion_metrics = evaluate_model(champion_model, data, device)
        result["champion_f1"] = champion_metrics["val_macro_f1"]
    except Exception as e:
        result["decision"] = "block"
        result["reason"] = f"Champion evaluation failed: {e}. Retaining champion."
        return result

    print(f"Champion Val Macro-F1:  {result['champion_f1']:.4f}")

    # ===== APPLY PROMOTION POLICY =====
    improvement = result["candidate_f1"] - result["champion_f1"]
    print(f"Improvement (delta):    {improvement:+.4f}")
    print(f"Required threshold:     +{PROMOTION_THRESHOLD:.4f}")

    if improvement >= PROMOTION_THRESHOLD:
        result["decision"] = "promote"
        result["reason"] = (
            f"Candidate F1 ({result['candidate_f1']:.4f}) exceeds champion "
            f"({result['champion_f1']:.4f}) by {improvement:.4f} >= {PROMOTION_THRESHOLD}. "
            f"Promoting candidate."
        )
        print(f"Decision: PROMOTE")
    elif improvement > 0:
        # Improvement exists but below threshold — treat as tie
        result["decision"] = "block"
        result["reason"] = (
            f"Candidate F1 ({result['candidate_f1']:.4f}) improves over champion "
            f"({result['champion_f1']:.4f}) by only {improvement:.4f} < {PROMOTION_THRESHOLD}. "
            f"Improvement below threshold. Retaining champion (tie behavior)."
        )
        print(f"Decision: BLOCK (improvement below threshold)")
    else:
        # Regression
        result["decision"] = "block"
        result["reason"] = (
            f"Candidate F1 ({result['candidate_f1']:.4f}) is worse than champion "
            f"({result['champion_f1']:.4f}) by {improvement:.4f}. "
            f"Regression detected. Retaining champion."
        )
        print(f"Decision: BLOCK (regression)")

    return result


def promote_model(candidate_dir: str, champion_dir: str):
    """
    Promote the candidate model to champion by replacing the champion directory.

    Args:
        candidate_dir: Source model package path.
        champion_dir: Destination champion path.
    """
    if os.path.exists(champion_dir):
        # Archive the old champion before replacement
        archive_dir = champion_dir + "_previous"
        if os.path.exists(archive_dir):
            shutil.rmtree(archive_dir)
        shutil.copytree(champion_dir, archive_dir)
        shutil.rmtree(champion_dir)
        print(f"Previous champion archived to: {archive_dir}")

    shutil.copytree(candidate_dir, champion_dir)
    print(f"Candidate promoted to champion at: {champion_dir}")


def rollback_model(champion_dir: str):
    """
    Rollback champion to the previous archived version.

    Args:
        champion_dir: Path to the current champion directory.
    """
    archive_dir = champion_dir + "_previous"
    if not os.path.exists(archive_dir):
        print("ERROR: No previous champion archive found. Cannot rollback.")
        return False

    if os.path.exists(champion_dir):
        shutil.rmtree(champion_dir)
    shutil.copytree(archive_dir, champion_dir)
    print(f"Rolled back to previous champion from: {archive_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Model validation gate")
    parser.add_argument(
        "--candidate-dir", type=str, default="models/latest",
        help="Path to candidate model package",
    )
    parser.add_argument(
        "--champion-dir", type=str, default="models/champion",
        help="Path to current champion model package",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Rollback champion to previous version instead of running gate",
    )
    args = parser.parse_args()

    if args.rollback:
        success = rollback_model(args.champion_dir)
        sys.exit(0 if success else 1)

    result = run_validation_gate(args.candidate_dir, args.champion_dir)

    print(f"\n{'='*60}")
    print(f"Gate Decision: {result['decision'].upper()}")
    print(f"Reason: {result['reason']}")
    print(f"{'='*60}")

    if result["decision"] in ("promote", "auto_promote"):
        promote_model(args.candidate_dir, args.champion_dir)
        sys.exit(0)
    else:
        print("Candidate was NOT promoted. Champion retained.")
        sys.exit(1)  # Non-zero exit code blocks CI pipeline


if __name__ == "__main__":
    main()
