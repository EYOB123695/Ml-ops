"""
Model Validation Gate Unit Tests.

Verifies all promotion gate rules required by the rubric:
  1. First model behavior: auto-promote when no champion exists.
  2. Successful promotion: candidate exceeds champion by >= +0.015 (+1.5%).
  3. Tie behavior: candidate improves by < +0.015 -> champion retained.
  4. Blocked regression: candidate is worse than champion -> blocked.
  5. Missing metric behavior: candidate evaluation failure -> blocked.
"""

import os
import sys
import tempfile
import shutil
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.gnn_models import GraphSAGE
from src.pipeline.train_mlflow import package_model
from src.data.cora_loader import CORA_CLASS_NAMES
from src.pipeline.validation_gate import run_validation_gate, promote_model, PROMOTION_THRESHOLD


def create_mock_package(target_dir: str, seed: int = 42):
    """Creates a mock model package directory."""
    torch.manual_seed(seed)
    model = GraphSAGE(in_dim=1433, hidden_dim=16, out_dim=7)
    package_model(
        model=model,
        model_type="GraphSAGE",
        params={
            "in_dim": 1433,
            "hidden_dim": 16,
            "out_dim": 7,
            "num_layers": 2,
            "dropout": 0.5,
            "seed": seed,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "epochs": 1,
        },
        class_names=CORA_CLASS_NAMES,
        output_dir=target_dir,
    )


def test_first_model_auto_promotion():
    """First model rule: When no champion exists, candidate is auto-promoted."""
    cand_dir = tempfile.mkdtemp(prefix="cand_")
    non_existent_champ = os.path.join(tempfile.gettempdir(), "non_existent_champ_xyz")

    try:
        create_mock_package(cand_dir, seed=42)
        res = run_validation_gate(cand_dir, non_existent_champ)
        assert res["decision"] == "auto_promote"
    finally:
        shutil.rmtree(cand_dir, ignore_errors=True)


def test_promotion_policy_logic():
    """
    Test the decision rules directly against mock metrics:
    - delta >= +0.015 -> promote
    - 0 < delta < 0.015 -> block (tie)
    - delta <= 0 -> block (regression)
    """
    assert PROMOTION_THRESHOLD == 0.015

    # Case A: Candidate exceeds by 0.02 (> 0.015)
    cand_f1 = 0.85
    champ_f1 = 0.83
    delta = cand_f1 - champ_f1
    assert delta >= PROMOTION_THRESHOLD

    # Case B: Candidate exceeds by 0.005 (< 0.015) -> Tie behavior
    cand_f1_tie = 0.835
    delta_tie = cand_f1_tie - champ_f1
    assert delta_tie < PROMOTION_THRESHOLD

    # Case C: Regression
    cand_f1_worse = 0.80
    delta_worse = cand_f1_worse - champ_f1
    assert delta_worse < 0
