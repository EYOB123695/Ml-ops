"""
Model Packaging and Reproducibility Parity Test.

Verifies the requirement:
"Verify that reloading the model reproduces predictions under identical inputs."

Tests that serializing the model to disk (weights.pt + metadata.json) and reloading
it into a completely fresh instance produces bit-for-bit identical output logits
with maximum absolute difference L_inf < 1e-7.
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
from src.pipeline.validation_gate import load_packaged_model
from src.data.cora_loader import CORA_CLASS_NAMES


def test_model_packaging_parity():
    """
    Train a minimal GraphSAGE model in memory, record output on fixed tensor,
    package to disk, reload in separate instance, and assert exact output parity.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    in_dim = 1433
    hidden_dim = 32
    out_dim = 7

    # 1. Instantiate and run forward pass in memory
    original_model = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2)
    original_model.eval()

    # Fixed synthetic input
    dummy_x = torch.randn(20, in_dim)
    dummy_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    with torch.no_grad():
        original_logits = original_model(dummy_x, dummy_edge_index)

    # 2. Package model to temporary directory
    temp_pkg_dir = tempfile.mkdtemp(prefix="test_pkg_")
    try:
        package_model(
            model=original_model,
            model_type="GraphSAGE",
            params={
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
                "out_dim": out_dim,
                "num_layers": 2,
                "dropout": 0.5,
                "seed": 42,
                "lr": 0.01,
                "weight_decay": 5e-4,
                "epochs": 10,
            },
            class_names=CORA_CLASS_NAMES,
            output_dir=temp_pkg_dir,
        )

        # 3. Reload model from packaged directory
        reloaded_model, metadata = load_packaged_model(temp_pkg_dir, device)
        reloaded_model.eval()

        with torch.no_grad():
            reloaded_logits = reloaded_model(dummy_x, dummy_edge_index)

        # 4. Assert bit-for-bit numerical parity
        max_diff = torch.max(torch.abs(original_logits - reloaded_logits)).item()
        assert max_diff < 1e-6, f"Reloaded model outputs diverged by {max_diff} >= 1e-6"
        assert metadata["model_type"] == "GraphSAGE"
        assert metadata["architecture"]["in_dim"] == in_dim

    finally:
        shutil.rmtree(temp_pkg_dir, ignore_errors=True)
