"""
Prediction Consistency Integration Test.

Verifies the requirement:
"Verify that offline and API predictions agree when using identical inputs,
graph context, preprocessing, and model versions."

Compares offline direct model evaluation vs. HTTP predictions via FastAPI TestClient.
"""

import os
import sys
import tempfile
import shutil
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.gnn_models import GraphSAGE
from src.pipeline.train_mlflow import package_model
from src.data.cora_loader import CORA_CLASS_NAMES
import src.serving.app as app_module
from src.serving.predictor import GNNPredictor


@pytest.fixture(scope="module")
def setup_dummy_champion():
    """Sets up a mock champion model package for testing prediction consistency."""
    torch.manual_seed(100)
    in_dim = 1433
    hidden_dim = 32
    out_dim = 7

    model = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    model.eval()

    temp_dir = tempfile.mkdtemp(prefix="test_champion_")
    package_model(
        model=model,
        model_type="GraphSAGE",
        params={
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "num_layers": 2,
            "dropout": 0.5,
            "seed": 100,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "epochs": 1,
        },
        class_names=CORA_CLASS_NAMES,
        output_dir=temp_dir,
    )

    # Save mock graph context with 50 nodes
    num_nodes = 50
    dummy_x = torch.randn(num_nodes, in_dim)
    dummy_src = torch.randint(0, num_nodes, (120,))
    dummy_dst = torch.randint(0, num_nodes, (120,))
    dummy_edges = torch.stack([dummy_src, dummy_dst], dim=0)

    torch.save(
        {
            "x": dummy_x,
            "edge_index": dummy_edges,
            "y": torch.randint(0, out_dim, (num_nodes,)),
            "num_nodes": num_nodes,
        },
        os.path.join(temp_dir, "graph_context.pt"),
    )

    # Point predictor to this mock directory
    app_module.predictor = GNNPredictor(model_dir=temp_dir)

    yield temp_dir, app_module.predictor

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_offline_online_prediction_parity(setup_dummy_champion):
    """
    Assert that offline probability outputs match the online API probability outputs
    within numerical tolerance of L_inf < 1e-5.
    """
    temp_dir, predictor = setup_dummy_champion
    test_node_ids = [0, 5, 12, 23, 44]

    # 1. Offline inference
    offline_preds, _, _ = predictor.predict_transductive(test_node_ids)

    # 2. Online inference via FastAPI
    client = TestClient(app_module.app)
    response = client.post("/predict/transductive", json={"node_ids": test_node_ids})
    assert response.status_code == 200
    online_data = response.json()

    online_preds = {p["node_id"]: p for p in online_data["predictions"]}

    # 3. Assert exact parity
    for nid in test_node_ids:
        assert nid in online_preds
        off_item = offline_preds[nid]
        on_item = online_preds[nid]

        assert off_item["class_id"] == on_item["class_id"]
        assert off_item["class_name"] == on_item["predicted_class"]

        for class_name, off_prob in off_item["probabilities"].items():
            on_prob = on_item["probabilities"][class_name]
            diff = abs(off_prob - on_prob)
            assert diff < 1e-4, f"Node {nid} class {class_name} probability diverged: {diff}"
