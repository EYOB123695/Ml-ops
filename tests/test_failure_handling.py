"""
Automated Test Suite for Defensive Failure Handling and Edge Cases.

Verifies the required scenarios:
  1. Isolated nodes (degree = 0)
  2. Duplicate target IDs
  3. Invalid edge endpoints (out of range)
  4. Incorrect feature dimensions (!= 1433)
  5. Unknown target IDs
  6. Model loading failure handling
"""

import os
import sys
import tempfile
import shutil
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.serving.app as app_module
from src.serving.app import app
from src.serving.predictor import GNNPredictor
from src.models.gnn_models import GraphSAGE
from src.pipeline.train_mlflow import package_model
from src.data.cora_loader import CORA_CLASS_NAMES

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_serving_predictor():
    """Ensure the FastAPI app has a valid loaded predictor for testing endpoints."""
    temp_dir = tempfile.mkdtemp(prefix="test_app_champ_")
    torch.manual_seed(42)
    in_dim = 1433
    hidden_dim = 32
    out_dim = 7

    model = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    package_model(
        model=model,
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
            "epochs": 1,
        },
        class_names=CORA_CLASS_NAMES,
        output_dir=temp_dir,
    )

    num_nodes = 50
    dummy_x = torch.randn(num_nodes, in_dim)
    dummy_src = torch.randint(0, num_nodes, (100,))
    dummy_dst = torch.randint(0, num_nodes, (100,))
    torch.save(
        {
            "x": dummy_x,
            "edge_index": torch.stack([dummy_src, dummy_dst], dim=0),
            "y": torch.randint(0, out_dim, (num_nodes,)),
            "num_nodes": num_nodes,
        },
        os.path.join(temp_dir, "graph_context.pt"),
    )

    app_module.predictor = GNNPredictor(model_dir=temp_dir)

    yield

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_isolated_node_handling():
    """
    Scenario 1: New node with degree 0 (no incident citations).
    Expected Behavior: The model processes the node using self-loop / node features
    without crashing or dividing by zero, returning a 200 OK with a clear warning.
    """
    payload = {
        "nodes": [
            {
                "temp_id": "isolated_test_paper_1",
                "features": [0.0] * 1433,
                "citations": [],  # Isolated node: 0 citations
            }
        ]
    }
    response = client.post("/predict/inductive", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["node_id"] == "isolated_test_paper_1"
    assert "class_id" in data["predictions"][0]
    assert 0 <= data["predictions"][0]["confidence"] <= 1.0

    # Ensure warning is reported about isolated node
    assert any("isolated" in w.lower() for w in data.get("warnings", []))


def test_duplicate_target_ids():
    """
    Scenario 2: Duplicate target IDs passed in transductive prediction request.
    Expected Behavior: The API deduplicates the target IDs internally and returns
    predictions for unique nodes with a warning, avoiding redundant computation.
    """
    payload = {"node_ids": [10, 10, 25, 25, 10]}
    response = client.post("/predict/transductive", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Should only return predictions for the 2 unique nodes
    assert len(data["predictions"]) == 2
    assert any("duplicate" in w.lower() for w in data.get("warnings", []))


def test_invalid_edge_endpoints():
    """
    Scenario 3: An inductive node provides citation links to non-existent node IDs.
    Expected Behavior: The system detects that the target citation ID (e.g. 999999) is out
    of range [0, num_nodes - 1], ignores the phantom edge, logs a warning, and completes inference.
    """
    payload = {
        "nodes": [
            {
                "temp_id": "phantom_edge_paper",
                "features": [0.05] * 1433,
                "citations": [999999],  # Out of range citation
            }
        ]
    }
    response = client.post("/predict/inductive", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert any("invalid ID" in w for w in data.get("warnings", []))


def test_incorrect_feature_dimensions():
    """
    Scenario 4: Node feature vector has incorrect dimensions (e.g. 50 instead of 1433).
    Expected Behavior: Pydantic validation rejects the malformed input with a 422
    Unprocessable Entity error, clearly explaining that 1433 features are required.
    """
    payload = {
        "nodes": [
            {
                "temp_id": "malformed_vector_paper",
                "features": [0.1] * 50,  # 50 dimensions instead of 1433
                "citations": [1, 2],
            }
        ]
    }
    response = client.post("/predict/inductive", json=payload)
    assert response.status_code == 422
    assert "1433" in response.text


def test_unknown_target_id():
    """
    Scenario 5: Requesting an unknown/out-of-bounds node ID in transductive mode.
    Expected Behavior: Returns 404 Not Found indicating the node ID does not exist in Cora.
    """
    payload = {"node_ids": [999999]}
    response = client.post("/predict/transductive", json=payload)
    assert response.status_code == 404
    assert "out of range" in response.json()["detail"].lower()


def test_model_loading_failure():
    """
    Scenario 6: Attempting to instantiate GNNPredictor with an invalid directory.
    Expected Behavior: Raises FileNotFoundError with clear description rather than hanging.
    """
    with pytest.raises(FileNotFoundError):
        GNNPredictor(model_dir="non_existent_model_dir_xyz")
