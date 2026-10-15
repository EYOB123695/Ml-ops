"""
Graph Structural and Embedding Drift Detection Tests.

Verifies the requirements:
  - Demonstrate a normal case (no false alert).
  - Demonstrate a detected change for structural drift (degree shift).
  - Demonstrate a detected change for embedding drift (distribution shift).
  - Justify alert thresholds via statistical tests (KS test alpha=0.05, MMD threshold).
"""

import numpy as np
import pytest
from src.monitoring.drift_detector import GraphDriftDetector


@pytest.fixture
def drift_setup():
    """Sets up synthetic baseline degrees and embeddings."""
    np.random.seed(42)
    num_nodes = 200
    embedding_dim = 32

    # Baseline: Poisson degree distribution (mean=4.0)
    baseline_degrees = np.random.poisson(lam=4.0, size=num_nodes)
    # Baseline: Standard normal embeddings
    baseline_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, embedding_dim))

    detector = GraphDriftDetector(
        baseline_degrees=baseline_degrees,
        baseline_embeddings=baseline_embeddings,
        ks_alpha=0.05,
        mmd_threshold=0.03,
    )
    return detector, num_nodes, embedding_dim


def test_normal_case_no_drift(drift_setup):
    """
    Normal Case: Incoming data sampled from the identical distribution as baseline.
    Expected: Both structural and embedding checks report no drift detected.
    """
    detector, num_nodes, embedding_dim = drift_setup
    np.random.seed(99)

    # Sample from same underlying distributions
    normal_degrees = np.random.poisson(lam=4.0, size=num_nodes)
    normal_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, embedding_dim))

    report = detector.run_full_diagnostic(normal_degrees, normal_embeddings)

    assert report["overall_status"] == "HEALTHY"
    assert report["structural_check"]["drift_detected"] is False
    assert report["embedding_check"]["drift_detected"] is False
    assert report["structural_check"]["p_value"] >= 0.05


def test_detected_structural_drift(drift_setup):
    """
    Detected Change: Influx of high-degree hub nodes shifts the degree distribution.
    Expected: KS test detects significant shift (p < 0.05) and flags structural drift.
    """
    detector, num_nodes, embedding_dim = drift_setup
    np.random.seed(123)

    # Drastic degree shift (e.g., dense cluster where mean degree = 15.0)
    drifted_degrees = np.random.poisson(lam=15.0, size=num_nodes)
    normal_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, embedding_dim))

    struct_res = detector.check_structural_drift(drifted_degrees)

    assert struct_res["drift_detected"] is True
    assert struct_res["p_value"] < 0.05
    assert "ALERT" in struct_res["message"]


def test_detected_embedding_drift(drift_setup):
    """
    Detected Change: Node feature / representation distribution shifts (mean shift).
    Expected: Maximum Mean Discrepancy (MMD) exceeds the alert threshold.
    """
    detector, num_nodes, embedding_dim = drift_setup
    np.random.seed(456)

    normal_degrees = np.random.poisson(lam=4.0, size=num_nodes)
    # Mean shift in embeddings (e.g. loc=2.5 instead of 0.0)
    drifted_embeddings = np.random.normal(loc=2.5, scale=1.0, size=(num_nodes, embedding_dim))

    embed_res = detector.check_embedding_drift(drifted_embeddings)

    assert embed_res["drift_detected"] is True
    assert embed_res["mmd_score"] > detector.mmd_threshold
    assert "ALERT" in embed_res["message"]
