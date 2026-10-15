"""
FastAPI Inference Service for GNN Node Classification.

Endpoints:
  GET  /health          - Health check with model version
  GET  /model/metadata  - Detailed model architecture and schema info
  POST /predict/transductive - Classify existing Cora nodes by ID
  POST /predict/inductive    - Classify unseen nodes with features + citations

All responses include model version and latency for auditing.
"""

import os
import sys
import time
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.serving.schemas import (
    TransductiveRequest,
    InductiveRequest,
    PredictionResponse,
    NodePrediction,
    HealthResponse,
    ModelMetadataResponse,
    ErrorResponse,
)
from src.serving.predictor import GNNPredictor

from contextlib import asynccontextmanager

MODEL_DIR = os.environ.get("MODEL_DIR", "models/champion")
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global predictor
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "weights.pt")):
        try:
            predictor = GNNPredictor(model_dir=MODEL_DIR, device="cpu")
            print(f"Model loaded from: {MODEL_DIR}")
        except Exception as e:
            print(f"Model initialization warning: {e}")
    else:
        print(f"INFO: No champion model found at '{MODEL_DIR}'. Serving in standby mode.")
    yield


app = FastAPI(
    title="GNN Node Classification API",
    description=(
        "Production inference API for Cora citation network node classification. "
        "Supports both transductive (existing nodes) and inductive (new nodes) inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== HEALTH CHECK =====
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Returns service status, loaded model version, and device info.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready.",
        )

    return HealthResponse(
        status="healthy",
        model_version=predictor.get_model_version(),
        model_type=predictor.model_type,
        num_graph_nodes=predictor.num_graph_nodes,
        device=str(predictor.device),
    )


# ===== MODEL METADATA =====
@app.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata():
    """
    Return detailed model architecture, feature schema, and class mapping.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return ModelMetadataResponse(
        model_type=predictor.model_type,
        architecture=predictor.metadata["architecture"],
        feature_schema=predictor.metadata.get("feature_schema", {}),
        class_mapping=predictor.class_mapping,
        num_graph_nodes=predictor.num_graph_nodes,
    )


# ===== TRANSDUCTIVE PREDICTION =====
@app.post("/predict/transductive", response_model=PredictionResponse)
def predict_transductive(request: TransductiveRequest):
    """
    Predict classes for existing nodes in the training graph.

    The model uses the full graph context (all training edges and features)
    to compute neighborhood-aggregated predictions for the requested nodes.

    Error Cases Handled:
      - Unknown node IDs -> 404
      - Duplicate node IDs -> Warning + deduplication
      - Model not loaded -> 503
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        predictions, warnings, latency = predictor.predict_transductive(
            request.node_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}"
        )

    pred_list = []
    for nid, pred in predictions.items():
        pred_list.append(
            NodePrediction(
                node_id=nid,
                predicted_class=pred["class_name"],
                class_id=pred["class_id"],
                confidence=round(pred["confidence"], 6),
                probabilities=pred["probabilities"],
            )
        )

    return PredictionResponse(
        model_version=predictor.get_model_version(),
        model_type=predictor.model_type,
        predictions=pred_list,
        latency_ms=round(latency, 3),
        warnings=warnings,
    )


# ===== INDUCTIVE PREDICTION =====
@app.post("/predict/inductive", response_model=PredictionResponse)
def predict_inductive(request: InductiveRequest):
    """
    Predict classes for unseen nodes introduced after training.

    Each new node must provide:
      - A 1433-dimensional feature vector
      - (Optional) Citation edges to existing Cora nodes

    The model constructs a temporary augmented graph, wires the new node's
    citations, and runs GraphSAGE neighborhood aggregation dynamically.

    Graph Context Requirements:
      - For a 2-layer GNN, the 2-hop neighborhood of the target node is
        needed for full representation quality.
      - If a new node has zero citations (isolated), the model degrades
        to feature-only projection (no neighborhood information).

    Inference Limitations:
      - Predictions for isolated nodes may be less accurate.
      - Very high-degree new nodes may slow inference.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    new_nodes = [
        {
            "temp_id": node.temp_id,
            "features": node.features,
            "citations": node.citations,
        }
        for node in request.nodes
    ]

    try:
        predictions, warnings, latency = predictor.predict_inductive(new_nodes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}"
        )

    pred_list = []
    for temp_id, pred in predictions.items():
        pred_list.append(
            NodePrediction(
                node_id=temp_id,
                predicted_class=pred["class_name"],
                class_id=pred["class_id"],
                confidence=round(pred["confidence"], 6),
                probabilities=pred["probabilities"],
            )
        )

    return PredictionResponse(
        model_version=predictor.get_model_version(),
        model_type=predictor.model_type,
        predictions=pred_list,
        latency_ms=round(latency, 3),
        warnings=warnings,
    )
