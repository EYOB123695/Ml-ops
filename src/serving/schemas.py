"""
Pydantic Schemas for the GNN Inference API.

Defines request and response models with strict validation for:
  - Transductive inference (existing Cora node IDs)
  - Inductive inference (new nodes with features and citations)
  - Health check and model metadata responses
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class TransductiveRequest(BaseModel):
    """Request for predicting classes of existing nodes in the training graph."""

    node_ids: List[int] = Field(
        ...,
        min_length=1,
        description="List of existing Cora node IDs to classify.",
    )

    @field_validator("node_ids")
    @classmethod
    def validate_no_negative_ids(cls, v):
        for nid in v:
            if nid < 0:
                raise ValueError(f"Node ID must be non-negative, got {nid}")
        return v


class InductiveNode(BaseModel):
    """A single unseen node introduced after training."""

    temp_id: str = Field(
        ...,
        description="Temporary identifier for this new node (e.g., 'new_paper_1').",
    )
    features: List[float] = Field(
        ...,
        description="Node feature vector. Must have exactly 1433 elements for Cora.",
    )
    citations: List[int] = Field(
        default_factory=list,
        description="List of existing Cora node IDs that this node cites or is cited by.",
    )

    @field_validator("features")
    @classmethod
    def validate_feature_length(cls, v):
        if len(v) != 1433:
            raise ValueError(
                f"Expected exactly 1433 features for Cora, got {len(v)}. "
                f"Each feature represents a binary bag-of-words indicator."
            )
        return v


class InductiveRequest(BaseModel):
    """Request for predicting classes of unseen nodes with their local context."""

    nodes: List[InductiveNode] = Field(
        ...,
        min_length=1,
        description="List of new nodes with features and citation edges.",
    )


class NodePrediction(BaseModel):
    """Prediction result for a single node."""

    node_id: Union[int, str] = Field(
        ..., description="The node ID (integer for existing, string for new)."
    )
    predicted_class: str = Field(
        ..., description="Human-readable class name."
    )
    class_id: int = Field(
        ..., description="Integer class label."
    )
    confidence: float = Field(
        ..., description="Softmax probability of the predicted class."
    )
    probabilities: Dict[str, float] = Field(
        ..., description="Mapping from class name to softmax probability."
    )


class PredictionResponse(BaseModel):
    """Standard prediction response including model version and latency."""

    model_version: str = Field(
        ..., description="Version identifier of the serving model."
    )
    model_type: str = Field(
        ..., description="Architecture name (e.g., GraphSAGE)."
    )
    predictions: List[NodePrediction] = Field(
        ..., description="List of predictions for each requested node."
    )
    latency_ms: float = Field(
        ..., description="Inference latency in milliseconds."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during inference.",
    )


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_version: str
    model_type: str
    num_graph_nodes: int
    device: str


class ModelMetadataResponse(BaseModel):
    """Detailed model metadata response."""

    model_type: str
    architecture: dict
    feature_schema: dict
    class_mapping: Dict[str, str]
    num_graph_nodes: int


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    detail: str
    status_code: int
