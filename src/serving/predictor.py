"""
Predictor Module: Offline and Online Inference Wrapper.

Loads a packaged model (weights + metadata + graph context) and provides
both transductive inference (existing nodes) and inductive inference
(unseen nodes with new features and citation edges).
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.models.gnn_models import build_model


class GNNPredictor:
    """
    Production inference wrapper for the GNN node classifier.

    Handles:
      - Loading packaged models (weights, metadata, graph context)
      - Transductive prediction for existing graph nodes
      - Inductive prediction for unseen nodes with local edge context
      - Input validation and defensive error handling
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        """
        Initialize the predictor by loading a packaged model.

        Args:
            model_dir: Path to directory containing weights.pt,
                       metadata.json, and graph_context.pt.
            device: Torch device string ("cpu" or "cuda").

        Raises:
            FileNotFoundError: If required files are missing.
            RuntimeError: If model loading fails.
        """
        self.device = torch.device(device)
        self.model_dir = model_dir

        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {model_dir}")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Build and load model
        arch = self.metadata["architecture"]
        self.model = build_model(
            model_type=arch["name"],
            in_dim=arch["in_dim"],
            hidden_dim=arch["hidden_dim"],
            out_dim=arch["out_dim"],
            num_layers=arch.get("num_layers", 2),
            dropout=arch.get("dropout", 0.5),
        )

        weights_path = os.path.join(model_dir, "weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights.pt not found in {model_dir}")

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Load graph context for transductive inference
        graph_path = os.path.join(model_dir, "graph_context.pt")
        if os.path.exists(graph_path):
            ctx = torch.load(graph_path, map_location=self.device, weights_only=False)
            self.graph_x = ctx["x"].to(self.device)
            self.graph_edge_index = ctx["edge_index"].to(self.device)
            self.num_graph_nodes = ctx["num_nodes"]
        else:
            self.graph_x = None
            self.graph_edge_index = None
            self.num_graph_nodes = 0

        # Class mapping
        self.class_mapping = self.metadata.get("class_mapping", {})
        self.model_type = arch["name"]
        self.model_version = self.metadata.get("training", {}).get(
            "seed", "unknown"
        )

    def get_model_version(self) -> str:
        """Return a human-readable model version string."""
        seed = self.metadata.get("training", {}).get("seed", "?")
        return f"{self.model_type}_v1_seed{seed}"

    def predict_transductive(self, node_ids: List[int]) -> Tuple[dict, List[str]]:
        """
        Predict classes for existing nodes in the training graph.

        Args:
            node_ids: List of integer node IDs present in the Cora graph.

        Returns:
            Tuple of (predictions_dict, warnings_list).
            predictions_dict maps node_id -> {class_id, class_name, confidence, probs}.

        Raises:
            ValueError: If graph context is not loaded.
        """
        warnings = []

        if self.graph_x is None:
            raise ValueError(
                "No graph context loaded. Cannot perform transductive inference. "
                "Ensure graph_context.pt is in the model directory."
            )

        # Validate and deduplicate node IDs
        valid_ids = []
        seen = set()
        for nid in node_ids:
            if nid in seen:
                warnings.append(f"Duplicate node ID {nid} — deduplicated.")
                continue
            seen.add(nid)

            if nid < 0 or nid >= self.num_graph_nodes:
                raise ValueError(
                    f"Node ID {nid} is out of range [0, {self.num_graph_nodes - 1}]. "
                    f"Use the inductive endpoint for new nodes."
                )
            valid_ids.append(nid)

        # Forward pass on full graph (transductive)
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(self.graph_x, self.graph_edge_index)
            probs = F.softmax(logits, dim=1)
        latency = (time.perf_counter() - start) * 1000

        # Extract predictions for requested nodes
        predictions = {}
        for nid in valid_ids:
            node_probs = probs[nid].cpu().numpy()
            class_id = int(node_probs.argmax())
            class_name = self.class_mapping.get(str(class_id), f"class_{class_id}")
            confidence = float(node_probs[class_id])

            prob_dict = {
                self.class_mapping.get(str(i), f"class_{i}"): round(float(p), 6)
                for i, p in enumerate(node_probs)
            }

            predictions[nid] = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "probabilities": prob_dict,
            }

        return predictions, warnings, latency

    def predict_inductive(
        self, new_nodes: List[dict]
    ) -> Tuple[dict, List[str], float]:
        """
        Predict classes for unseen nodes introduced after training.

        Each new node provides its feature vector and citation edges
        to existing graph nodes. The method constructs a temporary
        augmented graph and runs inference.

        Args:
            new_nodes: List of dicts with keys:
                - temp_id: str identifier
                - features: List[float] of length 1433
                - citations: List[int] of existing node IDs

        Returns:
            Tuple of (predictions_dict, warnings_list, latency_ms).
        """
        warnings = []

        if self.graph_x is None:
            raise ValueError("No graph context loaded for inductive inference.")

        in_dim = self.metadata["architecture"]["in_dim"]
        n_existing = self.num_graph_nodes
        n_new = len(new_nodes)

        # Build augmented feature matrix: [existing_nodes + new_nodes, in_dim]
        new_features = []
        temp_id_map = {}  # temp_id -> augmented_index

        for i, node in enumerate(new_nodes):
            aug_idx = n_existing + i
            temp_id_map[node["temp_id"]] = aug_idx

            feat = torch.tensor(node["features"], dtype=torch.float32)
            if feat.shape[0] != in_dim:
                raise ValueError(
                    f"Node '{node['temp_id']}' has {feat.shape[0]} features, "
                    f"expected {in_dim}."
                )
            new_features.append(feat)

        new_x = torch.stack(new_features).to(self.device)
        aug_x = torch.cat([self.graph_x, new_x], dim=0)

        # Build augmented edges: existing edges + new citation edges
        new_src = []
        new_dst = []
        for i, node in enumerate(new_nodes):
            aug_idx = n_existing + i
            for cite_id in node.get("citations", []):
                if cite_id < 0 or cite_id >= n_existing:
                    warnings.append(
                        f"Node '{node['temp_id']}' cites invalid ID {cite_id} "
                        f"(range [0, {n_existing - 1}]). Skipping this edge."
                    )
                    continue
                # Undirected: add both directions
                new_src.extend([aug_idx, cite_id])
                new_dst.extend([cite_id, aug_idx])

        if new_src:
            new_edges = torch.tensor([new_src, new_dst], dtype=torch.long).to(
                self.device
            )
            aug_edge_index = torch.cat(
                [self.graph_edge_index, new_edges], dim=1
            )
        else:
            aug_edge_index = self.graph_edge_index
            for node in new_nodes:
                if not node.get("citations"):
                    warnings.append(
                        f"Node '{node['temp_id']}' has no citations. "
                        f"Prediction relies on node features only (isolated node)."
                    )

        # Forward pass on augmented graph
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(aug_x, aug_edge_index)
            probs = F.softmax(logits, dim=1)
        latency = (time.perf_counter() - start) * 1000

        # Extract predictions for new nodes only
        predictions = {}
        for node in new_nodes:
            aug_idx = temp_id_map[node["temp_id"]]
            node_probs = probs[aug_idx].cpu().numpy()
            class_id = int(node_probs.argmax())
            class_name = self.class_mapping.get(str(class_id), f"class_{class_id}")
            confidence = float(node_probs[class_id])

            prob_dict = {
                self.class_mapping.get(str(i), f"class_{i}"): round(float(p), 6)
                for i, p in enumerate(node_probs)
            }

            predictions[node["temp_id"]] = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "probabilities": prob_dict,
            }

        return predictions, warnings, latency

    def get_embeddings(self, node_ids: List[int] = None) -> np.ndarray:
        """
        Extract node embeddings from the penultimate layer.
        Used by the drift detector for embedding distribution monitoring.

        Args:
            node_ids: Optional list of node IDs. If None, returns all.

        Returns:
            Numpy array of embeddings [num_nodes, hidden_dim].
        """
        if self.graph_x is None:
            raise ValueError("No graph context loaded.")

        with torch.no_grad():
            embeddings = self.model.get_embeddings(
                self.graph_x, self.graph_edge_index
            )

        if node_ids is not None:
            embeddings = embeddings[node_ids]

        return embeddings.cpu().numpy()
