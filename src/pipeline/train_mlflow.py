"""
MLflow-Tracked Training Pipeline for Cora Node Classification.

This script trains a GNN model (GCN or GraphSAGE) on the Cora dataset
with full experiment tracking via MLflow. Every run records:
  - Hyperparameters (model_type, hidden_dim, lr, dropout, seed, etc.)
  - Per-epoch metrics (train_loss, val_loss, val_acc, val_macro_f1)
  - Final test metrics (test_acc, test_macro_f1)
  - Artifacts (model weights, metadata.json, confusion_matrix.png)
  - Tags (git commit hash, dataset MD5, author)

Usage:
    python -m src.pipeline.train_mlflow --model_type GraphSAGE --seed 42
    python -m src.pipeline.train_mlflow --model_type GCN --seed 123 --hidden_dim 32
    python -m src.pipeline.train_mlflow --model_type GraphSAGE --seed 999 --lr 0.005
"""

import os
import sys
import json
import argparse
import time
import subprocess
import tempfile
import shutil

import torch
import torch.nn.functional as F
import numpy as np

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

    class LocalMLflowTracker:
        """Lightweight offline fallback tracker that mirrors the MLflow API."""
        def __init__(self):
            self.active_run_id = None
            self.run_data = {}
            self.exp_name = "default"
            self.base_dir = "./mlruns"
            self.run_dir = None

        def set_tracking_uri(self, uri):
            pass

        def set_experiment(self, name):
            self.exp_name = name

        def start_run(self, run_name=None):
            self.active_run_id = f"run_{int(time.time())}_{run_name or 'default'}"
            self.run_dir = os.path.join(self.base_dir, self.exp_name, self.active_run_id)
            os.makedirs(self.run_dir, exist_ok=True)
            self.run_data = {"run_id": self.active_run_id, "params": {}, "metrics": {}, "tags": {}}
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.run_dir:
                with open(os.path.join(self.run_dir, "run_summary.json"), "w") as f:
                    json.dump(self.run_data, f, indent=2)

        def log_params(self, params):
            self.run_data["params"].update(params)

        def log_param(self, k, v):
            self.run_data["params"][k] = v

        def log_metrics(self, metrics, step=None):
            for k, v in metrics.items():
                if k not in self.run_data["metrics"]:
                    self.run_data["metrics"][k] = []
                self.run_data["metrics"][k].append({"step": step, "value": v})

        def log_metric(self, k, v, step=None):
            self.log_metrics({k: v}, step=step)

        def set_tag(self, k, v):
            self.run_data["tags"][k] = v

        def log_artifacts(self, artifact_dir):
            if self.run_dir:
                dest = os.path.join(self.run_dir, "artifacts")
                shutil.copytree(artifact_dir, dest, dirs_exist_ok=True)

        def active_run(self):
            class RunInfo:
                def __init__(self, run_id):
                    self.run_id = run_id
            class Run:
                def __init__(self, run_id):
                    self.info = RunInfo(run_id)
            return Run(self.active_run_id)

    mlflow = LocalMLflowTracker()

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.cora_loader import load_cora, inspect_dataset, compute_dataset_hash, CORA_CLASS_NAMES
from src.models.gnn_models import build_model, count_parameters


def get_git_commit() -> str:
    """Retrieve the current Git commit SHA for provenance tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Evaluate model on a specific data split.

    Args:
        model: Trained PyTorch model.
        x: Node features.
        edge_index: Edge connectivity.
        y: Ground truth labels.
        mask: Boolean mask for the target split.

    Returns:
        Dictionary with accuracy, macro_f1, loss, predictions, and true labels.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[mask], y[mask])
        pred = logits[mask].argmax(dim=1).cpu().numpy()
        true = y[mask].cpu().numpy()

        acc = float((pred == true).mean())
        macro_f1 = float(f1_score(true, pred, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "loss": float(loss),
        "predictions": pred,
        "true_labels": true,
    }


def save_confusion_matrix(true_labels, predictions, class_names, filepath):
    """Save a confusion matrix plot as a PNG file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(true_labels, predictions)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(class_names.values()),
            yticklabels=list(class_names.values()),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping confusion matrix plot.")


def package_model(model, model_type, params, class_names, output_dir):
    """
    Package the trained model with all metadata required for deployment.

    Creates a directory containing:
      - weights.pt: Model state dictionary
      - metadata.json: Architecture, feature schema, class mapping, version info
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "weights.pt"))

    # Save comprehensive metadata
    metadata = {
        "model_type": model_type,
        "architecture": {
            "name": model_type,
            "num_layers": params.get("num_layers", 2),
            "in_dim": params["in_dim"],
            "hidden_dim": params["hidden_dim"],
            "out_dim": params["out_dim"],
            "dropout": params["dropout"],
        },
        "feature_schema": {
            "feature_length": params["in_dim"],
            "feature_type": "float32",
            "description": "Binary bag-of-words node features from Cora",
        },
        "class_mapping": {str(k): v for k, v in class_names.items()},
        "training": {
            "seed": params["seed"],
            "learning_rate": params["lr"],
            "weight_decay": params["weight_decay"],
            "epochs_trained": params["epochs"],
        },
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="MLflow-tracked GNN training on Cora")
    parser.add_argument("--model_type", type=str, default="GraphSAGE",
                        choices=["MLP", "GCN", "GraphSAGE"],
                        help="Model architecture to train")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden layer dimension")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay for regularization")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of GNN layers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--experiment_name", type=str, default="cora_node_classification",
                        help="MLflow experiment name")
    args = parser.parse_args()

    # If config file is provided, override defaults
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if hasattr(args, key) and value is not None:
                setattr(args, key, value)
        print(f"Loaded configuration from: {args.config}")

    # ===== REPRODUCIBILITY =====
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ===== LOAD DATA =====
    print("Loading Cora dataset...")
    data = load_cora()
    dataset_hash = compute_dataset_hash(data)
    dataset_stats = inspect_dataset(data)
    print(f"  Dataset hash: {dataset_hash}")
    print(f"  Homophily: {dataset_stats['homophily_ratio']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = data["x"].to(device)
    edge_index = data["edge_index"].to(device)
    y = data["y"].to(device)
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    # ===== BUILD MODEL =====
    model = build_model(
        model_type=args.model_type,
        in_dim=data["num_features"],
        hidden_dim=args.hidden_dim,
        out_dim=data["num_classes"],
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    param_count = count_parameters(model)
    print(f"  Model: {args.model_type}, Parameters: {param_count}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ===== MLFLOW TRACKING =====
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment_name)

    run_name = f"{args.model_type}_seed{args.seed}_h{args.hidden_dim}"

    with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters
        mlflow.log_params({
            "model_type": args.model_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
            "epochs": args.epochs,
            "optimizer": "Adam",
            "parameter_count": param_count,
        })

        # Log provenance tags
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dataset_name", "Cora")
        mlflow.set_tag("dataset_md5", dataset_hash)
        mlflow.set_tag("author", "mlops_pipeline")

        # ===== TRAINING LOOP =====
        best_val_f1 = 0.0
        best_state = None
        best_epoch = 0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            # --- Training step ---
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            train_loss = F.cross_entropy(out[train_mask], y[train_mask])
            train_loss.backward()
            optimizer.step()

            # --- Validation step ---
            val_result = evaluate(model, x, edge_index, y, val_mask)

            # Log metrics per epoch
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss.item()),
                    "val_loss": val_result["loss"],
                    "val_accuracy": val_result["accuracy"],
                    "val_macro_f1": val_result["macro_f1"],
                },
                step=epoch,
            )

            if epoch % 20 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | "
                    f"Val Acc: {val_result['accuracy']:.4f} | "
                    f"Val F1: {val_result['macro_f1']:.4f}"
                )

            # Checkpoint best model based on validation Macro-F1
            if val_result["macro_f1"] > best_val_f1:
                best_val_f1 = val_result["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_macro_f1", best_val_f1)

        print(f"\nTraining complete in {training_time:.1f}s")
        print(f"Best epoch: {best_epoch}, Best val F1: {best_val_f1:.4f}")

        # ===== FINAL TEST EVALUATION =====
        model.load_state_dict(best_state)
        model.to(device)
        test_result = evaluate(model, x, edge_index, y, test_mask)

        mlflow.log_metrics({
            "test_accuracy": test_result["accuracy"],
            "test_macro_f1": test_result["macro_f1"],
            "test_loss": test_result["loss"],
        })

        print(f"Test Accuracy:  {test_result['accuracy']:.4f}")
        print(f"Test Macro-F1:  {test_result['macro_f1']:.4f}")

        # ===== SAVE ARTIFACTS =====
        # Create a temporary directory for artifacts
        artifact_dir = tempfile.mkdtemp(prefix="mlops_artifacts_")

        try:
            # 1. Confusion matrix plot
            cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
            save_confusion_matrix(
                test_result["true_labels"],
                test_result["predictions"],
                CORA_CLASS_NAMES,
                cm_path,
            )

            # 2. Classification report
            report = classification_report(
                test_result["true_labels"],
                test_result["predictions"],
                target_names=list(CORA_CLASS_NAMES.values()),
                output_dict=True,
                zero_division=0,
            )
            report_path = os.path.join(artifact_dir, "classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # 3. Packaged model bundle (weights + metadata)
            model_pkg_dir = os.path.join(artifact_dir, "model_package")
            package_model(
                model=model,
                model_type=args.model_type,
                params={
                    "in_dim": data["num_features"],
                    "hidden_dim": args.hidden_dim,
                    "out_dim": data["num_classes"],
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": best_epoch,
                },
                class_names=CORA_CLASS_NAMES,
                output_dir=model_pkg_dir,
            )

            # 4. Save graph context for transductive serving
            graph_context = {
                "edge_index": data["edge_index"].cpu(),
                "x": data["x"].cpu(),
                "y": data["y"].cpu(),
                "num_nodes": data["num_nodes"],
            }
            torch.save(
                graph_context,
                os.path.join(model_pkg_dir, "graph_context.pt"),
            )

            # Log all artifacts to MLflow
            mlflow.log_artifacts(artifact_dir)

            # Also save a local copy for the FastAPI server
            local_model_dir = os.path.join("models", "latest")
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            shutil.copytree(model_pkg_dir, local_model_dir)
            print(f"\nModel package saved locally to: {local_model_dir}")

        finally:
            # Clean up temp directory
            shutil.rmtree(artifact_dir, ignore_errors=True)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print("View dashboard: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
