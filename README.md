# Production GNN MLOps Pipeline for Cora Node Classification

This repository contains an end-to-end, production-grade MLOps system for classifying scientific research papers on the Cora citation network using Graph Neural Networks (GraphSAGE and GCN).

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Raw Cora Dataset (2708 nodes, 1433 d) |
                      +---------------------------------------+
                                          |
                                          v
       +---------------------------------------------------------------------+
       | Untracked Baseline (train_baseline.py) vs MLflow Pipeline (train_mlflow.py) |
       +---------------------------------------------------------------------+
                                          |
                                          v
                   +---------------------------------------------+
                   | Packaged Model Artifact (models/latest)     |
                   | - weights.pt, metadata.json, graph_context  |
                   +---------------------------------------------+
                                          |
                                          v
                   +---------------------------------------------+
                   | Automated Validation Gate (validation_gate) |
                   | - Threshold: Delta Macro-F1 >= +1.5%        |
                   | - Tie: Retain Champion                      |
                   | - Regression: Block Candidate               |
                   +---------------------------------------------+
                            |                           |
             [Promote]      v                           v  [Block]
             +---------------------------+    +-----------------------+
             | Active Champion (champion)|    | Candidate Rejected    |
             +---------------------------+    +-----------------------+
                            |
                            v
       +---------------------------------------------------------+
       | FastAPI Serving Engine (src/serving/app.py)             |
       | - Transductive: /predict/transductive (Existing nodes)  |
       | - Inductive:    /predict/inductive    (New papers)     |
       +---------------------------------------------------------+
                                    |
                                    v
       +---------------------------------------------------------+
       | Drift Monitoring (src/monitoring/drift_detector.py)     |
       | - Structural: Two-Sample Kolmogorov-Smirnov Test        |
       | - Embedding:  Maximum Mean Discrepancy (MMD) via RBF    |
       +---------------------------------------------------------+
```

---

## 1. Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Untracked Baseline (Control Experiment)
```bash
python -m src.pipeline.train_baseline
```
*Note the 5 documented limitations*:
1. **Ephemeral Metrics**: All metrics are printed to stdout only; lost upon terminal exit.
2. **Lost Provenance**: Hyperparameters are hardcoded; no record of seed or learning rate.
3. **Silent Overwriting**: Silently replaces `model.pt` without version control.
4. **No Code/Data Lineage**: No link between model weights and Git commit or dataset hash.
5. **Orphaned Weights**: Saves only raw tensors without feature schema (1433) or class mapping.

---

### Step 3: Run MLflow Tracked Experiments
Run at least 3 distinct configurations to populate the experiment registry:

```bash
# Experiment 1: GraphSAGE Baseline (Seed 42)
python -m src.pipeline.train_mlflow --config configs/exp1_graphsage_baseline.yaml

# Experiment 2: GraphSAGE Tuned (Seed 123, Hidden Dim 128)
python -m src.pipeline.train_mlflow --config configs/exp2_graphsage_tuned.yaml

# Experiment 3: GCN Comparison Architecture (Seed 999)
python -m src.pipeline.train_mlflow --config configs/exp3_gcn_comparison.yaml
```

View the experiment comparison dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

### Step 4: Evaluate the Automated Validation Gate

```bash
# Compare candidate (models/latest) against current champion (models/champion)
python -m src.pipeline.validation_gate --candidate-dir models/latest --champion-dir models/champion
```

#### Promotion Gate Rules:
* **Metric**: Validation Macro-F1.
* **Threshold**: Candidate must beat Champion by at least $\Delta \ge +0.015$ (+1.5%).
* **Tie Behavior**: If $\Delta < 0.015$, Champion is retained (avoids unnecessary churn).
* **Regression**: If Candidate is worse ($\Delta \le 0$), promotion is blocked and non-zero exit code halts CI.
* **First Model Rule**: If no Champion exists, auto-promotes Candidate as Version 1.

---

### Step 5: Start the Serving API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://127.0.0.1:8000/docs`.

#### Endpoints:
1. `GET /health`: Health status, device, and active model version.
2. `GET /model/metadata`: Model architecture, feature dimensions, and class mappings.
3. `POST /predict/transductive`: Classify existing Cora nodes by ID.
4. `POST /predict/inductive`: Classify brand-new, unseen research papers by supplying their 1,433-dim bag-of-words vector and citation links.

---

### Step 6: Run the Automated Test Suite (13 Tests)

```bash
pytest tests/ -v
```

The test suite validates:
* **Prediction Consistency**: Bit-for-bit parity between offline inference and online HTTP responses ($L_\infty < 10^{-5}$).
* **Packaging Parity**: Reloading `weights.pt` + `metadata.json` reproduces in-memory outputs ($L_\infty < 10^{-6}$).
* **Failure Handling**:
  1. *Isolated nodes* (degree = 0) handled gracefully via self-loops.
  2. *Duplicate target IDs* deduplicated safely.
  3. *Invalid edge endpoints* ignored with warnings.
  4. *Incorrect feature dimensions* ($\ne 1433$) rejected with HTTP 422.
  5. *Unknown target IDs* return HTTP 404.
  6. *Model loading failure* raises clear exceptions.
* **Drift Detection**: Normal case verification, structural drift (KS test $p < 0.05$), and embedding drift (MMD).
* **Validation Gate**: Unit tests for first-model auto-promote, threshold checks, and regression blocks.

---

## 2. Emergency Rollback

If an approved model in production experiences regression:
```bash
python -m src.pipeline.rollback --champion-dir models/champion
```
Restores the prior champion from `models/champion_previous` with automatic health verification.

---

## 3. Drift Monitoring Details

* **Graph Structural Drift**: Two-sample Kolmogorov-Smirnov test on node degree distributions:
  $$\text{KS}(D_{\text{base}}, D_{\text{prod}}) \implies \text{Alert if } p\text{-value} < 0.05$$
* **Embedding Drift**: Maximum Mean Discrepancy (MMD) with RBF kernel using a **fixed, frozen encoder**:
  $$\text{MMD}^2(P, Q) = \mathbb{E}[k(X, X')] + \mathbb{E}[k(Y, Y')] - 2\mathbb{E}[k(X, Y)] \implies \text{Alert if } \text{MMD} > 0.03$$
* **Why raw embedding coordinates cannot be compared across different models**:
  Neural representations are invariant to orthogonal rotations, reflections, and neuron permutations. Two separate training runs can achieve identical loss while orienting the latent space in completely different directions. Therefore, drift must be computed using a single frozen encoder over time.
