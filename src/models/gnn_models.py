"""
GNN Model Definitions for Cora Node Classification (Pure PyTorch Implementation).

Contains MLP, GCN (Kipf & Welling), and GraphSAGE (Hamilton et al.) model classes.
Implemented entirely with standard PyTorch tensor operations so that it runs
without requiring external C++ compiled packages (torch-geometric / torch-scatter / torch-sparse),
while preserving full mathematical equivalence and API compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """
    Pure PyTorch implementation of Graph Convolutional Layer (Kipf & Welling).
    Computes:
      H = D_hat^(-1/2) * A_hat * D_hat^(-1/2) * X * W
    where A_hat = A + I_N (self-loops) and D_hat is the degree matrix of A_hat.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with symmetric degree normalization and message passing.

        Args:
            x: Node feature tensor [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].

        Returns:
            Convolved node representations [num_nodes, out_channels].
        """
        num_nodes = x.size(0)

        # 1. Add self-loops (A_hat = A + I)
        loop_idx = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
        loop_edges = torch.stack([loop_idx, loop_idx], dim=0)
        edges = torch.cat([edge_index, loop_edges], dim=1)

        row, col = edges[0], edges[1]

        # 2. Compute degrees and symmetric normalization coefficients
        deg = torch.zeros(num_nodes, device=x.device, dtype=torch.float32)
        deg.index_add_(0, row, torch.ones(row.size(0), device=x.device, dtype=torch.float32))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 3. Linear projection
        x_proj = self.lin(x)

        # 4. Message passing (neighborhood aggregation)
        out = torch.zeros_like(x_proj)
        out.index_add_(0, row, norm.unsqueeze(-1) * x_proj[col])

        return out


class SAGEConv(nn.Module):
    """
    Pure PyTorch implementation of GraphSAGE Layer (Hamilton et al.).
    Uses mean neighborhood aggregation:
      h_v = W_self * x_v + W_neigh * Mean_{u in N(v)}(x_u)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mean neighborhood aggregation.

        Args:
            x: Node feature tensor [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].

        Returns:
            Aggregated node representations [num_nodes, out_channels].
        """
        num_nodes = x.size(0)
        row, col = edge_index[0], edge_index[1]

        # Compute in-degree for mean aggregation
        deg = torch.zeros(num_nodes, device=x.device, dtype=torch.float32)
        deg.index_add_(0, row, torch.ones(row.size(0), device=x.device, dtype=torch.float32))
        deg_clamp = deg.clamp(min=1.0).unsqueeze(-1)

        # Aggregate neighbor features
        neigh_sum = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        neigh_sum.index_add_(0, row, x[col])
        neigh_mean = neigh_sum / deg_clamp

        # Combine self and neighborhood projections
        return self.lin_self(x) + self.lin_neigh(neigh_mean)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline for node classification.
    Uses only node features, ignores graph structure entirely.
    Serves as the control model to demonstrate the value of graph edges.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        return F.relu(self.fc1(x))


class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    Uses spectral neighborhood averaging via symmetric normalization.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class GraphSAGE(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017) with mean aggregation.
    Designed for inductive learning: can generalize to unseen nodes
    by aggregating features from their local neighborhoods.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


def build_model(
    model_type: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.5,
) -> nn.Module:
    """Factory function to instantiate a model by name."""
    model_type = model_type.upper()
    if model_type == "MLP":
        return MLP(in_dim, hidden_dim, out_dim, dropout)
    elif model_type == "GCN":
        return GCN(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif model_type == "GRAPHSAGE":
        return GraphSAGE(in_dim, hidden_dim, out_dim, num_layers, dropout)
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Choose from: MLP, GCN, GraphSAGE"
        )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
