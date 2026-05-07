import torch
import torch.nn as nn


class GatedDynamicGCN(nn.Module):
    def __init__(self, dim_in, dim_out, top_k=10):
        super().__init__()
        self.dim_out = dim_out
        self.top_k = top_k
        self.feature_projection = nn.Linear(dim_in, dim_out)
        self.gate_projection = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.activation = nn.ReLU()

    def _expand_graph(self, graph, batch_size, seq_len):
        if graph is None:
            raise ValueError("Graph input is required for GatedDynamicGCN.")

        if graph.dim() == 2:
            return graph.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        if graph.dim() == 3:
            return graph.unsqueeze(1).expand(batch_size, seq_len, -1, -1)
        if graph.dim() == 4:
            return graph
        raise ValueError(f"Unsupported graph shape: {tuple(graph.shape)}")

    def _keep_topk(self, graph):
        num_nodes = graph.shape[-1]
        if self.top_k >= num_nodes:
            return graph

        _, topk_indices = torch.topk(graph, self.top_k, dim=-1)
        mask = torch.zeros_like(graph)
        mask.scatter_(-1, topk_indices, 1.0)
        return graph * mask

    def forward(self, x, graph):
        batch_size, seq_len, num_nodes, _ = x.shape
        graph = self._expand_graph(graph, batch_size, seq_len)
        graph = self._keep_topk(graph)

        x_flat = x.reshape(batch_size * seq_len, num_nodes, -1)
        graph_flat = graph.reshape(batch_size * seq_len, num_nodes, num_nodes)

        hidden = self.feature_projection(x_flat)
        aggregated = torch.bmm(graph_flat, hidden)
        gate = torch.sigmoid(self.gate_projection(x_flat) + aggregated)
        out = hidden + gate * aggregated
        out = self.norm(out)
        out = self.activation(out)
        return out.reshape(batch_size, seq_len, num_nodes, self.dim_out)
