import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDynamicGCN(nn.Module):
    def __init__(self, dim_in, dim_out, top_k=10):
        super(GatedDynamicGCN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.top_k = top_k

        self.fc_x = nn.Linear(dim_in, dim_out)
        
        self.fc_gate = nn.Linear(dim_in, dim_out)

        self.ln = nn.LayerNorm(dim_out)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, graph):
        B, L, N, C = x.shape
        
        assert graph.shape[0] == B and graph.shape[1] == L, \
            f"Graph shape {graph.shape} mismatch with X shape {x.shape}"

        x_flat = x.reshape(B * L, N, C)         # [B*L, N, C]
        graph_flat = graph.reshape(B * L, N, N) # [B*L, N, N]

        if self.top_k < N:
            topk_values, topk_indices = torch.topk(graph_flat, self.top_k, dim=-1)
            
            mask = torch.zeros_like(graph_flat)
            mask.scatter_(-1, topk_indices, 1.0)
            
            curr_graph = graph_flat * mask
        else:
            curr_graph = graph_flat

        h = self.fc_x(x_flat)  # [B*L, N, Out]

        agg_info = torch.bmm(curr_graph, h)

        # Gate = Sigmoid( W_gate * X + Agg )
        gate_input = self.fc_gate(x_flat) + agg_info
        gate = self.sigmoid(gate_input)

        # Out = H + (Gate * Agg)
        out = h + (gate * agg_info)

        out = self.ln(out)
        out = self.relu(out)

        return out.reshape(B, L, N, self.dim_out)
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim,graph_type,version,top_k):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.graph_type = graph_type
        self.version = version
        self.top_k = top_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.reset_parameters()
        if (self.graph_type == 'static' or self.graph_type == 'time') and self.version == 3:
            self.conv_fuse = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False)
    def reset_parameters(self):
        """Initialize parameters to avoid NaN values"""
        # Initialize weights using xavier uniform
        nn.init.xavier_uniform_(self.weights_pool)
        # Initialize bias to zeros
        nn.init.zeros_(self.bias_pool)
    def asym_adj(self, adj):
        # D^{-1}A
        if adj.dim() == 2:
            rowsum = torch.sum(adj, dim=1) + 1e-6
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diag(d_inv)
            adj_normalized = torch.mm(d_mat_inv, adj)
        elif adj.dim() == 3:
            # rowsum = torch.sum(adj, dim=-1) + 1e-6  # (batch_size, N)
            # d_inv = torch.pow(rowsum, -1)  # (batch_size, N)
            # d_inv[torch.isinf(d_inv)] = 0.
            max_value, _ = torch.max(adj, dim=-1)  # (batch_size, N)
            d_inv = torch.pow(max_value + 1e-6, -1)  # (batch_size, N)
            d_inv[torch.isinf(d_inv)] = 0.
            adj_normalized = d_inv.unsqueeze(-1) * adj  # (batch_size, N, N)
        return adj_normalized
    def forward(self, x, node_embeddings, graph=None):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        batch_size = x.shape[0]  # B*L for version 4, B for others
        
        if self.graph_type == 'sg':
            supports = F.softmax(
                F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        elif self.graph_type == 'static' or self.graph_type == 'time':
            if self.version == 1:
                supports = F.softmax(F.relu(self.asym_adj(self.graph)), dim=1)
            elif self.version == 2:
                emb_adj = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))  # (N, N)
                supports = F.softmax(F.relu(torch.mul(graph, emb_adj)), dim=1)
            elif self.version == 3:
                emb_adj = torch.mm(node_embeddings, node_embeddings.transpose(0,1)) # (N,N)
                stack_adj = torch.stack([graph, emb_adj], dim=-1).unsqueeze(0)  # (1, N, N, 2)
                fuse_input = stack_adj.permute(0, 3, 1, 2)  # (1, 2, N, N)
                if next(self.conv_fuse.parameters()).device != fuse_input.device:
                    self.conv_fuse = self.conv_fuse.to(fuse_input.device)
                fused = self.conv_fuse(fuse_input).squeeze(0).squeeze(0)  # (N, N)
                supports = F.softmax(F.relu(fused), dim=1)
                # supports: (N, N)
            elif self.version == 4:
                Length = batch_size // graph.shape[0]
                if graph.dim() == 3:
                    graph = graph.repeat_interleave(Length, dim=0)  # (batch_size*Length, N, N)
                    k = self.top_k
                    _, topk_indices = torch.topk(graph, k, dim=-1)  # (batch_size*Length, N, k)
                    mask = torch.zeros_like(graph)  # (batch_size*Length, N, N)
                    mask.scatter_(-1, topk_indices, 1.0)
                    # graph = graph * mask  # (batch_size*Length, N, N)
                    # graph = (self.asym_adj(graph)) + 1e-6  # (batch_size*Length, N, N)

                    N = graph.shape[-1]

                    # supports = F.softmax(F.relu(graph), dim=-1)  # (batch_size*Length, N, N)
                    supports=graph+1e-6
                else:
                    B,L,N,_=graph.shape
                    graph=graph.reshape(B*L,N,N)
                    k = self.top_k
                    _, topk_indices = torch.topk(graph, k, dim=-1)  # (batch_size*Length, N, k)
                    mask = torch.zeros_like(graph)  # (batch_size*Length, N, N)
                    mask.scatter_(-1, topk_indices, 1.0)
                    graph = graph * mask  # (batch_size*Length, N, N)
                    # graph = (self.asym_adj(graph)) + 1e-6  # (batch_size*Length, N, N)
                    supports = graph+1e-6
            else:
                raise ValueError(f"Invalid version: {self.version}")
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type}")
        if supports.dim() == 3:  # version 4: (B*L, N, N)
            batch_len = supports.shape[0]
            eye = torch.eye(node_num).to(supports.device)  # (N, N)
            eye = eye.unsqueeze(0).expand(batch_len, -1, -1)  # (B*L, N, N)
            support_set = [eye, supports]  # eye: (B*L, N, N), supports: (B*L, N, N)
            # default cheb_k = 3
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(
                    2 * supports, support_set[-1]) - support_set[-2])
            supports = torch.stack(support_set, dim=1)  # (B*L, cheb_k, N, N)
            # N, cheb_k, dim_in, dim_out
            weights = torch.einsum(
                'nd,dkio->nkio', node_embeddings, self.weights_pool)
            bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
            x_g = torch.einsum("bknm,bmc->bknc", supports, x)  # (B*L, cheb_k, N, C)
            x_g = x_g.permute(0, 2, 1, 3)  # (B*L, N, cheb_k, C)
            x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                                   weights) + bias  # (B*L, N, dim_out)
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
            # default cheb_k = 3
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(
                    2 * supports, support_set[-1]) - support_set[-2])
            supports = torch.stack(support_set, dim=0)  # (cheb_k, N, N)
            # N, cheb_k, dim_in, dim_out
            weights = torch.einsum(
                'nd,dkio->nkio', node_embeddings, self.weights_pool)
            bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
            x_g = torch.einsum("knm,bmc->bknc", supports,
                               x)  # B, cheb_k, N, dim_in
            x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
            x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                                   weights) + bias  # b, N, dim_out
        return x_gconv
