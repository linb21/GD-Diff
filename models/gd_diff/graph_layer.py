import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDynamicGCN(nn.Module):
    def __init__(self, dim_in, dim_out, top_k=10):
        """
        Args:
            dim_in: 输入特征维度
            dim_out: 输出特征维度
            top_k: 动态图中保留的最大边数 (Top-K sparse)
        """
        super(GatedDynamicGCN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.top_k = top_k

        # 1. 特征变换 (Node-wise feature transformation)
        self.fc_x = nn.Linear(dim_in, dim_out)
        
        # 2. 门控参数 (Gate generation)
        self.fc_gate = nn.Linear(dim_in, dim_out)

        # 3. 归一化 (LayerNorm 对应处理非归一化权重的关键)
        self.ln = nn.LayerNorm(dim_out)

        # 4. 激活
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, graph):
        """
        Args:
            x:     [Batch, Length, Node, Dim_in]
            graph: [Batch, Length, Node, Node]  <-- 动态图输入
        """
        B, L, N, C = x.shape
        
        # 校验维度
        assert graph.shape[0] == B and graph.shape[1] == L, \
            f"Graph shape {graph.shape} mismatch with X shape {x.shape}"

        # --- 1. 维度展平 (Merge Batch and Time) ---
        # 我们将 B 和 L 合并，看作是 (B*L) 个独立的图进行并行计算
        x_flat = x.reshape(B * L, N, C)         # [B*L, N, C]
        graph_flat = graph.reshape(B * L, N, N) # [B*L, N, N]

        # --- 2. 动态图 Top-K 稀疏化 ---
        # 针对每个时间步的图，单独做 Top-K
        if self.top_k < N:
            # 在最后一个维度找 TopK
            topk_values, topk_indices = torch.topk(graph_flat, self.top_k, dim=-1)
            
            # 创建掩码
            mask = torch.zeros_like(graph_flat)
            mask.scatter_(-1, topk_indices, 1.0)
            
            # 应用掩码，保留物理权重值 (不归一化)
            curr_graph = graph_flat * mask
        else:
            curr_graph = graph_flat

        # --- 3. 线性变换 ---
        h = self.fc_x(x_flat)  # [B*L, N, Out]

        # --- 4. 空间聚合 (Spatial Aggregation) ---
        # 动态图矩阵乘法: (B*L, N, N) * (B*L, N, Out) -> (B*L, N, Out)
        # 这里正确利用了每个时间步不同的图权重
        agg_info = torch.bmm(curr_graph, h)

        # --- 5. 门控机制 (Gating) ---
        # Gate = Sigmoid( W_gate * X + Agg )
        # 动态决定每个节点在当前时刻应该接受多少邻居信息
        gate_input = self.fc_gate(x_flat) + agg_info
        gate = self.sigmoid(gate_input)

        # --- 6. 残差融合 ---
        # Out = H + (Gate * Agg)
        # 相比单纯的 H + Agg，Gate 可以防止未归一化的大权重导致梯度爆炸
        out = h + (gate * agg_info)

        # --- 7. 归一化与激活 ---
        # 必不可少
        out = self.ln(out)
        out = self.relu(out)

        # --- 8. 恢复维度 ---
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
        # 对邻接矩阵adj进行asymmetric归一化，返回归一化后的矩阵
        # D^{-1}A
        if adj.dim() == 2:
            rowsum = torch.sum(adj, dim=1) + 1e-6
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diag(d_inv)
            adj_normalized = torch.mm(d_mat_inv, adj)
        elif adj.dim() == 3:
            # 对于3维张量 (batch_size, N, N)，并行处理每个batch
            # 计算每行的和: (batch_size, N)
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
        
        # 对于version 4，如果graph是(batch_size, N, N)而x是(batch_size*Length, N, C)，需要扩展graph
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
                # 对于version 4，graph是(batch_size, N, N)，而x是(batch_size*Length, N, C)
                Length = batch_size // graph.shape[0]
                if graph.dim() == 3:
                    graph = graph.repeat_interleave(Length, dim=0)  # (batch_size*Length, N, N)
                    # 应用relu
                    k = self.top_k  # 保留top-k个值
                    # 对每一行进行topk，获取top-k个值的索引
                    _, topk_indices = torch.topk(graph, k, dim=-1)  # (batch_size*Length, N, k)
                    # 创建mask，只保留top-k个位置
                    mask = torch.zeros_like(graph)  # (batch_size*Length, N, N)
                    mask.scatter_(-1, topk_indices, 1.0)  # 在top-k位置设置为1
                    # 应用mask，只保留top-k个值
                    # graph = graph * mask  # (batch_size*Length, N, N)
                    # graph = (self.asym_adj(graph)) + 1e-6  # (batch_size*Length, N, N)

                    # TopK操作：对每一行只保留权重最高的k个值
                    N = graph.shape[-1]

                    # 归一化
                    # supports = F.softmax(F.relu(graph), dim=-1)  # (batch_size*Length, N, N)
                    supports=graph+1e-6
                else:
                    B,L,N,_=graph.shape
                    graph=graph.reshape(B*L,N,N)
                    k = self.top_k  # 保留top-k个值
                    # 对每一行进行topk，获取top-k个值的索引
                    _, topk_indices = torch.topk(graph, k, dim=-1)  # (batch_size*Length, N, k)
                    # 创建mask，只保留top-k个位置
                    mask = torch.zeros_like(graph)  # (batch_size*Length, N, N)
                    mask.scatter_(-1, topk_indices, 1.0)  # 在top-k位置设置为1
                    # 应用mask，只保留top-k个值
                    graph = graph * mask  # (batch_size*Length, N, N)
                    # graph = (self.asym_adj(graph)) + 1e-6  # (batch_size*Length, N, N)
                    supports = graph+1e-6
            else:
                raise ValueError(f"Invalid version: {self.version}")
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type}")
        # 根据supports的维度决定如何处理
        if supports.dim() == 3:  # version 4: (B*L, N, N)
            # 将torch.eye从(N, N)扩展到(B*L, N, N)
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
            # 修改einsum: bknm和bmc相乘得到bknc
            x_g = torch.einsum("bknm,bmc->bknc", supports, x)  # (B*L, cheb_k, N, C)
            x_g = x_g.permute(0, 2, 1, 3)  # (B*L, N, cheb_k, C)
            x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                                   weights) + bias  # (B*L, N, dim_out)
        else:  # 其他版本: supports是(N, N)
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