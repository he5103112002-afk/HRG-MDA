import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import copy
from torch.utils.checkpoint import checkpoint

CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
]

BRAIN_REGIONS = {
    'frontal': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'temporal_left': [14, 23, 32],
    'central': [15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30],
    'temporal_right': [22, 31, 40],
    'parietal': [33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'occipital': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
}

BRAIN_REGIONS_NAMES = {k: [CHANNEL_NAMES[i] for i in v] for k, v in BRAIN_REGIONS.items()}


def create_region_mask():
    mask = torch.zeros(len(CHANNEL_NAMES), len(CHANNEL_NAMES))
    for region_channels in BRAIN_REGIONS.values():
        for i in region_channels:
            for j in region_channels:
                mask[i, j] = 1.0
    return mask


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


# =========================================================
# 极其关键：RMSNorm 工具类
# =========================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


# =========================================================
# 核心修复版：多头差分注意力 (包含内部 Head-wise RMSNorm)
# =========================================================
class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads, lambda_init, dropout, temperature=1.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature

        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # 【核心修复】：将原版危险的 torch.randn 改为 torch.zeros
        # 这确保了点乘之和为 0，使得初始 lambda_val = exp(0) - exp(0) + lambda_init = lambda_init
        # 彻底杜绝训练第一步的梯度爆炸问题！
        self.lambda_q1 = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.zeros(num_heads, self.d_head))

        self.lambda_init = lambda_init
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5

        self._reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.attn_hook_helper = nn.Identity()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X, attn_mask=None):
        batch, N, d_model = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)

        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        # 由于初始化为0，这里起步严严格格就是 lambda_init (0.2)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        scaling = 1.0 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            A1 = A1 + attn_mask
            A2 = A2 + attn_mask

        attention1 = F.softmax(A1 / self.temperature, dim=-1)
        attention2 = F.softmax(A2 / self.temperature, dim=-1)

        # 核心差分相减
        attention = attention1 - lambda_val * attention2
        attention = self.attn_hook_helper(attention)
        attention = self.dropout(attention)

        O = torch.matmul(attention, V)

        # 【核心保留】：原著中极其关键的内部 Head-wise RMSNorm，防止方差震荡
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)
        O_normalized = O_normalized * (1.0 - self.lambda_init)

        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)
        out = self.dropout(self.W_o(O_concat))

        return out


# ---------------------------------------------------------
# Step 2: Region-aware Graph Representation Module (RGRM)
# ---------------------------------------------------------
class RegionAwareGraphModule(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(RegionAwareGraphModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.register_buffer('region_mask', create_region_mask())
        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 0:
                self.register_buffer(f'{region_name}_indices', torch.tensor(channels, dtype=torch.long))
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        self.alpha_cont = nn.Parameter(torch.tensor(0.5))
        self.alpha_sparse = nn.Parameter(torch.tensor(0.5))
        self.output_projection = nn.Linear(hidden_dim, 310)
        self.attention_weights_extractor = nn.Linear(310, 310)

    def regional_continuous_attention(self, Q, K, V):
        batch_size, num_channels, dim = Q.shape
        output = torch.zeros_like(V)
        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 0:
                region_indices = getattr(self, f'{region_name}_indices')
                region_values = V[:, region_indices, :]
                avg_values = region_values.mean(dim=1, keepdim=True)
                output[:, region_indices, :] = avg_values.expand(-1, len(channels), -1)
        return output

    def regional_sparse_attention(self, Q, K, V):
        batch_size, num_channels, dim = Q.shape
        output = torch.zeros_like(V)
        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 1:
                region_indices = getattr(self, f'{region_name}_indices')
                region_Q = Q[:, region_indices, :]
                region_K = K[:, region_indices, :]
                region_V = V[:, region_indices, :]
                similarity = torch.bmm(region_Q, region_K.transpose(1, 2))
                mask = torch.eye(len(channels), device=Q.device, dtype=torch.bool).unsqueeze(0)
                similarity.masked_fill_(mask, float('-inf'))
                _, max_indices = torch.max(similarity, dim=2)
                batch_indices = torch.arange(batch_size, device=Q.device).unsqueeze(1)
                channel_indices = torch.arange(len(channels), device=Q.device).unsqueeze(0)
                selected_indices = max_indices[batch_indices, channel_indices]
                for i, channel_idx in enumerate(region_indices):
                    output[:, channel_idx, :] = V[batch_indices.squeeze(), region_indices[selected_indices[:, i]], :]
            elif len(channels) == 1:
                region_indices = getattr(self, f'{region_name}_indices')
                output[:, region_indices, :] = V[:, region_indices, :]
        return output

    def forward(self, x, return_attention=False):
        batch_size, time_steps, feature_dim = x.shape
        if feature_dim != 310:
            raise ValueError(f"Expected feature dimension 310, got {feature_dim}")
        x_reshaped = x.view(batch_size, time_steps, 62, 5)
        enhanced_features = []
        attention_weights_list = []

        for t in range(time_steps):
            x_t = x_reshaped[:, t, :, :]
            Q = self.W_q(x_t)
            K = self.W_k(x_t)
            V = self.W_v(x_t)
            cont_output = self.regional_continuous_attention(Q, K, V)
            sparse_output = self.regional_sparse_attention(Q, K, V)
            alpha_cont_norm = torch.sigmoid(self.alpha_cont)
            alpha_sparse_norm = 1.0 - alpha_cont_norm
            fused_output = alpha_cont_norm * cont_output + alpha_sparse_norm * sparse_output
            pooled_graph_feature = fused_output.mean(dim=1)
            projected_feature = self.output_projection(pooled_graph_feature)
            enhanced_feature = x[:, t, :] + projected_feature
            enhanced_features.append(enhanced_feature)
            if return_attention:
                attention_weights = torch.sigmoid(self.attention_weights_extractor(enhanced_feature))
                attention_weights_list.append(attention_weights)

        enhanced_sequence = torch.stack(enhanced_features, dim=1)
        if return_attention:
            attention_weights_tensor = torch.stack(attention_weights_list, dim=1)
            aggregated_attention = attention_weights_tensor.mean(dim=1)
            return enhanced_sequence, aggregated_attention
        return enhanced_sequence


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class GCN(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = (F.relu(output + self.bias), adj)
        else:
            output = (F.relu(output), adj)
        return output

    def norm_adj(self, adj):
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1.0
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class ChebyNet(Module):
    def __init__(self, K, in_feature, out_feature):
        super(ChebyNet, self).__init__()
        self.K = K
        self.filter_weight, self.filter_bias = self.init_fliter(K, in_feature, out_feature)

    def init_fliter(self, K, feature, out, bias=True):
        weight = nn.Parameter(torch.FloatTensor(K, 1, feature, out), requires_grad=True)
        nn.init.normal_(weight, 0, 0.1)
        bias_ = None
        if bias:
            bias_ = nn.Parameter(torch.zeros((1, 1, out), dtype=torch.float32), requires_grad=True)
            nn.init.normal_(bias_, 0, 0.1)
        return weight, bias_

    def get_L(self, adj):
        degree = torch.sum(adj, dim=1)
        degree_norm = torch.div(1.0, torch.sqrt(degree) + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        L = - torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        return L

    def rescale_L(self, L):
        largest_eigval, _ = torch.linalg.eigh(L)
        largest_eigval = torch.max(largest_eigval)
        L = (2.0 / largest_eigval) * L - torch.eye(L.size(0), device=L.device, dtype=torch.float)
        return L

    def chebyshev(self, x, L):
        W = self.filter_weight.squeeze(1)
        x0 = x
        out = torch.matmul(x0, W[0])

        if self.K > 1:
            x1 = torch.matmul(L, x)
            out = out + torch.matmul(x1, W[1])

        for k in range(2, self.K):
            x2 = 2.0 * torch.matmul(L, x1) - x0
            out = out + torch.matmul(x2, W[k])
            x0, x1 = x1, x2

        out = F.relu(out + self.filter_bias)
        return out

    def forward(self, data):
        x, adj = data
        L = self.get_L(adj)
        out = self.chebyshev(x, L)
        out = (out, adj)
        return out


class GraphEncoder(nn.Module):
    def __init__(self, num_layers, num_node, in_features, out_features, K,
                 graph2token='Linear', encoder_type='GCN'):
        super(GraphEncoder, self).__init__()
        self.graph2token = graph2token
        self.K = K
        assert graph2token in ['Linear', 'AvgPool', 'MaxPool', 'Flatten', 'None']
        if graph2token == 'Linear':
            self.tokenizer = nn.Linear(num_node * out_features, out_features)
        else:
            self.tokenizer = None
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = self.get_layer(encoder_type, in_features, out_features)
            else:
                layer = self.get_layer(encoder_type, out_features, out_features)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)

    def get_layer(self, encoder_type, in_features, out_features):
        assert encoder_type in ['Cheby', 'GCN']
        if encoder_type == 'GCN':
            GNN = GCN(in_features, out_features)
        if encoder_type == 'Cheby':
            GNN = ChebyNet(self.K, in_features, out_features)
        return GNN

    def forward(self, x, adj):
        output = self.encoder((x, adj))
        x, _ = output
        if self.tokenizer is not None:
            x = x.view(x.size(0), -1)
            output = self.tokenizer(x)
        else:
            if self.graph2token == 'AvgPool':
                output = torch.mean(x, dim=-1)
            elif self.graph2token == 'MaxPool':
                output = torch.max(x, dim=-1)[0]
            elif self.graph2token == 'Flatten':
                output = x.view(x.size(0), -1)
            elif self.graph2token == 'None':
                output = x
        return output


class RMPG(nn.Module):
    def __init__(self, in_features=5, hidden_dim=16, num_nodes=62, num_adj=2, K=4):
        super(RMPG, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.GE1 = GraphEncoder(num_layers=1, num_node=num_nodes, in_features=in_features,
                                out_features=hidden_dim, K=K, graph2token='None', encoder_type='Cheby')
        self.GE2 = GraphEncoder(num_layers=2, num_node=num_nodes, in_features=in_features,
                                out_features=hidden_dim, K=K, graph2token='None', encoder_type='Cheby')

        self.adjs = nn.Parameter(torch.FloatTensor(num_adj, num_nodes, num_nodes), requires_grad=True)
        nn.init.xavier_uniform_(self.adjs)

        self.restore_proj = nn.Linear(hidden_dim, in_features)

    def get_adj(self, self_loop=False):
        num_nodes = self.adjs.shape[-1]
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(adj.device)
        return adj

    def run_ge1(self, feat, adj_mat):
        return self.GE1(feat, adj_mat)

    def run_ge2(self, feat, adj_mat):
        return self.GE2(feat, adj_mat)

    def forward(self, x):
        B, T, _ = x.shape
        x_reshaped = x.view(B * T, self.num_nodes, -1)

        adjs = self.get_adj(self_loop=False)

        if self.training:
            dummy_tensor = x_reshaped
            if not dummy_tensor.requires_grad:
                dummy_tensor.requires_grad_()
            x1 = checkpoint(self.run_ge1, dummy_tensor, adjs[0], use_reentrant=False)
            x2 = checkpoint(self.run_ge2, dummy_tensor, adjs[1], use_reentrant=False)
        else:
            x1 = self.run_ge1(x_reshaped, adjs[0])
            x2 = self.run_ge2(x_reshaped, adjs[1])

        x_fused = torch.stack((x1, x2), dim=1)
        x_graph = torch.mean(x_fused, dim=1)

        x_restored = self.restore_proj(x_graph)
        x_out = x_restored.view(B, T, -1)

        return x + x_out


# ---------------------------------------------------------
# Step 4: MultiScaleTemporalTransformer (带前置RMSNorm及残差)
# ---------------------------------------------------------
# ---------------------------------------------------------
# Step 4: MultiScaleTemporalTransformer (纯净去噪版：仅保留 Pre-Norm，舍弃残差)
# ---------------------------------------------------------
class MultiScaleTemporalTransformer(nn.Module):
    def __init__(self, input_dim=310, hidden_dim=64, num_heads=8, window_size=5, sparse_period=3):
        super(MultiScaleTemporalTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.sparse_period = sparse_period

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 保留外部 Pre-Norm，稳定输入差分注意力的特征方差
        self.local_norm = RMSNorm(hidden_dim)
        self.local_attention = MultiHeadDifferentialAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            lambda_init=0.2,  # 严格遵照第一层理论最优值0.2
            dropout=0.1
        )

        self.global_norm = RMSNorm(hidden_dim)
        self.global_attention = MultiHeadDifferentialAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            lambda_init=0.2,
            dropout=0.1
        )

        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._cached_masks = {}

    def get_local_mask(self, seq_len, device):
        cache_key = f"local_{seq_len}_{device}"
        if cache_key not in self._cached_masks:
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            for i in range(seq_len):
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                mask[i, start:end] = 0.0
            self._cached_masks[cache_key] = mask
        return self._cached_masks[cache_key]

    def get_sparse_mask(self, seq_len, device):
        cache_key = f"sparse_{seq_len}_{device}"
        if cache_key not in self._cached_masks:
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            for i in range(seq_len):
                mask[i, i] = 0.0
                for j in range(0, seq_len, self.sparse_period):
                    mask[i, j] = 0.0
            self._cached_masks[cache_key] = mask
        return self._cached_masks[cache_key]

    def forward(self, x, return_attention=False):
        batch_size, seq_len, input_dim = x.shape
        # x_proj 包含了未经时序提纯的原始特征和噪声
        x_proj = self.input_projection(x)

        local_mask = self.get_local_mask(seq_len, x.device)
        sparse_mask = self.get_sparse_mask(seq_len, x.device)

        # ==============================================================
        # 【核心修正】：只做 Pre-Norm 平稳化，坚决去掉了 `+ x_proj`
        # 此时 local_out 和 global_out 是经过差分相减后，绝对纯净的特征
        # ==============================================================
        local_out = self.local_attention(self.local_norm(x_proj), attn_mask=local_mask)
        global_out = self.global_attention(self.global_norm(x_proj), attn_mask=sparse_mask)

        # 将局部和全局的纯净特征进行拼接和线性融合
        fused_features = self.fusion(torch.cat([local_out, global_out], dim=-1))

        # 在纯净特征上计算纯净的时间注意力权重，下发给 CoDG 模块
        attention_scores = self.attention_pool(fused_features)
        attention_weights = F.softmax(attention_scores, dim=1).squeeze(-1)

        # 加权求和，得到最终的高保真序列级情绪表征
        sequence_repr = torch.sum(fused_features * attention_weights.unsqueeze(-1), dim=1)

        if return_attention:
            return sequence_repr, attention_weights
        return sequence_repr


class CollaborativeDomainGeneralization(nn.Module):
    def __init__(self, feature_dim=64, spatial_attention_dim=310, temporal_attention_dim=30, temperature=0.07):
        super(CollaborativeDomainGeneralization, self).__init__()
        self.temperature = temperature
        self.spatial_attention_encoder = nn.Sequential(
            nn.Linear(spatial_attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        self.temporal_attention_encoder = nn.Sequential(
            nn.Linear(temporal_attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        self.domain_invariant_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        self.orthogonal_projector = nn.Linear(feature_dim, feature_dim, bias=False)

    def compute_contrastive_loss(self, attention_emb, subject_ids):
        attention_emb_norm = F.normalize(attention_emb, dim=1)
        sim_matrix = torch.matmul(attention_emb_norm, attention_emb_norm.T) / self.temperature
        subject_mask = (subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)).float()
        subject_mask.fill_diagonal_(0.0)
        pos_sim = sim_matrix * subject_mask
        neg_sim = sim_matrix * (1.0 - subject_mask)
        pos_sim = pos_sim.sum(dim=1) / (subject_mask.sum(dim=1) + 1e-8)
        neg_sim = torch.logsumexp(neg_sim, dim=1)
        contrastive_loss = -pos_sim + neg_sim
        return contrastive_loss.mean()

    def compute_mmd_loss(self, features, subject_ids):
        unique_subjects = torch.unique(subject_ids)
        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=features.device)
        mmd_loss = 0.0
        count = 0
        for i, subject_i in enumerate(unique_subjects):
            for j, subject_j in enumerate(unique_subjects):
                if i >= j:
                    continue
                features_i = features[subject_ids == subject_i]
                features_j = features[subject_ids == subject_j]
                if len(features_i) > 0 and len(features_j) > 0:
                    mean_i = features_i.mean(dim=0)
                    mean_j = features_j.mean(dim=0)
                    mmd_loss += torch.norm(mean_i - mean_j, p=2)
                    count += 1
        return mmd_loss / max(count, 1)

    def compute_orthogonal_loss(self, features):
        features_centered = features - features.mean(dim=0, keepdim=True)
        correlation_matrix = torch.matmul(features_centered.T, features_centered) / (features.size(0) - 1.0)
        identity = torch.eye(correlation_matrix.size(0), device=features.device)
        orthogonal_loss = torch.norm(correlation_matrix * (1.0 - identity), p='fro') ** 2
        return orthogonal_loss

    def forward(self, features, spatial_attention_weights, temporal_attention_weights, subject_ids=None, labels=None):
        batch_size = features.size(0)
        domain_invariant_features = self.domain_invariant_projector(features)
        orthogonal_features = self.orthogonal_projector(domain_invariant_features)
        spatial_attention_emb = self.spatial_attention_encoder(spatial_attention_weights)
        temporal_attention_emb = self.temporal_attention_encoder(temporal_attention_weights)
        attention_fusion = torch.cat([spatial_attention_emb, temporal_attention_emb], dim=-1)
        losses = {}
        if self.training:
            orthogonal_loss = self.compute_orthogonal_loss(orthogonal_features)
            losses['feature_orthogonal'] = orthogonal_loss
            if subject_ids is not None:
                contrastive_loss = self.compute_contrastive_loss(attention_fusion, subject_ids)
                losses['attention_contrastive'] = contrastive_loss
                mmd_loss = self.compute_mmd_loss(orthogonal_features, subject_ids)
                losses['feature_mmd'] = mmd_loss
        return orthogonal_features, losses



# =========================================================
# Prompt-Guided DG：轻量 EEG 空频 Prompt Pool + MHCA 检索器
# 注意：这个模块不会在 RSMCoDGModel.__init__ 中默认实例化，
# 必须等 Stage 1 原始主干训练完成后，通过 enable_prompt_learning() 再创建。
# 这样可以保证 Stage 1 的随机初始化顺序、参数集合和原始代码完全一致。
# =========================================================
class EEGSpatialSpectralPromptPool(nn.Module):
    def __init__(
        self,
        num_source_subjects,
        time_steps=30,
        num_channels=62,
        num_bands=5,
        init_std=0.03,
        max_scale=1.0,
        scale_init=1.0,
        attention_temperature=1.0,
        num_heads=5,
        use_scale=False,
    ):
        super(EEGSpatialSpectralPromptPool, self).__init__()
        if num_source_subjects <= 0:
            raise ValueError(f"num_source_subjects must be positive, got {num_source_subjects}")
        if num_bands % num_heads != 0:
            raise ValueError(f"num_bands={num_bands} must be divisible by num_heads={num_heads}")

        self.num_source_subjects = int(num_source_subjects)
        self.time_steps = int(time_steps)
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)
        self.max_scale = float(max_scale)
        self.attention_temperature = float(attention_temperature)
        self.num_heads = int(num_heads)
        self.head_dim = self.num_bands // self.num_heads
        self.use_scale = bool(use_scale)

        # 每个源被试一个结构化 EEG Prompt：P_i ∈ R^{1 × 62 × 5}。
        # 输入 x 是 [B, 30, 310]，其中 310 = 62 × 5；forward 时沿时间维广播。
        # 按照原 PGDG 源码的强注入思想，这里默认 init_std=0.03，并且 use_scale=False 时直接做 x + prompt。
        self.prompts = nn.Parameter(
            torch.randn(self.num_source_subjects, 1, self.num_channels, self.num_bands) * float(init_std)
        )

        # 可选 sigmoid gate 控制 Prompt 幅度。
        # 当 use_scale=False 时，严格按照原 Prompt 代码的 x + prompt 方式，不再乘 prompt_scale。
        # 当 use_scale=True 时，实际 scale = sigmoid(prompt_gate) * max_scale。
        if self.use_scale:
            scale_init = max(min(float(scale_init), self.max_scale * 0.95), self.max_scale * 0.01)
            gate_init = math.log(scale_init / (self.max_scale - scale_init))
            self.prompt_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        else:
            self.prompt_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)

        # 轻量 MHCA：沿 5 个频带维做 Q/K 投影，电极维保留，不引入大矩阵。
        # Q 来自当前样本的 [62,5] 空频图，K/V 来自源域 prompts。
        self.query_proj = nn.Linear(self.num_bands, self.num_bands, bias=False)
        self.key_proj = nn.Linear(self.num_bands, self.num_bands, bias=False)
        self.out_proj = nn.Linear(self.num_bands, self.num_bands, bias=False)

    @property
    def prompt_scale(self):
        if not self.use_scale:
            return self.prompts.new_tensor(1.0)
        return torch.sigmoid(self.prompt_gate) * self.max_scale

    def prompt_parameters(self):
        yield self.prompts
        if self.use_scale:
            yield self.prompt_gate

    def attention_parameters(self):
        for param in self.query_proj.parameters():
            yield param
        for param in self.key_proj.parameters():
            yield param
        for param in self.out_proj.parameters():
            yield param

    def regularization_loss(self):
        if self.use_scale:
            return self.prompts.pow(2).mean() + 0.01 * self.prompt_scale.pow(2)
        return self.prompts.pow(2).mean()

    def _format_subject_ids(self, subject_ids, batch_size, device):
        if subject_ids is None:
            raise ValueError("subject_ids is required when prompt mode is 'subject'.")
        subject_ids = subject_ids.to(device=device, dtype=torch.long).view(-1)
        if subject_ids.numel() != batch_size:
            raise ValueError(f"subject_ids length {subject_ids.numel()} does not match batch size {batch_size}")
        if torch.any(subject_ids < 0) or torch.any(subject_ids >= self.num_source_subjects):
            raise ValueError(
                f"subject_ids must be in [0, {self.num_source_subjects - 1}], got "
                f"min={int(subject_ids.min().item())}, max={int(subject_ids.max().item())}"
            )
        return subject_ids

    def _subject_prompt(self, batch_size, subject_ids, device):
        subject_ids = self._format_subject_ids(subject_ids, batch_size, device)
        return self.prompts[subject_ids]

    def _attention_prompt(self, x, subject_ids=None, exclude_self=False):
        B, T, F_dim = x.shape
        if F_dim != self.num_channels * self.num_bands:
            raise ValueError(
                f"Expected feature dimension {self.num_channels * self.num_bands}, got {F_dim}"
            )

        # 当前样本 Query: [B, 62, 5]。用时间平均做稳定检索，避免为每个时间步都做大注意力。
        x_map = x.view(B, T, self.num_channels, self.num_bands).mean(dim=1)
        prompt_map = self.prompts.squeeze(1)  # [M, 62, 5]

        Q = self.query_proj(x_map)
        K = self.key_proj(prompt_map)
        V = prompt_map

        Q = Q.view(B, self.num_channels, self.num_heads, self.head_dim)
        K = K.view(self.num_source_subjects, self.num_channels, self.num_heads, self.head_dim)
        V = V.view(self.num_source_subjects, self.num_channels, self.num_heads, self.head_dim)

        # scores: [B, H, M]，每个 head 独立从 M 个源域 prompt 中取权重。
        scores = torch.einsum('bchd,mchd->bhm', Q, K) / math.sqrt(self.num_channels * self.head_dim)
        scores = scores / max(self.attention_temperature, 1e-6)

        if exclude_self and subject_ids is not None and self.num_source_subjects > 1:
            subject_ids = self._format_subject_ids(subject_ids, B, x.device)
            mask = F.one_hot(subject_ids, num_classes=self.num_source_subjects).bool().unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        mixed = torch.einsum('bhm,mchd->bchd', weights, V)
        mixed = mixed.contiguous().view(B, self.num_channels, self.num_bands)
        mixed = self.out_proj(mixed)
        return mixed.unsqueeze(1)

    def forward(self, x, subject_ids=None, mode='subject', exclude_self=False):
        B, T, F_dim = x.shape
        if mode in [None, 'none']:
            return x
        if F_dim != self.num_channels * self.num_bands:
            raise ValueError(
                f"Expected feature dimension {self.num_channels * self.num_bands}, got {F_dim}"
            )

        if mode == 'subject':
            prompt = self._subject_prompt(B, subject_ids, x.device)
        elif mode == 'attention':
            prompt = self._attention_prompt(x, subject_ids=subject_ids, exclude_self=exclude_self)
        else:
            raise ValueError(f"Unsupported prompt mode: {mode}")

        prompt = prompt.view(B, 1, F_dim).expand(-1, T, -1)
        if self.use_scale:
            return x + self.prompt_scale * prompt
        return x + prompt


class SourceSpecificClassifier(nn.Module):
    def __init__(self, feature_dim=64, number_of_category=3, dropout_rate=0.2):
        super(SourceSpecificClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, number_of_category, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class RSMCoDGModel(nn.Module):
    def __init__(self, cuda, number_of_category=3, dropout_rate=0.4, time_steps=30):
        super(RSMCoDGModel, self).__init__()
        self.noise_layer = GaussianNoise(std=0.12)

        self.region_graph_module = RegionAwareGraphModule(input_dim=5, hidden_dim=64)

        self.rmpg = RMPG(in_features=5, hidden_dim=16, num_nodes=62, num_adj=2, K=4)

        self.temporal_transformer = MultiScaleTemporalTransformer(
            input_dim=310,
            hidden_dim=64,
            num_heads=8
        )

        self.codg = CollaborativeDomainGeneralization(
            feature_dim=64,
            spatial_attention_dim=310,
            temporal_attention_dim=time_steps
        )

        self.cls_fc = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LayerNorm(64),  # 这里将 BatchNorm1d 替换为了 LayerNorm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),  # 这里将 BatchNorm1d 替换为了 LayerNorm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, number_of_category, bias=True)
        )

        # Prompt 相关模块默认不创建，保证 Stage 1 与原始模型初始化完全一致。
        self.number_of_category = number_of_category
        self.dropout_rate = dropout_rate
        self.time_steps = time_steps
        self.prompt_pool = None
        self.source_specific_classifiers = None

    def enable_prompt_learning(
        self,
        num_source_subjects,
        prompt_init_std=0.03,
        prompt_max_scale=1.0,
        prompt_scale_init=1.0,
        prompt_attention_temperature=1.0,
        prompt_heads=5,
        prompt_use_scale=False,
    ):
        self.prompt_pool = EEGSpatialSpectralPromptPool(
            num_source_subjects=num_source_subjects,
            time_steps=self.time_steps,
            num_channels=62,
            num_bands=5,
            init_std=prompt_init_std,
            max_scale=prompt_max_scale,
            scale_init=prompt_scale_init,
            attention_temperature=prompt_attention_temperature,
            num_heads=prompt_heads,
            use_scale=prompt_use_scale,
        )
        self.source_specific_classifiers = nn.ModuleList([
            SourceSpecificClassifier(
                feature_dim=64,
                number_of_category=self.number_of_category,
                dropout_rate=min(float(self.dropout_rate), 0.2),
            )
            for _ in range(num_source_subjects)
        ])
        return self

    def has_prompt_learning(self):
        return self.prompt_pool is not None and self.source_specific_classifiers is not None

    def compute_specific_logits(self, features, subject_ids):
        if self.source_specific_classifiers is None:
            raise RuntimeError("source_specific_classifiers is None. Call enable_prompt_learning() first.")
        subject_ids = subject_ids.to(device=features.device, dtype=torch.long).view(-1)
        logits = features.new_zeros(features.size(0), self.number_of_category)
        for subject_idx in torch.unique(subject_ids):
            idx = subject_ids == subject_idx
            logits[idx] = self.source_specific_classifiers[int(subject_idx.item())](features[idx])
        return logits

    def forward(
        self,
        x,
        subject_ids=None,
        labels=None,
        apply_noise=False,
        return_feats=False,
        use_prompt=False,
        prompt_mode='none',
        prompt_exclude_self=False,
        return_specific=False,
    ):
        if apply_noise and self.training:
            x = self.noise_layer(x)

        if use_prompt:
            if self.prompt_pool is None:
                raise RuntimeError("use_prompt=True but prompt_pool is None. Call enable_prompt_learning() first.")
            x = self.prompt_pool(
                x,
                subject_ids=subject_ids,
                mode=prompt_mode,
                exclude_self=prompt_exclude_self,
            )

        x_enhanced, spatial_attention_weights = self.region_graph_module(x, return_attention=True)
        x_topological = self.rmpg(x_enhanced)

        sequence_repr, a_temporal = self.temporal_transformer(x_topological, return_attention=True)

        domain_generalized_features, dg_losses = self.codg(
            sequence_repr, spatial_attention_weights, a_temporal, subject_ids, labels
        )

        x_logits = self.cls_fc(domain_generalized_features)
        x_pred = F.log_softmax(x_logits, dim=1)

        cls_loss = None
        if labels is not None:
            cls_loss = F.nll_loss(x_pred, labels)

        specific_pred = None
        specific_loss = None
        if return_specific:
            if subject_ids is None:
                raise ValueError("subject_ids is required when return_specific=True")
            specific_logits = self.compute_specific_logits(domain_generalized_features, subject_ids)
            specific_pred = F.log_softmax(specific_logits, dim=1)
            if labels is not None:
                specific_loss = F.nll_loss(specific_pred, labels)

        if return_feats and return_specific:
            return (
                x_pred,
                x_logits,
                dg_losses,
                cls_loss,
                domain_generalized_features,
                spatial_attention_weights,
                a_temporal,
                specific_pred,
                specific_loss,
            )

        if return_feats:
            return x_pred, x_logits, dg_losses, cls_loss, domain_generalized_features, spatial_attention_weights, a_temporal

        if return_specific:
            return x_pred, x_logits, dg_losses, cls_loss, specific_pred, specific_loss

        return x_pred, x_logits, dg_losses, cls_loss


class RSMCoDGTestModel(nn.Module):
    def __init__(self, baseModel, use_prompt=False, prompt_mode='none'):
        super(RSMCoDGTestModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
        self.use_prompt = use_prompt
        self.prompt_mode = prompt_mode

    def forward(self, x):
        x_pred, x_logits, _, _ = self.baseModel(
            x,
            subject_ids=None,
            labels=None,
            apply_noise=False,
            use_prompt=self.use_prompt,
            prompt_mode=self.prompt_mode,
        )
        return x_logits


def create_rsm_codg_model(num_classes=3, dropout_rate=0.4):
    return RSMCoDGModel(
        cuda=True,
        number_of_category=num_classes,
        dropout_rate=dropout_rate
    )


def calculate_total_loss(cls_loss, dg_losses, loss_weights=None):
    if loss_weights is None:
        loss_weights = {
            'feature_orthogonal': 0.05,
            'attention_contrastive': 0.2,
            'feature_mmd': 0.15
        }
    total_loss = cls_loss
    for loss_name, loss_value in dg_losses.items():
        if loss_name in loss_weights:
            total_loss += loss_weights[loss_name] * loss_value
    return total_loss