import copy

import torch
import torch.nn as nn
from STTNS.GCN_models import GCN, GraphCNN
from STTNS.One_hot_encoder import One_hot_encoder
from STTNS.layers import graph_constructor
from STTNS.utils import get_batch_edge_index
from torch_geometric.nn import GATv2Conv, GCNConv
import utils
import numpy as np
from math import sqrt


# model input shape:[1, N, T]
# model output shape:[N, T]
from STTNS import  layers1

class STTNSNet(nn.Module):
    def __init__(self, node, k, in_channels, embed_size, time_num,
                 num_layers=1, heads=1, dropout=0.2, graph='gat', forward_expansion=4):
        self.num_layers = num_layers
        super(STTNSNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)   #embed 代表25个节点
        self.transformer = Transformer(embed_size, heads, k, node, time_num, dropout, forward_expansion,graph)#forward_expansion 代表4维
        self.conv2 = nn.Conv2d(embed_size, in_channels, 1)

    def forward(self, x):
        # input x:[B, C, N, T]
        # 通道变换   C ->embedsize
        x = self.conv1(x)  # [B, embed_size, N, T]   (128,25,60,25)

        x = x.permute(0, 2, 3, 1)  # [B, N, T, embed_size]
        x = self.transformer(x, x, x, self.num_layers)  # [B, N, T, embed_size]

        # 预测时间T_dim，转换时间维数
        x = x.permute(0, 3, 1, 2)  # [B, embed_size, N, T]
        x = self.conv2(x)
        #减少维度
        out = x.squeeze(1)

        return out


class Transformer(nn.Module):
    def __init__(self, embed_size, heads, k, node, time_num, dropout, forward_expansion,graph):
        super(Transformer, self).__init__()
        self.sttnblock = STTNSNetBlock(embed_size, heads, k, node, time_num, dropout, forward_expansion,graph)

    def forward(self, query, key, value, num_layers):
        q, k, v = query, key, value
        for i in range(num_layers):
            out = self.sttnblock(q, k, v)
            q, k, v = out, out, out
        return out


# model input:[N, T, C]
# model output[N, T, C]

argsd_model = 64

from STTNS.layers1 import SpatioConvLayer

class STTNSNetBlock(nn.Module):
    def __init__(self, embed_size, heads, k, node, time_num, dropout, forward_expansion,graph):
        super(STTNSNetBlock, self).__init__()
        self.SpatialTansformer = STransformer(embed_size, heads, k, node, dropout, forward_expansion,graph)

        self.SpationConvlayer =SpatioConvLayer(3,32,32)

        #self.TemporalTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.Temporal = Crossformer( 7, 96, 24, 6, win_size = 4,
                  d_model=512,  baseline = False, device=torch.device('cuda:0'))

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        out1 = self.norm1(self.SpatialTansformer(query, key, value) + query)#移除时间块的消融实验直接注释掉out2

        #out1 = self.norm1(self.SpationConvlayer(query, key, value) )  # 移除时间块的消融实验直接注释掉out2
        #out2.shape (128,25,60,25)
        out2 = self.dropout(self.norm2(self.Temporal(out1) + out1))
        #out2=self.dropout(self.TemporalTransformer(query, key, value)+query)#移除空间块的消融实验

        return out2


# model input:[N, T, C]
# model output:[N, T, C]
class STransformer(nn.Module):
    def __init__(self, embed_size, heads, k, node, dropout, forward_expansion,graph):
        super(STransformer, self).__init__()
        self.node_idx = torch.arange(node).cuda()
        self.gen_adj = graph_constructor(node, k, embed_size).cuda()
        self.adj, _ = self.gen_adj(self.node_idx)
        self.D_S = nn.Parameter(self.adj)
        self.embed_linear = nn.Linear(node, embed_size)
        # self.attention = SSelfattention(embed_size, heads)
        self.attention = SSelfattention(60)
        self.norm1 = nn.LayerNorm(embed_size)  #标准化
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),  #forward:4 embed:25
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # 调用GCN
        if graph =='gcn':
            self.graph = GATv2Conv(embed_size, embed_size, heads=1)
        else:
            self.graph = GraphCNN(embed_size, embed_size * 2, embed_size, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Spatial Embedding 部分
        B, N, T, C = query.shape

        D_S = self.embed_linear(self.D_S)
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)   #进行T

        _, adj = self.gen_adj(self.node_idx)
        # GCN 部分
        adj = torch.tensor(adj, dtype=torch.long)
        adj = get_batch_edge_index(adj, B, N)
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)

        #对齐维度  torch.Tensor(维度四个)  X_G  (128,25,0,25)   最终T=60 Cat到上面去 -》（128,25,60,25）
        X_G = torch.Tensor(B, N, 0, C).cuda()
        for t in range(T):   #data(3200,25)   query(128,25,60,25) 128x25=3200
            data = query[:, :, t, :].reshape(-1, C).contiguous()
            o = self.graph(data, adj).view(B, -1, C).contiguous()
            #保证张量是连续的  o(128,25,25)
            o = o.unsqueeze(2)  #o(128,25,1,25)
            X_G = torch.cat((X_G, o), dim=2)

        # spatial transformer
        query = query + D_S
        value = value + D_S
        key = key + D_S
        # attn = self.attention(value, key, query)  # [N, T, C]
        attn = self.attention(query)
        M_s = self.dropout(self.norm1(attn + query))
        feedforward = self.feed_forward(M_s)
        U_s = self.dropout(self.norm2(feedforward + M_s))

        # 融合
        g = torch.sigmoid(self.fs(U_s) + self.fg(X_G))
        out = g * U_s + (1 - g) * X_G

        return out


# model input:[N,T,C]
# model output:[N,T,C]
#src
class SSelfattention1(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfattention1, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.values = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.queries = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.keys = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape
        query = values.reshape(B, N, T, self.heads, self.per_dim)
        keys = keys.reshape(B, N, T, self.heads, self.per_dim)
        values = values.reshape(B, N, T, self.heads, self.per_dim)

        # q, k, v:[N, T, heads, per_dim]
        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)

        # spatial self-attention
        attn = torch.einsum("bqthd, bkthd->bqkth", (queries, keys))  # [N, N, T, heads]
        attention = torch.softmax(attn / (self.embed_size ** (1 / 2)), dim=1)

        out = torch.einsum("bqkth,bkthd->bqthd", (attention, values))  # [N, T, heads, per_dim]
        out = out.reshape(B, N, T, self.heads * self.per_dim)  # [N, T, C]

        out = self.fc(out)

        return out
#空间块注意力

class SSelfattention(nn.Module):# MSCA的实现code   dim=60
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 1, stride=1, padding=0, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)  # [B, N, T, embed_size]
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        out = attn * u
        out = out.permute(0, 2, 1, 3)
        return out

# input[N, T, C]
class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        # Temporal embedding One hot time_num=60
        self.time_num = time_num
        self.one_hot = One_hot_encoder(embed_size, time_num).cuda()  # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size).cuda()  # temporal embedding选用nn.Embedding

        # self.attention = TSelfattention(embed_size, heads)
        self.attention = TSelfattention()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # q, k, v：[N, T, C]
        B, N, T, C = query.shape

        # D_T = self.one_hot(t, N, T)  # temporal embedding选用one-hot方式 或者
        D_T = self.temporal_embedding(torch.arange(0, T).cuda())  # temporal embedding选用nn.Embedding
        D_T = D_T.expand(N, T, C)

        # TTransformer
        x = D_T + query
        attention = self.attention(x, x, x, None)
        # attention = self.attention(x, x, x)
        M_t = self.dropout(self.norm1(attention + x))
        feedforward = self.feed_forward(M_t)
        U_t = self.dropout(self.norm2(M_t + feedforward))

        out = U_t + x + M_t

        return out


class TSelfattention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(TSelfattention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = utils.ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous()
class TSelfattention1(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfattention1, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = self.embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # q, k, v:[N, T, C]
        B, N, T, C = query.shape

        # q, k, v:[N,T,heads, per_dim]
        keys = key.reshape(B, N, T, self.heads, self.per_dim)
        queries = query.reshape(B, N, T, self.heads, self.per_dim)
        values = value.reshape(B, N, T, self.heads, self.per_dim)

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # compute temperal self-attention
        attnscore = torch.einsum("bnqhd, bnkhd->bnqkh", (queries, keys))  # [N, T, T, heads]
        attention = torch.softmax(attnscore / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("bnqkh, bnkhd->bnqhd", (attention, values))  # [N, T, heads, per_dim]
        out = out.reshape(B, N, T, self.embed_size)
        out = self.fc(out)

        return out
#
from math import ceil
class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4,
                 d_model=512, baseline=False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        #x_seq = self.enc_value_embedding(x_seq)
        #x_seq += self.enc_pos_embedding
        #x_seq = self.pre_norm(x_seq)

        #enc_out = self.encoder(x_seq)

        #dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        #predict_y = self.decoder(dec_in, enc_out)

        return base #+ predict_y[:, :self.out_len, :]

from einops import rearrange, repeat

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = 5#seg_len

        self.linear = nn.Linear(self.seg_len, d_model)

    def forward(self, x):
        # batch, ts_len, ts_dim = x.shape
        x = x.reshape(128, 25, -1)
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=5 )#self.seg_len)
        #czw
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=x.shape[0], d=x.shape[1])

        return x_embed

