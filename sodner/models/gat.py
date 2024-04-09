import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from opt_einsum import contract
from transformers import BertModel
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from sodner.models.mlp import MLP


class AGGCN(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_emb_dim: int,
                 feature_dim: int,
                 tree_prop: int = 1,
                 tree_dropout: float=0.0,
                 aggcn_heads: int=4,
                 aggcn_sublayer_first: int=2,
                 aggcn_sublayer_second: int=4,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AGGCN, self).__init__(vocab, regularizer)

        self.in_dim = span_emb_dim
        self.mem_dim = span_emb_dim

        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.num_layers = tree_prop

        self.layers = nn.ModuleList()

        self.heads = aggcn_heads
        self.sublayer_first = aggcn_sublayer_first
        self.sublayer_second = aggcn_sublayer_second

        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(GraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_first))
                self.layers.append(GraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_second))
            else:
                self.layers.append(MultiGraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(tree_dropout, self.mem_dim, self.sublayer_second, self.heads))

        self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim)

        # mlp output layer
        in_dim = span_emb_dim
        mlp_layers = [nn.Linear(in_dim, feature_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*mlp_layers)
        # initializer(self)


    # adj: (batch, sequence, sequence)
    # text_embeddings: (batch, sequence, emb_dim)
    # text_mask: (batch, sequence)
    def forward(self, adj, text_embeddings, text_mask):

        gcn_inputs = self.input_W_G(text_embeddings)
        text_mask = text_mask.unsqueeze(-2)
        #adj.shape,text_embeddings.shape,text_mask.shape torch.Size([1, 22, 22]) torch.Size([1, 22, 400]) torch.Size([1, 1, 22])

        layer_list = []
        outputs = gcn_inputs
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        for i in range(len(self.layers)):
            if i < 2:
                outputs = self.layers[i](adj, outputs)
                layer_list.append(outputs)
            else:
                attn_tensor = self.attn(outputs, outputs, text_embeddings, text_mask)
                #attn_tensor.shape [1,4,22,22]
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                outputs = self.layers[i](attn_adj_list, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)

        outputs = self.out_mlp(dcgcn_output)

        #text_embedding.shape [2,22,400]
        #print(text_embeddings.shape, text_embeddings.shape[2])

        #score.shape [1,22,22,2]
        return outputs


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, tree_dropout, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(tree_dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))


    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, tree_dropout, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(tree_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))


    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def attention(query, key, text_embedding, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    model = TriAttention(text_embedding.shape[2], 2)
    model.cuda()
    score = model.forward_self(text_embedding)
    # 创建一个全零张量，形状为 (4, 44, 44, 2)
    padding = torch.zeros(score.shape[0], score.shape[1], score.shape[2], score.shape[3])
    # 使用 torch.cat 连接这两个张量，指定 dim=3 连接在第 3 个维度上
    score = score.cuda()  # 将 score 移动到 GPU
    padding = padding.cuda()  # 将 padding 移动到 GPU
    score = torch.cat((score, padding), dim=3)
    score = score.view(p_attn.shape[0], p_attn.shape[1], p_attn.shape[2], p_attn.shape[3])
    #print(p_attn.shape, score.shape)
    combined_tensor = p_attn + score
    #print(p_attn.shape, score.shape, combined_tensor.shape)
    min_value = combined_tensor.min()
    max_value = combined_tensor.max()

    # 将 combined_tensor 缩放到 [0, 1] 范围内
    normalized_combined_tensor = (combined_tensor - min_value) / (max_value - min_value)
    p_attn1 = F.softmax(normalized_combined_tensor, dim=-1)
    if dropout is not None:
        #执行
        p_attn1 = dropout(p_attn1)
    return p_attn1


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, text_embeddings, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        #query,key [batch,sentence,embedding]
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key,text_embeddings, mask=mask, dropout=self.dropout)
        # attn.shape [1,4,8,8]


        return attn



class TriAffine(nn.Module):
    def __init__(self, n_in, num_class, bias_x=True, bias_y=True, scale="none", init_std=0.01):
        super().__init__()

        self.n_in = n_in
        self.num_class = num_class
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_in + bias_x,
                                                n_in,
                                                n_in + bias_y,
                                                num_class))
        self.init_std = init_std
        self.scale = scale
        self.calculate_scale_factor()
        self.reset_parameters()

    def calculate_scale_factor(self):
        if self.scale == "none":
            self.scale_factor = 1
        elif self.scale == "sqrt":
            self.scale_factor = self.n_in ** (-0.5)
        elif self.scale.find("tri") >= 0:
            self.scale_factor = self.n_in ** (-1.5) * self.init_std ** (-1)

    def extra_repr(self):
        s = f"n_in={self.n_in}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        s += f", num_class={self.num_class}"
        return s

    def reset_parameters(self):
        # nn.init.zeros_(self.weight)
        nn.init.normal_(self.weight, std=self.init_std)

    def forward(self, x, y, z):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, n_in]
            y (torch.Tensor): [batch_size, seq_len, n_in]
            z (torch.Tensor): [batch_size, seq_len, n_in]
        Returns:
            s (torch.Tensor): [batch_size, seq_len, seq_len, seq_len]
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # w = contract('bzk,ikjr->bzijr', z, self.weight) # bsz * seq * h * h * class
        # s = contract('bxi,bzijr,byj->bzxyr', x, w, y) # bsz * seq * seq * seq * class
        # s = contract('bxi,bzk,ikjr,byj->bzxyr', x, z, self.weight, y)
        # s = contract('bxi,bzk,ikjr,byj->bxzyr', x, z, self.weight, y)
        s = contract('bxi,bzk,ikjr,byj->bxyzr', x, z, self.weight, y)

        if self.num_class == 1:
            return s.squeeze(-1)

        if hasattr(self, 'scale_factor'):
            s = s * self.scale_factor
        return s

    def forward_query(self, x, y, z):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        s = contract('bxi,bzk,ikjr,bxj->bzxr', x, z, self.weight, y)

        if self.num_class == 1:
            return s.squeeze(-1)

        if hasattr(self, 'scale_factor'):
            s = s * self.scale_factor
        return s

class TriAttention(nn.Module):
    def __init__(self, hidden_dim, num_class, attention_dim=None, mask=True,
                 reduce_last=False, dropout=0.0):
        super(TriAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        if attention_dim is None or attention_dim == 0:
            if self.hidden_dim == 768:
                self._hidden_dim = 384
            else:
                self._hidden_dim = self.hidden_dim // 2
        else:
            self._hidden_dim = attention_dim
        self.parser = TriAffine(self._hidden_dim, self.num_class)
        self.mask = mask
        self.reduce_last = reduce_last
        self.dropout = dropout
        if not self.reduce_last:
            self.linear_h = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout)
        else:
            self.linear_h = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_t = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
            self.linear_m = MLP(self.hidden_dim, self.hidden_dim, self._hidden_dim, 2, self.dropout)
        self.V = nn.Linear(self.hidden_dim, self.num_class)

    def forward_self(self, memory):
        _, seq, _ = memory.size()
        head = self.linear_h(memory)
        tail = self.linear_t(memory)
        mid = self.linear_m(memory)
        #head.shape, tail.shape, mid.shape [2,22,200]

        score = self.parser(head, tail, mid)  # b * seq * seq * seq * type
        #score.shape [1,22,22,22,2]
        seq_t = torch.arange(seq).to(memory.device)
        if self.mask:
            seq_x = seq_t.unsqueeze(-1).unsqueeze(-1).repeat(1, seq, seq)
            seq_y = seq_t.unsqueeze(0).unsqueeze(-1).repeat(seq, 1, seq)
            seq_z = seq_t.unsqueeze(0).unsqueeze(0).repeat(seq, seq, 1)
            score.masked_fill_(torch.bitwise_or(seq_z > seq_y, seq_z < seq_x).unsqueeze(0).unsqueeze(-1), -1e6)
        alpha = F.softmax(score, dim=-2)

        type_span_h = torch.relu(contract('bkh,bijkr->bijrh', memory, alpha))
        score = contract('bijrh,rh->bijr', type_span_h, self.V.weight) + self.V.bias
        return score
