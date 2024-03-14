import torch.nn as nn
import torch
import torch.optim as optim
import copy
import math
import torch.nn.functional as F


def attention(query, key, mask=None, dropout=None):

    #mask:(2,1,1,22)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)#torch.Size([4, 4, 6, 6])
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class MultiHeadAttention(nn.Module):

    def __init__(self, span_emb_dim: int, heads, d_model, dropout=0.4, RoPE=True):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.h = heads
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.in_dim = span_emb_dim
        self.mem_dim = span_emb_dim
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)
        self.RoPE = RoPE
        self.conv1d = nn.Conv1d(in_channels=heads, out_channels=d_model, kernel_size=3, padding=1)


    def forward(self, text_embeddings, text_mask):
        text_embeddings = text_embeddings
        gcn_inputs = self.input_W_G(text_embeddings)
        text_mask = text_mask.unsqueeze(-2)
        outputs = gcn_inputs
        attn_tensor = self.mutiattention(outputs, outputs, text_mask)
        #attn_tensor.shape 2,4,22,22
        # 将多头自注意力的输出传递给卷积层
        #conv_output = self.conv1d(attn_tensor)  # 调整维度以匹配卷积层的输入
        # 选择第一个通道（channel）进行卷积
        conv_output = self.conv1d(attn_tensor[:, :, 0, :])

        # 在这里可以执行其他操作，如激活函数、池化等
        conv_output = F.relu(conv_output)
        #print("conv_output.shape")
        #conv_output.shape[2,400,22]


        return conv_output



    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1) #生成绝对位置信息

        indices = torch.arange(0, output_dim // 2, dtype=torch.float) #由Sinusoidal公式可知 i的范围是0-d/2
        indices = torch.pow(10000, -2 * indices / output_dim) #公式计算得到theta_i
        embeddings = position_ids * indices #生成带theta的embedding
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) #引入cosm sinm在最后维度进行堆叠
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape)))) #扩展到整个batch_size
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim)) #修改为输出维度
        #print("embeddings")
        #print(embeddings)
        embeddings = embeddings.cuda()
        return embeddings

    def mutiattention(self, query, key, mask=None, dropout=None):
        #query.shape : （2，22，400）batch, max_sentence_length,emb_dim
        #key.shape ： （2，22，400）
        #mask.shape ： （2，1，22）

        #mask:(2,1,22)
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        #nbatches = 2  #query.shape,key.shape : (2,4,22,100)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(query.size(0), query.size(2), query.size(3))
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)    #将奇数列信息抽取出来也就是cosm拿出来并复制
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)  #将偶数列信息抽出来并复制
            cos_pos = cos_pos.permute(0, 2, 1, 3)
            sin_pos = sin_pos.permute(0, 2, 1, 3)
            #...操作表示自动判断其中得到维度区间，None增加一维，::2两个冒号直接写表示从所有的数据中隔行取数据，从0开始，1::2两个冒号直接写表示从所有数据中隔行取数据。从1开始

            qw2 = torch.stack([query[..., 1::2], query[..., ::2]], -1)  #奇数列加上负号，得到第二个q的矩阵
            qw2 = qw2.reshape(query.shape)
            qw = query * cos_pos + qw2 * sin_pos  #最后融入位置信息
            kw2 = torch.stack([-key[..., 1::2], key[..., ::2]], -1)
            kw2 = kw2.reshape(key.shape)
            kw = key * cos_pos + kw2 * sin_pos

        attn = attention(qw, kw, mask=mask, dropout=self.dropout)#attn.shape:[4,4,6,6]

        return attn





















