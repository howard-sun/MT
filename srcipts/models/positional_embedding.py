# *_*coding:utf-8 *_*
import math
import torch
import torch.nn as nn

#使用pytorch实现Transformer需要自己实现position Embedding
class PositionalEmbedding(nn.Module):
    #位置编码器的初始化函数
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        #embed_dim: 词嵌入的维度，论文默认是512；
        #max_len: 文本序列的最大长度；

        assert embed_dim % 2 == 0
        # 根据论文的公式，构造出PE矩阵
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        # scale = 1/(10000^(i/(hd-1)))
        scale = torch.exp(-torch.arange(half_dim).float() / (half_dim - 1) *math.log(10000.))
        # pe = pos/(10000^(i/(hd-1)))
        pe = torch.arange(max_len).float()[:, None] * scale[None, :]
        # 偶数列使用sin，奇数列使用cos
        # pe = sin(pos/(10000^(i/(hd-1)))) ， cos(pos/(10000^(i/(hd-1))))
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1)
        self.register_buffer('pe', pe)

    # add&norm
    def forward(self, tokens, shift=0):

        out = self.pe[shift:shift + tokens.shape[1], :]
        out = out.unsqueeze(0)
        return out.detach()