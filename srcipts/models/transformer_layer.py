# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention



class EncoderLayer(nn.Module):
    #两部分组成
    #第一部分是一个multi-head self-attention mechanism
    #第二部分是一个position-wise feed-forward network全连接层
    def __init__(self, embed_dim, ffn_dim, num_heads, attn_dropout=0., act_dropout=0., dropout=0., layernorm_before=False):

        super().__init__()

        self.embed_dim = embed_dim
        self.layernorm_before = layernorm_before
        self.act_dropout = act_dropout
        self.dropout = dropout
        #nn.MultiheadAttention的用法
        self.self_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout)
        #将embedding分割成num_heads份，每一份分别拿来做一下attention
        self.attn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)
        #embed-dim：词嵌入维度 ,eps：为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.reset_parameters()

    def reset_parameters(self):
        #使用Xavier 初始化，避免反向传播的梯度消失等问题
        nn.init.xavier_uniform_(self.ffn1.weight)
        nn.init.xavier_uniform_(self.ffn2.weight)
        if self.ffn1.bias is not None:
            nn.init.constant_(self.ffn1.bias, 0.0)
            nn.init.constant_(self.ffn2.bias, 0.0)

    def forward(self, x, mask):
        # self-attention mechanism
        identity = x
        if self.layernorm_before:
            x = self.attn_layernorm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.attn_layernorm(x)

        # position-wise feed-forward network全连接层
        identity = x
        if self.layernorm_before:
            x = self.ffn_layernorm(x)
        x = F.relu(self.ffn1(x))
        x = F.dropout(x, p=self.act_dropout, training=self.training)
        x = self.ffn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.ffn_layernorm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, attn_dropout=0., act_dropout=0., dropout=0., layernorm_before=False):
        #三部分组成
        #第一个部分是multi-head self-attention mechanism
        #第二部分是multi-head context-attention mechanism
        #第三部分是一个position-wise feed-forward network
        super().__init__()
        self.embed_dim = embed_dim
        self.layernorm_before = layernorm_before
        self.act_dropout = act_dropout
        self.dropout = dropout

        self.self_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout)
        self.attn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)


        self.enc_dec_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout)
        self.enc_dec_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)


        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ffn1.weight)
        nn.init.xavier_uniform_(self.ffn2.weight)

        if self.ffn1.bias is not None:
            nn.init.constant_(self.ffn1.bias, 0.0)
            nn.init.constant_(self.ffn2.bias, 0.0)

    def forward(self,x,encoder_out,src_key_padding_mask,tgt_key_padding_mask,tgt_mask,prev_x=None):

        # multi-head self-attention mechanism
        identity = x
        if prev_x is None:
            kv = x
        else:
            kv = torch.cat([prev_x, x], dim=0)
        if self.layernorm_before:
            x = self.attn_layernorm(x)
            kv = self.attn_layernorm(kv)

        x = self.self_attn(query=x, key=kv, value=kv,key_padding_mask=tgt_key_padding_mask,attn_mask=tgt_mask)[0]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.attn_layernorm(x)

        # multi-head context-attention mechanism
        identity = x
        if self.layernorm_before:
            x = self.enc_dec_layernorm(x)
        x, _ = self.enc_dec_attention(query=x,key=encoder_out,value=encoder_out,key_padding_mask=src_key_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.enc_dec_layernorm(x)

        # position-wise feed-forward network
        identity = x
        if self.layernorm_before:
            x = self.ffn_layernorm(x)
        x = F.relu(self.ffn1(x))
        x = F.dropout(x, p=self.act_dropout, training=self.training)
        x = self.ffn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.ffn_layernorm(x)
        return x