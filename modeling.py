import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, activation=lambda x: torch.softmax(x, dim=-1)):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_o = nn.Linear(embed_dim, embed_dim)
        self.activation = activation
        
        
    def forward(self, query, key, value):
        B, _, _ = query.shape
        
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        
        Q = Q.view(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        K = K.permute(0,1,3,2)
        QK = torch.matmul(Q, K) / torch.sqrt(self.head_dim)
        
        A = torch.matmul(self.activation(QK), V)
        A = A.permute(0,2,1,3)  
        A = A.contiguous().view(B, -1, self.embed_dim)
        A = self.linear_o(A)

        return A
    

class LinearHeadAttention(MultiheadAttention):
    def forward(self, query, key, value):
        B, _, _ = query.shape

        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        
        Q = Q.view(B, -1, self.num_heads, self.head_dim)
        K = K.view(B, -1, self.num_heads, self.head_dim).permute(0,2,3,1)
        V = V.view(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        Q = self.activation(Q)
        K = self.activation(K)

        KV = torch.matmul(K, V)
        K = K.permute(0, 3, 1, 2)
        Z = 1/(torch.einsum("BLHD,BHD->BLH", Q, K.sum(dim=1))+1e-6)
        A = torch.einsum("BLHD,BHCD,BLH->BLHC", Q, KV, Z).contiguous()

        A = A.view(B, -1, self.embed_dim)
        A = self.linear_o(A)
        return A
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, attn, d_model, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = attn
        self.linear_in = nn.Linear(d_model, dim_feedforward)
        self.linear_out = nn.Linear(dim_feedforward, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.norm_in(x)
        x = x + self.dropout(self.self_attn(x, x, x))
        out = self.norm_out(x)
        out = self.dropout(self.activation(self.linear_in(out)))
        out = self.dropout(self.linear_out(out))
        return x + out
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
