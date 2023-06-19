import torch
import torch.nn as nn
import torch.optim as optim
import math


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, num_heads, dropout):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        output = self.output_projection(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attended, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attended)
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

# 하이퍼파라미터 설정
input_size = 30         # 입력 시계열 데이터의 차원
output_size = 5         # 출력 시계열 데이터의 차원
num_layers = 6          # 트랜스포머 레이어 수
hidden_size = 10        # 트랜스포머 내부 임베딩 차원
num_heads = 8           # TODO: 멀티 ?
