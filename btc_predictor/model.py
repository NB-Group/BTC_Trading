# btc_predictor/model.py

import torch
import torch.nn as nn
import math
from typing import cast, Union

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float, task: str = 'regression'):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.task = task
        if self.task == 'regression':
            self.fc = nn.Linear(hidden_dim, 1)
        elif self.task == 'classification':
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1, :, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BTCPriceTransformer(nn.Module):
    output_layer: Union[nn.Linear, nn.Sequential]  # 明确类型，彻底消除linter报错
    def __init__(self, input_dim: int, d_model: int, nhead: int, nlayers: int, dim_feedforward: int, dropout: float = 0.5, task: str = 'regression'):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        
        # 关键修复：必须在init_weights之前定义task和output_layer
        self.task = task
        if self.task == 'regression':
            self.output_layer = nn.Linear(d_model, 1)
        elif self.task == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )

        self.init_weights()

    @staticmethod
    def _init_sequential_weights(seq: nn.Sequential):  # type: ignore
        if len(seq) > 0:
            first_layer = seq[0]
            if hasattr(first_layer, 'bias') and hasattr(first_layer, 'weight'):
                first_layer.bias.data.zero_()
                first_layer.weight.data.uniform_(-0.1, 0.1)

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if self.task == 'regression':
            if isinstance(self.output_layer, nn.Linear):
                self.output_layer.bias.data.zero_()
                self.output_layer.weight.data.uniform_(-initrange, initrange)
        elif self.task == 'classification':
            if isinstance(self.output_layer, nn.Sequential):
                self._init_sequential_weights(self.output_layer)


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]  # 取最后一个时间步
        # output_layer一定是nn.Module，可以直接调用
        assert isinstance(self.output_layer, nn.Module), 'output_layer类型错误，必须为nn.Module'
        return self.output_layer(output)

def create_model(
    input_dim: int, 
    model_type: str = 'transformer', 
    task: str = 'regression',
    # Transformer特定默认值
    d_model: int = 128,
    n_head: int = 8,
    n_layers: int = 4,
    dim_feedforward: int = 256,
    # LSTM特定默认值
    hidden_dim: int = 128,
    # 通用
    dropout: float = 0.1,
    **kwargs) -> nn.Module:
    
    if model_type == 'transformer':
        model = BTCPriceTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=n_head,
            nlayers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            task=task
        )
    elif model_type == 'lstm':
        model = SimpleLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            task=task
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    return model 