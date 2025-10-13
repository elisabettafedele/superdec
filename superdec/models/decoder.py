import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer
import math
from typing import List

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device='cuda'):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len+1, d_model).to(device)
        position = torch.arange(0, max_len+1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device='cuda'):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(max_len+1, d_model)).to(device)
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)  # Initialize learnable positional encodings

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.positional_encoding[:seq_len, :]

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layers: List[TransformerDecoderLayer], n_layers, max_len, masked_attention, pos_encoding_type='sinusoidal'):
        super(TransformerDecoder, self).__init__()

        # Updated Decoder Layers
        self.layers = nn.ModuleList(decoder_layers)
        self.n_layers = n_layers
        self.d_model = decoder_layers[0].linear1.in_features
        self.max_len = max_len
        self.masked_attention = masked_attention

        # Choose between sinusoidal and learnable positional encodings
        if pos_encoding_type == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(self.d_model, max_len)
        elif pos_encoding_type == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(self.d_model, max_len)
        else:
            raise ValueError("encoding_type must be 'sinusoidal' or 'learnable'")

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = self.positional_encoding(tgt.repeat(memory.shape[0], 1, 1))

        intermediate_outputs = []  # To store outputs of all intermediate layers
        assign_matrices = []
        mask = None

        for layer in self.layers:
            output = layer(output, memory, memory_mask=mask)
            projected_queries_layer = self.project_queries(output)
            assign_matrix = memory @ projected_queries_layer.transpose(-1,-2)[...,:-1]
            if (self.masked_attention):
                softmaxed_assign_matrix = torch.nn.functional.softmax(assign_matrix, dim=-1)
                mask = softmaxed_assign_matrix > 0.5
                mask = mask.transpose(-1,-2)
                mask = torch.concatenate((mask, torch.ones((mask.shape[0], 1, mask.shape[2]), dtype=torch.bool, device=mask.device)), dim=-2)
            intermediate_outputs.append(output)  # Append intermediate output
            assign_matrices.append(assign_matrix)  # Append projected queries

        return intermediate_outputs, assign_matrices  # Return the list of all intermediate outputs
