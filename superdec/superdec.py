import torch
import torch.nn as nn

from superdec.models.decoder import TransformerDecoder
from superdec.models.decoder_layer import DecoderLayer
from superdec.models.point_encoder import StackedPVConv
from superdec.models.heads import SuperDecHead
from superdec.lm_optimization.lm_optimizer import LMOptimizer

class SuperDec(nn.Module):
    def __init__(self, ctx):
        super(SuperDec, self).__init__()
        self.n_layers = ctx.decoder.n_layers
        self.n_heads = ctx.decoder.n_heads
        self.n_queries = ctx.decoder.n_queries
        self.deep_supervision = ctx.decoder.deep_supervision
        self.pos_encoding_type = ctx.decoder.pos_encoding_type
        self.dim_feedforward = ctx.decoder.dim_feedforward
        self.emb_dims = ctx.point_encoder.l3.out_channels # output dimension of pvcnn
        self.lm_optimization = False
        if self.lm_optimization:
            self.lm_optimizer = LMOptimizer()

        self.point_encoder = StackedPVConv(ctx.point_encoder)

        decoder_layers = [DecoderLayer(d_model=self.emb_dims,
                                        nhead=self.n_heads,
                                        dim_feedforward=self.dim_feedforward,
                                        batch_first=True,
                                        swapped_attention=ctx.decoder.swapped_attention,)
                                        for _ in range(self.n_layers)]

        self.layers = TransformerDecoder(decoder_layers=decoder_layers, n_layers=self.n_layers,
                                         max_len=self.n_queries, pos_encoding_type=self.pos_encoding_type,
                                         masked_attention=ctx.decoder.masked_attention)

        self.layers.project_queries = nn.Sequential(
            nn.Linear(self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims),
        )
        self.heads = SuperDecHead(emb_dims=self.emb_dims)
        init_queries = torch.zeros(self.n_queries + 1, self.emb_dims)
        self.register_buffer('init_queries', init_queries) # TODO double check -> new codebase

    def forward(self, x):
        point_features = self.point_encoder(x)

        refined_queries_list, assign_matrices = self.layers(self.init_queries, point_features)
        outdict_list = []

        # TODO remove this in the final version. there is no need to compute the output for all of them
        thred = 24
        for i, q in enumerate(refined_queries_list):
            outdict_list += [self.heads(q[:,:-1,...])]
            assign_matrix = assign_matrices[i]
            assign_matrix = torch.softmax(assign_matrix, dim=2)
            outdict_list[i]['assign_matrix'] = assign_matrix
            # outdict_list[i]['exist'] = (assign_matrix.sum(1) > thred).to(torch.float32).detach()[...,None]

        if self.lm_optimization:
            outdict_list[-1] = self.lm_optimizer(outdict_list[-1], x)

        return outdict_list[-1]
