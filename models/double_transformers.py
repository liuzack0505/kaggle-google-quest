import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AlbertModel

from models.head import Head
from models.siamese_transformers import avg_pool_forward, mha_pool_forward
from common import *


class DoubleTransformer(nn.Module):
    def __init__(self, AvgPooledModel, pretrained_model_name='./albert-base-v2', use_attention_pooling=False):
        super().__init__()
        # We instantiate TWO separate transformers (Question & Answer)
        self.q_transformer = AvgPooledModel.from_pretrained(
            pretrained_model_name, use_attention_pooling=use_attention_pooling)
        self.a_transformer = AvgPooledModel.from_pretrained(
            pretrained_model_name, use_attention_pooling=use_attention_pooling)
        self.head = Head(n_h=256, n_feats=5, n_bert=768, dropout=0.2)

    def forward(self, x_feats, q_ids, a_ids, seg_q_ids=None, seg_a_ids=None):
        x_q = self.q_transformer(q_ids, seg_q_ids)
        x_a = self.a_transformer(a_ids, seg_a_ids)
        return self.head(x_feats, x_q, x_a)


class PooledAlbert(AlbertModel):
    def __init__(self, config, use_attention_pooling=False):
        super().__init__(config)
        self.use_attention_pooling = use_attention_pooling

        if use_attention_pooling:
            # MHA Implementation
            self.mha_layer = nn.MultiheadAttention(
                config.hidden_size, num_heads=8, batch_first=True)
            self.pooling_query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size))

        self.init_weights()

    # Removed redundant from_pretrained; parent class handles kwargs automatically

    def forward(self, ids, seg_ids=None):
        if self.use_attention_pooling:
            return mha_pool_forward(self, PooledAlbert, ids, seg_ids)
        return avg_pool_forward(self, PooledAlbert, ids, seg_ids)


class DoubleAlbert(DoubleTransformer):
    def __init__(self, use_attention_pooling=False):
        super().__init__(PooledAlbert, './albert-base-v2', use_attention_pooling)
