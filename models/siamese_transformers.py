import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, XLNetModel, RobertaModel

from models.head import Head
from common import *


def avg_pool_forward(model, ModelClass, ids, seg_ids=None):
    att_mask = (ids > 0).float()
    x_out = super(ModelClass, model).forward(
        ids, att_mask, token_type_ids=seg_ids)[0]
    att_mask = att_mask.unsqueeze(-1)
    return (x_out * att_mask).sum(dim=1) / att_mask.sum(dim=1)


def mha_pool_forward(model, ModelClass, ids, seg_ids=None):
    """
    Minimal MHA Pooling implementation.
    Uses the model's self.mha_layer and self.pooling_query.
    """
    att_mask = (ids > 0).float()
    x_out = super(ModelClass, model).forward(
        ids, attention_mask=att_mask, token_type_ids=seg_ids)[0]

    # 1. Prepare Query: Expand the learnable query parameter to match batch size
    # Query Shape: [batch, 1, hidden]
    batch_size = x_out.size(0)
    query = model.pooling_query.repeat(batch_size, 1, 1)

    # 2. Prepare Mask for MHA
    # PyTorch MHA expects True for positions to IGNORE (padding).
    # ids==0 gives us the padding positions.
    key_padding_mask = (ids == 0)

    # 3. Run Multihead Attention
    # attn_output shape: [batch, 1, hidden]
    attn_output, _ = model.mha_layer(
        query, x_out, x_out, key_padding_mask=key_padding_mask)

    # 4. Squeeze to get [batch, hidden]
    return attn_output.squeeze(1)


class SiameseTransformer(nn.Module):
    def __init__(self, AvgPooledModel, pretrained_model_name='./bert-base-uncased', use_attention_pooling=False):
        super().__init__()
        self.transformer = AvgPooledModel.from_pretrained(
            pretrained_model_name, use_attention_pooling=use_attention_pooling)
        self.head = Head(n_h=256, n_feats=5, n_bert=768, dropout=0.2)

    def forward(self, x_feats, q_ids, a_ids, seg_q_ids=None, seg_a_ids=None):
        x_q = self.transformer(q_ids, seg_q_ids)
        x_a = self.transformer(a_ids, seg_a_ids)
        return self.head(x_feats, x_q, x_a)


class PooledBert(BertModel):
    def __init__(self, config, use_attention_pooling=False):
        super().__init__(config)
        self.use_attention_pooling = use_attention_pooling

        if use_attention_pooling:
            # Define MHA components directly here to avoid extra classes
            self.mha_layer = nn.MultiheadAttention(
                config.hidden_size, num_heads=8, batch_first=True)
            self.pooling_query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size))

        self.init_weights()

    def forward(self, ids, seg_ids=None):
        if self.use_attention_pooling:
            return mha_pool_forward(self, PooledBert, ids, seg_ids)
        else:
            return avg_pool_forward(self, PooledBert, ids, seg_ids)


class SiameseBert(SiameseTransformer):
    def __init__(self, use_attention_pooling=False):
        super().__init__(PooledBert, './bert-base-uncased', use_attention_pooling)


class PooledXLNet(XLNetModel):
    def __init__(self, config, use_attention_pooling=False):
        super().__init__(config)
        self.use_attention_pooling = use_attention_pooling

        if use_attention_pooling:
            # XLNet uses 'd_model'
            self.mha_layer = nn.MultiheadAttention(
                config.d_model, num_heads=8, batch_first=True)
            self.pooling_query = nn.Parameter(
                torch.randn(1, 1, config.d_model))

        self.init_weights()

    def forward(self, ids, seg_ids=None):
        if self.use_attention_pooling:
            return mha_pool_forward(self, PooledXLNet, ids, seg_ids)
        else:
            return avg_pool_forward(self, PooledXLNet, ids, seg_ids)


class SiameseXLNet(SiameseTransformer):
    def __init__(self, use_attention_pooling=False):
        super().__init__(PooledXLNet, 'xlnet-base-cased', use_attention_pooling)


class PooledRoberta(RobertaModel):
    def __init__(self, config, use_attention_pooling=False):
        super().__init__(config)
        self.use_attention_pooling = use_attention_pooling

        if use_attention_pooling:
            self.mha_layer = nn.MultiheadAttention(
                config.hidden_size, num_heads=8, batch_first=True)
            self.pooling_query = nn.Parameter(
                torch.randn(1, 1, config.hidden_size))

        self.init_weights()

    def forward(self, ids, seg_ids=None):
        if self.use_attention_pooling:
            return mha_pool_forward(self, PooledRoberta, ids, seg_ids)
        else:
            return avg_pool_forward(self, PooledRoberta, ids, seg_ids)

    def resize_type_embeddings(self, new_num_types):
        old_embeddings = self.embeddings.token_type_embeddings
        model_embeds = self._get_resized_embeddings(
            old_embeddings, new_num_types)
        self.embeddings.token_type_embeddings = model_embeds
        self.config.type_vocab_size = new_num_types
        self.type_vocab_size = new_num_types


class SiameseRoberta(SiameseTransformer):
    def __init__(self, use_attention_pooling=False):
        super().__init__(PooledRoberta, 'roberta-base', use_attention_pooling)
        self.transformer.resize_type_embeddings(2)
