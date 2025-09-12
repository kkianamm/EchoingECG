import torch.nn as nn
from transformers import BertModel

from src.model.gpo import GPO, l2norm


class ProbEncoderTextBert(nn.Module):
    def __init__(self, embed_size, var_norm=False, mean_norm=True, gpo_dim=32, **kwargs):
        super().__init__()
        """ Language Model with BERT (from VSE infty)
        original code: https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py
        """
        self.embed_size = embed_size

        self.backbone = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
        backbone_embed_dim = self.backbone.config.hidden_size
        self.linear = nn.Linear(backbone_embed_dim, embed_size)
        self.gpool = GPO(gpo_dim, gpo_dim)
        self.var_norm = var_norm
        self.std_linear = nn.Linear(backbone_embed_dim, embed_size)
        self.std_gpool = GPO(gpo_dim, gpo_dim)
        self.mean_norm = mean_norm

    @property
    def get_embed_dim(self):
        return self.embed_size

    def forward(self, x, attention_mask, lengths):
        """Handles variable size captions"""
        # Embed word ids to vectors
        # bert_attention_mask = (x != 0).float()
        x = x.squeeze(dim=1) if x.ndim == 4 else x
        bert_emb = self.backbone(x, attention_mask=attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        pooled_features, _ = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if self.mean_norm:
            pooled_features = l2norm(pooled_features, dim=-1)

        std_cap_emb = self.std_linear(bert_emb)
        if self.var_norm:
            std_cap_emb = l2norm(std_cap_emb, dim=-1)
        std_pooled_features, _ = self.std_gpool(std_cap_emb, cap_len.to(cap_emb.device))

        return {"mean": pooled_features, "std": std_pooled_features}
