import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTFaceEmbedder(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", embed_dim=512):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0
        )

        out_dim = self.backbone.num_features

        self.embedding = nn.Sequential(
            nn.Linear(out_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        feats = self.backbone(x)
        emb = self.embedding(feats)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
