import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    """
    Stable ArcFace implementation (no acos)
    phi = cos(theta + m) = cosθ*cosm - sinθ*sinm
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.W = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=1)

        cosine = F.linear(embeddings, W).clamp(-1.0, 1.0)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-9))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s
        return logits
