import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=1)

        cosine = F.linear(embeddings, W).clamp(-1.0, 1.0)
        theta = torch.acos(cosine)
        target_cosine = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = cosine * (1 - one_hot) + target_cosine * one_hot
        return logits * self.s
