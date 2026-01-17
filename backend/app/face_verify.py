import sys
import os

# ✅ FIX: Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from io import BytesIO

from face_module.src.model import ViTFaceEmbedder


class FaceVerifier:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device

        self.model = ViTFaceEmbedder(embed_dim=512)
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Load only backbone + embedding
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def _preprocess(self, img_bytes):
        img = Image.open(img_bytes).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def verify(self, id_image_bytes, selfie_image_bytes):
        id_tensor = self._preprocess(id_image_bytes)
        selfie_tensor = self._preprocess(selfie_image_bytes)

        emb_id = self.model(id_tensor)
        emb_selfie = self.model(selfie_tensor)

        similarity = F.cosine_similarity(emb_id, emb_selfie).item()
        return similarity
