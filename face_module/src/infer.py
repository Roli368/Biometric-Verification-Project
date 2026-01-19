import os
import torch
from PIL import Image
from torchvision import transforms

from face_module.src.model import ViTFaceEmbedder
from face_module.src.config import IMG_SIZE, EMBED_DIM, CHECKPOINT_PATH, GALLERY_PATH

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Image transform
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------- Model loader (cached) --------
_MODEL = None

def load_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")

    model = ViTFaceEmbedder(embed_dim=EMBED_DIM).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    _MODEL = model
    return model

# -------- Inference helpers --------
def embed_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x)

    return emb

def predict(img_path):
    model = load_model()
    emb = embed_image(model, img_path)

    return {
        "status": "ok",
        "embedding_shape": list(emb.shape),
        "embedding_norm": float(emb.norm(dim=1).cpu())
    }
