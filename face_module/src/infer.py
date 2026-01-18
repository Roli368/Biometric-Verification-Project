import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import ViTFaceEmbedder
from src.config import IMG_SIZE, EMBED_DIM, CHECKPOINT_PATH, GALLERY_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")

    model = ViTFaceEmbedder(embed_dim=EMBED_DIM).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

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
