import os, sys, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_DIR = "/kaggle/working/FaceDetectionViT"
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from src.config import *
from src.dataset import WebFaceDataset
from src.model import ViTFaceEmbedder
from src.arcface import ArcFace

os.makedirs(f"{PROJECT_DIR}/models/checkpoints", exist_ok=True)

class Trainer:
    def __init__(self, model, arcface, optimizer, device):
        self.model = model
        self.arcface = arcface
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def train_one_epoch(self, loader, epoch, epochs, max_steps=50):
        self.model.train()
        self.arcface.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        total_loss = 0.0

        for step, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            emb = self.model(imgs)
            logits = self.arcface(emb, labels)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / step
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

            if step >= max_steps:
                break

        return total_loss / step

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = json.load(open(SPLIT_JSON))
    idmap  = json.load(open(IDMAP_JSON))

    root = splits["dataset_root"]
    train_ids = splits["train_ids"][:300]

    trainset = WebFaceDataset(
        root=root,
        id_list=train_ids,
        id2label=idmap["id2label"],
        img_size=IMG_SIZE,
        augment=True
    )

    loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = ViTFaceEmbedder(embed_dim=EMBED_DIM).to(device)
    arc   = ArcFace(in_features=EMBED_DIM, out_features=len(idmap["id2label"]), s=ARC_S, m=ARC_M).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(arc.parameters()), lr=LR)

    trainer = Trainer(model, arc, optimizer, device)

    loss = trainer.train_one_epoch(loader, 1, 1, max_steps=50)
    print(f" Mini Train Done | Avg Loss: {loss:.4f}")

    save_path = f"{PROJECT_DIR}/models/checkpoints/ckpt_latest.pth"
    torch.save({
        "model": model.state_dict(),
        "arcface": arc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 1
    }, save_path)

    print(" Saved:", save_path)

if __name__ == "__main__":
    main()