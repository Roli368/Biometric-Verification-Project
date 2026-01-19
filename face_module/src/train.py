import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import *
from src.dataset import WebFaceDataset
from src.model import ViTFaceEmbedder
from src.arcface import ArcFace

def save_ckpt(epoch, model, arcface, optimizer):
    os.makedirs(os.path.dirname(CKPT_LATEST), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "arcface": arcface.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CKPT_LATEST)

def load_ckpt(model, arcface, optimizer, device):
    ckpt = torch.load(CKPT_LATEST, map_location=device)
    model.load_state_dict(ckpt["model"])
    arcface.load_state_dict(ckpt["arcface"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"] + 1

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Device:", device)

    splits = json.load(open(SPLIT_JSON))
    idmap  = json.load(open(IDMAP_JSON))

    trainset = WebFaceDataset(
        root=splits["dataset_root"],
        id_list=splits["train_ids"],
        id2label=idmap["id2label"],
        img_size=IMG_SIZE,
        augment=True
    )

    loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    model = ViTFaceEmbedder(model_name=BACKBONE, embed_dim=EMBED_DIM).to(device)
    arcface = ArcFace(EMBED_DIM, out_features=len(idmap["id2label"]), s=ARC_S, m=ARC_M).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device == "cuda"))

    start_epoch = 1
    if os.path.exists(CKPT_LATEST):
        print("✅ Resuming from:", CKPT_LATEST)
        start_epoch = load_ckpt(model, arcface, optimizer, device)
        print("✅ Start epoch:", start_epoch)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        arcface.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        total = 0.0

        for step, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and device == "cuda")):
                emb = model(imgs)
                logits = arcface(emb, labels)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()

            if GRAD_CLIP and GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total/step:.4f}")

        save_ckpt(epoch, model, arcface, optimizer)
        print("💾 Saved checkpoint:", CKPT_LATEST)

if __name__ == "__main__":
    main()
