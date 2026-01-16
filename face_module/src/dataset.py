import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WebFaceDataset(Dataset):
    def __init__(self, root, id_list, id2label, img_size=224, augment=False):
        self.root = root
        self.id_list = id_list
        self.id2label = id2label

        self.samples = []
        for pid in self.id_list:
            pdir = os.path.join(self.root, pid)
            if not os.path.isdir(pdir):
                continue
            imgs = [x for x in os.listdir(pdir) if x.lower().endswith((".jpg", ".jpeg", ".png"))]
            for img in imgs:
                self.samples.append((os.path.join(pdir, img), self.id2label[pid]))

        base_tf = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                *base_tf
            ])
        else:
            self.transform = transforms.Compose(base_tf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
