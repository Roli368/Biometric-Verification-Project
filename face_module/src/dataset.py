import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class WebFaceDataset(Dataset):
    def __init__(self, root, id_list, id2label, img_size=224, augment=True):
        self.root = root
        self.id_list = id_list
        self.id2label = id2label
        self.img_size = img_size
        self.augment = augment

        self.samples = []
        for pid in self.id_list:
            person_dir = os.path.join(self.root, pid)
            if not os.path.isdir(person_dir):
                continue
            for fn in os.listdir(person_dir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(person_dir, fn), pid))

        aug = []
        if augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
            ]

        self.tf = transforms.Compose(
            aug + [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = int(self.id2label[pid])
        return x, torch.tensor(y, dtype=torch.long)
