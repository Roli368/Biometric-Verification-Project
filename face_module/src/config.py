import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # face_module/

IMG_SIZE = 224
EMBED_DIM = 512
BACKBONE = "vit_small_patch16_224"

CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "ckpt_latest.pth")
GALLERY_PATH    = os.path.join(BASE_DIR, "models", "exported", "gallery.pth")
