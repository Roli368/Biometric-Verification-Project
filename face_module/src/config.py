PROJECT_DIR = "/kaggle/working/FaceDetectionViT"

DATASET_ROOT = "/kaggle/input/webface-112x112/webface_112x112"
SPLIT_JSON   = f"{PROJECT_DIR}/data/annotations/split.json"
IDMAP_JSON   = f"{PROJECT_DIR}/data/annotations/id_map.json"

IMG_SIZE     = 224
EMBED_DIM    = 512
BACKBONE     = "vit_small_patch16_224"

BATCH_SIZE   = 128
LR           = 3e-4
WEIGHT_DECAY = 0.05
EPOCHS       = 10
NUM_WORKERS  = 4

ARC_S        = 30.0
ARC_M        = 0.5

USE_AMP      = True
GRAD_CLIP    = 1.0

CKPT_LATEST  = f"{PROJECT_DIR}/models/checkpoints/ckpt_latest.pth"
