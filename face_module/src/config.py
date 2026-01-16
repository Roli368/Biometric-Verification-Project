PROJECT_DIR = "/kaggle/working/FaceDetectionViT"

SPLIT_JSON   = f"{PROJECT_DIR}/data/annotations/split.json"
IDMAP_JSON   = f"{PROJECT_DIR}/data/annotations/id_map.json"

IMG_SIZE     = 224
EMBED_DIM    = 512

BATCH_SIZE   = 64
LR           = 1e-4
EPOCHS       = 1
NUM_WORKERS  = 2

ARC_S        = 30.0
ARC_M        = 0.5
