import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms

from face_module.src.model import ViTFaceEmbedder


class FaceVerifier:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device

        # -------------------------------------------------
        # Load face recognition model
        # -------------------------------------------------
        self.model = ViTFaceEmbedder(embed_dim=512)
        ckpt = torch.load(checkpoint_path, map_location=device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=True)
        else:
            self.model.load_state_dict(ckpt, strict=True)

        self.model.to(device)
        self.model.eval()

        # -------------------------------------------------
        # MediaPipe face detector (KEY FIX)
        # -------------------------------------------------
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6
        )

        # -------------------------------------------------
        # Image preprocessing for face embedding
        # -------------------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    # -------------------------------------------------
    # Face detection (for green rectangle)
    # -------------------------------------------------
    def detect_face_bbox(self, img_bytes):
        """
        Detect face and return bounding box in original image pixels.
        """
        img = Image.open(img_bytes).convert("RGB")
        img_np = np.array(img)
        h, w, _ = img_np.shape

        results = self.face_detector.process(img_np)

        if not results.detections:
            return None

        box = results.detections[0].location_data.relative_bounding_box

        return {
            "x": int(box.xmin * w),
            "y": int(box.ymin * h),
            "w": int(box.width * w),
            "h": int(box.height * h),
            "img_w": w,
            "img_h": h,
        }

    # -------------------------------------------------
    # Internal preprocessing
    # -------------------------------------------------
    def _preprocess(self, img_bytes):
        img = Image.open(img_bytes).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    # -------------------------------------------------
    # Face verification (cosine similarity)
    # -------------------------------------------------
    @torch.no_grad()
    def verify(self, id_image_bytes, selfie_image_bytes):
        """
        Returns cosine similarity between ID image and selfie.
        """
        id_tensor = self._preprocess(id_image_bytes)
        selfie_tensor = self._preprocess(selfie_image_bytes)

        emb_id = self.model(id_tensor)
        emb_selfie = self.model(selfie_tensor)

        similarity = F.cosine_similarity(emb_id, emb_selfie).item()
        return similarity
