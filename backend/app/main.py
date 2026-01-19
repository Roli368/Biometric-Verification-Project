import sys
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import numpy as np

# -------------------------------------------------
# Make project root visible (so face_module imports work)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.liveness import UltimateLiveness10
from app.face_verify import FaceVerifier

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(title="Biometric Verification API")

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "face_module",
    "models",
    "checkpoints",
    "ckpt_latest.pth"
)

# -------------------------------------------------
# Lazy-loaded face verifier
# -------------------------------------------------
_face_verifier = None

def get_face_verifier():
    global _face_verifier

    if not os.path.exists(MODEL_PATH):
        return None

    if _face_verifier is None:
        _face_verifier = FaceVerifier(
            checkpoint_path=MODEL_PATH,
            device="cpu"
        )

    return _face_verifier


# -------------------------------------------------
# Schemas
# -------------------------------------------------
class LivenessRequest(BaseModel):
    rgb_data: List[List[List[int]]]


# -------------------------------------------------
# Liveness Endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "Biometric Verification API is running",
        "endpoints": ["/verify-face", "/verify-liveness", "/docs"]
    }





@app.post("/verify-liveness")
def verify_liveness(req: LivenessRequest):
    try:
        if not req.rgb_data or len(req.rgb_data) < 10:
            return {
                "is_alive": False,
                "verdict": "INSUFFICIENT_DATA",
                "trust_score": 0.0
            }

        engine = UltimateLiveness10()
        frames = [np.array(f, dtype=np.uint8) for f in req.rgb_data]

        verdict, trust = None, 0.0
        for frame in frames:
            verdict, trust = engine.verify(frame)

        return {
            "is_alive": verdict == "LIVE HUMAN",
            "verdict": verdict,
            "trust_score": round(float(trust), 4)
        }

    except Exception as e:
        return {
            "is_alive": False,
            "verdict": "ERROR",
            "error": str(e)
        }


# -------------------------------------------------
# Face Verification Endpoint (UPDATED)
# -------------------------------------------------
@app.post("/verify-face")
def verify_face(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):
    verifier = get_face_verifier()

    # -------------------------------------------------
    # Face detection (for green rectangle)
    # -------------------------------------------------
    bbox = None
    if verifier is not None:
        bbox = verifier.detect_face_bbox(id_image.file)

    # Reset file pointer (IMPORTANT)
    id_image.file.seek(0)
    selfie_image.file.seek(0)

    if bbox is None:
        return {
            "match": False,
            "error": "No face detected",
            "bbox": None
        }

    # -------------------------------------------------
    # Model not ready yet
    # -------------------------------------------------
    if verifier is None:
        return {
            "match": False,
            "bbox": bbox,
            "error": "Face model not available yet"
        }

    # -------------------------------------------------
    # Face verification
    # -------------------------------------------------
    try:
        similarity = verifier.verify(
            id_image.file,
            selfie_image.file
        )

        return {
            "similarity": round(similarity, 4),
            "match": similarity > 0.5,
            "bbox": bbox
        }

    except Exception as e:
        return {
            "match": False,
            "bbox": bbox,
            "error": str(e)
        }
