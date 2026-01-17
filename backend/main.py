from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np

from app.liveness import POSPulseExtractor

app = FastAPI()

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== PREFLIGHT ==================
@app.options("/{path:path}")
async def options_handler(path: str, request: Request):
    return JSONResponse(
        status_code=200,
        content={"ok": True},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ================== LIVENESS ENGINE ==================
extractor = POSPulseExtractor()

# ================== VERIFY ENDPOINT ==================
@app.post("/verify")
async def verify(
    id_card: UploadFile,
    selfie: UploadFile,
    liveness: str = Form(...)
):
    """
    liveness = {
      fps: number,
      rgb_frames: [[r,g,b], ...]
    }
    """

    # ---------- LIVENESS ----------
    data = json.loads(liveness)
    rgb_frames = data["rgb_frames"]
    fps = data["fps"]

    # ✅ FIX: fps passed correctly
    pulse, is_screen = extractor.extract_pulse(rgb_frames, fps)

    peaks = np.where(np.diff(np.sign(np.diff(pulse))) < 0)[0]
    bpm = (len(peaks) / 3) * 60

    is_alive = (45 < bpm < 160) and not is_screen

    # ---------- FACE VERIFICATION (DUMMY) ----------
    similarity = 0.78  # dummy similarity (78%)

    match_confidence = round(similarity * 100, 2)

    # ---------- FINAL VERDICT ----------
    verified = is_alive and similarity > 0.6

    return {
        "status": "success",
        "liveness": is_alive,
        "bpm": int(bpm),
        "match_confidence": match_confidence,
        "verdict": "VERIFIED" if verified else "REJECTED"
    }

# ================== ROOT ==================
@app.get("/")
def root():
    return {"message": "Backend is running"}
