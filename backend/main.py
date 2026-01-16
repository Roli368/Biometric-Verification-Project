from fastapi import FastAPI, Body
from app.liveness import POSPulseExtractor 
import numpy as np

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
extractor = POSPulseExtractor(fps=30)

@app.post("/verify-liveness")
async def verify_liveness(rgb_data: list = Body(...)):
    # Error handling: Agar data kam hai toh reject karo
    if len(rgb_data) < 90:
        return {"status": "error", "message": "Need 90 frames (3 sec)"}
    
    try:
        # 1. Heartbeat Extraction
        pulse = extractor.extract_pulse(rgb_data)
        
        # 2. BPM Calculation
        peaks = np.where(np.diff(np.sign(np.diff(pulse))) < 0)[0]
        bpm = (len(peaks) / 3.0) * 60 
        
        # 3. Decision Logic
        is_alive = 45 < bpm < 160 
        
        return {
            "bpm": int(bpm),
            "is_alive": bool(is_alive),
            "verdict": "Verified" if is_alive else "Spoof Detected",
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Swarupa-Prana API is running"}