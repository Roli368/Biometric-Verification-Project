from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from app.liveness import POSPulseExtractor 
import numpy as np

# 1. Sabse pehle app initialize karo (Nahi toh NameError aayega)
app = FastAPI()

# 2. CORS middleware setup karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = POSPulseExtractor(fps=30)

# 3. Ab endpoints define karo
@app.post("/verify-liveness")
async def verify_liveness(rgb_data: list = Body(...)):
    if len(rgb_data) < 90:
        return {"status": "error", "message": "Need 90 frames (3 sec)"}
    
    try:
        pulse = extractor.extract_pulse(rgb_data)
        
        # Screen Detection logic
        if extractor.is_screen:
            return {
                "bpm": 0, 
                "is_alive": False, 
                "verdict": "Spoof: Screen Detected", 
                "status": "success"
            }
            
        peaks = np.where(np.diff(np.sign(np.diff(pulse))) < 0)[0]
        bpm = (len(peaks) / 3.0) * 60 
        is_alive = 45 < bpm < 160 
        
        return {
            "bpm": int(bpm),
            "is_alive": bool(is_alive),
            "verdict": "Verified" if is_alive else "Spoof: Irregular Heartbeat",
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Swarupa-Prana API is running"}