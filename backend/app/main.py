from fastapi import FastAPI, Body
import numpy as np
import time
from app.liveness import UltimateLiveness10

# TECH: FastAPI production server with Ultra-Security Liveness Engine.
app = FastAPI(title="ULTIMATE NO-MERCY LIVENESS — 100/100")
engine = UltimateLiveness10()

@app.post("/verify-liveness")
async def verify_liveness(rgb_data: list = Body(...)):
    """
    TECH: Processes 200-frame deep analysis for biological rPPG and temporal hashing.
    MATH: Bayesian trust scoring based on 7-layer verification.
    """
    try:
        # Step: Reconstruct frames from incoming pixel arrays
        frames = [np.array(f, dtype=np.uint8) for f in rgb_data]
        
        # Step: Pipeline processing for each frame in sequence
        for frame in frames:
            verdict, trust = engine.verify(frame)
        
        return {
            "is_alive": verdict == "LIVE HUMAN",
            "verdict": verdict,
            "trust_score": round(float(trust), 4),
            "challenge_active": engine.challenge,
            "fps_performance": round(engine.fps, 2),
            "buffer_status": f"{len(engine.rgb_buf)}/200"
        }
        
    except Exception as e:
        return {"is_alive": False, "verdict": "ERROR", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Initializing production server
    uvicorn.run(app, host="127.0.0.1", port=8000)