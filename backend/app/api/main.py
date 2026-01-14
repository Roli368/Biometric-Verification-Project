from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Biometric Verification Engine")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend is running"}

@app.post("/verify")
async def verify_identity(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):
    """
    TEMPORARY MOCK RESPONSE
    ML logic will be plugged here
    """
    return {
        "match_score": 0.92,
        "liveness": "REAL",
        "final_decision": "ACCEPT"
    }
