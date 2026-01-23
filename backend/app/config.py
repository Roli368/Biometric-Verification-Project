import os

# Thresholds and config
TRUST_THRESHOLD = float(os.getenv('TRUST_THRESHOLD', 0.05))
BLUR_LAPLACIAN_MIN = float(os.getenv('BLUR_LAPLACIAN_MIN', 80.0))
BRIGHTNESS_MIN = int(os.getenv('BRIGHTNESS_MIN', 50))
BRIGHTNESS_MAX = int(os.getenv('BRIGHTNESS_MAX', 200))
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', 0.5))
SWAP_THRESHOLD = float(os.getenv('SWAP_THRESHOLD', 0.8))
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 180))
CAPTURE_INTERVAL_MS = int(os.getenv('CAPTURE_INTERVAL_MS', 300))
MIN_BUFFER = int(os.getenv('MIN_BUFFER', 30))
ID_FACE_PADDING = float(os.getenv('ID_FACE_PADDING', 0.25))

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
