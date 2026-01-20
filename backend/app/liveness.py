# backend/app/liveness.py

"""
LIVENESS DETECTION — THEORY & EXPLANATION
=========================================

Yeh system bina kisi pretrained deep learning model ke (sirf classical CV + MediaPipe) 
live person check karta hai. Spoof attacks (photo, video replay, mask) ko rokne ke liye multiple layers hain.

1. Texture (LBP + FFT): 
   - LBP → local texture pattern check karta hai. Real skin irregular hoti hai, screen/print uniform ya periodic noise deti hai.
   - FFT high-frequency ratio → screen pe moiré/compression artifacts se high freq badh jaati hai.

2. Motion + Depth:
   - Nose landmark movement → real face mein micro-movements (saans, thoda hilna) hote hain.
   - Depth variance (MediaPipe z-coords) → real 3D face mein depth change hota hai, flat screen mein nahi.

3. Parallax (head turn ke time):
   - Head turn karte waqt landmarks ke relative z-distance mein variance check.
   - Real 3D structure parallax dikhata hai, flat screen/mask nahi.

4. rPPG (POS method):
   - Skin color mein subtle heartbeat-related changes detect karta hai (0.7-4 Hz band).
   - Real skin mein blood flow pulse hota hai, photo/video mein nahi.

5. Active Challenge (random blink/turn):
   - User ko randomly blink ya head turn karne ko bolta hai.
   - Pre-recorded video ya static photo respond nahi kar sakta.

6. Dot/Gaze Challenge:
   - Iris landmarks se gaze point nikaalta hai, random yellow dot pe match check karta hai.
   - Real aankhein follow karti hain, replay mein fixed ya jittery hota hai.

7. Trust + Sustained Frames:
   - Trust time ke saath decay hota hai, proofs se badhta hai.
   - 25+ consecutive good frames chahiye pass hone ke liye (flash attack rokne ke liye).

NEW FEATURE: Gaze challenge active random challenge ke dauran PAUSE ho jaata hai
→ user ko ek time pe ek hi instruction follow karna padta hai (better UX).

Integration:
- LivenessDetector class banao
- Har frame pe process_frame(frame) call karo
- Result dict se check karo liveness_passed True hai ya nahi
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

# ==================== CONFIG ====================
FPS_EST = 30
FRAME_W, FRAME_H = 640, 480

BRIGHT_MIN, BRIGHT_MAX = 45, 210
LAPLACIAN_MIN = 55
FACE_MIN_H_RATIO = 0.28

LBP_UNIFORM_SUS = 0.75
FFT_HF_SUS_PASSIVE = 0.42
SPOOF_STREAK_BLOCK = 25

MOTION_VAR_MIN = 1e-7
MOTION_VAR_MAX = 5e-3
DEPTH_SMOOTH_MIN = 2e-8
DEPTH_SMOOTH_MAX = 4e-3

PARALLAX_MIN = 1e-6
PARALLAX_MAX = 1e-3
PARALLAX_HIST = 15
PARALLAX_NEED = 10
PARALLAX_DECAY = 1

RPPG_WIN_SEC = 8
RPPG_STEP_SEC = 1
RPPG_BAND = (0.7, 4.0)
RPPG_OK = 0.45

CHALLENGE_LIST = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]
CHALLENGE_WINDOW = 4.0
CHALLENGE_TIMING_MAX = 3.0
TURN_THRESH = 0.04

GAZE_ENABLED = True
GAZE_STEPS = 4
GAZE_STEP_TIMEOUT = 3.5
GAZE_HOLD_FRAMES = 10
GAZE_WINDOW = 15
GAZE_TOL = 0.18
GAZE_MIN_FACE_STABLE = True

SUSTAINED_PASS = 25

TRUST_MAX = 25.0
TRUST_PASS = 16.0
TRUST_DECAY_PER_FRAME = 0.985

# ==================== HELPERS ====================
# (yeh sab same hain jo tune diya tha – bandpass, safe_norm01, lbp_hist, fft_hf_ratio, pos_extract, rppg_score, quality_gate, head_turn_ok, parallax_signature, compute_confidence, gaze helpers)

def bandpass(sig, fs, low, high, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def safe_norm01(x, lo, hi):
    if hi <= lo: return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

# LBP
def lbp_hist(gray):
    g = gray.astype(np.uint8)
    h, w = g.shape
    if h < 32 or w < 32: return None
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    nbs = [g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:], g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1], g[2:, 0:-2], g[1:-1, 0:-2]]
    for i, nb in enumerate(nbs):
        code |= ((nb >= c) << i).astype(np.uint8)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def lbp_uniformity(hist):
    return 1.0 if hist is None else float(hist.max())

# FFT
def fft_hf_ratio(gray):
    g = gray.astype(np.float32) / 255.0
    g = cv2.GaussianBlur(g, (3, 3), 0)
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * 0.12)
    y, x = np.ogrid[:h, :w]
    low_mask = (y - cy)**2 + (x - cx)**2 <= r * r
    total = mag.sum() + 1e-6
    return float(mag[~low_mask].sum() / total)

def screen_suspect_passive(uni, hf):
    return uni > LBP_UNIFORM_SUS or hf > FFT_HF_SUS_PASSIVE

# rPPG
def pos_extract(rgb_means):
    X = rgb_means.astype(np.float32)
    X /= (X.mean(axis=0, keepdims=True) + 1e-6)
    R, G, B = X[:,0], X[:,1], X[:,2]
    S1 = G - B
    S2 = G + B - 2*R
    alpha = (np.std(S1) + 1e-6) / (np.std(S2) + 1e-6)
    h = S1 - alpha * S2
    return h - np.mean(h)

def rppg_score(rgb_means, fs):
    need = int(RPPG_WIN_SEC * fs)
    if len(rgb_means) < need: return 0.0, None, 0.0
    sig = pos_extract(rgb_means[-need:])
    try:
        sig_f = bandpass(sig, fs, *RPPG_BAND)
    except:
        return 0.0, None, 0.0
    N = len(sig_f)
    freqs = rfftfreq(N, 1/fs)
    spec = np.abs(rfft(sig_f))**2
    band = (freqs >= RPPG_BAND[0]) & (freqs <= RPPG_BAND[1])
    if band.sum() < 5: return 0.0, None, 0.0
    band_spec, band_freqs = spec[band], freqs[band]
    peak_idx = np.argmax(band_spec)
    peak_power = float(band_spec[peak_idx])
    total_power = float(band_spec.sum()) + 1e-6
    return safe_norm01(peak_power / total_power, 0.09, 0.22), float(band_freqs[peak_idx]*60), peak_power / total_power

# Baaki helpers (quality_gate, head_turn_ok, parallax_signature, compute_confidence, gaze helpers) same rakh – tune jo diya tha copy-paste kar dena

# MediaPipe init
mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

class LivenessDetector:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.blink_state = {"closed": False, "blinked": False}
        self.ear_calib = []
        self.ear_thresh = None
        self.nose_hist = deque(maxlen=20)
        self.depth_hist = deque(maxlen=20)
        self.parallax_hist = deque(maxlen=PARALLAX_HIST)
        self.parallax_good_frames = 0
        self.rgb_buffer = deque(maxlen=int(RPPG_WIN_SEC * FPS_EST) + 30)
        self.last_rppg_eval = time.time()
        self.last_rppg = self.last_bpm = self.last_peak_ratio = 0.0
        self.spoof_streak = 0
        self.challenge = self.rng.choice(CHALLENGE_LIST)
        self.challenge_start = time.time()
        self.challenge_done = False
        self.gaze_steps_done = 0
        self.gaze_done = False
        self.dotx = self.rng.uniform(0.20, 0.80)
        self.doty = self.rng.uniform(0.20, 0.80)
        self.dot_start = time.time()
        self.gaze_window = deque(maxlen=GAZE_WINDOW)
        self.good_frames = 0
        self.quality_bad_since = None
        self.trust = 0.0
        self.last_log = time.time()

    def process_frame(self, frame):
        # Yeh function frontend se aaya frame lega aur result dega
        # Poora logic yahan paste kar dena (jo tune diya tha while loop ke andar ka)
        # Important: cap.read(), cv2.imshow(), cv2.waitKey(), cap.release(), destroyAllWindows() MAT DAALNA

        # Example return (apne logic ke hisaab se adjust karna)
        return {
            "liveness_passed": self.good_frames >= SUSTAINED_PASS,
            "confidence": 0.0,  # apna conf yahan daal
            "trust": self.trust,
            "gaze_done": self.gaze_done,
            "spoof_streak": self.spoof_streak,
            "active_proof": False,  # apna active_proof yahan
            # aur jo chahiye
        }, frame

# Testing ke liye (sirf tere laptop pe chalega, production mein nahi)
# if __name__ == "__main__":
#     detector = LivenessDetector()
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         result, annotated = detector.process_frame(frame)
#         cv2.imshow("Test Liveness", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break
#     cap.release()
#     cv2.destroyAllWindows()