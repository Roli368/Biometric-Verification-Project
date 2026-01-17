import cv2
import time
import random
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy

# TECH: Dynamic buffers for temporal, biological, and geometric consistency.
BUFFER_RPPG = 200
BUFFER_BLINK = 100
BUFFER_TEMPORAL = 50
MIN_BUFFER = 30

# ================= UTILS =================
def adaptive_bandpass(sig, fps, env_var):
    """
    TECH: Signal Phase Extraction via Analytic Signal (Hilbert).
    MATH: Adjusts frequency cutoffs based on environment noise (env_var).
    Hilbert transform calculates the instantaneous phase (unwrap(angle)).
    Standard deviation of phase difference identifies synthetic signals.
    """
    low = max(0.5, 0.7 - env_var * 0.2)
    high = min(5.0, 4.0 + env_var * 0.5)
    nyq = 0.5 * fps
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    filt = filtfilt(b, a, sig)
    phase = np.unwrap(np.angle(hilbert(filt)))
    return filt, np.std(np.diff(phase))

def fft_metrics(sig, fps):
    """
    TECH: Spectral Sharpness & Peak Analysis.
    MATH: Sharpness ratio (yf[idx]/sidebands) distinguishes a clean heartbeat 
    from a flat-line spoof or random sensor noise.
    """
    yf = np.abs(rfft(sig))
    xf = rfftfreq(len(sig), 1/fps)
    idx = np.argmax(yf)
    bpm = xf[idx] * 60
    side_low = yf[:max(1, idx//2)]
    side_high = yf[min(len(yf)-1, idx*2):]
    sidebands = np.mean(np.concatenate([side_low, side_high]))
    sharpness = yf[idx] / (sidebands + 1e-6)
    return bpm, entropy(yf + 1e-6), sharpness

def dense_flow(prev, curr):
    """
    TECH: Farneback Dense Optical Flow.
    MATH: Computes motion vectors for every pixel. 
    Magnitude (sqrt(x^2 + y^2)) detects the subtle skin micro-motions 
    missing in static photo replays.
    """
    if prev is None: return 0.0
    p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    return float(np.mean(mag))

def frame_lsb_bits(frame):
    """
    TECH: Bit-Plane Hash Generation.
    MATH: Packs Least Significant Bits (LSB). 
    Used to detect virtual camera injections by checking temporal noise.
    """
    lsb = frame & 1
    return np.packbits(lsb.flatten())

def hamming(a, b):
    """MATH: Bitwise XOR count to find distance between frame hashes."""
    return np.count_nonzero(a != b)

# ================= ENGINE =================
class UltimateLiveness10:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.rgb_buf, self.blink_buf, self.pulse_env_buf = [], [], []
        self.z_buf, self.pose_buf, self.hash_chain = [], [], []
        self.prev_roi = None
        self.start = time.perf_counter()
        self.frames = 0
        self.fps, self.env_var = 30, 0.0
        self.challenge = random.choice(["BLINK", "SMILE", "TURN"])
        self.challenge_start = time.time()

    def skin_roi(self, frame, lm):
        h, w, _ = frame.shape
        idx = [10, 338, 297, 332]
        pts = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in idx])
        x,y,w1,h1 = cv2.boundingRect(pts)
        return frame[y:y+h1, x:x+w1]

    def rppg(self):
        rgb = np.array(self.rgb_buf)
        mean = np.mean(rgb, axis=0)
        norm = rgb / (mean + 1e-6)
        s1, s2 = norm[:,1]-norm[:,2], -2*norm[:,0]+norm[:,1]+norm[:,2]
        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        pulse_raw = s1 + alpha*s2
        pulse, phase_std = adaptive_bandpass(pulse_raw, self.fps, self.env_var)
        bpm, ent, sharp = fft_metrics(pulse, self.fps)
        self.pulse_env_buf.append(abs(pulse[-1]))
        if len(self.pulse_env_buf) > 50: self.pulse_env_buf.pop(0)
        return bpm, ent, sharp, phase_std

    def blink_entropy(self, lm):
        ear = (abs(lm[159].y - lm[145].y) + abs(lm[386].y - lm[374].y)) / 2
        self.blink_buf.append(ear)
        if len(self.blink_buf) > BUFFER_BLINK: self.blink_buf.pop(0)
        hist,_ = np.histogram(self.blink_buf, bins=10, density=True)
        return entropy(hist + 1e-6)

    def depth_score(self, lm):
        """MATH: Variance of difference in Z-mean to detect static depth masks."""
        z = np.mean([lm[i].z for i in [10, 33, 263]])
        self.z_buf.append(z)
        if len(self.z_buf) > BUFFER_BLINK: self.z_buf.pop(0)
        return np.var(np.diff(self.z_buf)) if len(self.z_buf) >= 10 else 0.0

    def temporal_score(self, roi):
        """TECH: Motion + Anti-Injection Hash Chain Analysis."""
        flow = dense_flow(self.prev_roi, roi)
        self.prev_roi = roi
        h = frame_lsb_bits(roi)
        self.hash_chain.append(h)
        if len(self.hash_chain) > BUFFER_TEMPORAL: self.hash_chain.pop(0)
        if len(self.hash_chain) > 1:
            jumps = sum(hamming(self.hash_chain[i-1], self.hash_chain[i]) > 2 for i in range(1,len(self.hash_chain)))
            if jumps / (len(self.hash_chain)-1) > 0.2: return 0.0
        return flow

    def challenge_ok(self, lm):
        """TECH: Behavioral Challenge Logic (Smile/Turn/Blink)."""
        now = time.time()
        if now - self.challenge_start > 8:
            self.challenge, self.challenge_start = random.choice(["BLINK","SMILE","TURN"]), now
            return 0
        if self.challenge == "BLINK": return int(np.std(self.blink_buf[-20:]) > 0.03)
        if self.challenge == "SMILE": return int(abs(lm[13].y - lm[14].y) > 0.04)
        return int(abs(lm[1].x - (lm[234].x + lm[454].x)/2) > 0.03)

    def verify(self, frame):
        self.frames += 1
        elapsed = time.perf_counter() - self.start
        self.fps = self.frames / max(elapsed,1e-6)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks: return "NO FACE", 0.0
        lm = res.multi_face_landmarks[0].landmark
        roi = self.skin_roi(frame, lm)
        self.rgb_buf.append(np.mean(roi, axis=(0,1)))
        if len(self.rgb_buf) > BUFFER_RPPG: self.rgb_buf.pop(0)
        if len(self.rgb_buf) < MIN_BUFFER: 
            self.env_var = np.std(self.rgb_buf)
            return "CALIBRATING", 0.0

        bpm, ent, sharp, phase = self.rppg()
        blink_e, depth, temp, chall = self.blink_entropy(lm), self.depth_score(lm), self.temporal_score(roi), self.challenge_ok(lm)

        # Hard veto: Biological and Physical consistency checks
        if depth < 1e-5 or phase > 1.0 or temp < 0.3: return "SPOOF", 0.0

        scores = [45 < bpm < 160, ent > 1.2, sharp > 3, blink_e > 1.3, depth > 1e-5, temp > 0.5, chall]
        trust = sum(scores) / len(scores)
        return ("LIVE HUMAN" if trust > 0.9 else "SPOOF"), trust

class POSPulseExtractor:
    def __init__(self, low_hz=0.7, high_hz=4.0):
        self.low_hz = low_hz
        self.high_hz = high_hz

    def butter_bandpass_filter(self, data, fps):
        nyq = 0.5 * fps
        low = self.low_hz / nyq
        high = self.high_hz / nyq
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, data)

    def extract_pulse(self, rgb_buffer, fps):
        """
        Args:
            rgb_buffer: List of [R, G, B] averages per frame
            fps: Frames per second of capture

        Returns:
            filtered_pulse: rPPG signal
            is_screen: True if screen replay detected
        """

        rgb_array = np.array(rgb_buffer, dtype=np.float32)

        # Normalize RGB
        mean_rgb = np.mean(rgb_array, axis=0)
        norm_rgb = rgb_array / (mean_rgb + 1e-6)

        # POS method
        s1 = norm_rgb[:, 1] - norm_rgb[:, 2]
        s2 = -2 * norm_rgb[:, 0] + norm_rgb[:, 1] + norm_rgb[:, 2]

        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        raw_h = s1 + alpha * s2

        # Bandpass filter
        filtered_pulse = self.butter_bandpass_filter(raw_h, fps)

        # FFT-based screen detection
        n = len(filtered_pulse)
        yf = fft(filtered_pulse)
        xf = fftfreq(n, 1 / fps)

        high_freq_energy = np.sum(np.abs(yf[xf > 5.0]))
        low_freq_energy = np.sum(
            np.abs(yf[(xf >= self.low_hz) & (xf <= self.high_hz)])
        )

        is_screen = (high_freq_energy / (low_freq_energy + 1e-6)) > 0.5

        return filtered_pulse, is_screen
