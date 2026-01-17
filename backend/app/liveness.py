import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

class POSPulseExtractor:
    def __init__(self, fps=30):
        self.fps = fps
        self.low_hz, self.high_hz = 0.7, 4.0

    def butter_bandpass_filter(self, data):
        nyq = 0.5 * self.fps
        low, high = self.low_hz / nyq, self.high_hz / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)

    def extract_pulse(self, rgb_buffer):
        rgb_array = np.array(rgb_buffer)
        mean_rgb = np.mean(rgb_array, axis=0)
        norm_rgb = rgb_array / (mean_rgb + 1e-6)
        
        # Plane-Orthogonal-to-Skin (POS) Math
        s1 = norm_rgb[:, 1] - norm_rgb[:, 2]
        s2 = -2 * norm_rgb[:, 0] + norm_rgb[:, 1] + norm_rgb[:, 2]
        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        raw_h = s1 + (alpha * s2)
        
        filtered_pulse = self.butter_bandpass_filter(raw_h)
        
        # --- NEW: FFT Screen Detection ---
        n = len(filtered_pulse)
        yf = fft(filtered_pulse)
        xf = fftfreq(n, 1 / self.fps)
        
        # Screen noise usually lives in very high frequencies
        high_freq_energy = np.sum(np.abs(yf[np.where(xf > 5.0)])) 
        low_freq_energy = np.sum(np.abs(yf[np.where((xf >= 0.7) & (xf <= 4.0))]))
        
        # If noise/pulse ratio is too high, it's a digital screen
        self.is_screen = (high_freq_energy / (low_freq_energy + 1e-6)) > 0.5
        
        return filtered_pulse