import cv2
import numpy as np

# ---------- BLUR CHECK ----------
def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def is_blurry(image, threshold=100):
    score = blur_score(image)
    return score < threshold, score


# ---------- LIGHT CHECK ----------
def lighting_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def bad_lighting(image, low=60, high=200):
    score = lighting_score(image)
    return score < low or score > high, score


# ---------- DRAW WARNING ----------
def draw_warning(frame, text):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)