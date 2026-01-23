import cv2
import numpy as np
from typing import Optional
from ..config import ID_FACE_PADDING


def extract_face_from_id(input_path: str, output_path: str) -> Optional[str]:
    """
    Extract face from ID document image
    Returns output path if successful, None otherwise
    """
    img = cv2.imread(input_path)
    if img is None:
        return None
    
    # Use Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None
    
    # Take the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Add padding
    padding = int(w * ID_FACE_PADDING)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    # Crop face
    face = img[y1:y2, x1:x2]
    
    # Save
    cv2.imwrite(output_path, face)
    return output_path
