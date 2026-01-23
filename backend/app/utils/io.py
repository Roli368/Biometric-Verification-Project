import cv2
import numpy as np
from typing import Optional


def imdecode_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to numpy array"""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def imencode_image(img: np.ndarray, ext: str = '.jpg') -> Optional[bytes]:
    """Encode numpy array to image bytes"""
    success, encoded = cv2.imencode(ext, img)
    if success:
        return encoded.tobytes()
    return None
