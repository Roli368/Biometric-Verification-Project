import cv2
import numpy as np
from typing import Tuple, Dict, Any
from ..config import BLUR_LAPLACIAN_MIN, BRIGHTNESS_MIN, BRIGHTNESS_MAX


def calculate_blur(image: np.ndarray) -> float:
    """Calculate blur metric using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate average brightness"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def pre_check_quality(image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    """
    Check frame quality before processing
    Returns: (is_ok, result_dict)
    """
    blur = calculate_blur(image)
    brightness = calculate_brightness(image)
    
    metrics = {
        'blur': blur,
        'brightness': brightness
    }
    
    # Check blur
    if blur < BLUR_LAPLACIAN_MIN:
        return False, {
            'message': 'Image too blurry',
            'reason': 'blur',
            'metrics': metrics
        }
    
    # Check brightness
    if brightness < BRIGHTNESS_MIN:
        return False, {
            'message': 'Too dark',
            'reason': 'brightness_low',
            'metrics': metrics
        }
    
    if brightness > BRIGHTNESS_MAX:
        return False, {
            'message': 'Too bright',
            'reason': 'brightness_high',
            'metrics': metrics
        }
    
    return True, {
        'message': 'Quality OK',
        'metrics': metrics
    }
