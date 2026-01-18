import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def run_guardrails(image):
    """The Pre-Processing Fortress (Guardrail)"""
    # Laplacian Variance check for blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # HSV-Value check for light
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    avg_brightness = np.mean(v)
    
    # Validation
    if laplacian_var < 100:
        return False, f"REJECTED: Blurry (Var: {laplacian_var:.1f})"
    if avg_brightness < 50:
        return False, f"REJECTED: Too Dark (Val: {avg_brightness:.1f})"
        
    return True, "PASSED"

def align_face(image):
    """Recursive Alignment using Affine Warp"""
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.detections:
        return None
    
    keypoints = results.detections[0].location_data.relative_keypoints
    right_eye = (keypoints[0].x * image.shape[1], keypoints[0].y * image.shape[0])
    left_eye = (keypoints[1].x * image.shape[1], keypoints[1].y * image.shape[0])
    
    # Calculate angle for Affine Warp
    dY = left_eye[1] - right_eye[1]
    dX = left_eye[0] - right_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    

    eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return aligned