import cv2
import numpy as np
from typing import Optional
import mediapipe as mp


# Initialize face detection
mp_face_detection = mp.solutions.face_detection


def infer_embedding(image: np.ndarray) -> np.ndarray:
    """
    Generate face embedding from image
    For production, use a proper face recognition model (ArcFace, FaceNet, etc.)
    This is a placeholder using basic features
    """
    # Resize to standard size
    face = cv2.resize(image, (112, 112))
    
    # Convert to grayscale and flatten as basic "embedding"
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Simple feature extraction (histogram of oriented gradients would be better)
    # For now, just normalize the flattened image
    embedding = gray.flatten().astype(np.float32)
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def verify_faces(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.5) -> bool:
    """Verify if two face embeddings match"""
    similarity = cosine_sim(embedding1, embedding2)
    return similarity >= threshold
