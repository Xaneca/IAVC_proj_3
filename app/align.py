"""Alinhamento da cara com os 5 landmarks → crop 112×112 (template ArcFace).

Usado pela aplicação E pelo treino, para que ambos vejam exatamente o mesmo tipo de crop.
"""
import numpy as np
from insightface.utils.face_align import norm_crop

CROP_SIZE = 112


def align(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Devolve a cara alinhada (112, 112, 3) em BGR."""
    return norm_crop(frame, landmarks, image_size=CROP_SIZE)
