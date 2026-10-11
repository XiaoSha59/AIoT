from __future__ import annotations

import cv2
import numpy as np


class DetectionError(RuntimeError):
    pass


class FaceDetector:
    def __init__(self, path: str):
        try:
            self.detector = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.85, 0.3, 5000)
        except Exception as exc:
            raise DetectionError(f"YuNet cannot load {path}: {exc}") from exc

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        try:
            self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = self.detector.detect(bgr)
            return [] if faces is None else list(faces)
        except Exception as exc:
            raise DetectionError(str(exc)) from exc


def crop_face(bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Rotate around eyes when landmarks permit, then use a padded bounding box.

    The same saved RGB crops feed training. No SFace/embedding model is required.
    """
    x, y, w, h = map(float, face[:4])
    if w <= 0 or h <= 0:
        raise DetectionError("Invalid face bounding box")
    rotated = bgr
    corners = np.asarray([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    if len(face) >= 8:
        eyes = sorted([np.asarray(face[4:6]), np.asarray(face[6:8])], key=lambda p: p[0])
        delta = eyes[1] - eyes[0]
        if np.isfinite(delta).all() and np.linalg.norm(delta) > 5:
            angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
            if abs(angle) <= 45:
                center = tuple(map(float, (eyes[0] + eyes[1]) / 2))
                matrix = cv2.getRotationMatrix2D(center, angle, 1)
                rotated = cv2.warpAffine(bgr, matrix, (bgr.shape[1], bgr.shape[0]), flags=cv2.INTER_LINEAR)
                corners = cv2.transform(corners[None], matrix)[0]
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    pad = (hi-lo) * 0.10
    x1, y1 = np.maximum(0, np.floor(lo-pad)).astype(int)
    x2, y2 = np.minimum([bgr.shape[1], bgr.shape[0]], np.ceil(hi+pad)).astype(int)
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        raise DetectionError("Face outside image")
    return crop
