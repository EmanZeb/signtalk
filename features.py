import numpy as np


def extract_features(lm):
    """Translate to wrist, scale-normalise -> 63 floats (translation/scale invariant)."""
    pts = np.asarray(lm, dtype=np.float32).copy()
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1)) + 1e-6
    pts /= scale
    return pts.flatten()


def mirror_features(feat):
    f = feat.reshape(21, 3).copy()
    f[:, 0] *= -1
    return f.flatten()
