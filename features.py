"""Hand-crafted, interpretable patch features (labels never used)."""
import numpy as np
from PIL import Image
from scipy import ndimage

FEATURE_NAMES = [
    "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
    "Hue_mean", "Sat_mean", "Sat_std", "Val_mean", "Val_std",
    "ExG_mean (greenness)", "ExG_std", "GR_index (veg)", "GB_index (water)",
    "Edge_mean", "Edge_std", "Laplacian_var (texture)", "Gray_entropy",
]

def extract(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((64, 64))
    a = np.asarray(img, float) / 255.0
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    hsv = np.asarray(img.convert("HSV"), float) / 255.0
    exg = 2 * G - R - B
    gr = (G - R) / (G + R + 1e-6)
    gb = (G - B) / (G + B + 1e-6)
    gray = a.mean(2)
    edge = np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
    lap = ndimage.laplace(gray).var()
    hist, _ = np.histogram(gray, bins=16, range=(0, 1))
    p = hist / hist.sum(); p = p[p > 0]
    ent = -(p * np.log2(p)).sum()
    return np.array([
        R.mean(), G.mean(), B.mean(), R.std(), G.std(), B.std(),
        hsv[..., 0].mean(), hsv[..., 1].mean(), hsv[..., 1].std(),
        hsv[..., 2].mean(), hsv[..., 2].std(),
        exg.mean(), exg.std(), gr.mean(), gb.mean(),
        edge.mean(), edge.std(), lap, ent,
    ])
