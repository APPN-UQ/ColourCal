"""Compute and apply a colour correction matrix from ColorChecker measurements."""
import json
from pathlib import Path

import numpy as np

from .reference import COLORCHECKER_SRGB


def colour_matrix(measured_rgb, reference_rgb=None):
    """Least-squares 3x3 colour correction matrix: corrected = measured @ M.

    measured_rgb: (N, 3) float64 — mean patch values from the camera.
    reference_rgb: (N, 3) float64 — known sRGB values (default: ColorChecker Classic).
    Returns M: (3, 3) float64.
    """
    if reference_rgb is None:
        reference_rgb = COLORCHECKER_SRGB
    M, _, _, _ = np.linalg.lstsq(measured_rgb, reference_rgb, rcond=None)
    return M


def apply_correction(img, M):
    """Apply a 3x3 colour correction matrix to a BGR image.

    The matrix operates in RGB space; this handles the BGR<->RGB conversion.
    """
    rgb = img[..., ::-1].astype(np.float64)
    corrected = rgb @ M
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected[..., ::-1]  # back to BGR


def save_calibration(path, M, measured_rgb, error_stats=None):
    """Save calibration to a JSON file."""
    data = {
        "matrix": M.tolist(),
        "measured_rgb": measured_rgb.tolist(),
        "reference_rgb": COLORCHECKER_SRGB.tolist(),
    }
    if error_stats:
        data["error"] = error_stats
    Path(path).write_text(json.dumps(data, indent=2))


def load_calibration(path):
    """Load calibration from a JSON file. Returns M: (3, 3) float64."""
    data = json.loads(Path(path).read_text())
    return np.array(data["matrix"], dtype=np.float64)


def calibration_error(measured_rgb, M, reference_rgb=None):
    """Mean and max colour error (Euclidean in sRGB) after correction."""
    if reference_rgb is None:
        reference_rgb = COLORCHECKER_SRGB
    corrected = measured_rgb @ M
    err = np.sqrt(((corrected - reference_rgb) ** 2).sum(axis=1))
    return {"mean": float(err.mean()), "max": float(err.max()), "per_patch": err.tolist()}
