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


# Acceptance limits. cv2.mcc happily "finds" a chart in a scene without one (24 near-identical
# patches); the fit then produces a wild matrix with a huge residual. A real chart, even under
# a colour cast or poor exposure, keeps these properties.
MIN_SPREAD = 40.0        # range of patch brightness (0-255): white vs black patch
MIN_CORRELATION = 0.85   # measured vs reference values, all 72 numbers
MAX_MEAN_ERROR = 25.0    # fit residual (sRGB units) that still counts as a calibration
MAX_MAX_ERROR = 75.0


def validate_measurement(measured_rgb, reference_rgb=None):
    """Is this plausibly a ColorChecker? (ok, reason).

    Three checks: the patches span a real brightness range, their values
    correlate with the reference chart, and the grey ramp (bottom row) gets
    darker from left to right.
    """
    if reference_rgb is None:
        reference_rgb = COLORCHECKER_SRGB
    m = np.asarray(measured_rgb, dtype=np.float64)
    if m.shape != reference_rgb.shape:
        return False, f"expected {reference_rgb.shape[0]} patches, got {m.shape[0]}"
    lum = m.mean(axis=1)
    spread = float(np.ptp(lum))
    if spread < MIN_SPREAD:
        return False, (f"the 24 'patches' are all about the same brightness (range {spread:.0f} of 255): "
                       "no ColorChecker in the frame")
    corr = float(np.corrcoef(m.ravel(), reference_rgb.ravel())[0, 1])
    if not corr >= MIN_CORRELATION:
        return False, f"patch colours do not match a ColorChecker (correlation {corr:.2f}): wrong object, or upside down?"
    greys = lum[18:24]
    if not np.all(np.diff(greys) < 0):
        return False, "the grey ramp is not in order: chart rotated or partly covered?"
    return True, f"chart plausible (brightness range {spread:.0f}, correlation {corr:.2f})"


def validate_fit(error_stats):
    """Is the fitted correction good enough to use? (ok, reason)."""
    mean, mx = error_stats["mean"], error_stats["max"]
    if mean > MAX_MEAN_ERROR or mx > MAX_MAX_ERROR:
        return False, (f"poor fit (mean error {mean:.0f}, max {mx:.0f} sRGB units): glare on the chart, "
                       "uneven light, or not a chart")
    return True, f"mean error {mean:.1f}, max {mx:.1f} (sRGB units)"
