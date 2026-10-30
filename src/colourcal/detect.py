"""Detect a ColorChecker Classic (24-patch) in an image and extract patch colours."""
import cv2
import numpy as np


def detect_checker(img, chart_type=cv2.mcc.MCC24):
    """Detect a ColorChecker in a BGR image.

    Returns (measured_rgb, corners) or (None, None) if not found.
    measured_rgb: (24, 3) float64 array of mean patch RGB values.
    corners: (4, 2) float64 array of the chart's outer corners (for drawing).
    """
    det = cv2.mcc.CCheckerDetector.create()
    if not det.process(img, chart_type):
        return None, None
    checker = det.getBestColorChecker()
    box = np.array(checker.getBox(), dtype=np.float64)
    raw = np.array(checker.getChartsRGB(), dtype=np.float64)
    # (24*3, 5): each patch has 3 rows (R, G, B), each [count, mean, std, min, max].
    measured_rgb = raw[:, 1].reshape(24, 3)  # column 1 is the channel mean
    return measured_rgb, box


def detect_checker_region(img, roi):
    """Detect within a region of interest: (x, y, w, h)."""
    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    measured, corners = detect_checker(crop)
    if corners is not None:
        corners[:, 0] += x
        corners[:, 1] += y
    return measured, corners
