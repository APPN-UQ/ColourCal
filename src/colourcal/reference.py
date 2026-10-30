"""Reference sRGB values for the X-Rite / Calibrite ColorChecker Classic (24 patches).

Values are D50-adapted sRGB (0-255), the widely used BabelColor average measurements.
Patch order follows the cv2.mcc layout (row-major, top-left to bottom-right).
"""
import numpy as np

# fmt: off
COLORCHECKER_SRGB = np.array([
    [115, 82, 68],    # dark skin
    [194, 150, 130],  # light skin
    [98, 122, 157],   # blue sky
    [87, 108, 67],    # foliage
    [133, 128, 177],  # blue flower
    [103, 189, 170],  # bluish green
    [214, 126, 44],   # orange
    [80, 91, 166],    # purplish blue
    [193, 90, 99],    # moderate red
    [94, 60, 108],    # purple
    [157, 188, 64],   # yellow green
    [224, 163, 46],   # orange yellow
    [56, 61, 150],    # blue
    [70, 148, 73],    # green
    [175, 54, 60],    # red
    [231, 199, 31],   # yellow
    [187, 86, 149],   # magenta
    [8, 133, 161],    # cyan
    [243, 243, 242],  # white
    [200, 200, 200],  # neutral 8
    [160, 160, 160],  # neutral 6.5
    [122, 122, 121],  # neutral 5
    [85, 85, 85],     # neutral 3.5
    [52, 52, 52],     # black
], dtype=np.float64)
# fmt: on
