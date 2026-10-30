"""ColourCal — colour correction from a ColorChecker chart."""
from .detect import detect_checker
from .correct import colour_matrix, apply_correction
