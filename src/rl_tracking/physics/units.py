"""
Unit conversion constants and utilities.

TrackML standard:
- Positions (x, y, z): millimeters (mm)
- Momentum (px, py, pz): GeV/c
- Magnetic field: Tesla

This codebase uses:
- Positions: centimeters (cm) for consistency with layer_info.csv
- Conversion factor: 1 cm = 10 mm
"""

# Conversion constants
MM_TO_CM = 0.1  # 1 mm = 0.1 cm
CM_TO_MM = 10.0  # 1 cm = 10 mm

# TrackML data is in mm, we convert to cm
def trackml_to_cm(value_mm):
    """Convert TrackML position from mm to cm."""
    return value_mm * MM_TO_CM

def cm_to_trackml(value_cm):
    """Convert position from cm to TrackML mm."""
    return value_cm * CM_TO_MM

