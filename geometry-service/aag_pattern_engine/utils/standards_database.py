"""Manufacturing standards database (ISO, ANSI, DIN)"""

from typing import Optional, Tuple

# ISO Metric Thread Standards (M1 - M24)
ISO_METRIC_THREADS = {
    # diameter (m): pitch (m)
    0.001: 0.00025,    # M1 × 0.25
    0.0012: 0.00025,   # M1.2 × 0.25
    0.0016: 0.00035,   # M1.6 × 0.35
    0.002: 0.0004,     # M2 × 0.4
    0.0025: 0.00045,   # M2.5 × 0.45
    0.003: 0.0005,     # M3 × 0.5
    0.004: 0.0007,     # M4 × 0.7
    0.005: 0.0008,     # M5 × 0.8
    0.006: 0.001,      # M6 × 1.0
    0.008: 0.00125,    # M8 × 1.25
    0.010: 0.0015,     # M10 × 1.5
    0.012: 0.00175,    # M12 × 1.75
    0.016: 0.002,      # M16 × 2.0
    0.020: 0.0025,     # M20 × 2.5
    0.024: 0.003,      # M24 × 3.0
}

# ISO 3601 O-Ring Standards (AS568)
O_RING_STANDARDS = {
    # inner_diameter (m): (outer_diameter (m), cross_section (m))
    0.0015: (0.0021, 0.0012),   # -001
    0.0020: (0.0028, 0.0016),   # -002
    0.0025: (0.0034, 0.0020),   # -003
    0.0030: (0.0041, 0.0024),   # -004
    0.0040: (0.0055, 0.0032),   # -005
    0.0050: (0.0069, 0.0040),   # -006
    0.0060: (0.0083, 0.0048),   # -007
    0.0070: (0.0097, 0.0056),   # -008
}

# ISO 4762 Counterbore Standards (Socket Head Cap Screws)
COUNTERBORE_STANDARDS = {
    # thread_diameter (m): (counterbore_diameter (m), counterbore_depth (m))
    0.003: (0.006, 0.003),      # M3
    0.004: (0.0075, 0.0035),    # M4
    0.005: (0.009, 0.004),      # M5
    0.006: (0.011, 0.0045),     # M6
    0.00635: (0.0127, 0.00508), # 1/4"
    0.00794: (0.0159, 0.00635), # 5/16"
    0.00952: (0.0191, 0.00762), # 3/8"
    0.01270: (0.0254, 0.01016), # 1/2"
}

# DIN 6885 / ISO R773 Keyway Standards
KEYWAY_STANDARDS = {
    # shaft_diameter (m): (keyway_width (m), keyway_depth (m))
    0.006: (0.002, 0.001),      # 6mm shaft
    0.008: (0.003, 0.0015),     # 8mm shaft
    0.010: (0.004, 0.002),      # 10mm shaft
    0.012: (0.004, 0.002),      # 12mm shaft
    0.016: (0.005, 0.003),      # 16mm shaft
    0.020: (0.006, 0.004),      # 20mm shaft
    0.025: (0.008, 0.005),      # 25mm shaft
    0.030: (0.010, 0.006),      # 30mm shaft
}

def get_thread_standard(diameter: float, tolerance: float = 0.0005) -> Optional[Tuple[float, str]]:
    """
    Get standard thread pitch for diameter
    
    Args:
        diameter: Thread major diameter in meters
        tolerance: Matching tolerance in meters (default 0.5mm)
    
    Returns:
        Tuple of (pitch, standard_name) or None if not found
    """
    for std_diameter, pitch in ISO_METRIC_THREADS.items():
        if abs(diameter - std_diameter) < tolerance:
            return (pitch, f"M{int(std_diameter * 1000)}×{pitch * 1000:.2f}")
    return None

def get_o_ring_standard(inner_diameter: float, tolerance: float = 0.0005) -> Optional[Tuple[float, float, str]]:
    """
    Get standard O-ring dimensions
    
    Args:
        inner_diameter: O-ring inner diameter in meters
        tolerance: Matching tolerance in meters
    
    Returns:
        Tuple of (outer_diameter, cross_section, standard_name) or None
    """
    for std_id, (od, cs) in O_RING_STANDARDS.items():
        if abs(inner_diameter - std_id) < tolerance:
            size_code = f"AS568-{list(O_RING_STANDARDS.keys()).index(std_id) + 1:03d}"
            return (od, cs, size_code)
    return None
