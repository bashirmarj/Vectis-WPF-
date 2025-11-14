"""
Industrial Manufacturing Routing Selector
Implements logic similar to Mastercam/Siemens NX for process selection
"""
import logging

logger = logging.getLogger(__name__)

def select_routings_industrial(desc, material):
    """
    Select manufacturing routings based on industrial standards
    
    Args:
        desc: Geometry descriptor dictionary with keys:
            - bounding_box: [width_mm, height_mm, depth_mm]
            - volume_cm3: Part volume
            - is_cylindrical: Boolean
            - has_flat_surfaces: Boolean
            - holes_count: Number of holes detected
            - grooves_count: Number of grooves detected
            - complexity_score: 1-10 complexity rating
            - tolerance: Tolerance requirement in mm (optional)
        material: Material name string
    
    Returns:
        Dictionary with:
            - recommended_routings: List of process names in order
            - reasoning: List of reasoning strings explaining selections
    """
    routings = []
    reasons = []
    
    # Extract geometry features
    bbox = desc.get("bounding_box", [0, 0, 0])
    largest_dim = max(bbox) if bbox else 0
    volume = desc.get("volume_cm3", 0)
    complexity = desc.get("complexity_score", 5)
    holes = desc.get("holes_count", 0)
    grooves = desc.get("grooves_count", 0)
    tolerance = desc.get("tolerance", 0.05)  # Default 0.05mm
    
    is_cylindrical = desc.get("is_cylindrical", False)
    has_flats = desc.get("has_flat_surfaces", False)
    
    logger.info(f"Routing selection: cylindrical={is_cylindrical}, largest_dim={largest_dim}mm, holes={holes}, complexity={complexity}")
    
    # ============= CYLINDRICAL PART LOGIC =============
    if is_cylindrical:
        # Primary turning operation
        if largest_dim < 500:  # Parts under 500mm
            routings.append("CNC Lathe")
            reasons.append("Cylindrical geometry under 500mm — CNC lathe preferred for precision turning.")
        else:  # Large cylindrical parts
            routings.append("Boring Mill")
            reasons.append("Large cylindrical part (>500mm) — boring mill required for capacity.")
        
        # Secondary operations for holes
        if holes > 0:
            if holes >= 4 or complexity >= 7:
                routings.append("VMC Machining")
                reasons.append(f"{holes} hole(s) detected — VMC for secondary drilling/tapping operations.")
            else:
                # Small hole count can be done on lathe
                reasons.append(f"{holes} hole(s) can be drilled on the lathe during turning cycle.")
        
        # Keyway/groove operations
        if grooves > 0:
            routings.append("Keyway Machine")
            reasons.append(f"{grooves} groove(s)/keyway(s) detected — dedicated broaching/slotting required.")
    
    # ============= PRISMATIC PART LOGIC =============
    elif has_flats:
        # Primary milling operation
        if largest_dim < 1000:  # Parts under 1 meter
            routings.append("VMC Machining")
            reasons.append("Prismatic geometry within 1000mm — suitable for vertical machining center.")
        else:  # Large flat parts
            routings.append("Boring Mill")
            reasons.append("Large prismatic part (>1000mm) — horizontal boring mill for large bed capacity.")
        
        # Hole drilling
        if holes > 0:
            if "VMC Machining" not in routings:
                routings.append("VMC Machining")
                reasons.append(f"{holes} hole(s) detected — VMC for drilling/boring operations.")
            else:
                reasons.append(f"{holes} hole(s) will be machined in the primary VMC cycle.")
        
        # Complex features requiring EDM
        if complexity >= 8:
            routings.append("Wire EDM")
            reasons.append("High complexity (score ≥8) — Wire EDM for intricate features and sharp internal corners.")
    
    # ============= MATERIAL-BASED ROUTING =============
    material_lower = material.lower()
    
    # Hard materials requiring EDM
    if "stainless" in material_lower or "hardened" in material_lower or "tool steel" in material_lower:
        if "Wire EDM" not in routings:
            routings.append("Wire EDM")
            reasons.append("Hard/abrasive material (stainless/hardened steel) — EDM recommended for precision finishing and tool life.")
    
    # Aluminum - note faster machining
    if "aluminum" in material_lower or "aluminium" in material_lower:
        reasons.append("Aluminum material — high-speed machining strategies will reduce cycle time.")
    
    # Brass/Copper - excellent machinability
    if "brass" in material_lower or "copper" in material_lower:
        reasons.append("Brass/copper material — excellent machinability, fast feeds possible.")
    
    # ============= TOLERANCE-DRIVEN FINISHING =============
    if tolerance < 0.01:  # Tight tolerance < 10 microns
        if "Wire EDM" not in routings:
            routings.append("Wire EDM")
            reasons.append(f"Tight tolerance requirement (±{tolerance}mm) — Wire EDM finishing for precision.")
        else:
            reasons.append(f"Tight tolerance (±{tolerance}mm) will be achieved through EDM finishing passes.")
    elif tolerance < 0.02:  # Medium-tight tolerance
        reasons.append(f"Tolerance requirement (±{tolerance}mm) — finish machining passes required.")
    
    # ============= SURFACE FINISH REQUIREMENTS =============
    # Note: Surface finish could be passed in descriptor if needed
    # For now, infer from material and tolerance
    if tolerance < 0.02 or "stainless" in material_lower:
        reasons.append("Surface finish consideration — final passes with reduced feed/depth for Ra < 1.6μm.")
    
    # ============= FALLBACK FOR UNCLASSIFIED PARTS =============
    if len(routings) == 0:
        routings.append("VMC Machining")
        reasons.append("General machining — VMC selected as versatile default for unclassified geometry.")
    
    # Remove duplicate routings while preserving order
    ordered_routings = []
    for r in routings:
        if r not in ordered_routings:
            ordered_routings.append(r)
    
    logger.info(f"Selected routings: {ordered_routings}")
    
    return {
        "recommended_routings": ordered_routings,
        "reasoning": reasons
    }
