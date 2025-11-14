"""
Machining Time & Cost Estimator
Calculates machining time and cost based on geometry, material, and routing
"""
import logging

logger = logging.getLogger(__name__)

# Material Removal Rates (cm³/min) by process
# Based on typical industrial CNC equipment
MRR_BY_PROCESS = {
    "VMC Machining": 15.0,      # Vertical Machining Center
    "CNC Lathe": 20.0,          # CNC Turning Center
    "Boring Mill": 10.0,        # Horizontal Boring Mill (slower but larger)
    "Keyway Machine": 5.0,      # Broaching/Slotting (slower, precise)
    "Wire EDM": 0.8             # Wire Electrical Discharge Machining (very slow, precise)
}

# Hourly machine rates ($/hour) including labor, overhead, tooling
HOURLY_RATE_BY_PROCESS = {
    "VMC Machining": 80.0,
    "CNC Lathe": 75.0,
    "Boring Mill": 90.0,
    "Keyway Machine": 70.0,
    "Wire EDM": 120.0
}

# Material difficulty multipliers (affects cycle time)
MATERIAL_FACTORS = {
    "aluminum": 0.8,        # Easy to machine, fast
    "aluminium": 0.8,
    "brass": 0.9,           # Good machinability
    "copper": 1.0,
    "mild steel": 1.0,      # Baseline
    "carbon steel": 1.1,
    "cold rolled steel": 1.1,
    "stainless": 1.3,       # Harder, work-hardens
    "stainless steel": 1.3,
    "tool steel": 1.4,
    "hardened": 1.5,        # Very hard, slow machining
    "titanium": 1.6,        # Difficult to machine
    "inconel": 1.8          # Extremely difficult
}

def estimate_machining_time_and_cost(desc, material, routings):
    """
    Calculate machining time and cost for each routing
    
    Args:
        desc: Geometry descriptor with volume_cm3, complexity_score, etc.
        material: Material name string
        routings: List of routing names (e.g., ["VMC Machining", "Wire EDM"])
    
    Returns:
        Dictionary with:
            - machining_summary: List of dicts with routing, time, cost
            - total_cost_usd: Total machining cost
    """
    volume = desc.get("volume_cm3", 0)
    complexity = desc.get("complexity_score", 5)
    holes = desc.get("holes_count", 0)
    bbox = desc.get("bounding_box", [0, 0, 0])
    largest_dim = max(bbox) if bbox else 0
    
    # Determine material difficulty factor
    material_lower = material.lower()
    material_factor = 1.0
    for material_key, factor in MATERIAL_FACTORS.items():
        if material_key in material_lower:
            material_factor = factor
            break
    
    # Complexity factor: scale machining time based on feature complexity
    # Complexity ranges 1-10, baseline is 5
    complexity_factor = 1.0 + (complexity - 5) * 0.05  # ±25% range
    
    logger.info(f"Estimation params: volume={volume}cm³, material_factor={material_factor}, complexity_factor={complexity_factor}")
    
    # Calculate for each routing
    time_estimates = []
    total_cost = 0.0
    
    for routing in routings:
        mrr = MRR_BY_PROCESS.get(routing, 10.0)  # Default 10 cm³/min
        rate = HOURLY_RATE_BY_PROCESS.get(routing, 75.0)  # Default $75/hr
        
        # Base machining time from volume removal
        if routing == "Wire EDM":
            # EDM cuts perimeter, not bulk volume
            # Estimate based on surface area or part size
            perimeter_cm = (bbox[0] + bbox[1]) / 5  # Rough perimeter estimate in cm
            machining_time_min = (perimeter_cm * 10) * material_factor  # EDM is slow
        elif routing == "Keyway Machine":
            # Keyway/groove is localized, not full volume
            # Estimate based on number of grooves
            grooves = desc.get("grooves_count", 1)
            machining_time_min = (grooves * 15) * material_factor  # 15 min per groove
        else:
            # Standard machining: volume-based
            machining_time_min = (volume / mrr) * material_factor * complexity_factor
        
        # Add setup and positioning overhead
        setup_overhead_min = 10.0  # 10 minutes per setup
        if largest_dim > 500:
            setup_overhead_min = 20.0  # Larger parts take longer to set up
        
        total_time_min = machining_time_min + setup_overhead_min
        
        # Calculate cost
        machining_cost = (total_time_min / 60) * rate
        total_cost += machining_cost
        
        time_estimates.append({
            "routing": routing,
            "machining_time_min": round(total_time_min, 2),
            "machining_cost": round(machining_cost, 2)
        })
        
        logger.info(f"{routing}: {total_time_min:.1f} min, ${machining_cost:.2f}")
    
    return {
        "machining_summary": time_estimates,
        "total_cost_usd": round(total_cost, 2)
    }
