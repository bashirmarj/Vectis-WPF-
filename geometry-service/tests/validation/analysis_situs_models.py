"""
Complete Analysis Situs Data Models
===================================

Mirrors the complete Analysis Situs JSON structure.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ============================================================================
# HOLE FEATURES
# ============================================================================

@dataclass
class ASBore:
    """Bore component of a hole."""
    face_ids: List[int]
    diameter: float
    depth: float
    bore_type: str = "cylindrical"  # cylindrical, counterbore, etc.


@dataclass
class ASConicalBottom:
    """Conical bottom of a hole (drill point)."""
    face_ids: List[int]
    angle: Optional[float] = None  # Cone angle in degrees


@dataclass
class ASCounterbore:
    """Counterbore feature within a hole."""
    face_ids: List[int]
    diameter: float
    depth: float


@dataclass
class ASCountersink:
    """Countersink feature within a hole."""
    face_ids: List[int]
    diameter: float
    angle: float


@dataclass
class ASHole:
    """Complete hole feature."""
    face_ids: List[int]
    fully_recognized: bool
    total_depth: float
    bores: List[ASBore] = field(default_factory=list)
    
    # Optional components
    conical_bottom: Optional[ASConicalBottom] = None
    counterbores: List[ASCounterbore] = field(default_factory=list)
    countersinks: List[ASCountersink] = field(default_factory=list)
    
    # Metadata
    hole_type: str = "cylindricalHole"
    is_through: bool = False
    axis: Optional[List[float]] = None


# ============================================================================
# POCKET FEATURES
# ============================================================================

@dataclass
class ASPrismaticConfig:
    """Prismatic milling configuration (pocket)."""
    axis: List[float]  # Milling axis direction
    depth: float
    face_ids: List[int]
    
    # Optional details
    bottom_faces: List[int] = field(default_factory=list)
    wall_faces: List[int] = field(default_factory=list)
    entry_type: Optional[str] = None  # plunge, ramp, etc.


@dataclass
class ASPocket:
    """Pocket/prismatic milling feature."""
    face_ids: List[int]
    configurations: List[ASPrismaticConfig] = field(default_factory=list)
    
    # Geometry
    depth: Optional[float] = None
    area: Optional[float] = None
    volume: Optional[float] = None
    
    # Type
    pocket_type: str = "prismaticMilling"
    is_closed: bool = True
    is_through: bool = False


# ============================================================================
# FILLET & CHAMFER FEATURES
# ============================================================================

@dataclass
class ASFilletChain:
    """Fillet chain (connected blend surfaces)."""
    face_ids: List[int]
    radius: float
    total_length: float
    contour_length: float
    convex: bool  # True = external round, False = internal fillet
    
    # Optional
    variable_radius: bool = False
    min_radius: Optional[float] = None
    max_radius: Optional[float] = None


@dataclass
class ASChamferChain:
    """Chamfer chain."""
    face_ids: List[int]
    angle: float  # Degrees
    distance: float  # mm
    total_length: float
    
    chamfer_type: str = "linear"  # linear, circular


# ============================================================================
# OTHER FEATURES
# ============================================================================

@dataclass
class ASShoulder:
    """Shoulder/step feature."""
    face_ids: List[int]
    height: float
    axis: List[float]
    
    shoulder_type: str = "step"


@dataclass
class ASShaft:
    """Shaft/boss feature."""
    face_ids: List[int]
    diameter: float
    length: float
    axis: List[float]


@dataclass
class ASThread:
    """Thread feature."""
    face_ids: List[int]
    major_diameter: float
    pitch: float
    length: float
    is_internal: bool


@dataclass
class ASFreeFlatFace:
    """Free flat surface (simple planar face)."""
    face_id: int
    area: float
    normal: List[float]
    accessible: bool = True


# ============================================================================
# ACCESSIBILITY & MACHINING DATA
# ============================================================================

@dataclass
class ASSideMillingAxis:
    """Side milling accessibility axis."""
    axis: List[float]  # Direction vector
    face_ids: List[int]  # Faces accessible from this axis


@dataclass
class ASEndMillingAxis:
    """End milling accessibility axis."""
    axis: List[float]
    face_ids: List[int]


@dataclass
class ASMilledFace:
    """Individual milled face with machining info."""
    face_id: int
    surface_type: str  # plane, cylinder, cone, etc.
    area: float
    accessible_from: List[str]  # Axis directions
    requires_3_axis: bool
    requires_4_axis: bool = False
    requires_5_axis: bool = False


# ============================================================================
# WARNINGS & VALIDATION
# ============================================================================

@dataclass
class ASSemanticWarning:
    """DFM/semantic warning."""
    code: int  # e.g., 2201
    label: str  # e.g., "cncCode_PartBody_Warn_ImpossibleCorner"
    face_ids: List[int]
    vertex_ids: List[int] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    message: Optional[str] = None
    severity: str = "warning"  # warning, error, info


# ============================================================================
# SUMMARY & METADATA
# ============================================================================

@dataclass
class ASSummary:
    """Part summary statistics."""
    num_vertices: int = 0
    num_edges: int = 0
    num_faces: int = 0
    num_3d_milled_faces: int = 0
    num_inaccessible_faces: int = 0
    num_warnings: int = 0
    bounding_box: Optional[Dict[str, float]] = None


# ============================================================================
# COMPLETE GROUND TRUTH
# ============================================================================

@dataclass
class ASGroundTruth:
    """
    Complete Analysis Situs recognition output.
    
    This mirrors the entire AS JSON structure.
    """
    # Primary features
    holes: List[ASHole] = field(default_factory=list)
    pockets: List[ASPocket] = field(default_factory=list)
    fillets: List[ASFilletChain] = field(default_factory=list)
    chamfers: List[ASChamferChain] = field(default_factory=list)
    
    # Additional features
    shoulders: List[ASShoulder] = field(default_factory=list)
    shafts: List[ASShaft] = field(default_factory=list)
    threads: List[ASThread] = field(default_factory=list)
    
    # Surface data
    free_flat_faces: List[ASFreeFlatFace] = field(default_factory=list)
    milled_faces: List[ASMilledFace] = field(default_factory=list)
    
    # Accessibility
    side_milling_axes: List[ASSideMillingAxis] = field(default_factory=list)
    end_milling_axes: List[ASEndMillingAxis] = field(default_factory=list)
    
    # Warnings & validation
    semantic_warnings: List[ASSemanticWarning] = field(default_factory=list)
    
    # Summary
    summary: ASSummary = field(default_factory=ASSummary)
    
    # Metadata
    processing_time: float = 0.0
    sdk_version: str = ""
    sdk_hash: str = ""
    file_path: str = ""
    unit_scale_factor: float = 1.0
