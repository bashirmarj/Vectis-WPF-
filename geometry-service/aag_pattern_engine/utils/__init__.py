"""Utility functions package"""

from .geometric_utils import (
    compute_distance,
    compute_angle,
    is_perpendicular,
    is_parallel,
    normalize_vector,
    compute_cross_product,
    compute_dot_product
)
from .mesh_utils import (
    tessellate_shape,
    compute_vertex_normals
)
from .step_utils import (
    load_step_file,
    load_step_with_metadata
)
from .standards_database import (
    ISO_METRIC_THREADS,
    O_RING_STANDARDS,
    COUNTERBORE_STANDARDS,
    KEYWAY_STANDARDS,
    get_thread_standard,
    get_o_ring_standard
)
from .logging_config import setup_logging
from .vexity_helpers import (
    is_depression_edge,
    is_protrusion_edge,
    is_vertical_wall_transition,
    is_smooth_blend,
    requires_strict_concave
)

__all__ = [
    "compute_distance",
    "compute_angle",
    "is_perpendicular",
    "is_parallel",
    "normalize_vector",
    "compute_cross_product",
    "compute_dot_product",
    "tessellate_shape",
    "compute_vertex_normals",
    "load_step_file",
    "load_step_with_metadata",
    "ISO_METRIC_THREADS",
    "O_RING_STANDARDS",
    "COUNTERBORE_STANDARDS",
    "KEYWAY_STANDARDS",
    "get_thread_standard",
    "get_o_ring_standard",
    "setup_logging",
    "is_depression_edge",
    "is_protrusion_edge",
    "is_vertical_wall_transition",
    "is_smooth_blend",
    "requires_strict_concave",
]
