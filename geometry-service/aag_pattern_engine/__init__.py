"""
AAG Pattern Matching System
Production-grade machining feature recognition for CNC manufacturing

Complete MFCAD++ coverage (25 classes) with industrial validation
"""

from .version import __version__
from .pattern_matcher import (
    AAGPatternMatcher,
    RecognitionResult,
    RecognitionStatus,
    PartType,
    RecognitionMetrics
)
from .feature_validator import (
    FeatureValidator,
    ValidationReport,
    ValidationLevel,
    ValidationIssue,
    FeatureConflict,
    QualityMetrics
)
from .graph_builder import (
    AAGGraphBuilder,
    GraphNode,
    GraphEdge,
    SurfaceType,
    Vexity
)

__all__ = [
    "__version__",
    "AAGPatternMatcher",
    "RecognitionResult",
    "RecognitionStatus",
    "PartType",
    "RecognitionMetrics",
    "FeatureValidator",
    "ValidationReport",
    "ValidationLevel",
    "ValidationIssue",
    "FeatureConflict",
    "QualityMetrics",
    "AAGGraphBuilder",
    "GraphNode",
    "GraphEdge",
    "SurfaceType",
    "Vexity",
]

__author__ = "Your Company"
__email__ = "engineering@yourcompany.com"
__license__ = "MIT"
__version__ = "1.0.0"
