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
from .graph_builder import AAGGraphBuilder
from .machining_configuration_detector import MachiningConfigurationDetector
from .tool_accessibility_analyzer import ToolAccessibilityAnalyzer

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
    "MachiningConfigurationDetector",
    "ToolAccessibilityAnalyzer",
]

__author__ = "Vectis Machining"
__email__ = "engineering@vectismachining.com"
__license__ = "MIT"
__version__ = "2.0.0"
