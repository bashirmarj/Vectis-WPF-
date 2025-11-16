"""
AAG Pattern Matching System
Production-grade machining feature recognition for CNC manufacturing

Complete MFCAD++ coverage with industrial validation
"""

from .version import __version__
from .pattern_matcher import AAGPatternMatcher, RecognitionResult
from .feature_validator import FeatureValidator, ValidationReport
from .graph_builder import AAGGraphBuilder

# Main exports
__all__ = [
    "__version__",
    "AAGPatternMatcher",
    "RecognitionResult",
    "FeatureValidator",
    "ValidationReport",
    "AAGGraphBuilder",
]

# Package metadata
__author__ = "Your Company"
__email__ = "engineering@yourcompany.com"
__license__ = "MIT"
