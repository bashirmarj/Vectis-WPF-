"""Validate AAG Pattern Matching installation"""

import sys

def validate():
    print("Validating installation...")
    
    # Test imports
    try:
        from aag_pattern_engine import AAGPatternMatcher
        print("✓ AAGPatternMatcher imported")
    except ImportError as e:
        print(f"✗ Failed to import AAGPatternMatcher: {e}")
        return False
    
    try:
        from aag_pattern_engine import FeatureValidator
        print("✓ FeatureValidator imported")
    except ImportError as e:
        print(f"✗ Failed to import FeatureValidator: {e}")
        return False
    
    # Test recognizers
    try:
        from aag_pattern_engine.recognizers import HoleRecognizer
        print("✓ HoleRecognizer imported")
    except ImportError as e:
        print(f"✗ Failed to import HoleRecognizer: {e}")
        return False
    
    # Test initialization
    try:
        matcher = AAGPatternMatcher()
        print("✓ AAGPatternMatcher initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False
    
    print("\n✅ All validation checks passed!")
    print("\nSystem is ready for production use.")
    return True

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
