"""
production_recognizer_integration.py
====================================

Integration module for production feature recognizer with vectismachining app.py

This module bridges the new production recognizer with your existing
crash_free_geometric_recognizer.py system.
"""

from production_feature_recognizer import ProductionFeatureRecognizer
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ProductionRecognizerIntegration:
    """
    Integration wrapper for production feature recognizer.

    Provides compatibility layer with existing vectismachining format.
    """

    def __init__(self):
        self.recognizer = ProductionFeatureRecognizer()

    def recognize_features(self, step_file_path: str) -> Dict:
        """
        Main entry point - compatible with existing API.

        Returns format compatible with crash_free_geometric_recognizer.py:
        {
            'status': 'success' | 'failed',  # ✅ CRITICAL: Required by app.py
            'instances': [...],  # List of feature dictionaries
            'num_features_detected': int,
            'overall_confidence': float,
            'part_family': str,
            'recognition_method': str
        }
        """
        try:
            # Run production recognizer
            result = self.recognizer.recognize(step_file_path)

            if not result['success']:
                logger.error(f"Production recognizer failed: {result['errors']}")
                return self._empty_result()

            # Convert to vectismachining format
            instances = self._convert_to_instances(result)

            # ✅ FIX: Add 'status' key that app.py expects
            return {
                'status': 'success',  # ✅ CRITICAL: app.py checks for this
                'instances': instances,
                'num_features_detected': len(instances),
                'overall_confidence': result['confidence'],
                'avg_confidence': result['confidence'],  # ✅ Alias for compatibility
                'part_family': result['part_family'],
                'rotation_axis': result.get('rotation_axis'),
                'recognition_method': 'production_feature_recognizer',
                'processing_time_seconds': result['processing_time_seconds'],
                'inference_time_sec': result['processing_time_seconds'],  # ✅ Alias for compatibility
                'feature_summary': self._generate_feature_summary(instances),
                'num_faces_analyzed': result.get('num_faces_analyzed', 0),
                'raw_result': result  # Keep original for reference
            }

        except Exception as e:
            logger.error(f"Integration error: {e}", exc_info=True)
            return self._empty_result()

    def _convert_to_instances(self, result: Dict) -> List[Dict]:
        """
        Convert production recognizer output to instances format.
        
        Maps production recognizer feature types to vectismachining format:
        - hole → hole
        - pocket → pocket
        - slot → slot
        - turning_feature → boss/step/groove (based on subtype)
        """
        instances = []

        # Convert holes
        for hole in result.get('holes', []):
            instance = {
                'type': 'hole',
                'subtype': hole.get('subtype', 'through'),
                'confidence': hole.get('confidence', 0.0),
                'parameters': {
                    'diameter': hole.get('diameter', 0.0),
                    'depth': hole.get('depth', 0.0),
                    'location': hole.get('location', [0, 0, 0]),
                    'axis': hole.get('axis', [0, 0, 1])
                },
                'face_indices': hole.get('face_indices', []),
                'detection_method': hole.get('detection_method', 'production')
            }
            instances.append(instance)

        # Convert pockets
        for pocket in result.get('pockets', []):
            instance = {
                'type': 'pocket',
                'subtype': pocket.get('subtype', 'rectangular'),
                'confidence': pocket.get('confidence', 0.0),
                'parameters': pocket.get('dimensions', {}),
                'location': pocket.get('location', [0, 0, 0]),
                'face_indices': pocket.get('face_indices', []),
                'detection_method': pocket.get('detection_method', 'production')
            }
            instances.append(instance)

        # Convert slots
        for slot in result.get('slots', []):
            instance = {
                'type': 'slot',
                'subtype': slot.get('subtype', 'rectangular'),
                'confidence': slot.get('confidence', 0.0),
                'parameters': slot.get('dimensions', {}),
                'location': slot.get('location', [0, 0, 0]),
                'direction': slot.get('direction', [1, 0, 0]),
                'is_through': slot.get('is_through', False),
                'face_indices': slot.get('face_indices', []),
                'detection_method': slot.get('detection_method', 'production')
            }
            instances.append(instance)

        # Convert turning features
        for turning in result.get('turning_features', []):
            subtype = turning.get('subtype', 'unknown')

            # Map turning features to appropriate types
            if subtype == 'base_cylinder':
                feature_type = 'boss'
            elif subtype == 'step':
                feature_type = 'step'
            elif subtype == 'groove':
                feature_type = 'groove'
            elif subtype == 'taper':
                feature_type = 'taper'
            else:
                feature_type = 'turning_feature'

            instance = {
                'type': feature_type,
                'subtype': subtype,
                'confidence': turning.get('confidence', 0.0),
                'parameters': turning.get('dimensions', {}),
                'location': turning.get('location', [0, 0, 0]),
                'axis': turning.get('axis', [1, 0, 0]),
                'face_indices': turning.get('face_indices', []),
                'detection_method': turning.get('detection_method', 'production')
            }
            instances.append(instance)

        return instances

    def _generate_feature_summary(self, instances: List[Dict]) -> Dict[str, int]:
        """
        Generate feature summary for logging and debugging.
        
        Returns count of each feature type/subtype combination.
        """
        summary = {}
        for instance in instances:
            feature_type = instance.get('type', 'unknown')
            subtype = instance.get('subtype')
            
            if subtype:
                key = f"{feature_type}_{subtype}"
            else:
                key = feature_type
                
            summary[key] = summary.get(key, 0) + 1
        
        return summary

    def _empty_result(self) -> Dict:
        """Return empty result on failure"""
        return {
            'status': 'failed',  # ✅ CRITICAL: app.py checks for this
            'instances': [],
            'num_features_detected': 0,
            'overall_confidence': 0.0,
            'avg_confidence': 0.0,
            'part_family': 'unknown',
            'recognition_method': 'production_feature_recognizer',
            'processing_time_seconds': 0.0,
            'inference_time_sec': 0.0,
            'feature_summary': {},
            'num_faces_analyzed': 0,
            'error': 'Feature recognition failed'
        }


def integrate_with_app():
    """
    Returns example integration code for app.py
    """

    example = """
# ============================================================================
# ADD TO app.py - INTEGRATION WITH PRODUCTION RECOGNIZER
# ============================================================================

# At the top of app.py, add import:
from production_recognizer_integration import ProductionRecognizerIntegration

# In your recognition endpoint (around line 2000-2100), REPLACE:

# OLD CODE:
# recognizer = ExtendedCrashFreeRecognizer()
# feature_result = recognizer.recognize_features(step_file_path)

# NEW CODE:
try:
    # Try production recognizer first
    production_recognizer = ProductionRecognizerIntegration()
    feature_result = production_recognizer.recognize_features(step_file_path)

    logger.info(f"✅ Production recognizer: {feature_result['num_features_detected']} features")
    logger.info(f"   Part family: {feature_result['part_family']}")
    logger.info(f"   Confidence: {feature_result['overall_confidence']*100:.1f}%")

    # If production recognizer fails or low confidence, fallback
    if feature_result.get('status') != 'success' or feature_result['num_features_detected'] == 0 or feature_result['overall_confidence'] < 0.3:
        logger.warning("⚠️ Production recognizer low confidence, using fallback...")

        # Fallback to original recognizer
        from crash_free_geometric_recognizer import FlaskCrashFreeRecognizer
        fallback_recognizer = FlaskCrashFreeRecognizer()
        feature_result = fallback_recognizer.recognize_features(step_file_path)

except Exception as e:
    logger.error(f"❌ Production recognizer error: {e}")

    # Fallback to original recognizer
    logger.info("🔄 Falling back to original recognizer...")
    from crash_free_geometric_recognizer import FlaskCrashFreeRecognizer
    fallback_recognizer = FlaskCrashFreeRecognizer()
    feature_result = fallback_recognizer.recognize_features(step_file_path)

# Continue with existing code (machining translation, etc.)
# feature_result now has the same format, so no other changes needed!
"""

    return example


if __name__ == "__main__":
    print(integrate_with_app())
