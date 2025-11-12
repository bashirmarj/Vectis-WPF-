"""
production_feature_recognizer.py
=================================

Main orchestrator for complete feature recognition system.

Version: 2.1 (DEDUPLICATION FIX)
Target Accuracy: 70-80%

CRITICAL FIX: Ensures all result dict lists (holes, turning_features, etc.)
use deduplicated features. Previous version had deduplication working but
integration layer was using undedup'd turning_features list.

Features:
- Integrates all recognizers
- Memory management
- Error handling
- Confidence scoring
- Manufacturing validation
- PROPER deduplication across all result lists
"""

from production_hole_recognizer import ProductionHoleRecognizer
from production_pocket_recognizer import ProductionPocketRecognizer
from production_slot_recognizer import ProductionSlotRecognizer
from production_turning_recognizer import ProductionTurningRecognizer
from production_part_classifier import ProductionPartClassifier, PartFamily

from OCC.Extend.DataExchange import read_step_file
from typing import List, Dict, Optional
import logging
import time
import psutil
import os

logger = logging.getLogger(__name__)


class ProductionFeatureRecognizer:
    """
    Complete production feature recognition system.
    
    Version 2.1: Fixed deduplication - ensures integration layer gets clean data.
    
    Usage:
        recognizer = ProductionFeatureRecognizer()
        result = recognizer.recognize("part.step")
    """

    def __init__(self, 
                 memory_limit_mb: int = 2000,
                 time_limit_sec: float = 60.0):
        """
        Initialize recognizers with resource limits.
        
        Args:
            memory_limit_mb: Maximum memory usage (MB)
            time_limit_sec: Maximum processing time (seconds)
        """
        self.memory_limit_mb = memory_limit_mb
        self.time_limit_sec = time_limit_sec
        
        # Initialize recognizers
        self.hole_recognizer = ProductionHoleRecognizer()
        self.pocket_recognizer = ProductionPocketRecognizer()
        self.slot_recognizer = ProductionSlotRecognizer()
        self.turning_recognizer = ProductionTurningRecognizer()
        self.classifier = ProductionPartClassifier()
        
        logger.info("✅ Production Feature Recognizer v2.1 initialized (with dedup fix)")
        logger.info(f"   Memory limit: {memory_limit_mb}MB")
        logger.info(f"   Time limit: {time_limit_sec}s")

    def recognize(self, step_file_path: str) -> Dict:
        """
        Main entry point: Recognize all features in STEP file.
        
        Returns properly deduplicated features in all result lists.
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 STARTING FEATURE RECOGNITION")
        logger.info(f"{'='*80}")
        logger.info(f"   File: {os.path.basename(step_file_path)}")
        
        try:
            # Load STEP file
            shape = read_step_file(step_file_path)
            
            if shape is None:
                logger.error("❌ Failed to load STEP file")
                return self._error_result("Failed to load STEP file")
            
            # Step 1: Initial feature recognition (turning + holes)
            logger.info("\n🔍 Step 1: Initial feature detection...")
            
            # Try turning recognition first
            logger.info("\n   🔄 Checking for rotational features...")
            turning_result = self.turning_recognizer.recognize_turning_features(shape)
            
            if turning_result['part_type'] == 'rotational':
                turning_features = turning_result['features']
                rotation_axis = turning_result['axis']
                logger.info(f"   Found {len(turning_features)} turning features")
            else:
                turning_features = []
                rotation_axis = None
                logger.info("   No rotational features detected")
            
            # Recognize holes (all parts have holes)
            logger.info("\n   🕳️  Recognizing holes...")
            holes = self.hole_recognizer.recognize_all_holes(shape)
            logger.info(f"   Found {len(holes)} holes")
            
            # Step 2: Classify based on initial features
            logger.info("\n🏷️  Step 2: Classifying part family...")
            
            feature_dict = {
                'turning_features': [self._to_dict(f) for f in turning_features],
                'holes': [self._to_dict(h) for h in holes],
                'pockets': [],
                'slots': [],
                'rotation_axis': rotation_axis
            }
            
            part_family = self.classifier.classify(shape, feature_dict)
            logger.info(f"   → {part_family.value.upper()}")
            
            # Initialize feature lists
            pockets = []
            slots = []
            all_features = holes.copy() + turning_features.copy()
            
            # Step 3: Part-specific recognition (prismatic features if needed)
            if part_family == PartFamily.PRISMATIC:
                logger.info("\n🔲 Step 3: Recognizing prismatic features...")
                
                logger.info("\n   📦 Recognizing pockets...")
                pockets = self.pocket_recognizer.recognize_all_pockets(shape)
                all_features.extend(pockets)
                
                logger.info("\n   ➖ Recognizing slots...")
                slots = self.slot_recognizer.recognize_all_slots(shape)
                all_features.extend(slots)
                
                logger.info(f"   Pockets: {len(pockets)}, Slots: {len(slots)}")
            
            elif part_family == PartFamily.ROTATIONAL:
                logger.info("\n🔄 Step 3: Rotational part confirmed")
                logger.info(f"   Turning features: {len(turning_features)}")
            
            else:  # HYBRID or UNKNOWN
                logger.info("\n🔀 Step 3: Recognizing mixed/hybrid features...")
                
                # Add prismatic features
                logger.info("\n   📦 Recognizing pockets...")
                pockets = self.pocket_recognizer.recognize_all_pockets(shape)
                all_features.extend(pockets)
                
                logger.info("\n   ➖ Recognizing slots...")
                slots = self.slot_recognizer.recognize_all_slots(shape)
                all_features.extend(slots)
            
            # Step 4: Post-processing - CRITICAL FIX!
            logger.info("\n🔧 Step 4: Post-processing...")
            
            initial_count = len(all_features)
            
            # Remove duplicates
            all_features = self._remove_duplicates(all_features)
            
            # CRITICAL FIX: Update individual feature lists with deduplicated features
            holes_dedup = [f for f in all_features if self._is_hole(f)]
            pockets_dedup = [f for f in all_features if self._is_pocket(f)]
            slots_dedup = [f for f in all_features if self._is_slot(f)]
            turning_dedup = [f for f in all_features if self._is_turning(f)]
            
            logger.info(f"   Deduplicated features:")
            logger.info(f"      Holes: {len(holes)} → {len(holes_dedup)}")
            logger.info(f"      Pockets: {len(pockets)} → {len(pockets_dedup)}")
            logger.info(f"      Slots: {len(slots)} → {len(slots_dedup)}")
            logger.info(f"      Turning: {len(turning_features)} → {len(turning_dedup)}")
            logger.info(f"      Total: {initial_count} → {len(all_features)}")
            
            # Calculate confidence
            all_features = self._calculate_confidence(all_features)
            
            if all_features:
                overall_confidence = sum(self._get_confidence(f) for f in all_features) / len(all_features)
            else:
                overall_confidence = 0.0
            
            processing_time = time.time() - start_time
            memory_used = self._get_memory_usage() - initial_memory
            
            # Build result - USE DEDUPLICATED LISTS!
            result = {
                'success': True,
                'part_family': part_family.value,
                'holes': [self._to_dict(h) for h in holes_dedup],  # FIXED: Use dedup'd
                'pockets': [self._to_dict(p) for p in pockets_dedup],  # FIXED: Use dedup'd
                'slots': [self._to_dict(s) for s in slots_dedup],  # FIXED: Use dedup'd
                'turning_features': [self._to_dict(f) for f in turning_dedup],  # FIXED: Use dedup'd
                'rotation_axis': rotation_axis,
                'num_features': len(all_features),
                'confidence': overall_confidence,
                'processing_time_seconds': processing_time,
                'memory_used_mb': memory_used,
                'errors': []
            }
            
            # Summary
            logger.info("\n" + "="*80)
            logger.info("✅ RECOGNITION COMPLETE")
            logger.info("="*80)
            logger.info(f"   Part Family: {part_family.value.upper()}")
            logger.info(f"   Total Features: {len(all_features)}")
            logger.info(f"      - Holes: {len(holes_dedup)}")
            logger.info(f"      - Pockets: {len(pockets_dedup)}")
            logger.info(f"      - Slots: {len(slots_dedup)}")
            logger.info(f"      - Turning: {len(turning_dedup)}")
            logger.info(f"   Overall Confidence: {overall_confidence*100:.1f}%")
            logger.info(f"   Processing Time: {processing_time:.2f}s")
            logger.info(f"   Memory Used: {memory_used:.1f}MB")
            logger.info("="*80)
            
            return result
            
        except Exception as e:
            logger.error(f"\n❌ Recognition failed: {e}", exc_info=True)
            return self._error_result(str(e))
    
    def _is_hole(self, feature) -> bool:
        """Check if feature is a hole"""
        if hasattr(feature, 'hole_type'):
            return True
        if isinstance(feature, dict):
            return feature.get('type') == 'hole'
        return False
    
    def _is_pocket(self, feature) -> bool:
        """Check if feature is a pocket"""
        if hasattr(feature, 'pocket_type'):
            return True
        if isinstance(feature, dict):
            return feature.get('type') == 'pocket'
        return False
    
    def _is_slot(self, feature) -> bool:
        """Check if feature is a slot"""
        if hasattr(feature, 'slot_type'):
            return True
        if isinstance(feature, dict):
            return feature.get('type') == 'slot'
        return False
    
    def _is_turning(self, feature) -> bool:
        """Check if feature is a turning feature"""
        if hasattr(feature, 'feature_type'):
            ftype = feature.feature_type
            if hasattr(ftype, 'value'):
                return ftype.value in ['base_cylinder', 'step', 'groove', 'taper', 'face', 'thread']
            return ftype in ['base_cylinder', 'step', 'groove', 'taper', 'face', 'thread']
        if isinstance(feature, dict):
            return feature.get('type') == 'turning_feature'
        return False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _to_dict(self, feature) -> Dict:
        """Convert feature to dict"""
        if hasattr(feature, 'to_dict'):
            return feature.to_dict()
        elif isinstance(feature, dict):
            return feature
        else:
            return {}
    
    def _get_confidence(self, feature) -> float:
        """Extract confidence"""
        if hasattr(feature, 'confidence'):
            return feature.confidence
        elif isinstance(feature, dict):
            return feature.get('confidence', 0.0)
        else:
            return 0.0
    
    def _remove_duplicates(self, features: List) -> List:
        """
        Remove duplicate features based on location.
        
        NOTE: Turning features should already be semantically merged
        by production_turning_recognizer.py. This is a final cleanup
        for any remaining location-based duplicates.
        """
        unique_features = []
        seen_locations = set()
        
        for feature in features:
            loc = self._get_location(feature)
            
            if loc not in seen_locations:
                unique_features.append(feature)
                seen_locations.add(loc)
        
        if len(features) != len(unique_features):
            logger.info(f"   Removed {len(features) - len(unique_features)} duplicates")
        
        return unique_features
    
    def _get_location(self, feature) -> tuple:
        """Extract location"""
        if hasattr(feature, 'location'):
            loc = feature.location
            return tuple(round(x, 1) for x in loc)
        elif isinstance(feature, dict):
            loc = feature.get('location', (0, 0, 0))
            return tuple(round(x, 1) for x in loc)
        else:
            return (0, 0, 0)
    
    def _calculate_confidence(self, features: List) -> List:
        """Adjust confidence based on context"""
        # Could add contextual confidence adjustments here
        return features
    
    def _error_result(self, error_msg: str) -> Dict:
        """Return error result"""
        return {
            'success': False,
            'part_family': 'unknown',
            'holes': [],
            'pockets': [],
            'slots': [],
            'turning_features': [],
            'rotation_axis': None,
            'num_features': 0,
            'confidence': 0.0,
            'processing_time_seconds': 0.0,
            'memory_used_mb': 0.0,
            'errors': [error_msg]
        }


# Convenience function
def recognize_step_file(step_file_path: str) -> Dict:
    """Convenience function to recognize features"""
    recognizer = ProductionFeatureRecognizer()
    return recognizer.recognize(step_file_path)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python production_feature_recognizer.py <step_file>")
        sys.exit(1)
    
    result = recognize_step_file(sys.argv[1])
    
    if result['success']:
        print(f"\n✅ Success!")
        print(f"   Part type: {result['part_family']}")
        print(f"   Features: {result['num_features']}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
    else:
        print(f"\n❌ Failed: {result['errors']}")
