"""
Edge-Based Detection + Taxonomy Integration
Combines enhanced edge extraction, edge-based feature detection, and taxonomy mapping

This is the main entry point that app.py will use.
"""

import logging
import time
from typing import Dict, List

from enhanced_edge_extractor import extract_enhanced_edges
from edge_feature_detector import detect_features_from_edges
from enhanced_recognizer_wrapper import TaxonomyMapper

logger = logging.getLogger(__name__)


class EdgeBasedFeatureRecognizer:
    """
    Complete edge-based feature recognition with taxonomy integration
    
    Pipeline:
    1. Extract enhanced edge data (geometry + topology)
    2. Detect features from edges
    3. Map to taxonomy
    4. Return standardized result
    """
    
    def __init__(self):
        self.taxonomy_mapper = TaxonomyMapper()
    
    def recognize_features(self, shape) -> Dict:
        """
        Main recognition pipeline
        
        Args:
            shape: OCC TopoDS_Shape
        
        Returns:
            Dict with instances, statistics, taxonomy info
        """
        correlation_id = f"edge_{int(time.time() * 1000)}"
        
        try:
            start_time = time.time()
            
            logger.info(f"\n[{correlation_id}] " + "="*70)
            logger.info(f"[{correlation_id}] 🚀 EDGE-BASED FEATURE RECOGNITION")
            logger.info(f"[{correlation_id}] " + "="*70)
            
            # Step 1: Extract enhanced edge data
            logger.info(f"\n[{correlation_id}] Step 1: Enhanced edge extraction...")
            edge_result = extract_enhanced_edges(shape)
            
            if edge_result.get('error'):
                return self._error_response(correlation_id, edge_result['error'])
            
            enhanced_edges = edge_result['edges']
            edge_stats = edge_result['statistics']
            
            logger.info(f"[{correlation_id}]    ✅ {edge_stats['total_edges']} edges extracted")
            logger.info(f"[{correlation_id}]    📊 Feature boundaries: {edge_stats['feature_boundaries']}")
            logger.info(f"[{correlation_id}]    📊 By type: {dict(edge_stats['by_type'])}")
            logger.info(f"[{correlation_id}]    📊 By hint: {dict(edge_stats['by_hint'])}")
            
            # Step 2: Detect features from edges
            logger.info(f"\n[{correlation_id}] Step 2: Edge-based feature detection...")
            detection_result = detect_features_from_edges(enhanced_edges)
            
            if detection_result.get('error'):
                return self._error_response(correlation_id, detection_result['error'])
            
            raw_features = detection_result['features']
            detection_stats = detection_result['statistics']
            
            logger.info(f"[{correlation_id}]    ✅ {detection_stats['total_features']} features detected")
            logger.info(f"[{correlation_id}]    📊 By category: {dict(detection_stats['by_category'])}")
            logger.info(f"[{correlation_id}]    📊 Avg confidence: {detection_stats['avg_confidence']:.2f}")
            
            # Step 3: Enhance with taxonomy
            logger.info(f"\n[{correlation_id}] Step 3: Taxonomy mapping...")
            enhanced_features = self.taxonomy_mapper.enhance_feature_list(raw_features)
            taxonomy_summary = self.taxonomy_mapper.generate_taxonomy_summary(enhanced_features)
            
            logger.info(f"[{correlation_id}]    ✅ Taxonomy enhanced")
            logger.info(f"[{correlation_id}]    📊 Categories: {taxonomy_summary['by_category']}")
            logger.info(f"[{correlation_id}]    📊 Types: {len(taxonomy_summary['by_type'])} unique")
            
            # Build result
            elapsed = time.time() - start_time
            
            result = {
                'status': 'success',
                'correlation_id': correlation_id,
                'instances': enhanced_features,
                'num_features_detected': len(enhanced_features),
                'num_faces_analyzed': 0,  # Edge-based doesn't analyze faces
                'num_edges_analyzed': len(enhanced_edges),
                'avg_confidence': detection_stats.get('avg_confidence', 0.0),
                'confidence_score': detection_stats.get('avg_confidence', 0.0),
                'inference_time_sec': elapsed,
                'recognition_method': 'edge_based_enhanced',
                'feature_summary': dict(detection_stats.get('by_type', {})),
                'taxonomy_summary': taxonomy_summary,
                'edge_statistics': edge_stats
            }
            
            logger.info(f"\n[{correlation_id}] " + "="*70)
            logger.info(f"[{correlation_id}] ✅ SUCCESS: {result['num_features_detected']} features in {elapsed:.2f}s")
            logger.info(f"[{correlation_id}] " + "="*70 + "\n")
            
            return result
        
        except Exception as e:
            logger.error(f"[{correlation_id}] ❌ Edge-based recognition failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._error_response(correlation_id, str(e))
    
    def _error_response(self, correlation_id: str, error: str) -> Dict:
        """Generate error response"""
        return {
            'status': 'failed',
            'correlation_id': correlation_id,
            'error': error,
            'instances': [],
            'num_features_detected': 0,
            'confidence_score': 0.0,
            'recognition_method': 'edge_based_enhanced'
        }


# Flask wrapper for app.py compatibility
class FlaskEdgeRecognizer:
    """
    Flask-compatible wrapper for edge-based feature recognition
    Drop-in replacement for existing recognizers
    """
    
    def __init__(self, time_limit: float = 30.0):
        self.time_limit = time_limit
        self.recognizer = EdgeBasedFeatureRecognizer()
    
    def recognize_features(self, shape) -> Dict:
        """
        Recognize features from shape
        Compatible with existing app.py interface
        """
        return self.recognizer.recognize_features(shape)
