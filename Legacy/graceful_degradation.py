"""
Graceful degradation handlers for feature recognition
Provides fallback processing tiers when full B-Rep processing fails
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ProcessingTier(Enum):
    """Processing tier with associated confidence level"""
    TIER_1_BREP = "tier_1_brep"          # Full B-Rep + rule-based recognition (confidence: 0.95)
    TIER_2_MESH = "tier_2_mesh"          # Mesh-based processing (confidence: 0.75)
    TIER_3_POINT_CLOUD = "tier_3_point"  # Point cloud fallback (confidence: 0.60)
    TIER_4_BASIC = "tier_4_basic"        # Basic mesh only (confidence: 0.40)


class GracefulDegradation:
    """
    Handles graceful degradation through processing tiers.
    
    Tier 1 (confidence 0.95): Full B-Rep processing with rule-based recognition
    Tier 2 (confidence 0.75): Mesh-based feature detection
    Tier 3 (confidence 0.60): Point cloud processing
    Tier 4 (confidence 0.40): Basic mesh data only
    """
    
    # Confidence multipliers for each tier
    tier_confidence_multipliers = {
        ProcessingTier.TIER_1_BREP: 0.95,
        ProcessingTier.TIER_2_MESH: 0.75,
        ProcessingTier.TIER_3_POINT_CLOUD: 0.60,
        ProcessingTier.TIER_4_BASIC: 0.40
    }
    
    @staticmethod
    def select_tier(
        feature_recognition_available: bool,
        quality_score: float,
        circuit_breaker_state: str
    ) -> ProcessingTier:
        """
        Select appropriate processing tier based on system state.
        
        Args:
            feature_recognition_available: Whether feature recognizer is loaded
            quality_score: CAD file quality score (0.0 to 1.0)
            circuit_breaker_state: Circuit breaker state ("CLOSED", "OPEN", "HALF_OPEN")
            
        Returns:
            ProcessingTier enum value
        """
        
        # If feature recognition unavailable or circuit breaker open, skip to Tier 2
        if not feature_recognition_available:
            logger.info("⚠️ Feature recognition not available, selecting Tier 2 (mesh-based)")
            return ProcessingTier.TIER_2_MESH
        
        if circuit_breaker_state == "OPEN":
            logger.warning("⚠️ Circuit breaker OPEN, selecting Tier 2 (mesh-based)")
            return ProcessingTier.TIER_2_MESH
        
        # If quality score is very low, go directly to Tier 2
        if quality_score < 0.5:
            logger.warning(f"⚠️ Low quality score ({quality_score:.2f}), selecting Tier 2 (mesh-based)")
            return ProcessingTier.TIER_2_MESH
        
        # If circuit breaker is HALF_OPEN, be cautious - use Tier 2
        if circuit_breaker_state == "HALF_OPEN":
            logger.info("⚠️ Circuit breaker testing recovery, selecting Tier 2 (mesh-based)")
            return ProcessingTier.TIER_2_MESH
        
        # Default: Use Tier 1 (full B-Rep + rule-based recognition)
        logger.info("✅ System healthy, selecting Tier 1 (B-Rep + rule-based recognition)")
        return ProcessingTier.TIER_1_BREP
    
    @staticmethod
    def classify_confidence(confidence: float) -> str:
        """
        Classify recognition confidence level.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            'high' (>0.9), 'medium' (0.7-0.9), or 'low' (<0.7)
        """
        if confidence >= 0.9:
            return 'high'
        elif confidence >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    @staticmethod
    def tier_1_brep_processing(
        feature_recognizer,
        step_file_path: str,
        shape
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """
        Tier 1: Full B-Rep processing with rule-based recognition.
        
        Returns:
            (ml_features, confidence, tier)
        """
        try:
            logger.info("Attempting Tier 1: Full B-Rep + rule-based processing")
            
            result = feature_recognizer.recognize_features(step_file_path)
            
            if result and result.get('status') == 'success':
                ml_features = {
                    'instances': result.get('instances', []),
                    'num_features_detected': result.get('num_features_detected', 0),
                    'num_faces_analyzed': result.get('num_faces_analyzed', 0),
                    'inference_time_sec': result.get('inference_time_sec', 0),
                    'confidence_score': result.get('avg_confidence', 0.0),
                    'recognition_method': 'rule_based',
                    'feature_summary': result.get('feature_summary', {}),
                    'processing_tier': ProcessingTier.TIER_1_BREP.value,
                    'confidence_level': 'high'
                }
                
                logger.info(f"✅ Tier 1 successful: {ml_features['num_features_detected']} features detected")
                return ml_features, 0.95, ProcessingTier.TIER_1_BREP.value
            else:
                logger.warning(f"⚠️ Tier 1 failed: {result.get('error') if result else 'No result'}")
                return None, 0.0, ProcessingTier.TIER_1_BREP.value
                
        except Exception as e:
            logger.warning(f"⚠️ Tier 1 exception: {str(e)}")
            return None, 0.0, ProcessingTier.TIER_1_BREP.value
    
    @staticmethod
    def tier_2_mesh_processing(
        mesh_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """
        Tier 2: Mesh-based feature heuristics.
        
        Fallback when rule-based recognition fails but mesh generation succeeded.
        Uses geometric heuristics to detect basic features.
        
        Returns:
            (ml_features, confidence, tier)
        """
        try:
            logger.info("Attempting Tier 2: Mesh-based heuristic processing")
            
            # Simple heuristics from mesh data
            feature_instances = []
            
            # Detect cylindrical surfaces (potential holes)
            cylindrical_faces = [
                i for i, cls in enumerate(mesh_data.get('face_classifications', []))
                if cls == 'cylindrical'
            ]
            
            if cylindrical_faces:
                feature_instances.append({
                    'feature_type': 'potential_hole',
                    'face_ids': cylindrical_faces,
                    'bottom_faces': [],
                    'confidence': 0.75
                })
            
            # Detect planar pockets (groups of coplanar faces below stock)
            planar_faces = [
                i for i, cls in enumerate(mesh_data.get('face_classifications', []))
                if cls == 'planar'
            ]
            
            if len(planar_faces) > 2:
                feature_instances.append({
                    'feature_type': 'potential_pocket',
                    'face_ids': planar_faces[:5],  # Limit to first 5
                    'bottom_faces': [],
                    'confidence': 0.70
                })
            
            ml_features = {
                'feature_instances': feature_instances,
                'num_features_detected': len(feature_instances),
                'num_faces_analyzed': len(mesh_data.get('face_classifications', [])),
                'inference_time_sec': 0,
                'recognition_method': 'mesh_heuristics',
                'processing_tier': ProcessingTier.TIER_2_MESH.value,
                'confidence_level': 'medium',
                'warning': 'Reduced accuracy: mesh-based heuristics only'
            }
            
            logger.info(f"✅ Tier 2 successful: {len(feature_instances)} potential features detected")
            return ml_features, 0.75, ProcessingTier.TIER_2_MESH.value
            
        except Exception as e:
            logger.warning(f"⚠️ Tier 2 exception: {str(e)}")
            return None, 0.0, ProcessingTier.TIER_2_MESH.value
    
    @staticmethod
    def tier_3_point_cloud_processing(
        mesh_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """
        Tier 3: Point cloud-based basic detection.
        
        Last resort before returning mesh-only data.
        
        Returns:
            (ml_features, confidence, tier)
        """
        try:
            logger.info("Attempting Tier 3: Point cloud processing")
            
            ml_features = {
                'feature_instances': [],
                'num_features_detected': 0,
                'num_faces_analyzed': len(mesh_data.get('face_classifications', [])),
                'inference_time_sec': 0,
                'recognition_method': 'point_cloud_basic',
                'processing_tier': ProcessingTier.TIER_3_POINT_CLOUD.value,
                'confidence_level': 'low',
                'warning': 'Very limited accuracy: point cloud analysis only'
            }
            
            logger.info("✅ Tier 3 complete: No features detected, returning mesh data only")
            return ml_features, 0.60, ProcessingTier.TIER_3_POINT_CLOUD.value
            
        except Exception as e:
            logger.warning(f"⚠️ Tier 3 exception: {str(e)}")
            return None, 0.0, ProcessingTier.TIER_3_POINT_CLOUD.value
    
    @staticmethod
    def tier_4_basic_mesh(
        mesh_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """
        Tier 4: Return mesh data only without feature recognition.
        
        Returns:
            (ml_features, confidence, tier)
        """
        logger.info("Using Tier 4: Basic mesh only (no feature recognition)")
        
        ml_features = {
            'feature_instances': [],
            'num_features_detected': 0,
            'num_faces_analyzed': 0,
            'inference_time_sec': 0,
            'recognition_method': 'none',
            'processing_tier': ProcessingTier.TIER_4_BASIC.value,
            'confidence_level': 'none',
            'warning': 'Feature recognition failed, mesh visualization only'
        }
        
        return ml_features, 0.40, ProcessingTier.TIER_4_BASIC.value
    
    @staticmethod
    def process_with_fallback(
        feature_recognizer,
        step_file_path: str,
        shape,
        mesh_data: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """
        Execute processing with graceful degradation through tiers.
        
        Args:
            feature_recognizer: Feature recognizer instance
            step_file_path: Path to STEP file
            shape: OpenCascade shape object
            mesh_data: Generated mesh data
            
        Returns:
            (ml_features, confidence, processing_tier)
        """
        
        # Try Tier 1: Full B-Rep + rule-based recognition
        ml_features, confidence, tier = GracefulDegradation.tier_1_brep_processing(
            feature_recognizer, step_file_path, shape
        )
        
        if ml_features:
            return ml_features, confidence, tier
        
        # Tier 1 failed, try Tier 2: Mesh-based
        logger.warning("⚠️ Tier 1 failed, falling back to Tier 2 (mesh-based)")
        ml_features, confidence, tier = GracefulDegradation.tier_2_mesh_processing(mesh_data)
        
        if ml_features:
            return ml_features, confidence, tier
        
        # Tier 2 failed, try Tier 3: Point cloud
        logger.warning("⚠️ Tier 2 failed, falling back to Tier 3 (point cloud)")
        ml_features, confidence, tier = GracefulDegradation.tier_3_point_cloud_processing(mesh_data)
        
        if ml_features:
            return ml_features, confidence, tier
        
        # All recognition failed, return Tier 4: mesh only
        logger.error("❌ All feature recognition tiers failed, returning mesh only")
        ml_features, confidence, tier = GracefulDegradation.tier_4_basic_mesh(mesh_data)
        
        return ml_features, confidence, tier
