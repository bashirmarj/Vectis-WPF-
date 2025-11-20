"""
Complete Analysis Situs Log Parser
===================================

Parses the full AS JSON structure into Python objects.
"""

import json
from typing import Dict, List, Optional
from .analysis_situs_models import *


class AnalysisSitusParser:
    """Parse complete Analysis Situs JSON logs."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.raw_data = None
        
    def parse(self) -> ASGroundTruth:
        """Parse complete log file."""
        with open(self.log_path, 'r') as f:
            self.raw_data = json.load(f)
        
        ground_truth = ASGroundTruth()
        
        # Extract metadata
        ground_truth.processing_time = self.raw_data.get('processingTime', 0.0)
        ground_truth.sdk_version = self.raw_data.get('sdkVersion', '')
        ground_truth.sdk_hash = self.raw_data.get('sdkHash', '')
        ground_truth.file_path = self.raw_data.get('file', '')
        ground_truth.unit_scale_factor = self.raw_data.get('unitScaleFactor', 1.0)
        
        # Parse parts
        parts = self.raw_data.get('parts', [])
        if parts:
            body = parts[0].get('bodies', [{}])[0]
            
            # Parse summary
            ground_truth.summary = self._parse_summary(body.get('summary', {}))
            
            # Parse semantic warnings
            semantic_codes = body.get('semanticCodes', {})
            ground_truth.semantic_warnings = self._parse_warnings(semantic_codes)
            
            # Parse features
            features = body.get('features', {})
            ground_truth.holes = self._parse_holes(features)
            ground_truth.pockets = self._parse_pockets(features)
            ground_truth.shoulders = self._parse_shoulders(features)
            ground_truth.shafts = self._parse_shafts(features)
            ground_truth.threads = self._parse_threads(features)
            
            # Parse face data
            ground_truth.free_flat_faces = self._parse_free_flat(body.get('freeFlat', []))
            ground_truth.milled_faces = self._parse_milled_faces(body.get('milledFaces3Axes', []))
            
            # Parse accessibility
            ground_truth.side_milling_axes = self._parse_side_milling(body.get('sideMilling', []))
            ground_truth.end_milling_axes = self._parse_end_milling(body.get('endMilling', []))
        
        # Parse fillet chains (top level)
        ground_truth.fillets = self._parse_fillet_chains(
            self.raw_data.get('filletChains', [])
        )
        
        # Parse chamfer chains (top level)
        ground_truth.chamfers = self._parse_chamfer_chains(
            self.raw_data.get('chamferChains', [])
        )
        
        return ground_truth
    
    # ========================================================================
    # HOLE PARSING
    # ========================================================================
    
    def _parse_holes(self, features: Dict) -> List[ASHole]:
        """Parse all hole features."""
        holes = []
        
        holes_data = features.get('holes', [])
        for hole_data in holes_data:
            hole = ASHole(
                face_ids=hole_data.get('faceIds', []),
                fully_recognized=hole_data.get('fullyRecognized', False),
                total_depth=hole_data.get('totalDepth', 0.0),
                hole_type=hole_data.get('type', 'cylindricalHole'),
                is_through=hole_data.get('isThrough', False),
                axis=hole_data.get('axis')
            )
            
            # Parse bores
            bores_data = hole_data.get('bores', [])
            for bore_data in bores_data:
                bore = ASBore(
                    face_ids=bore_data.get('faceIds', []),
                    diameter=bore_data.get('diameter', 0.0),
                    depth=bore_data.get('depth', 0.0),
                    bore_type=bore_data.get('type', 'cylindrical')
                )
                hole.bores.append(bore)
            
            # Parse conical bottom
            bottom_data = hole_data.get('bottom')
            if bottom_data and bottom_data.get('type') == 'conical':
                hole.conical_bottom = ASConicalBottom(
                    face_ids=bottom_data.get('faceIds', []),
                    angle=bottom_data.get('angle')
                )
            
            # Parse counterbores
            counterbores_data = hole_data.get('counterbores', [])
            for cb_data in counterbores_data:
                cb = ASCounterbore(
                    face_ids=cb_data.get('faceIds', []),
                    diameter=cb_data.get('diameter', 0.0),
                    depth=cb_data.get('depth', 0.0)
                )
                hole.counterbores.append(cb)
            
            # Parse countersinks
            countersinks_data = hole_data.get('countersinks', [])
            for cs_data in countersinks_data:
                cs = ASCountersink(
                    face_ids=cs_data.get('faceIds', []),
                    diameter=cs_data.get('diameter', 0.0),
                    angle=cs_data.get('angle', 90.0)
                )
                hole.countersinks.append(cs)
            
            holes.append(hole)
        
        return holes
    
    # ========================================================================
    # POCKET PARSING
    # ========================================================================
    
    def _parse_pockets(self, features: Dict) -> List[ASPocket]:
        """Parse pocket/prismatic milling features."""
        pockets = []
        
        pockets_data = features.get('prismaticMilling', [])
        for pocket_data in pockets_data:
            pocket = ASPocket(
                face_ids=pocket_data.get('faceIds', []),
                depth=pocket_data.get('depth'),
                area=pocket_data.get('area'),
                volume=pocket_data.get('volume'),
                is_closed=pocket_data.get('isClosed', True),
                is_through=pocket_data.get('isThrough', False)
            )
            
            # Parse configurations
            configs_data = pocket_data.get('configurations', [])
            for config_data in configs_data:
                config = ASPrismaticConfig(
                    axis=config_data.get('axis', [0, 0, 1]),
                    depth=config_data.get('depth', 0.0),
                    face_ids=config_data.get('faceIds', []),
                    bottom_faces=config_data.get('bottomFaces', []),
                    wall_faces=config_data.get('wallFaces', []),
                    entry_type=config_data.get('entryType')
                )
                pocket.configurations.append(config)
            
            pockets.append(pocket)
        
        return pockets
    
    # ========================================================================
    # FILLET & CHAMFER PARSING
    # ========================================================================
    
    def _parse_fillet_chains(self, chains: List[Dict]) -> List[ASFilletChain]:
        """Parse fillet chains."""
        fillets = []
        
        for chain in chains:
            fillet = ASFilletChain(
                face_ids=chain.get('faceIds', []),
                radius=chain.get('radius', 0.0),
                total_length=chain.get('totalLength', 0.0),
                contour_length=chain.get('contourLength', 0.0),
                convex=chain.get('convex', False),
                variable_radius=chain.get('variableRadius', False),
                min_radius=chain.get('minRadius'),
                max_radius=chain.get('maxRadius')
            )
            fillets.append(fillet)
        
        return fillets
    
    def _parse_chamfer_chains(self, chains: List[Dict]) -> List[ASChamferChain]:
        """Parse chamfer chains."""
        chamfers = []
        
        for chain in chains:
            chamfer = ASChamferChain(
                face_ids=chain.get('faceIds', []),
                angle=chain.get('angle', 45.0),
                distance=chain.get('distance', 0.0),
                total_length=chain.get('totalLength', 0.0),
                chamfer_type=chain.get('type', 'linear')
            )
            chamfers.append(chamfer)
        
        return chamfers
    
    # ========================================================================
    # OTHER FEATURES
    # ========================================================================
    
    def _parse_shoulders(self, features: Dict) -> List[ASShoulder]:
        """Parse shoulder/step features."""
        shoulders = []
        
        shoulders_data = features.get('shoulders', [])
        for shoulder_data in shoulders_data:
            shoulder = ASShoulder(
                face_ids=shoulder_data.get('faceIds', []),
                height=shoulder_data.get('height', 0.0),
                axis=shoulder_data.get('axis', [0, 0, 1]),
                shoulder_type=shoulder_data.get('type', 'step')
            )
            shoulders.append(shoulder)
        
        return shoulders
    
    def _parse_shafts(self, features: Dict) -> List[ASShaft]:
        """Parse shaft/boss features."""
        shafts = []
        
        shafts_data = features.get('shafts', [])
        for shaft_data in shafts_data:
            shaft = ASShaft(
                face_ids=shaft_data.get('faceIds', []),
                diameter=shaft_data.get('diameter', 0.0),
                length=shaft_data.get('length', 0.0),
                axis=shaft_data.get('axis', [0, 0, 1])
            )
            shafts.append(shaft)
        
        return shafts
    
    def _parse_threads(self, features: Dict) -> List[ASThread]:
        """Parse thread features."""
        threads = []
        
        threads_data = features.get('threads', [])
        for thread_data in threads_data:
            thread = ASThread(
                face_ids=thread_data.get('faceIds', []),
                major_diameter=thread_data.get('majorDiameter', 0.0),
                pitch=thread_data.get('pitch', 0.0),
                length=thread_data.get('length', 0.0),
                is_internal=thread_data.get('isInternal', True)
            )
            threads.append(thread)
        
        return threads
    
    # ========================================================================
    # FACE & ACCESSIBILITY DATA
    # ========================================================================
    
    def _parse_free_flat(self, free_flat_data: List[Dict]) -> List[ASFreeFlatFace]:
        """Parse free flat faces."""
        faces = []
        
        for face_data in free_flat_data:
            face = ASFreeFlatFace(
                face_id=face_data.get('faceId', 0),
                area=face_data.get('area', 0.0),
                normal=face_data.get('normal', [0, 0, 1]),
                accessible=face_data.get('accessible', True)
            )
            faces.append(face)
        
        return faces
    
    def _parse_milled_faces(self, milled_data: List[int]) -> List[ASMilledFace]:
        """Parse milled faces (currently just IDs in AS log)."""
        faces = []
        
        for face_id in milled_data:
            face = ASMilledFace(
                face_id=face_id,
                surface_type="unknown",
                area=0.0,
                accessible_from=[],
                requires_3_axis=True
            )
            faces.append(face)
        
        return faces
    
    def _parse_side_milling(self, side_data: List[Dict]) -> List[ASSideMillingAxis]:
        """Parse side milling accessibility."""
        axes = []
        
        for axis_data in side_data:
            axis = ASSideMillingAxis(
                axis=axis_data.get('axis', [1, 0, 0]),
                face_ids=axis_data.get('faceIds', [])
            )
            axes.append(axis)
        
        return axes
    
    def _parse_end_milling(self, end_data: List[Dict]) -> List[ASEndMillingAxis]:
        """Parse end milling accessibility."""
        axes = []
        
        for axis_data in end_data:
            axis = ASEndMillingAxis(
                axis=axis_data.get('axis', [0, 0, 1]),
                face_ids=axis_data.get('faceIds', [])
            )
            axes.append(axis)
        
        return axes
    
    # ========================================================================
    # WARNINGS & SUMMARY
    # ========================================================================
    
    def _parse_warnings(self, semantic_codes: Dict) -> List[ASSemanticWarning]:
        """Parse semantic warnings."""
        warnings = []
        
        warnings_data = semantic_codes.get('warnings', [])
        for warning_data in warnings_data:
            warning = ASSemanticWarning(
                code=warning_data.get('code', 0),
                label=warning_data.get('label', ''),
                face_ids=warning_data.get('faceIds', []),
                vertex_ids=warning_data.get('vertexIds', []),
                edge_ids=warning_data.get('edgeIds', []),
                message=warning_data.get('message'),
                severity='warning'
            )
            warnings.append(warning)
        
        return warnings
    
    def _parse_summary(self, summary_data: Dict) -> ASSummary:
        """Parse part summary."""
        return ASSummary(
            num_vertices=summary_data.get('numVertices', 0),
            num_edges=summary_data.get('numEdges', 0),
            num_faces=summary_data.get('numFaces', 0),
            num_3d_milled_faces=summary_data.get('num3dMilledFaces', 0),
            num_inaccessible_faces=summary_data.get('numInaccessibleFaces', 0),
            num_warnings=summary_data.get('numWarnings', 0),
            bounding_box=summary_data.get('boundingBox')
        )


def load_ground_truth(log_path: str) -> ASGroundTruth:
    """Load and parse Analysis Situs log."""
    parser = AnalysisSitusParser(log_path)
    return parser.parse()
