"""
Analysis Situs-like Local Viewer

Web-based CAD viewer with feature recognition using trame + VTK.
Opens in browser at http://localhost:8080
"""

import sys
import os
import logging
import threading
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# VTK imports
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

# Trame imports
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, vtk as vtk_widgets

# Add geometry-service to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from local_viewer.tessellator import tessellate_to_vtk, get_face_center
from OCP.STEPControl import STEPControl_Reader

# Feature colors (RGB 0-1)
FEATURE_COLORS = {
    'blind_hole': (1.0, 0.2, 0.2),
    'through_hole': (0.8, 0.1, 0.1),
    'counterbored_hole': (1.0, 0.4, 0.2),
    'pocket': (0.2, 0.4, 1.0),
    'slot': (0.3, 0.6, 1.0),
    'fillet': (0.2, 0.8, 0.3),
    'chamfer': (0.5, 0.8, 0.2),
    'boss': (0.8, 0.6, 0.2),
    'step': (0.6, 0.4, 0.8),
    'island': (0.9, 0.5, 0.7),
    'unrecognized': (0.7, 0.7, 0.7),
}


class FeatureViewerApp:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        # State variables
        self.state.status = "Ready - Load a STEP file to begin"
        self.state.file_path = ""
        self.state.num_faces = 0
        self.state.num_features = 0
        self.state.processing_time = 0
        self.state.log_messages = []
        self.state.features = []
        self.state.selected_face_id = None
        self.state.selected_face_info = {}
        self.state.is_analyzing = False

        # VTK pipeline
        self.polydata = None
        self.shape = None
        self.renderer = None
        self.render_window = None
        self.render_window_interactor = None
        self.actor = None
        self.mapper = None
        self.lut = None

        # Setup UI first - VTK will be initialized when needed
        self._setup_ui()

    def _create_lookup_table(self):
        """Create VTK lookup table for face colors"""
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        # Default gray
        for i in range(256):
            lut.SetTableValue(i, 0.7, 0.7, 0.7, 1.0)
        lut.Build()
        return lut

    def _log(self, message):
        """Add message to log"""
        with self.state:
            self.state.log_messages = self.state.log_messages + [message]
            # Keep last 100 messages
            if len(self.state.log_messages) > 100:
                self.state.log_messages = self.state.log_messages[-100:]

    def load_step_file(self):
        """Open file dialog and load STEP file"""
        # Use tkinter file dialog (runs in subprocess to avoid blocking)
        def select_file():
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            filepath = filedialog.askopenfilename(
                title="Select STEP File",
                filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
                initialdir=os.path.expanduser("~")
            )
            root.destroy()
            if filepath:
                self._load_step(filepath)

        threading.Thread(target=select_file, daemon=True).start()

    def _load_step(self, filepath):
        """Load STEP file and tessellate"""
        self._log(f"Loading: {filepath}")
        self.state.status = "Loading STEP file..."

        try:
            # Read STEP
            reader = STEPControl_Reader()
            status = reader.ReadFile(filepath)
            if status != 1:
                raise ValueError(f"Failed to read STEP file (status={status})")

            reader.TransferRoots()
            self.shape = reader.OneShape()

            if self.shape.IsNull():
                raise ValueError("Failed to extract shape from STEP file")

            # Tessellate
            self._log("Tessellating...")
            self.polydata = tessellate_to_vtk(self.shape, linear_deflection=0.005)

            # Update VTK pipeline
            if self.actor:
                self.renderer.RemoveActor(self.actor)

            self.mapper = vtkPolyDataMapper()
            self.mapper.SetInputData(self.polydata)
            self.mapper.SetLookupTable(self.lut)
            self.mapper.SetScalarModeToUseCellData()
            self.mapper.SetColorModeToMapScalars()
            self.mapper.SetScalarRange(0, self.polydata.GetNumberOfCells())

            self.actor = vtkActor()
            self.actor.SetMapper(self.mapper)
            self.renderer.AddActor(self.actor)
            self.renderer.ResetCamera()

            # Update state
            num_faces = int(self.polydata.GetCellData().GetArray("face_id").GetRange()[1]) + 1
            with self.state:
                self.state.file_path = filepath
                self.state.num_faces = num_faces
                self.state.num_features = 0
                self.state.features = []
                self.state.status = f"Loaded: {num_faces} faces"

            self._log(f"Loaded successfully: {num_faces} faces, {self.polydata.GetNumberOfCells()} triangles")
            self.ctrl.view_update()

        except Exception as e:
            self._log(f"ERROR: {str(e)}")
            self.state.status = f"Error loading file"

    def run_analysis(self):
        """Run AAG feature recognition"""
        if not self.shape:
            self._log("No shape loaded - load a STEP file first")
            return

        self._log("Starting AAG feature recognition...")
        self.state.status = "Analyzing..."
        self.state.is_analyzing = True

        def analyze():
            try:
                import time
                start = time.time()

                # Import AAG pattern matcher
                from aag_pattern_engine.pattern_matcher import AAGPatternMatcher

                matcher = AAGPatternMatcher(tolerance=1e-6)
                result = matcher.recognize_all_features(self.shape)

                elapsed = time.time() - start

                # Extract features
                features = result.get_all_features() if hasattr(result, 'get_all_features') else []

                # Color faces by feature type
                self._color_by_features(features)

                with self.state:
                    self.state.features = features
                    self.state.num_features = len(features)
                    self.state.processing_time = round(elapsed, 2)
                    self.state.status = f"Analysis complete: {len(features)} features in {elapsed:.1f}s"
                    self.state.is_analyzing = False

                self._log(f"Analysis complete: {len(features)} features detected")
                for feat in features[:10]:  # Log first 10
                    ftype = feat.get('type', 'unknown')
                    conf = feat.get('confidence', 0)
                    self._log(f"  - {ftype} (confidence: {conf:.2f})")

                self.ctrl.view_update()

            except Exception as e:
                self._log(f"Analysis ERROR: {str(e)}")
                import traceback
                self._log(traceback.format_exc())
                self.state.is_analyzing = False
                self.state.status = "Analysis failed"

        threading.Thread(target=analyze, daemon=True).start()

    def _color_by_features(self, features):
        """Update lookup table to color faces by feature type"""
        if not self.polydata:
            return

        # Build face_id -> feature_type map
        face_to_feature = {}
        for feature in features:
            ftype = feature.get('type', 'unrecognized')
            for fid in feature.get('face_ids', []):
                face_to_feature[fid] = ftype

        # Update lookup table
        face_id_array = self.polydata.GetCellData().GetArray("face_id")
        for cell_id in range(self.polydata.GetNumberOfCells()):
            face_id = int(face_id_array.GetValue(cell_id))
            ftype = face_to_feature.get(face_id, 'unrecognized')
            color = FEATURE_COLORS.get(ftype, FEATURE_COLORS['unrecognized'])
            self.lut.SetTableValue(cell_id, color[0], color[1], color[2], 1.0)

        self.lut.Modified()
        self.mapper.Modified()

    def _setup_ui(self):
        """Setup trame UI layout"""
        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("AAG Feature Recognition Viewer")

            # Toolbar
            with layout.toolbar as toolbar:
                toolbar.density = "compact"
                v3.VBtn("Load STEP", click=self.load_step_file, color="primary", size="small")
                v3.VSpacer()
                v3.VBtn(
                    "Run Analysis",
                    click=self.run_analysis,
                    color="success",
                    size="small",
                    disabled=("!file_path",),
                    loading=("is_analyzing",)
                )
                v3.VSpacer()
                v3.VBtn("Clear", click=self.clear_scene, color="error", size="small")

            # Main content
            with layout.content:
                with v3.VContainer(fluid=True, classes="fill-height pa-0"):
                    with v3.VRow(no_gutters=True, classes="fill-height"):
                        # Left panel
                        with v3.VCol(cols=2, classes="pa-2"):
                            v3.VCard(
                                children=[
                                    v3.VCardTitle("Info"),
                                    v3.VCardText(children=[
                                        v3.VChip("Faces: {{ num_faces }}", size="small", classes="mb-2"),
                                        v3.VChip("Features: {{ num_features }}", size="small", classes="mb-2"),
                                        v3.VChip("Time: {{ processing_time }}s", size="small"),
                                    ])
                                ]
                            )

                        # Center - 3D viewer (placeholder - VTK widget integration TBD)
                        with v3.VCol(cols=8):
                            with v3.VCard(height="600px"):
                                v3.VCardText(
                                    "3D Viewer - VTK integration in progress\n\nThe tessellator and rendering pipeline are ready.\nVTK widget context initialization needs further investigation.",
                                    classes="text-center pa-16",
                                    style="white-space: pre-line;"
                                )

                        # Right panel - Feature list
                        with v3.VCol(cols=2, classes="pa-2"):
                            with v3.VCard():
                                v3.VCardTitle("Features")
                                with v3.VCardText():
                                    with v3.VList(density="compact", v_if="features.length > 0"):
                                        v3.VListItem(
                                            v_for="(feat, idx) in features",
                                            key="idx",
                                            children=[
                                                "{{ feat.type }} ({{ feat.confidence?.toFixed(2) || 'N/A' }})"
                                            ]
                                        )
                                    v3.VChip("No features detected", v_if="features.length === 0", size="small")

            # Footer - Status and Log
            with layout.footer:
                with v3.VContainer(fluid=True, classes="pa-2"):
                    v3.VChip("{{ status }}", size="small", color="blue-grey", classes="mb-2")
                    v3.VTextarea(
                        label="Processing Log",
                        v_model=("log_messages.join('\\n')",),
                        readonly=True,
                        rows=4,
                        density="compact",
                        hide_details=True
                    )

        # Setup controller methods after UI is defined
        self.ctrl.view_update = lambda: None
        self.ctrl.view_reset_camera = lambda: None

    def clear_scene(self):
        """Clear loaded geometry and results"""
        if self.actor:
            self.renderer.RemoveActor(self.actor)
        self.polydata = None
        self.shape = None
        with self.state:
            self.state.file_path = ""
            self.state.num_faces = 0
            self.state.num_features = 0
            self.state.features = []
            self.state.log_messages = []
            self.state.status = "Ready - Load a STEP file to begin"
        self.ctrl.view_update()


def main():
    """Main entry point"""
    server = get_server(client_type="vue3")
    app = FeatureViewerApp(server)

    print("="*60)
    print("AAG Feature Recognition Viewer")
    print("="*60)
    print("Starting server...")
    print("Open http://localhost:8080 in your browser")
    print("="*60)

    server.start()


if __name__ == "__main__":
    main()
