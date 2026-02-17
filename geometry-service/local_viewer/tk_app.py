"""
Tkinter-based AAG Feature Recognition Viewer
Standalone desktop application using built-in Tkinter
"""
import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading

# Add geometry-service to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OCP.STEPControl import STEPControl_Reader

# VTK imports
from vtkmodules.vtkRenderingCore import (
    vtkRenderer, vtkActor, vtkPolyDataMapper, vtkCellPicker
)
from vtkmodules.vtkRenderingOpenGL2 import vtkWin32OpenGLRenderWindow
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkUnsignedCharArray
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkPolyDataNormals
from PIL import Image, ImageTk
import io
import numpy as np

# Feature color mapping
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


class AAGViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AAG Feature Recognition Viewer")
        self.root.geometry("1200x800")

        self.shape = None
        self.features = []

        # VTK components
        self.renderer = None
        self.polydata = None
        self.actor = None
        self.mapper = None
        self.lut = None
        self.render_window = None
        self.canvas_label = None
        self.canvas_image = None

        # Face picking and edge display
        self.highlight_mapper = None
        self.highlight_actor = None
        self.edge_actor = None
        self._rendering = False  # Guard against re-entrant render calls

        # Mouse interaction state
        self.last_x = 0
        self.last_y = 0
        self.is_dragging = False
        self.is_panning = False
        self.selected_face_id = None

        # Feature tree state
        self.feature_tree = None
        self.feature_index = {}            # tree iid -> index in self.features
        self.face_to_feature_idx = {}      # face_id -> index in self.features
        self._tree_select_guard = False

        self._setup_ui()

    def _setup_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill=tk.X)

        self.load_btn = ttk.Button(toolbar, text="Load STEP File", command=self.load_step)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(toolbar, text="Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(toolbar, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Main content area (3D viewer and side panels)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        # Left panel - Info
        left_frame = ttk.LabelFrame(main_frame, text="Info", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        self.faces_label = ttk.Label(left_frame, text="Faces: 0")
        self.faces_label.pack(anchor=tk.W, pady=2)

        self.features_label = ttk.Label(left_frame, text="Features: 0")
        self.features_label.pack(anchor=tk.W, pady=2)

        self.time_label = ttk.Label(left_frame, text="Time: 0s")
        self.time_label.pack(anchor=tk.W, pady=2)

        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Navigation help
        help_label = ttk.Label(left_frame, text="Navigation:", font=('Arial', 9, 'bold'))
        help_label.pack(anchor=tk.W, pady=(5, 2))

        nav_text = tk.Text(left_frame, height=6, width=20, wrap=tk.WORD, font=('Arial', 8),
                          bg='#f0f0f0', relief=tk.FLAT, padx=5, pady=5)
        nav_text.insert('1.0',
            "• Left Click = Select\n"
            "• Middle Drag = Rotate\n"
            "• Shift+Middle = Pan\n"
            "• Wheel = Zoom")
        nav_text.config(state=tk.DISABLED)
        nav_text.pack(anchor=tk.W, pady=2)

        # Center - 3D viewer with VTK (offscreen)
        center_frame = ttk.LabelFrame(main_frame, text="3D Viewer", padding=10)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Create canvas for displaying VTK renders
        self.canvas_label = tk.Label(center_frame, bg='#33333f', cursor='hand2')
        self.canvas_label.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events for SolidWorks-style navigation
        # SolidWorks: Middle drag = Rotate, Shift+Middle = Pan, Wheel = Zoom
        self.canvas_label.bind('<ButtonPress-1>', self._on_left_down)
        self.canvas_label.bind('<ButtonRelease-1>', self._on_left_up)
        self.canvas_label.bind('<B1-Motion>', self._on_left_drag)
        self.canvas_label.bind('<ButtonPress-2>', self._on_middle_down)
        self.canvas_label.bind('<ButtonRelease-2>', self._on_middle_up)
        self.canvas_label.bind('<B2-Motion>', self._on_middle_drag)
        self.canvas_label.bind('<MouseWheel>', self._on_mouse_wheel)

        # Setup VTK renderer (offscreen)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.25)  # Dark gray background

        self.render_window = vtkWin32OpenGLRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000, 800)  # Larger viewport to prevent cropping

        # Right panel - Features and Log
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))

        # Features section (top part of right panel)
        features_frame = ttk.LabelFrame(right_frame, text="Detected Features", padding=10)
        features_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Feature treeview with scrollbar
        tree_frame = ttk.Frame(features_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.feature_tree = ttk.Treeview(
            tree_frame,
            columns=('details', 'confidence'),
            show='tree headings',
            yscrollcommand=tree_scroll.set,
            selectmode='browse'
        )
        self.feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.feature_tree.yview)

        self.feature_tree.heading('#0', text='Feature', anchor=tk.W)
        self.feature_tree.heading('details', text='Dimensions', anchor=tk.W)
        self.feature_tree.heading('confidence', text='Conf', anchor=tk.CENTER)

        self.feature_tree.column('#0', width=150, minwidth=100)
        self.feature_tree.column('details', width=120, minwidth=80)
        self.feature_tree.column('confidence', width=50, minwidth=40, anchor=tk.CENTER)

        self.feature_tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        # Configure color tags for each feature type
        for ftype, rgb in FEATURE_COLORS.items():
            brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            if brightness > 0.6:
                r, g, b = int(rgb[0] * 153), int(rgb[1] * 153), int(rgb[2] * 153)
            else:
                r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            self.feature_tree.tag_configure(ftype, foreground=f'#{r:02x}{g:02x}{b:02x}')

        # Log section (bottom part of right panel)
        log_frame = ttk.LabelFrame(right_frame, text="Processing Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready - Load a STEP file to begin", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, padx=5, pady=(0, 5))

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()  # Safer UI update

    def load_step(self):
        filepath = filedialog.askopenfilename(
            title="Select STEP File",
            initialdir=str(Path.home()),
            filetypes=[("STEP Files", "*.step *.stp"), ("All Files", "*.*")]
        )

        if not filepath:
            return

        self.log(f"Loading: {filepath}")
        self.status_label.config(text="Loading STEP file...")
        self.root.update()

        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(filepath)

            if status != 1:
                raise ValueError(f"Failed to read STEP file (status={status})")

            reader.TransferRoots()
            self.shape = reader.OneShape()

            if self.shape.IsNull():
                raise ValueError("Failed to extract shape from STEP file")

            # Count faces
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE

            explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
            num_faces = 0
            while explorer.More():
                num_faces += 1
                explorer.Next()

            self.faces_label.config(text=f"Faces: {num_faces}")
            self.features_label.config(text="Features: 0")
            self.time_label.config(text="Time: 0s")

            self.log(f"Loaded successfully: {num_faces} faces")

            # Tessellate using custom tessellator (OCP bindings don't expose VIS pipeline methods)
            self.log("Tessellating geometry...")
            self.status_label.config(text="Tessellating...")
            self.root.update()

            from local_viewer.tessellator import tessellate_to_vtk
            self.polydata = tessellate_to_vtk(self.shape, linear_deflection=0.002, angular_deflection=5.0)

            self.log(f"Tessellated: {self.polydata.GetNumberOfPoints()} vertices, "
                    f"{self.polydata.GetNumberOfCells()} triangles")

            # Generate smooth normals for shading — prevents flat-triangle seam artifacts.
            # ConsistencyOff + AutoOrientOff: don't touch winding order (OCC handles it correctly).
            # SplittingOn + FeatureAngle 40°: smooth shading on curved surfaces,
            # hard crease at genuine sharp edges (walls, pockets, chamfers).
            normals = vtkPolyDataNormals()
            normals.SetInputData(self.polydata)
            normals.SetFeatureAngle(40.0)
            normals.SplittingOn()
            normals.ConsistencyOff()
            normals.AutoOrientNormalsOff()
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOff()
            normals.Update()

            # Create mapper and actor
            self.mapper = vtkPolyDataMapper()
            self.mapper.SetInputData(normals.GetOutput())
            self.mapper.ScalarVisibilityOff()  # Disable scalar coloring initially

            # Remove old actor if exists
            if self.actor:
                self.renderer.RemoveActor(self.actor)

            self.actor = vtkActor()
            self.actor.SetMapper(self.mapper)

            # Set default appearance
            prop = self.actor.GetProperty()
            prop.SetColor(0.8, 0.8, 0.8)  # Light gray default
            prop.SetAmbient(0.3)
            prop.SetDiffuse(0.7)
            prop.SetSpecular(0.2)
            prop.SetSpecularPower(20)

            # Push main geometry back slightly so highlight/edges always win depth test
            self.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            self.mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

            self.renderer.AddActor(self.actor)

            # Clean up old edge actor if any
            if self.edge_actor:
                self.renderer.RemoveActor(self.edge_actor)
                self.edge_actor = None

            self.renderer.ResetCamera()

            # Fix clipping planes to prevent cropping when rotating
            camera = self.renderer.GetActiveCamera()
            camera.SetClippingRange(0.01, 10000.0)  # Much wider range

            self._update_render()

            self.log(f"Rendered {self.polydata.GetNumberOfCells()} triangles")
            self.status_label.config(text=f"Loaded: {num_faces} faces")

            self.analyze_btn.config(state=tk.NORMAL)

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.status_label.config(text="Error loading file")
            messagebox.showerror("Error", f"Failed to load STEP file:\n{str(e)}")

    def run_analysis(self):
        if not self.shape:
            return

        self.log("Starting AAG feature recognition...")
        self.status_label.config(text="Analyzing...")

        self.analyze_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)

        # Run analysis in background thread
        thread = threading.Thread(target=self._analyze_thread, daemon=True)
        thread.start()

    def _analyze_thread(self):
        try:
            import time
            from aag_pattern_engine.pattern_matcher import AAGPatternMatcher

            start = time.time()
            matcher = AAGPatternMatcher(tolerance=1e-6)
            result = matcher.recognize_all_features(self.shape)
            elapsed = time.time() - start

            features = result.get_all_features() if hasattr(result, 'get_all_features') else []

            # Update UI from main thread
            self.root.after(0, self._on_analysis_complete, features, elapsed)

        except Exception as e:
            self.root.after(0, self._on_analysis_error, str(e))

    def _on_analysis_complete(self, features, elapsed):
        self.features = features

        self.features_label.config(text=f"Features: {len(features)}")
        self.time_label.config(text=f"Time: {elapsed:.2f}s")

        # Populate feature tree
        self._populate_feature_tree()

        self.log(f"Analysis complete: {len(features)} features in {elapsed:.1f}s")
        for i, feat in enumerate(features[:10]):
            ftype = feat.get('type', 'unknown')
            conf = feat.get('confidence', 0)
            self.log(f"  - {ftype} (confidence: {conf:.2f})")

        if len(features) > 10:
            self.log(f"  ... and {len(features) - 10} more")

        self.status_label.config(text=f"Analysis complete: {len(features)} features in {elapsed:.1f}s")

        # Color-code the features
        self._color_by_features(features)

        self.analyze_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)

    def _update_render(self):
        """Update the VTK render and display in Tkinter"""
        if not self.render_window or self._rendering:
            return

        self._rendering = True
        try:
            # Resize render window to match canvas exactly — ensures 1:1 pixel mapping for picking
            canvas_w = self.canvas_label.winfo_width()
            canvas_h = self.canvas_label.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                cur_w, cur_h = self.render_window.GetSize()
                if cur_w != canvas_w or cur_h != canvas_h:
                    self.render_window.SetSize(canvas_w, canvas_h)

            self.render_window.Render()

            # Prepare GPU frame for CPU readback (required on Windows with WGL)
            self.render_window.CopyResultFrame()

            width, height = self.render_window.GetSize()

            # Extract pixel data directly — avoids vtkWindowToImageFilter segfault on Windows
            array = vtkUnsignedCharArray()
            result = self.render_window.GetRGBACharPixelData(
                0, 0, width - 1, height - 1, 0, array, 0
            )

            if not result or array.GetSize() == 0:
                self.log("Render: pixel data empty, skipping display update")
                return

            # Convert to numpy array
            np_array = np.frombuffer(np.asarray(array), dtype=np.uint8).copy()
            np_array = np_array.reshape((height, width, 4))
            np_array = np.flipud(np_array)           # VTK origin is bottom-left
            np_array = np_array[:, :, [2, 1, 0]]     # BGRA → RGB

            pil_image = Image.fromarray(np_array[:, :, :3], 'RGB')

            self.canvas_image = ImageTk.PhotoImage(pil_image)
            self.canvas_label.config(image=self.canvas_image)

        except Exception as e:
            self.log(f"Render update error: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self._rendering = False

    def _color_by_features(self, features):
        """Update colors based on detected features"""
        self.log(f"_color_by_features called with {len(features)} features")

        if not self.polydata:
            self.log("ERROR: No polydata available for coloring")
            return

        # Create lookup table
        self.lut = vtkLookupTable()
        self.lut.SetNumberOfTableValues(self.polydata.GetNumberOfCells())
        self.log(f"Created lookup table for {self.polydata.GetNumberOfCells()} cells")

        # Build face_id -> feature_type map
        face_to_feature = {}
        for feature in features:
            ftype = feature.get('type', 'unrecognized')
            for fid in feature.get('face_ids', []):
                face_to_feature[fid] = ftype

        # Color each cell based on its face_id
        face_id_array = self.polydata.GetCellData().GetArray("face_id")
        for cell_id in range(self.polydata.GetNumberOfCells()):
            face_id = int(face_id_array.GetValue(cell_id))
            ftype = face_to_feature.get(face_id, 'unrecognized')
            color = FEATURE_COLORS.get(ftype, FEATURE_COLORS['unrecognized'])
            self.lut.SetTableValue(cell_id, color[0], color[1], color[2], 1.0)

        self.lut.Build()

        # Apply lookup table to mapper
        self.mapper.ScalarVisibilityOn()  # Enable scalar coloring
        self.mapper.SetLookupTable(self.lut)
        self.mapper.SetScalarModeToUseCellData()
        self.mapper.SetColorModeToMapScalars()
        self.mapper.SetScalarRange(0, self.polydata.GetNumberOfCells())

        self._update_render()
        self.log("Applied feature color-coding")

    def _on_analysis_error(self, error_msg):
        self.log(f"Analysis ERROR: {error_msg}")
        self.status_label.config(text="Analysis failed")

        messagebox.showerror("Analysis Error", f"Failed to analyze:\n{error_msg}")

        self.analyze_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)

    def _get_feature_details(self, feature):
        """Extract key dimension string for tree display."""
        parts = []
        if 'diameter' in feature and feature['diameter']:
            parts.append(f"\u00d8{feature['diameter']:.2f}mm")
        if 'depth' in feature and feature['depth']:
            parts.append(f"d={feature['depth']:.2f}mm")
        if 'radius' in feature and feature['radius']:
            parts.append(f"R{feature['radius']:.2f}mm")
        if 'width' in feature and feature['width']:
            parts.append(f"w={feature['width']:.2f}mm")
        if 'length' in feature and feature['length']:
            parts.append(f"L={feature['length']:.2f}mm")
        if 'height' in feature and feature['height']:
            parts.append(f"h={feature['height']:.2f}mm")
        if 'area' in feature and feature['area']:
            parts.append(f"A={feature['area']:.1f}mm\u00b2")
        if 'angle' in feature and feature['angle']:
            parts.append(f"{feature['angle']:.0f}\u00b0")
        return ', '.join(parts) if parts else ''

    def _populate_feature_tree(self):
        """Build hierarchical feature tree from self.features."""
        self.feature_tree.delete(*self.feature_tree.get_children())
        self.feature_index = {}
        self.face_to_feature_idx = {}

        if not self.features:
            return

        # Build face_id -> feature index reverse lookup
        for idx, feat in enumerate(self.features):
            for fid in feat.get('face_ids', []):
                self.face_to_feature_idx[fid] = idx

        # Group features by type
        from collections import OrderedDict
        groups = OrderedDict()
        for idx, feat in enumerate(self.features):
            ftype = feat.get('type', 'unrecognized')
            if ftype not in groups:
                groups[ftype] = []
            groups[ftype].append((idx, feat))

        TYPE_LABELS = {
            'blind_hole': 'Blind Holes', 'through_hole': 'Through Holes',
            'counterbored_hole': 'Counterbored Holes', 'threaded_hole': 'Threaded Holes',
            'pocket': 'Pockets', 'slot': 'Slots', 'through_slot': 'Through Slots',
            'fillet': 'Fillets', 'chamfer': 'Chamfers',
            'boss': 'Bosses', 'step': 'Steps', 'island': 'Islands',
            'unrecognized': 'Unrecognized',
        }

        for ftype, feat_list in groups.items():
            label = TYPE_LABELS.get(ftype, ftype.replace('_', ' ').title() + 's')
            cat_iid = f"cat_{ftype}"

            self.feature_tree.insert(
                '', 'end', iid=cat_iid,
                text=f"{label} ({len(feat_list)})",
                tags=(ftype,), open=True
            )

            for seq, (idx, feat) in enumerate(feat_list):
                face_ids = feat.get('face_ids', [])
                conf = feat.get('confidence', 0)
                details = self._get_feature_details(feat)

                item_text = f"{ftype.replace('_', ' ').title()} #{seq + 1} ({len(face_ids)}f)"
                item_iid = f"feat_{idx}"

                self.feature_tree.insert(
                    cat_iid, 'end', iid=item_iid,
                    text=item_text,
                    values=(details, f"{conf:.0%}"),
                    tags=(ftype,)
                )
                self.feature_index[item_iid] = idx

    def _on_tree_select(self, event):
        """Tree selection -> highlight corresponding faces in 3D."""
        if self._tree_select_guard:
            return

        selection = self.feature_tree.selection()
        if not selection:
            return

        item_iid = selection[0]

        # Category node -> highlight all faces of that type
        if item_iid.startswith('cat_'):
            all_face_ids = []
            for child_iid in self.feature_tree.get_children(item_iid):
                child_idx = self.feature_index.get(child_iid)
                if child_idx is not None:
                    all_face_ids.extend(self.features[child_idx].get('face_ids', []))
            if all_face_ids:
                self._highlight_faces(all_face_ids)
                ftype = item_iid.replace('cat_', '')
                self.log(f"Category: {ftype} - {len(all_face_ids)} faces")
            return

        # Feature node -> highlight its faces
        feat_idx = self.feature_index.get(item_iid)
        if feat_idx is None:
            return

        feature = self.features[feat_idx]
        face_ids = feature.get('face_ids', [])
        if face_ids:
            self._highlight_faces(face_ids)
            ftype = feature.get('type', 'unknown')
            details = self._get_feature_details(feature)
            self.log(f"Selected: {ftype} - {details} - {len(face_ids)} faces")
            self.status_label.config(
                text=f"Selected: {ftype.replace('_', ' ').title()} - {details} - {len(face_ids)} faces"
            )

    def _select_tree_item(self, feat_idx):
        """Programmatically select and scroll to a feature in the tree."""
        item_iid = f"feat_{feat_idx}"
        if not self.feature_tree.exists(item_iid):
            return
        self._tree_select_guard = True
        try:
            parent = self.feature_tree.parent(item_iid)
            if parent:
                self.feature_tree.item(parent, open=True)
            self.feature_tree.selection_set(item_iid)
            self.feature_tree.see(item_iid)
            self.feature_tree.focus(item_iid)
        finally:
            self._tree_select_guard = False

    # SolidWorks-style mouse controls:
    # - Left click = Select face
    # - Middle drag = Rotate (no shift)
    # - Shift + Middle drag = Pan
    # - Mouse wheel = Zoom

    def _on_left_down(self, event):
        """Left mouse down - prepare for selection"""
        self.last_x = event.x
        self.last_y = event.y
        self.is_dragging = False

    def _on_left_up(self, event):
        """Left mouse up - select face if not dragging"""
        if not self.is_dragging:
            self._pick_face(event)

    def _on_left_drag(self, event):
        """Left drag - mark as dragging (no selection on release)"""
        dx = abs(event.x - self.last_x)
        dy = abs(event.y - self.last_y)
        if dx > 3 or dy > 3:
            self.is_dragging = True

    def _on_middle_down(self, event):
        """Middle mouse down - check for shift modifier"""
        self.last_x = event.x
        self.last_y = event.y
        self.is_panning = (event.state & 0x0001) != 0  # Shift key pressed

    def _on_middle_up(self, event):
        """Middle mouse up"""
        self.is_panning = False

    def _on_middle_drag(self, event):
        """Middle drag - rotate if no shift, pan if shift pressed"""
        if not self.renderer:
            return

        dx = event.x - self.last_x
        dy = event.y - self.last_y

        # Check if shift is held (pan mode)
        is_shift = (event.state & 0x0001) != 0

        camera = self.renderer.GetActiveCamera()

        if is_shift:
            # Pan mode (Shift + Middle)
            focal_point = camera.GetFocalPoint()
            position = camera.GetPosition()

            renderer = self.renderer
            renderer.SetWorldPoint(focal_point[0], focal_point[1], focal_point[2], 1.0)
            renderer.ComputeWorldToDisplay()
            display_point = renderer.GetDisplayPoint()

            renderer.SetDisplayPoint(display_point[0] - dx, display_point[1] + dy, display_point[2])
            renderer.ComputeDisplayToWorld()
            world_point = renderer.GetWorldPoint()

            if world_point[3] != 0:
                new_focal = [world_point[i] / world_point[3] for i in range(3)]
                motion = [new_focal[i] - focal_point[i] for i in range(3)]
                camera.SetFocalPoint([focal_point[i] - motion[i] for i in range(3)])
                camera.SetPosition([position[i] - motion[i] for i in range(3)])
        else:
            # Rotate mode (Middle only) - FIXED INVERSION
            camera.Azimuth(-dx * 0.5)  # Inverted horizontal
            camera.Elevation(dy * 0.5)  # Inverted vertical
            camera.OrthogonalizeViewUp()

            # Reset clipping range after rotation to prevent cropping
            self.renderer.ResetCameraClippingRange()

        self.last_x = event.x
        self.last_y = event.y
        self._update_render()

    def _on_mouse_wheel(self, event):
        """Mouse wheel - zoom (SolidWorks style)"""
        if not self.renderer:
            return

        camera = self.renderer.GetActiveCamera()

        # Zoom in/out based on wheel direction
        if event.delta > 0:
            camera.Zoom(1.15)
        else:
            camera.Zoom(0.87)

        self._update_render()

    def _pick_face(self, event):
        """Pick and highlight a face using vtkCellPicker"""
        self.log(f"Click detected at ({event.x}, {event.y})")

        if not self.polydata or not self.renderer:
            self.log("ERROR: No polydata or renderer available")
            return

        # Render window is always sized to match the canvas (see _update_render),
        # so canvas pixels map 1:1 to render pixels — only need to flip Y axis.
        render_width, render_height = self.render_window.GetSize()
        x_render = float(event.x)
        y_render = float(render_height - event.y)  # Flip Y: VTK origin is bottom-left

        self.log(f"Pick at canvas ({event.x},{event.y}) -> render ({x_render:.0f},{y_render:.0f}) in {render_width}x{render_height}")

        # Use vtkCellPicker — try progressively larger tolerances for small faces
        picker = vtkCellPicker()
        cell_id = -1
        for tolerance in (0.005, 0.01, 0.02):
            picker.SetTolerance(tolerance)
            result = picker.Pick(x_render, y_render, 0, self.renderer)
            if result != 0 and picker.GetCellId() >= 0:
                cell_id = picker.GetCellId()
                break

        if cell_id < 0:
            self.log("No face picked")
            return

        # Read face_id directly from polydata cell data (0-based, matches tessellator)
        face_id_array = self.polydata.GetCellData().GetArray("face_id")
        if not face_id_array:
            self.log("ERROR: No face_id array in polydata")
            return

        face_id = int(face_id_array.GetValue(cell_id))
        self.selected_face_id = face_id

        # Find feature for this face and highlight entire feature
        feat_idx = self.face_to_feature_idx.get(face_id)
        if feat_idx is not None:
            feature = self.features[feat_idx]
            feature_type = feature.get('type', 'unrecognized')
            face_ids = feature.get('face_ids', [])
            self.log(f"Picked cell {cell_id} -> Face {face_id} - {feature_type} ({len(face_ids)} faces)")
            self._highlight_faces(face_ids)
            self._select_tree_item(feat_idx)
            details = self._get_feature_details(feature)
            self.status_label.config(
                text=f"Selected: {feature_type.replace('_', ' ').title()} - {details} - {len(face_ids)} faces"
            )
        else:
            self.log(f"Picked cell {cell_id} -> Face {face_id} - unrecognized")
            self._highlight_faces([face_id])

    def _highlight_faces(self, face_ids):
        """Highlight multiple faces by extracting cells with matching face_ids."""
        if not self.polydata:
            return

        # Remove old highlight
        if self.highlight_actor:
            self.renderer.RemoveActor(self.highlight_actor)
            self.highlight_actor = None

        if not face_ids:
            self._update_render()
            return

        try:
            from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
            from vtkmodules.vtkCommonCore import vtkPoints

            face_id_array = self.polydata.GetCellData().GetArray("face_id")
            if not face_id_array:
                self.log("ERROR: No face_id array in polydata")
                return

            face_id_set = set(face_ids)
            src_points = self.polydata.GetPoints()

            new_pts = vtkPoints()
            new_cells = vtkCellArray()
            point_map = {}

            cells_found = 0
            for cell_id in range(self.polydata.GetNumberOfCells()):
                if int(face_id_array.GetValue(cell_id)) not in face_id_set:
                    continue
                cell = self.polydata.GetCell(cell_id)
                new_ids = []
                for i in range(cell.GetNumberOfPoints()):
                    old_id = cell.GetPointId(i)
                    if old_id not in point_map:
                        point_map[old_id] = new_pts.InsertNextPoint(src_points.GetPoint(old_id))
                    new_ids.append(point_map[old_id])
                new_cells.InsertNextCell(3, new_ids)
                cells_found += 1

            if cells_found == 0:
                self.log(f"WARNING: No cells found for face_ids {face_ids[:5]}...")
                return

            highlight_polydata = vtkPolyData()
            highlight_polydata.SetPoints(new_pts)
            highlight_polydata.SetPolys(new_cells)

            self.log(f"Highlight: {cells_found} cells for {len(face_ids)} faces")

            self.highlight_mapper = vtkPolyDataMapper()
            self.highlight_mapper.SetInputData(highlight_polydata)

            self.highlight_actor = vtkActor()
            self.highlight_actor.SetMapper(self.highlight_mapper)

            prop = self.highlight_actor.GetProperty()
            prop.SetColor(0.0, 0.9, 0.0)
            prop.SetOpacity(0.85)
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
            prop.SetRepresentationToSurface()
            prop.BackfaceCullingOff()
            prop.LightingOff()

            self.highlight_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            self.highlight_mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)

            self.renderer.AddActor(self.highlight_actor)
            self._update_render()

        except Exception as e:
            self.log(f"Error highlighting faces: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    def clear(self):
        self.shape = None
        self.features = []

        # Clear VTK scene
        if self.actor:
            self.renderer.RemoveActor(self.actor)
        if self.highlight_actor:
            self.renderer.RemoveActor(self.highlight_actor)
        if self.edge_actor:
            self.renderer.RemoveActor(self.edge_actor)

        self.polydata = None
        self.actor = None
        self.mapper = None
        self.lut = None
        self.highlight_actor = None
        self.highlight_mapper = None
        self.edge_actor = None

        if self.render_window:
            self._update_render()

        self.faces_label.config(text="Faces: 0")
        self.features_label.config(text="Features: 0")
        self.time_label.config(text="Time: 0s")

        self.feature_tree.delete(*self.feature_tree.get_children())
        self.feature_index = {}
        self.face_to_feature_idx = {}
        self.log_text.delete(1.0, tk.END)

        self.analyze_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready - Load a STEP file to begin")


def main():
    root = tk.Tk()
    app = AAGViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
