"""
Qt-based AAG Feature Recognition Viewer
Standalone desktop application similar to Analysis Situs
"""
import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QListWidget, QFileDialog,
    QSplitter, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Add geometry-service to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OCP.STEPControl import STEPControl_Reader


class AnalysisWorker(QThread):
    """Background worker for AAG analysis"""
    finished = pyqtSignal(list, float)
    error = pyqtSignal(str)

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def run(self):
        try:
            import time
            from aag_pattern_engine.pattern_matcher import AAGPatternMatcher

            start = time.time()
            matcher = AAGPatternMatcher(tolerance=1e-6)
            result = matcher.recognize_all_features(self.shape)
            elapsed = time.time() - start

            features = result.get_all_features() if hasattr(result, 'get_all_features') else []
            self.finished.emit(features, elapsed)
        except Exception as e:
            self.error.emit(str(e))


class AAGViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.shape = None
        self.features = []
        self.worker = None

        self.setWindowTitle("AAG Feature Recognition Viewer")
        self.setGeometry(100, 100, 1200, 800)

        self._setup_ui()

    def _setup_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        layout = QVBoxLayout(central)

        # Toolbar
        toolbar = QHBoxLayout()

        self.load_btn = QPushButton("Load STEP File")
        self.load_btn.clicked.connect(self.load_step)
        toolbar.addWidget(self.load_btn)

        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        toolbar.addWidget(self.analyze_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(self.clear_btn)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Content area
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Info
        left_panel = QGroupBox("Info")
        left_layout = QVBoxLayout()

        self.faces_label = QLabel("Faces: 0")
        self.features_label = QLabel("Features: 0")
        self.time_label = QLabel("Time: 0s")

        left_layout.addWidget(self.faces_label)
        left_layout.addWidget(self.features_label)
        left_layout.addWidget(self.time_label)
        left_layout.addStretch()

        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(200)
        splitter.addWidget(left_panel)

        # Center - 3D viewer placeholder
        center_panel = QGroupBox("3D Viewer")
        center_layout = QVBoxLayout()

        placeholder = QLabel("3D Viewer\n\nVTK integration pending\n\nTessellator ready")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("font-size: 14pt; color: gray;")

        center_layout.addWidget(placeholder)
        center_panel.setLayout(center_layout)
        splitter.addWidget(center_panel)

        # Right panel - Features
        right_panel = QGroupBox("Detected Features")
        right_layout = QVBoxLayout()

        self.feature_list = QListWidget()
        right_layout.addWidget(self.feature_list)

        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(300)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 1)

        layout.addWidget(splitter)

        # Bottom - Log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        layout.addWidget(log_group)

        # Status bar
        self.statusBar().showMessage("Ready - Load a STEP file to begin")

    def log(self, message):
        self.log_text.append(message)

    def load_step(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select STEP File",
            str(Path.home()),
            "STEP Files (*.step *.stp);;All Files (*.*)"
        )

        if not filepath:
            return

        self.log(f"Loading: {filepath}")
        self.statusBar().showMessage("Loading STEP file...")

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

            self.faces_label.setText(f"Faces: {num_faces}")
            self.features_label.setText("Features: 0")
            self.time_label.setText("Time: 0s")

            self.log(f"Loaded successfully: {num_faces} faces")
            self.statusBar().showMessage(f"Loaded: {num_faces} faces")

            self.analyze_btn.setEnabled(True)

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.statusBar().showMessage("Error loading file")
            QMessageBox.critical(self, "Error", f"Failed to load STEP file:\n{str(e)}")

    def run_analysis(self):
        if not self.shape:
            return

        self.log("Starting AAG feature recognition...")
        self.statusBar().showMessage("Analyzing...")

        self.analyze_btn.setEnabled(False)
        self.load_btn.setEnabled(False)

        # Run analysis in background thread
        self.worker = AnalysisWorker(self.shape)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_complete(self, features, elapsed):
        self.features = features

        self.features_label.setText(f"Features: {len(features)}")
        self.time_label.setText(f"Time: {elapsed:.2f}s")

        # Update feature list
        self.feature_list.clear()
        for feat in features:
            ftype = feat.get('type', 'unknown')
            conf = feat.get('confidence', 0)
            self.feature_list.addItem(f"{ftype} ({conf:.2f})")

        self.log(f"Analysis complete: {len(features)} features in {elapsed:.1f}s")
        for i, feat in enumerate(features[:10]):
            ftype = feat.get('type', 'unknown')
            conf = feat.get('confidence', 0)
            self.log(f"  - {ftype} (confidence: {conf:.2f})")

        if len(features) > 10:
            self.log(f"  ... and {len(features) - 10} more")

        self.statusBar().showMessage(f"Analysis complete: {len(features)} features in {elapsed:.1f}s")

        self.analyze_btn.setEnabled(True)
        self.load_btn.setEnabled(True)

    def on_analysis_error(self, error_msg):
        self.log(f"Analysis ERROR: {error_msg}")
        self.statusBar().showMessage("Analysis failed")

        QMessageBox.critical(self, "Analysis Error", f"Failed to analyze:\n{error_msg}")

        self.analyze_btn.setEnabled(True)
        self.load_btn.setEnabled(True)

    def clear(self):
        self.shape = None
        self.features = []

        self.faces_label.setText("Faces: 0")
        self.features_label.setText("Features: 0")
        self.time_label.setText("Time: 0s")

        self.feature_list.clear()
        self.log_text.clear()

        self.analyze_btn.setEnabled(False)
        self.statusBar().showMessage("Ready - Load a STEP file to begin")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = AAGViewerWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
