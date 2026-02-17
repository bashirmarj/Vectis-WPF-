#!/usr/bin/env python
"""
Local AAG Feature Recognition Viewer Launcher

Starts the trame-based web viewer for local feature recognition testing.
Opens in browser at http://localhost:8080
"""

import sys
import os

# Ensure geometry-service/ is on path BEFORE any imports
# This allows the OCC shim to be found and used
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Now safe to import the app
from local_viewer.app import main

if __name__ == "__main__":
    main()
