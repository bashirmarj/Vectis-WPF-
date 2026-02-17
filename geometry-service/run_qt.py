#!/usr/bin/env python
"""
Launch Tkinter-based AAG Feature Recognition Viewer
"""
import sys
import os

# Ensure geometry-service is on path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from local_viewer.tk_app import main

if __name__ == "__main__":
    main()
