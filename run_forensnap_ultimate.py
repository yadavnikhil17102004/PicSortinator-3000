#!/usr/bin/env python3
"""
ForenSnap Ultimate Launcher
===========================

Main launcher for ForenSnap Ultimate with organized directory structure.
"""

import sys
import os
from pathlib import Path

# Setup paths for organized directory structure
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" 
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Add src directory to Python path
sys.path.insert(0, str(SRC_DIR))

# Import the main application from the complete version
try:
    from forensnap_ultimate_complete import main
    print("✅ Using complete ForenSnap Ultimate with enhanced features")
except ImportError:
    print("⚠️  Complete version not found, falling back to basic version")
    try:
        from forensnap_ultimate import main
    except ImportError:
        print("❌ No ForenSnap main modules found! Check src/ directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
