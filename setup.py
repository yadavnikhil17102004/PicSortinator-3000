#!/usr/bin/env python3
"""
PicSortinator 3000 - Quick Setup Script
======================================
Automates the installation and setup process for new users.
"""

import os
import sys
import subprocess
import platform

def main():
    """Main setup function."""
    print("🎆 PicSortinator 3000 Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print("✅ Python version:", sys.version.split()[0])
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
        print("✅ Virtual environment created")
    
    # Install dependencies
    print("📥 Installing dependencies...")
    if platform.system() == "Windows":
        pip_path = os.path.join('venv', 'Scripts', 'pip')
    else:
        pip_path = os.path.join('venv', 'bin', 'pip')
    
    try:
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("✅ Directories created")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    if platform.system() == "Windows":
        print("1. Activate virtual environment: venv\\Scripts\\activate")
    else:
        print("1. Activate virtual environment: source venv/bin/activate")
    print("2. Install Tesseract OCR (see README.md)")
    print("3. Run: python main.py scan /path/to/your/photos")
    print("4. Run: python main.py process")
    
    return True

if __name__ == "__main__":
    main()
