#!/usr/bin/env python3
"""
ForenSnap Dependency Test Script
Tests which dependencies are available for Python 3.13 compatibility
"""

import sys
import importlib

def test_import(module_name, description=""):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description} - ERROR: {str(e)}")
        return False

def main():
    print("🔬 ForenSnap Ultimate - Dependency Test")
    print(f"🐍 Python Version: {sys.version}")
    print("=" * 60)
    
    # Core dependencies
    print("\n📦 CORE DEPENDENCIES:")
    core_success = 0
    core_total = 0
    
    deps = [
        ("PIL", "Pillow - Image processing"),
        ("cv2", "OpenCV - Computer vision"),
        ("numpy", "NumPy - Numerical computing"),
        ("requests", "Requests - HTTP library"),
    ]
    
    for module, desc in deps:
        core_total += 1
        if test_import(module, desc):
            core_success += 1
    
    # OCR dependencies  
    print("\n👁️ OCR DEPENDENCIES:")
    ocr_success = 0
    ocr_total = 0
    
    ocr_deps = [
        ("pytesseract", "Tesseract OCR wrapper"),
        ("easyocr", "EasyOCR - Neural OCR"),
        ("langdetect", "Language detection"),
    ]
    
    for module, desc in ocr_deps:
        ocr_total += 1
        if test_import(module, desc):
            ocr_success += 1
    
    # AI/ML dependencies
    print("\n🤖 AI/ML DEPENDENCIES:")
    ai_success = 0
    ai_total = 0
    
    ai_deps = [
        ("torch", "PyTorch - Deep learning framework"),
        ("torchvision", "PyTorch Vision - Image models"),
        ("transformers", "Hugging Face Transformers"),
        ("spacy", "spaCy - NLP library"),
    ]
    
    for module, desc in ai_deps:
        ai_total += 1
        if test_import(module, desc):
            ai_success += 1
    
    # GUI dependencies
    print("\n🖥️ GUI DEPENDENCIES:")
    gui_success = 0
    gui_total = 0
    
    gui_deps = [
        ("tkinter", "Tkinter - GUI toolkit"),
        ("tkinter.ttk", "Tkinter themed widgets"),
    ]
    
    for module, desc in gui_deps:
        gui_total += 1
        if test_import(module, desc):
            gui_success += 1
    
    # Optional advanced dependencies
    print("\n⭐ ADVANCED DEPENDENCIES:")
    advanced_success = 0
    advanced_total = 0
    
    advanced_deps = [
        ("face_recognition", "Face recognition library"),
        ("dlib", "dlib - Machine learning toolkit"),
        ("clip", "OpenAI CLIP model"),
        ("sqlalchemy", "SQLAlchemy - Database ORM"),
        ("fastapi", "FastAPI - Web framework"),
        ("reportlab", "ReportLab - PDF generation"),
    ]
    
    for module, desc in advanced_deps:
        advanced_total += 1
        if test_import(module, desc):
            advanced_success += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEPENDENCY SUMMARY:")
    print(f"Core Dependencies: {core_success}/{core_total}")
    print(f"OCR Dependencies: {ocr_success}/{ocr_total}")  
    print(f"AI/ML Dependencies: {ai_success}/{ai_total}")
    print(f"GUI Dependencies: {gui_success}/{gui_total}")
    print(f"Advanced Dependencies: {advanced_success}/{advanced_total}")
    
    total_success = core_success + ocr_success + ai_success + gui_success
    total_needed = core_total + ocr_total + ai_total + gui_total
    
    print(f"\nTotal Essential: {total_success}/{total_needed}")
    print(f"Total with Advanced: {total_success + advanced_success}/{total_needed + advanced_total}")
    
    if total_success >= total_needed - 1:  # Allow 1 missing dependency
        print("\n✅ ForenSnap should work with basic functionality!")
    elif total_success >= total_needed - 3:  # Allow 3 missing dependencies
        print("\n⚠️ ForenSnap may work with limited functionality")
    else:
        print("\n❌ Too many dependencies missing - manual installation needed")
    
    print("\n🔧 To install missing dependencies, try:")
    print("   pip install opencv-python pillow numpy requests pytesseract easyocr torch transformers")

if __name__ == "__main__":
    main()
