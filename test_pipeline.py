#!/usr/bin/env python3
"""
PicSortinator 3000 - Pipeline Test Script
=========================================

Test script to demonstrate the full ML pipeline.
Creates some test images and processes them through the system.
"""

import os
import sys
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add modules to path
sys.path.append(str(Path(__file__).parent / 'modules'))

try:
    from modules.loader import ImageLoader
    from modules.database import DatabaseManager
    from modules.tagging import ImageTagger
    from modules.ocr import TextExtractor
    from modules.model_manager import ModelManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_images():
    """Create some test images for pipeline testing."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    print("🎨 Creating test images...")
    
    # Create a simple photo-like image
    photo_img = Image.new('RGB', (400, 300), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(photo_img)
    # Draw a simple landscape
    draw.rectangle([0, 200, 400, 300], fill=(34, 139, 34))  # Green ground
    draw.ellipse([50, 50, 120, 120], fill=(255, 255, 0))    # Yellow sun
    photo_img.save(test_dir / "landscape_photo.jpg")
    
    # Create a text-heavy image (screenshot-like)
    text_img = Image.new('RGB', (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(text_img)
    try:
        # Try to use a font, fallback to default if not available
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add some text content
    text_lines = [
        "PicSortinator 3000 Test Document",
        "This is a sample screenshot with text content.",
        "Machine learning will classify this as a document.",
        "OCR will extract this readable text.",
        "Keywords: test, document, screenshot, OCR"
    ]
    
    y_pos = 50
    for line in text_lines:
        draw.text((50, y_pos), line, fill=(0, 0, 0), font=font)
        y_pos += 50
    
    text_img.save(test_dir / "text_document.png")
    
    # Create a colorful abstract image
    abstract_img = Image.new('RGB', (300, 300), color=(255, 0, 255))
    draw = ImageDraw.Draw(abstract_img)
    # Draw some colorful shapes
    for i in range(10):
        x, y = np.random.randint(0, 250, 2)
        color = tuple(np.random.randint(0, 255, 3))
        draw.ellipse([x, y, x+50, y+50], fill=color)
    abstract_img.save(test_dir / "abstract_art.png")
    
    print(f"✅ Created {len(list(test_dir.glob('*')))} test images in {test_dir}")
    return test_dir

def test_ml_tagging():
    """Test the ML-powered image tagging."""
    print("\n🤖 Testing ML Image Tagging...")
    
    try:
        tagger = ImageTagger()
        test_dir = Path("test_images")
        
        for image_path in test_dir.glob("*.jpg"):
            print(f"\n📸 Analyzing: {image_path.name}")
            tags = tagger.generate_tags(str(image_path))
            print(f"🏷️  Tags: {tags}")
            
            # Get detailed analysis
            detailed = tagger.get_detailed_analysis(str(image_path))
            print(f"💬 AI Comment: {detailed.get('ai_comment', 'No comment')}")
            print(f"📊 Top prediction: {detailed.get('top_predictions', [{}])[0].get('label', 'Unknown')} ({detailed.get('top_predictions', [{}])[0].get('confidence', 0):.2f})")
    
    except Exception as e:
        print(f"❌ ML Tagging test failed: {e}")
        print("💡 Make sure TensorFlow is installed: pip install tensorflow")

def test_ocr_extraction():
    """Test OCR text extraction."""
    print("\n📖 Testing OCR Text Extraction...")
    
    try:
        extractor = TextExtractor()
        test_dir = Path("test_images")
        
        for image_path in test_dir.glob("*.png"):  # Focus on PNG files (likely to have text)
            print(f"\n📄 Extracting text from: {image_path.name}")
            text = extractor.extract_text(str(image_path))
            print(f"📝 Extracted text: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            # Test keyword extraction
            keywords = extractor.extract_keywords(str(image_path))
            if keywords:
                print(f"🔍 Keywords: {keywords}")
            
            # Test if it's text-heavy
            is_text_heavy = extractor.is_text_heavy_image(str(image_path))
            print(f"📚 Text-heavy image: {is_text_heavy}")
    
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        print("💡 Make sure Tesseract is installed and in your PATH")

def test_database_integration():
    """Test database integration with loader."""
    print("\n💾 Testing Database Integration...")
    
    try:
        # Initialize components
        loader = ImageLoader()
        db = DatabaseManager("test_database.db")
        test_dir = Path("test_images")
        
        print(f"📁 Scanning directory: {test_dir}")
        image_files = loader.scan_directory(str(test_dir))
        print(f"🔍 Found {len(image_files)} images")
        
        # Add images to database
        added_count = 0
        for image_path in image_files:
            metadata = loader.extract_metadata(image_path)
            if metadata:
                file_hash = loader.get_file_hash(image_path)
                metadata['file_hash'] = file_hash
                
                image_id = db.add_image(metadata)
                if image_id:
                    added_count += 1
                    print(f"✅ Added to DB: {image_path.name} (ID: {image_id})")
        
        print(f"📊 Added {added_count} images to database")
        
        # Get statistics
        stats = db.get_image_stats()
        print(f"📈 Database stats: {stats}")
        
        # Cleanup
        db.close()
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")

def test_model_manager():
    """Test model manager functionality."""
    print("\n🧠 Testing Model Manager...")
    
    try:
        manager = ModelManager()
        
        # Get model info
        info = manager.get_model_info()
        print(f"📦 Models directory: {info['models_directory']}")
        print(f"🎯 Available models: {info['available_models']}")
        print(f"💾 Cached models: {len(info['cached_models'])}")
        
        # Test label loading
        labels = manager.load_imagenet_labels()
        print(f"🏷️  Loaded {len(labels)} ImageNet labels")
        print(f"🔤 Sample labels: {list(labels.values())[:10]}")
        
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")

def main():
    """Run all pipeline tests."""
    print("🚀 PicSortinator 3000 - Pipeline Test Suite")
    print("=" * 60)
    
    # Create test images
    test_dir = create_test_images()
    
    # Run tests
    test_model_manager()
    test_ml_tagging()
    test_ocr_extraction()
    test_database_integration()
    
    print("\n🎉 Pipeline test complete!")
    print(f"📁 Test images created in: {test_dir}")
    print("💡 To run the full application:")
    print("   python main.py scan test_images")
    print("   python main.py process")
    print("   python main.py search landscape")

if __name__ == "__main__":
    main()
