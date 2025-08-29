#!/usr/bin/env python3
"""
Basic PicSortinator 3000 Workflow Example
=========================================
Demonstrates the complete workflow from scanning to search.
"""

import sys
import os

# Add the parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import DatabaseManager
from modules.loader import ImageLoader
from modules.tagging import ImageTagger
from modules.ocr import TextExtractor
from modules.faces import FaceDetector

def basic_workflow_example():
    """Demonstrate basic PicSortinator workflow."""
    
    print("🎆 PicSortinator 3000 - Basic Workflow Example")
    print("=" * 60)
    
    # Initialize components
    print("🚀 Initializing components...")
    db = DatabaseManager()
    loader = ImageLoader()
    tagger = ImageTagger()
    ocr = TextExtractor()
    face_detector = FaceDetector()
    
    # Example image path (update this to your test images)
    image_directory = "test_images"  # Change this to your image directory
    
    if not os.path.exists(image_directory):
        print(f"❌ Directory {image_directory} not found.")
        print("Please create a 'test_images' directory with some sample images.")
        return
    
    # Step 1: Scan for images
    print(f"\n📸 Scanning {image_directory} for images...")
    images = loader.scan_directory(image_directory)
    print(f"Found {len(images)} images")
    
    # Step 2: Add to database
    print("\n💾 Adding images to database...")
    for img_path in images[:5]:  # Process first 5 images for demo
        try:
            img_id = db.add_image(img_path)
            print(f"  Added: {os.path.basename(img_path)} (ID: {img_id})")
        except Exception as e:
            print(f"  Error adding {img_path}: {e}")
    
    # Step 3: Process images
    print("\n🤖 Processing images with AI...")
    unprocessed = db.get_unprocessed_images()
    
    for image_data in unprocessed[:3]:  # Process first 3 for demo
        img_id, img_path = image_data[0], image_data[1]
        print(f"\n  Processing: {os.path.basename(img_path)}")
        
        try:
            # Generate tags
            tags = tagger.tag_image(img_path)
            print(f"    🏷️  Tags: {tags}")
            
            # Extract text
            text = ocr.extract_text(img_path)
            if text and len(text.strip()) > 3:
                print(f"    📄 Text: {text[:50]}...")
            else:
                print("    📄 No readable text found")
            
            # Detect faces
            faces = face_detector.detect_faces(img_path)
            print(f"    👥 Faces: {len(faces)} detected")
            
            # Update database
            db.update_image_processing(img_id, tags, text, len(faces))
            print("    ✅ Database updated")
            
        except Exception as e:
            print(f"    ❌ Error processing: {e}")
    
    # Step 4: Search examples
    print("\n🔍 Search Examples:")
    
    # Search by tag
    results = db.search_by_tags(["person"])
    print(f"  Images with 'person' tag: {len(results)}")
    
    # Search by text
    results = db.search_by_text("test")
    print(f"  Images containing 'test' text: {len(results)}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\n📊 Database Statistics:")
    print(f"  Total images: {stats.get('total_images', 0)}")
    print(f"  Processed images: {stats.get('processed_images', 0)}")
    print(f"  Unique tags: {stats.get('unique_tags', 0)}")
    
    print("\n🎉 Workflow complete!")
    print("You can now search and organize your images using the main CLI.")

if __name__ == "__main__":
    basic_workflow_example()
