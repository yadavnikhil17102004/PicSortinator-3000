#!/usr/bin/env python3
"""
PicSortinator 3000 - Main Application
====================================

Military-grade image organization with ML superpowers.
Sort, tag, and laugh at your messy photo collection — all offline.

🚀 Features:
- ML-powered image classification using MobileNetV2
- Advanced OCR text extraction with preprocessing
- Face detection and clustering (coming soon)
- Duplicate image detection
- Full-text search and metadata indexing
- Sarcastic commentary throughout

"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('picsortinator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules with graceful error handling
try:
    from modules.loader import ImageLoader
    from modules.database import DatabaseManager
    # Import ML modules with fallback
    try:
        from modules.tagging import ImageTagger
        from modules.model_manager import ModelManager
        ML_AVAILABLE = True
    except ImportError as ml_error:
        logger.warning(f"ML modules not available: {ml_error}")
        print("⚠️  ML features disabled - TensorFlow not installed")
        ImageTagger = None
        ModelManager = None
        ML_AVAILABLE = False
    
    from modules.ocr import TextExtractor
    from modules.faces import FaceDetector
    
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    print(f"💥 Import Error: {e}")
    print("🔧 Please make sure basic dependencies are installed:")
    print("   pip install opencv-python pillow numpy")
    print("\n📚 For full ML features, also install:")
    print("   pip install tensorflow imagehash")
    print("\n💡 For TensorFlow issues on Windows:")
    print("   pip install tensorflow-cpu")
    sys.exit(1)

class PicSortinator:
    """Military-grade image organization with ML superpowers and attitude."""
    
    def __init__(self, db_path: str = "data/picsortinator.db"):
        """Initialize PicSortinator 3000 with all systems online."""
        print("🚀 Initializing PicSortinator 3000...")
        
        self.db_path = db_path
        self.loader = ImageLoader()
        self.database = DatabaseManager(db_path)
        
        # Initialize ML modules if available
        if ML_AVAILABLE and ImageTagger is not None:
            self.tagger = ImageTagger()
            self.model_manager = ModelManager()
            print("🤖 ML tagging system online")
        else:
            self.tagger = None
            self.model_manager = None
            print("⚠️  ML tagging disabled (TensorFlow not found)")
        
        self.text_extractor = TextExtractor()
        self.face_detector = FaceDetector()
        
        # Funny initialization messages
        self.funny_messages = {
            'scanning': [
                "🔍 Scanning your photo collection like a forensic investigator...",
                "📸 Loading images and judging your photography skills...",
                "🧐 Analyzing your digital memories with AI precision..."
            ],
            'processing': [
                "🤖 AI is now critiquing your life choices through images...",
                "📊 Processing complete! Your photos have been sorted and mocked.",
                "🎯 Mission accomplished! Another photo collection tamed."
            ]
        }
        
        logger.info("✅ PicSortinator 3000 initialized successfully")
        print("✅ All systems online! Ready to sort and judge your photos.")
        
    def scan_directory(self, directory_path: str) -> int:
        """Scan directory with ML-powered metadata extraction."""
        import numpy as np
        
        print(f"\n🔍 Scanning directory: {directory_path}")
        print(np.random.choice(self.funny_messages['scanning']))
        
        # Use new loader to scan directory
        try:
            image_files = self.loader.scan_directory(directory_path)
            total_images = len(image_files)
            
            if total_images == 0:
                print("😅 No images found. Either this directory is empty or you're looking in the wrong place.")
                return 0
            
            print(f"✅ Found {total_images} images to process")
            
            added_count = 0
            for idx, image_path in enumerate(image_files, 1):
                try:
                    # Extract comprehensive metadata
                    metadata = self.loader.extract_metadata(image_path)
                    if metadata:
                        # Add file hash for duplicate detection
                        metadata['file_hash'] = self.loader.get_file_hash(image_path)
                        
                        # Add to database
                        image_id = self.database.add_image(metadata)
                        if image_id:
                            added_count += 1
                        
                        # Show progress
                        if idx % 10 == 0 or idx == total_images:
                            progress = (idx / total_images) * 100
                            print(f"📊 Progress: {progress:.1f}% ({idx}/{total_images}) - {image_path.name}")
                
                except Exception as e:
                    logger.error(f"💥 Error processing {image_path}: {e}")
            
            print(f"\n✅ Successfully added {added_count} images to database")
            if added_count > 50:
                print("📸 Wow, that's a lot of photos! Did you empty your entire phone?")
            
            return added_count
            
        except Exception as e:
            logger.error(f"💥 Directory scanning failed: {e}")
            print(f"❌ Scanning failed: {e}")
            return 0
    
    def process_images(self, limit=None):
        """Process unprocessed images with ML tagging and OCR."""
        import numpy as np
        
        print(f"\n🤖 Starting ML-powered image processing...")
        print(np.random.choice(self.funny_messages['processing']))
        
        # Get unprocessed images from database
        unprocessed = self.database.get_unprocessed_images(limit)
        total = len(unprocessed)
        
        if total == 0:
            print("✅ No unprocessed images found. Your collection is perfectly sorted!")
            return 0
            
        print(f"🎯 Found {total} images to process with AI...")
        
        processed_count = 0
        for idx, (image_id, image_path) in enumerate(unprocessed, 1):
            try:
                print(f"\n📸 Processing ({idx}/{total}): {Path(image_path).name}")
                
                # Initialize processing results
                tags = []
                extracted_text = ""
                faces = []
                
                # ML-powered image tagging (if available)
                if self.tagger and ML_AVAILABLE:
                    try:
                        tags = self.tagger.tag_image(image_path)
                        if tags:
                            print(f"   🏷️  ML Tags: {', '.join(tags[:5])}")
                        else:
                            print("   🤷 ML couldn't identify anything meaningful")
                    except Exception as e:
                        logger.error(f"ML tagging failed for {image_path}: {e}")
                        print(f"   ⚠️  ML tagging failed: {e}")
                else:
                    print("   ⚠️  ML tagging disabled (TensorFlow not available)")
                
                # OCR text extraction
                try:
                    extracted_text = self.text_extractor.extract_text(image_path)
                    if extracted_text and len(extracted_text) > 10:
                        print(f"   📄 Text found: {extracted_text[:50]}...")
                    elif "not available" in extracted_text.lower():
                        print(f"   ⚠️  {extracted_text}")
                    else:
                        print("   📄 No readable text found")
                except Exception as e:
                    logger.error(f"OCR failed for {image_path}: {e}")
                    print(f"   ⚠️  OCR failed: {e}")
                    extracted_text = "OCR failed"
                
                # Face detection
                try:
                    faces = self.face_detector.detect_faces(image_path)
                    if faces:
                        print(f"   👥 Found {len(faces)} face(s)")
                    else:
                        print("   👥 No faces detected")
                except Exception as e:
                    logger.error(f"Face detection failed for {image_path}: {e}")
                    print(f"   ⚠️  Face detection failed: {e}")
                
                # Update database with processing results
                try:
                    success = self.database.update_image_processing(
                        image_id=image_id,
                        tags=tags,
                        extracted_text=extracted_text,
                        faces=len(faces)
                    )
                    
                    if success:
                        processed_count += 1
                        print("   ✅ Processing complete and saved")
                    else:
                        print("   ❌ Failed to save processing results")
                        
                except Exception as e:
                    logger.error(f"Database update failed for {image_path}: {e}")
                    print(f"   ❌ Database update failed: {e}")
                
                # Show progress periodically
                if idx % 5 == 0 or idx == total:
                    progress = (idx / total) * 100
                    print(f"\n📊 Progress: {progress:.1f}% ({idx}/{total}) processed")
                    
            except Exception as e:
                logger.error(f"💥 Unexpected error processing {image_path}: {e}")
                print(f"   💥 Unexpected error: {e}")
        
        # Final summary
        print(f"\n🎉 Processing complete!")
        print(f"✅ Successfully processed: {processed_count}/{total} images")
        
        if processed_count > 20:
            print("🤖 AI has finished judging your photo collection. Results may be brutally honest.")
        elif processed_count > 0:
            print("📸 Your images have been sorted and tagged. AI approval rating: pending.")
        
        return processed_count
    
    def search_images(self, query, search_type="all"):
        """Search for images based on tags, text, or faces using AI-powered database."""
        print(f"\n🔍 Searching for '{query}' in {search_type}...")
        
        try:
            # Use the database manager's search functionality
            results = self.database.search_images(query, search_type)
            
            if not results:
                print("😅 Nothing found. Maybe try 'selfies'—I bet you have plenty.")
                print("💡 Try searching for:")
                print("   - Common objects: 'person', 'car', 'food', 'building'")
                print("   - Text content: any word you remember seeing")
                print("   - Locations: place names from EXIF data")
                return []
            
            print(f"✅ Found {len(results)} matching images!")
            
            # Show sample results
            for i, result in enumerate(results[:5], 1):
                print(f"\n{i}. {result.get('filename', 'Unknown')}")
                print(f"   📂 Path: {result.get('path', 'Unknown')}")
                if result.get('tags'):
                    print(f"   🏷️  Tags: {result['tags'][:100]}..." if len(result['tags']) > 100 else f"   🏷️  Tags: {result['tags']}")
                if result.get('extracted_text'):
                    text_preview = result['extracted_text'][:50] + "..." if len(result['extracted_text']) > 50 else result['extracted_text']
                    print(f"   📄 Text: {text_preview}")
            
            if len(results) > 5:
                print(f"\n📋 ...and {len(results) - 5} more results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            print(f"❌ Search failed: {e}")
            return []
    
    def organize_by_tag(self, output_dir="output"):
        """Organize images into folders based on their AI-generated tags."""
        import shutil
        
        print(f"\n📰 Organizing images by tags into '{output_dir}'...")
        
        try:
            # Get all processed images from database
            images = self.database.get_processed_images()
            
            if not images:
                print("😅 No processed images to organize. Run processing first!")
                return 0
            
            print(f"🏷️  Found {len(images)} processed images with tags")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            organized_count = 0
            tag_stats = {}
            
            for image in images:
                tags = image.get('tags', '')
                if not tags:
                    continue
                    
                # Get the primary tag (first one in the list)
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
                primary_tag = tag_list[0] if tag_list else "uncategorized"
                
                # Clean tag name for folder
                clean_tag = "".join(c for c in primary_tag if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_tag = clean_tag if clean_tag else "uncategorized"
                
                # Create tag directory
                tag_dir = Path(output_dir) / clean_tag
                tag_dir.mkdir(exist_ok=True)
                
                # Copy the file
                try:
                    source_path = Path(image['path'])
                    dest_path = tag_dir / source_path.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    while dest_path.exists():
                        stem = source_path.stem
                        suffix = source_path.suffix
                        dest_path = tag_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(source_path, dest_path)
                    organized_count += 1
                    
                    # Track stats
                    tag_stats[clean_tag] = tag_stats.get(clean_tag, 0) + 1
                    
                except Exception as e:
                    logger.error(f"Error copying {image['path']}: {e}")
                    print(f"   ⚠️  Failed to copy {source_path.name}: {e}")
            
            # Show results
            print(f"\n✅ Successfully organized {organized_count} images!")
            print(f"📁 Created {len(tag_stats)} tag folders:")
            
            # Show folder stats
            for tag, count in sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   🏷️  {tag}: {count} images")
            
            if len(tag_stats) > 10:
                print(f"   ...and {len(tag_stats) - 10} more folders")
            
            if organized_count > 100:
                print("🤖 AI has sorted your entire digital life. You're welcome.")
            elif organized_count > 0:
                print("📸 Photos organized by AI intelligence. Much better than your manual sorting!")
            
            return organized_count
            
        except Exception as e:
            logger.error(f"Organization failed: {e}")
            print(f"❌ Organization failed: {e}")
            return 0
    
    def export_data(self, format="csv", output_path=None):
        """Export AI-processed image data to CSV, JSON, or HTML."""
        print(f"\n📄 Exporting image data as {format.upper()}...")
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/picsortinator_export_{timestamp}.{format}"
        
        try:
            # Get all images from database
            images = self.database.get_all_images()
            
            if not images:
                print("😅 No images to export. Add some images first!")
                return None
            
            print(f"📋 Found {len(images)} images to export")
            
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == "csv":
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "ID", "Filename", "Path", "Size (bytes)", "Width", "Height",
                        "AI Tags", "Extracted Text", "Face Count", "Creation Date", 
                        "Camera Make", "Camera Model", "GPS Location", "Processed"
                    ])
                    
                    for image in images:
                        writer.writerow([
                            image.get('id', ''),
                            image.get('filename', ''),
                            image.get('path', ''),
                            image.get('size', ''),
                            image.get('width', ''),
                            image.get('height', ''),
                            image.get('tags', ''),
                            image.get('extracted_text', ''),
                            image.get('face_count', 0),
                            image.get('creation_date', ''),
                            image.get('camera_make', ''),
                            image.get('camera_model', ''),
                            image.get('gps_location', ''),
                            'Yes' if image.get('processed', 0) else 'No'
                        ])
                        
            elif format == "json":
                import json
                export_data = {
                    "export_info": {
                        "generated_by": "PicSortinator 3000",
                        "timestamp": datetime.now().isoformat(),
                        "total_images": len(images),
                        "processed_images": len([img for img in images if img.get('processed', 0)])
                    },
                    "images": images
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
            elif format == "html":
                processed_count = len([img for img in images if img.get('processed', 0)])
                
                html = f"""<!DOCTYPE html>
<html><head>
<title>PicSortinator 3000 - Image Export</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #4CAF50; color: white; }}
    .processed {{ background-color: #d4edda; }}
    .unprocessed {{ background-color: #f8d7da; }}
</style>
</head><body>
<h1>🤖 PicSortinator 3000 - AI Image Analysis Report</h1>
<p><strong>Export Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><strong>Total Images:</strong> {len(images)} | <strong>AI Processed:</strong> {processed_count}</p>
<table>
<tr><th>Filename</th><th>AI Tags</th><th>Text Found</th><th>Faces</th><th>Size</th><th>Status</th></tr>"""
                
                for image in images:
                    status_class = "processed" if image.get('processed', 0) else "unprocessed"
                    status_text = "✅ AI Processed" if image.get('processed', 0) else "⚠️  Pending"
                    
                    tags_display = image.get('tags', 'None')[:50] + "..." if len(image.get('tags', '')) > 50 else image.get('tags', 'None')
                    text_display = image.get('extracted_text', 'None')[:30] + "..." if len(image.get('extracted_text', '')) > 30 else image.get('extracted_text', 'None')
                    
                    html += f"""<tr class="{status_class}">
<td>{image.get('filename', 'Unknown')}</td>
<td>{tags_display}</td>
<td>{text_display}</td>
<td>{image.get('face_count', 0)}</td>
<td>{image.get('size', 0) // 1024 if image.get('size') else 0} KB</td>
<td>{status_text}</td>
</tr>"""
                    
                html += "</table></body></html>"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
            else:
                print(f"❌ Unsupported format: {format}")
                return None
            
            print(f"✅ Successfully exported data to: {output_path}")
            print(f"📁 File size: {Path(output_path).stat().st_size // 1024} KB")
            
            if format == "html":
                print(f"🌍 Open in browser to view the AI analysis report!")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            print(f"❌ Export failed: {e}")
            return None
        
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'database') and self.database:
            self.database.close()
            logger.info("✅ Database connection closed")
            print("👋 PicSortinator 3000 shutting down. Your photos are now properly sorted!")

def main():
    """Main entry point for the application."""
    print("\n" + "="*70)
    print("🎆 PICSORTINATOR 3000 - ML-POWERED PHOTO SORTER 🎆")
    print("="*70)
    print("🤖 Sort, tag, and laugh at your messy photo collection")
    print("🚀 Now with 100% more machine learning and sarcasm!")
    print("="*70)
    
    # Create PicSortinator instance
    try:
        sorter = PicSortinator()
    except Exception as e:
        print(f"❌ Failed to initialize PicSortinator: {e}")
        return
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python main.py scan <directory>   - Scan directory for images")
        print("  python main.py process [limit]    - Process unprocessed images")
        print("  python main.py search <query>     - Search for images")
        print("  python main.py organize           - Organize images by tag")
        print("  python main.py export [format]    - Export data (csv, json, html)")
        return
    
    command = sys.argv[1].lower()
    
    if command == "scan" and len(sys.argv) >= 3:
        directory = sys.argv[2]
        sorter.scan_directory(directory)
        
    elif command == "process":
        limit = int(sys.argv[2]) if len(sys.argv) >= 3 else None
        sorter.process_images(limit)
        
    elif command == "search" and len(sys.argv) >= 3:
        query = sys.argv[2]
        search_type = sys.argv[3] if len(sys.argv) >= 4 else "all"
        results = sorter.search_images(query, search_type)
        
        # Display results
        for i, (image_id, filename, path, metadata) in enumerate(results[:10], 1):
            print(f"{i}. {filename}")
            print(f"   Path: {path}")
            print(f"   Metadata: {metadata[:50]}...")
            print()
            
        if len(results) > 10:
            print(f"...and {len(results) - 10} more results")
            
    elif command == "organize":
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "output"
        sorter.organize_by_tag(output_dir)
        
    elif command == "export":
        format = sys.argv[2] if len(sys.argv) >= 3 else "csv"
        sorter.export_data(format)
        
    else:
        print("Unknown command or missing arguments")
    
    # Close database connection
    sorter.close()

if __name__ == "__main__":
    main()
