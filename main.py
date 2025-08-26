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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('picsortinator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules
try:
    from modules.loader import ImageLoader
    from modules.database import DatabaseManager
    from modules.tagging import ImageTagger
    from modules.ocr import TextExtractor
    from modules.faces import FaceDetector
    from modules.model_manager import ModelManager
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    print(f"💥 Import Error: {e}")
    print("🔧 Please make sure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    print("💡 For TensorFlow issues on Windows, try:")
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
        self.tagger = ImageTagger()
        self.text_extractor = TextExtractor()
        self.face_detector = FaceDetector()
        self.model_manager = ModelManager()
        
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
        """Process unprocessed images in the database."""
        cursor = self.conn.cursor()
        
        # Get unprocessed images
        if limit:
            cursor.execute("SELECT id, path FROM images WHERE processed = 0 LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT id, path FROM images WHERE processed = 0")
            
        unprocessed = cursor.fetchall()
        total = len(unprocessed)
        
        if total == 0:
            print("No unprocessed images found.")
            return 0
            
        print(f"Processing {total} images...")
        
        for idx, (image_id, image_path) in enumerate(unprocessed, 1):
            try:
                print(f"Processing ({idx}/{total}): {os.path.basename(image_path)}")
                
                # Generate tags
                tags = self.tagger.generate_tags(image_path)
                tags_str = ",".join(tags)
                
                # Extract text
                extracted_text = self.text_extractor.extract_text(image_path)
                
                # Detect faces
                faces = self.face_detector.detect_faces(image_path)
                faces_str = ",".join([f"person_{i}" for i in range(len(faces))])
                
                # Update database
                cursor.execute(
                    "UPDATE images SET tags = ?, extracted_text = ?, faces = ?, processed = 1 WHERE id = ?",
                    (tags_str, extracted_text, faces_str, image_id)
                )
                
                if idx % 5 == 0:
                    self.conn.commit()
                    print(funny_message("processing", count=idx))
                    
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                
        self.conn.commit()
        print(f"Finished processing {total} images")
        print(funny_message("complete", count=total))
        return total
    
    def search_images(self, query, search_type="all"):
        """Search for images based on tags, text, or faces."""
        cursor = self.conn.cursor()
        results = []
        
        if search_type == "tags" or search_type == "all":
            cursor.execute("SELECT id, filename, path, tags FROM images WHERE tags LIKE ?", (f"%{query}%",))
            results.extend(cursor.fetchall())
            
        if search_type == "text" or search_type == "all":
            cursor.execute("SELECT id, filename, path, extracted_text FROM images WHERE extracted_text LIKE ?", (f"%{query}%",))
            results.extend(cursor.fetchall())
            
        if search_type == "faces" or search_type == "all":
            cursor.execute("SELECT id, filename, path, faces FROM images WHERE faces LIKE ?", (f"%{query}%",))
            results.extend(cursor.fetchall())
        
        # Remove duplicates (by id)
        unique_results = []
        seen_ids = set()
        for result in results:
            if result[0] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result[0])
        
        if not unique_results:
            print("Nothing found. Maybe try 'selfies'—I bet you have plenty.")
        else:
            print(f"Found {len(unique_results)} matching images")
        
        return unique_results
    
    def organize_by_tag(self, output_dir="output"):
        """Organize images into folders based on their tags."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, filename, path, tags FROM images WHERE processed = 1")
        images = cursor.fetchall()
        
        if not images:
            print("No processed images to organize.")
            return 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        organized_count = 0
        for image_id, filename, path, tags in images:
            if not tags:
                continue
                
            # Get the primary tag (first one)
            tag_list = tags.split(",")
            primary_tag = tag_list[0] if tag_list else "uncategorized"
            
            # Create tag directory
            tag_dir = os.path.join(output_dir, primary_tag)
            os.makedirs(tag_dir, exist_ok=True)
            
            # Copy the file
            try:
                import shutil
                dest_path = os.path.join(tag_dir, filename)
                shutil.copy2(path, dest_path)
                organized_count += 1
            except Exception as e:
                logger.error(f"Error copying {path}: {e}")
        
        print(f"Organized {organized_count} images into {output_dir}")
        if organized_count > 0:
            print(funny_message("organize", count=organized_count))
        
        return organized_count
    
    def export_data(self, format="csv", output_path=None):
        """Export image data to CSV, JSON, or HTML."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/image_data_{timestamp}.{format}"
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, filename, path, size, tags, extracted_text, faces, creation_date FROM images")
        images = cursor.fetchall()
        
        if not images:
            print("No images to export.")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Filename", "Path", "Size (bytes)", "Tags", "Extracted Text", "Faces", "Creation Date"])
                writer.writerows(images)
                
        elif format == "json":
            import json
            data = []
            for image in images:
                data.append({
                    "id": image[0],
                    "filename": image[1],
                    "path": image[2],
                    "size": image[3],
                    "tags": image[4].split(",") if image[4] else [],
                    "extracted_text": image[5],
                    "faces": image[6].split(",") if image[6] else [],
                    "creation_date": str(image[7])
                })
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        elif format == "html":
            html = "<html><head><title>Image Sorter Export</title></head><body>"
            html += "<h1>Image Sorter Export</h1>"
            html += "<table border='1'><tr><th>ID</th><th>Filename</th><th>Tags</th><th>Text</th><th>Faces</th></tr>"
            
            for image in images:
                html += f"<tr><td>{image[0]}</td><td>{image[1]}</td><td>{image[4]}</td><td>{image[5]}</td><td>{image[6]}</td></tr>"
                
            html += "</table></body></html>"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        print(f"Exported data to {output_path}")
        return output_path
        
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
