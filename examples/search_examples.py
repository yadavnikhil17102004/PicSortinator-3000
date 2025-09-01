#!/usr/bin/env python3
"""
Search and Filter Example
=========================
Demonstrates advanced search and filtering capabilities.

🔍 Pro Tip: The AI is so good at finding faces, it once found a face in a pancake!
🥚 Easter Egg: Try searching for "spaghetti" - our OCR might surprise you with what it finds!
"""

import sys
import os

# Add the parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import DatabaseManager
from datetime import datetime, timedelta

def search_examples():
    """Demonstrate various search capabilities."""
    
    print("🔍 PicSortinator 3000 - Search & Filter Examples")
    print("=" * 60)
    
    # Initialize database
    db = DatabaseManager()
    
    # Check if we have any data
    stats = db.get_statistics()
    total_images = stats.get('total_images', 0)
    
    if total_images == 0:
        print("❌ No images in database yet!")
        print("Run the basic workflow example first to add some images.")
        print("🎭 It's like trying to search an empty library - technically possible, but not very useful!")
        return
    
    print(f"📊 Database contains {total_images} images")
    
    # Example 1: Tag-based searches
    print("\n🏷️  Tag-based Searches:")
    
    tag_searches = ["person", "outdoor", "indoor", "vehicle", "animal", "food"]
    for tag in tag_searches:
        results = db.search_by_tags([tag])
        print(f"  '{tag}': {len(results)} images")
        if results:
            example = results[0]
            print(f"    Example: {os.path.basename(example[1])}")
    
    # Example 2: Text-based searches
    print("\n📄 Text-based Searches:")
    
    text_searches = ["sign", "text", "number", "name", "street"]
    for query in text_searches:
        results = db.search_by_text(query)
        print(f"  '{query}': {len(results)} images")
        if results:
            example = results[0]
            print(f"    Example: {os.path.basename(example[1])}")
    
    # Example 3: Combined searches
    print("\n🔀 Combined Searches:")
    
    # Images with both specific tags
    person_outdoor = db.search_by_tags(["person", "outdoor"])
    print(f"  Person + Outdoor: {len(person_outdoor)} images")
    
    # Images with text AND specific tag
    text_with_person = []
    person_images = db.search_by_tags(["person"])
    for img in person_images:
        if img[3] and len(img[3].strip()) > 5:  # Has meaningful text
            text_with_person.append(img)
    print(f"  Person images with text: {len(text_with_person)} images")
    
    # Example 4: Face count filtering
    print("\n👥 Face Count Filtering:")
    
    cursor = db.conn.cursor()
    
    # Images with no faces
    cursor.execute("SELECT COUNT(*) FROM images WHERE face_count = 0")
    no_faces = cursor.fetchone()[0]
    print(f"  No faces: {no_faces} images")
    
    # Images with 1 face
    cursor.execute("SELECT COUNT(*) FROM images WHERE face_count = 1")
    one_face = cursor.fetchone()[0]
    print(f"  One face: {one_face} images")
    
    # Images with multiple faces
    cursor.execute("SELECT COUNT(*) FROM images WHERE face_count > 1")
    multiple_faces = cursor.fetchone()[0]
    print(f"  Multiple faces: {multiple_faces} images")
    
    # Example 5: Date-based filtering
    print("\n📅 Date-based Filtering:")
    
    # Images from last 7 days
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    cursor.execute("SELECT COUNT(*) FROM images WHERE scan_date > ?", (week_ago,))
    recent = cursor.fetchone()[0]
    print(f"  Last 7 days: {recent} images")
    
    # Images from last 30 days
    month_ago = (datetime.now() - timedelta(days=30)).isoformat()
    cursor.execute("SELECT COUNT(*) FROM images WHERE scan_date > ?", (month_ago,))
    recent_month = cursor.fetchone()[0]
    print(f"  Last 30 days: {recent_month} images")
    
    # Example 6: Advanced custom queries
    print("\n🚀 Advanced Custom Queries:")
    
    # Images with long text descriptions
    cursor.execute("""
        SELECT path, LENGTH(extracted_text) as text_length 
        FROM images 
        WHERE extracted_text IS NOT NULL AND LENGTH(extracted_text) > 50
        ORDER BY text_length DESC 
        LIMIT 3
    """)
    long_text = cursor.fetchall()
    print(f"  Images with lots of text: {len(long_text)}")
    for img_path, text_len in long_text:
        print(f"    {os.path.basename(img_path)}: {text_len} characters")
    
    # Images with many tags
    cursor.execute("""
        SELECT path, tags
        FROM images 
        WHERE tags IS NOT NULL AND LENGTH(tags) - LENGTH(REPLACE(tags, ',', '')) > 5
        LIMIT 3
    """)
    many_tags = cursor.fetchall()
    print(f"  Images with many tags: {len(many_tags)}")
    for img_path, tags in many_tags:
        tag_count = len(tags.split(',')) if tags else 0
        print(f"    {os.path.basename(img_path)}: {tag_count} tags")
    
    # Example 7: Similarity search simulation
    print("\n🎯 Tag Similarity Examples:")
    
    # Find images similar to a reference image by comparing tags
    all_images = cursor.execute("SELECT id, path, tags FROM images WHERE tags IS NOT NULL").fetchall()
    
    if len(all_images) >= 2:
        reference_img = all_images[0]
        ref_tags = set(reference_img[2].split(',')) if reference_img[2] else set()
        
        similarities = []
        for img in all_images[1:]:
            img_tags = set(img[2].split(',')) if img[2] else set()
            common_tags = ref_tags.intersection(img_tags)
            similarity = len(common_tags) / max(len(ref_tags.union(img_tags)), 1)
            similarities.append((img[1], similarity, len(common_tags)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Reference: {os.path.basename(reference_img[1])}")
        print(f"  Similar images:")
        for img_path, similarity, common in similarities[:3]:
            print(f"    {os.path.basename(img_path)}: {similarity:.2f} similarity ({common} common tags)")
    
    print("\n🎉 Search examples complete!")
    print("These queries can be integrated into a full search interface.")

if __name__ == "__main__":
    search_examples()
