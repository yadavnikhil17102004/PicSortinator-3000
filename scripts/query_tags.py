import sqlite3

# Connect to the database
conn = sqlite3.connect('data/picsortinator.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Find images with specific object tags
print("\n--- Images with specific object tags ---")
cursor.execute('''
    SELECT id, filename, tags 
    FROM images 
    WHERE tags LIKE '%phone%' 
       OR tags LIKE '%kitchen%' 
       OR tags LIKE '%computer%' 
       OR tags LIKE '%tree%'
       OR tags LIKE '%flower%'
''')
for row in cursor.fetchall():
    print(f"Image ID: {row['id']}, Filename: {row['filename']}, Tags: {row['tags']}")

# Get statistics on the tag types
print("\n--- Tag Category Statistics ---")
cursor.execute("SELECT COUNT(*) FROM images WHERE tags LIKE '%landscape_orientation%'")
landscape_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM images WHERE tags LIKE '%portrait_orientation%'")
portrait_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM images WHERE tags LIKE '%square_format%'")
square_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM images WHERE tags NOT LIKE '%class%'")
without_class_tags = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM images")
total_images = cursor.fetchone()[0]

print(f"Total images: {total_images}")
print(f"Landscape orientation: {landscape_count} ({landscape_count/total_images*100:.1f}%)")
print(f"Portrait orientation: {portrait_count} ({portrait_count/total_images*100:.1f}%)")
print(f"Square format: {square_count} ({square_count/total_images*100:.1f}%)")
print(f"Images without class tags: {without_class_tags} ({without_class_tags/total_images*100:.1f}%)")

# Show some sample images with extracted text
print("\n--- Sample Images with Extracted Text ---")
cursor.execute("SELECT id, filename, tags, extracted_text FROM images LIMIT 5")
for row in cursor.fetchall():
    print(f"Image ID: {row['id']}, Filename: {row['filename']}")
    print(f"Tags: {row['tags']}")
    print(f"Text Sample: {row['extracted_text'][:100]}...")
    print("-" * 50)

# Images with faces
print("\n--- Images with Faces ---")
cursor.execute("SELECT id, filename, faces FROM images WHERE faces IS NOT NULL AND faces != ''")
face_images = cursor.fetchall()
print(f"Found {len(face_images)} images with faces")
for row in face_images[:5]:  # Show first 5 examples
    print(f"Image ID: {row['id']}, Filename: {row['filename']}, Faces: {row['faces']}")

# Close the connection
conn.close()
