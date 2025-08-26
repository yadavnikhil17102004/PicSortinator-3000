import sqlite3
import sys

def get_image_details(image_id=None, filename=None):
    conn = sqlite3.connect('data/picsortinator.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if image_id:
        cursor.execute('''
            SELECT id, filename, path, tags, extracted_text, faces, processed
            FROM images 
            WHERE id = ?
        ''', (image_id,))
    elif filename:
        cursor.execute('''
            SELECT id, filename, path, tags, extracted_text, faces, processed
            FROM images 
            WHERE filename LIKE ?
        ''', ('%' + filename + '%',))
    else:
        print("Please provide either image_id or filename")
        return
    
    row = cursor.fetchone()
    
    if not row:
        print("No image found")
        return
    
    print(f"Image ID: {row['id']}")
    print(f"Filename: {row['filename']}")
    print(f"Path: {row['path']}")
    print(f"Processed: {'Yes' if row['processed'] else 'No'}")
    print("\nTags:")
    if row['tags']:
        for tag in row['tags'].split(','):
            print(f"  - {tag}")
    else:
        print("  No tags")
    
    print("\nExtracted Text:")
    if row['extracted_text']:
        print(f"  {row['extracted_text'][:200]}...")
        if len(row['extracted_text']) > 200:
            print("  (truncated for display)")
    else:
        print("  No text extracted")
    
    print("\nFaces:")
    if row['faces']:
        print(f"  {row['faces']}")
    else:
        print("  No faces detected")
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_image.py [id <image_id> | file <filename>]")
        sys.exit(1)
    
    if sys.argv[1] == "id" and len(sys.argv) >= 3:
        get_image_details(image_id=sys.argv[2])
    elif sys.argv[1] == "file" and len(sys.argv) >= 3:
        get_image_details(filename=sys.argv[2])
    else:
        print("Usage: python view_image.py [id <image_id> | file <filename>]")
