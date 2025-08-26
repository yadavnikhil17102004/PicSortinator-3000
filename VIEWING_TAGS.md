# How to View Image Tags in PicSortinator-3000

There are several ways to view and search the tags that PicSortinator-3000 has generated for your images:

## 1. Using view_image.py

This script allows you to view detailed information about a specific image, including its tags, extracted text, and face detection results.

```bash
# View an image by ID
python view_image.py id <image_id>

# View an image by filename (partial match works)
python view_image.py file <filename>
```

Example:
```bash
python view_image.py id 10
python view_image.py file beach
```

## 2. Using query_tags.py

This script queries the database for images with specific tags and provides statistics about your collection.

```bash
python query_tags.py
```

You can modify this script to search for specific tags by editing the SQL query.

## 3. Using the search command

```bash
python main.py search "<tag_name>"
```

Example:
```bash
python main.py search "phone"
python main.py search "landscape_orientation"
```

## 4. Direct SQL Queries

You can also run direct SQL queries against the database:

```bash
python -c "import sqlite3; conn = sqlite3.connect('data/picsortinator.db'); cursor = conn.cursor(); cursor.execute('SELECT id, filename, tags FROM images WHERE tags LIKE \"%phone%\"'); rows = cursor.fetchall(); for row in rows: print(f'Image ID: {row[0]}, Filename: {row[1]}, Tags: {row[2]}'); conn.close()"
```

## 5. Common Tags to Search For

The system generates several types of tags:

- **Orientation Tags**: `landscape_orientation`, `portrait_orientation`, `square_format`
- **Format Tags**: `jpeg_format`, `png_format`, etc.
- **Resolution Tags**: `high_resolution`, `low_resolution`
- **Object Tags**: `phone`, `kitchen`, `computer`, `tree`, `flower`, etc.
- **Class Tags**: `class NNN` (these are ImageNet class IDs)

## 6. Organizing Images by Tags

To organize your images into folders based on their primary tag:

```bash
python main.py organize
```

This will create folders in the `output` directory for each tag and copy the images there.

## Need more help?

Check the README.md file for more information on how to use PicSortinator-3000.
