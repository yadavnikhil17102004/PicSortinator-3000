"""
Utility functions for the Image Sorter application.
"""

import os
import sqlite3
import random
from pathlib import Path

def setup_database(db_path):
    """Create and set up the SQLite database if it doesn't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            path TEXT UNIQUE NOT NULL,
            size INTEGER,
            tags TEXT,
            extracted_text TEXT,
            faces TEXT,
            creation_date TIMESTAMP,
            processed BOOLEAN DEFAULT 0
        )
    ''')
    
    conn.commit()
    return conn

def get_image_files(directory_path):
    """Get all image files from a directory."""
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Define valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # Find all image files
    image_files = []
    for file_path in directory.glob('**/*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_files.append(file_path)
    
    return image_files

def funny_message(message_type, count=0):
    """Return a funny message based on the context."""
    messages = {
        "loading": [
            f"Holy pixels! You've got {count} images. Do you even delete anything?",
            f"Found {count} images. That's a lot of memories... or memes. Probably memes.",
            f"Loading {count} images. This is like your phone's gallery, but with fewer selfies.",
            f"Scanning {count} pics. I hope they're rated PG..."
        ],
        "processing": [
            "Processing images... finding all those embarrassing photos you forgot about.",
            "Working hard or hardly working? Unlike you, I'm actually processing these images.",
            f"Processed {count} images so far. The things I've seen cannot be unseen.",
            "Still processing... this is taking longer than your Instagram scrolling sessions."
        ],
        "complete": [
            f"Done! Processed {count} images. You're welcome.",
            f"All {count} images processed. I've seen things you people wouldn't believe...",
            f"Finished with {count} images. I need therapy after some of those.",
            f"Processing complete! {count} down, 0 to go. Freedom at last!"
        ],
        "organize": [
            "Images organized! Marie Kondo would be so proud.",
            f"Sorted {count} images. I'm basically a digital janitor for your mess.",
            "Images are now in folders. Unlike your real desk, this is actually organized.",
            f"Organized {count} photos. Your cat photos now live in their own kingdom."
        ],
        "error": [
            "Error! Something went wrong. Have you tried turning it off and on again?",
            "Houston, we have a problem. And by Houston, I mean your computer.",
            "Task failed successfully. Wait, that doesn't make sense...",
            "Error 404: Sense of humor not found."
        ]
    }
    
    # If count is over 1000, add a special message
    if count > 1000 and message_type == "complete":
        return "Congratulations! You're officially a digital hoarder."
    
    # Return a random message from the appropriate category
    category = messages.get(message_type, messages["error"])
    return random.choice(category)
