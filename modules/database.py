#!/usr/bin/env python3
"""
PicSortinator 3000 - Database Module
====================================

Handles SQLite database operations for storing image metadata and analysis results.

💾 Fun Fact: This database has indexed more memories than your brain can store!
🔍 Easter Egg: Full-text search so powerful, it can find that one meme from 2019.
📊 Warning: May become sentient and start organizing your life outside of photos.
"""

import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for PicSortinator 3000."""
    
    def __init__(self, db_path="data/picsortinator.db"):
        """
        Initialize database manager.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._ensure_data_directory()
        self._initialize_database()
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(exist_ok=True)
    
    def _initialize_database(self):
        """Initialize database connection and create tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self._create_tables()
            logger.info(f"Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                path TEXT UNIQUE NOT NULL,
                file_hash TEXT,
                size INTEGER,
                width INTEGER,
                height INTEGER,
                format TEXT,
                mode TEXT,
                creation_date TIMESTAMP,
                modified_date TIMESTAMP,
                scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                has_exif BOOLEAN DEFAULT FALSE,
                camera_make TEXT,
                camera_model TEXT,
                datetime_taken TIMESTAMP,
                orientation INTEGER,
                tags TEXT,
                extracted_text TEXT,
                faces TEXT,
                ml_confidence REAL,
                notes TEXT
            )
        ''')
        
        # Tags table for normalized tag storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT UNIQUE NOT NULL,
                category TEXT,
                usage_count INTEGER DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Image-tag relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_tags (
                image_id INTEGER,
                tag_id INTEGER,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (image_id, tag_id),
                FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
            )
        ''')
        
        # Face detection results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                face_number INTEGER,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                confidence REAL,
                encoding BLOB,
                person_id TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_path ON images (path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_hash ON images (file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_processed ON images (processed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags (tag_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_image ON faces (image_id)')
        
        self.conn.commit()
    
    def add_image(self, metadata):
        """
        Add a new image to the database.
        
        Args:
            metadata (dict): Image metadata dictionary
            
        Returns:
            int: ID of the inserted image, or None if failed
        """
        try:
            cursor = self.conn.cursor()
            
            # Check if image already exists
            cursor.execute("SELECT id FROM images WHERE path = ?", (metadata.get('path'),))
            if cursor.fetchone():
                logger.warning(f"Image already exists in database: {metadata.get('path')}")
                return None
            
            # Insert new image
            cursor.execute('''
                INSERT INTO images (
                    filename, path, file_hash, size, width, height, format, mode,
                    creation_date, modified_date, has_exif, camera_make, camera_model,
                    datetime_taken, orientation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.get('filename'),
                metadata.get('path'),
                metadata.get('file_hash'),
                metadata.get('size'),
                metadata.get('width'),
                metadata.get('height'),
                metadata.get('format'),
                metadata.get('mode'),
                metadata.get('creation_date'),
                metadata.get('modified_date'),
                metadata.get('has_exif', False),
                metadata.get('camera_make'),
                metadata.get('camera_model'),
                metadata.get('datetime_taken'),
                metadata.get('orientation')
            ))
            
            image_id = cursor.lastrowid
            self.conn.commit()
            logger.info(f"Added image to database: {metadata.get('filename')} (ID: {image_id})")
            return image_id
            
        except Exception as e:
            logger.error(f"Failed to add image to database: {e}")
            return None
    
    def update_image_analysis(self, image_id, tags=None, extracted_text=None, faces=None, confidence=None):
        """
        Update image with analysis results.
        
        Args:
            image_id (int): Image ID
            tags (list): List of tags
            extracted_text (str): OCR extracted text
            faces (list): List of face data
            confidence (float): ML confidence score
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert tags to string if provided
            tags_str = ",".join(tags) if tags else None
            faces_str = ",".join([f"person_{i}" for i in range(len(faces))]) if faces else None
            
            cursor.execute('''
                UPDATE images 
                SET tags = ?, extracted_text = ?, faces = ?, ml_confidence = ?, processed = TRUE
                WHERE id = ?
            ''', (tags_str, extracted_text, faces_str, confidence, image_id))
            
            self.conn.commit()
            logger.info(f"Updated analysis for image ID: {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to update image analysis: {e}")
            
    def update_image_processing(self, image_id, tags=None, extracted_text=None, faces=None, confidence=None):
        """
        Update image with processing results.
        
        Args:
            image_id (int): Image ID
            tags (list): List of tags
            extracted_text (str): OCR extracted text
            faces (int or list): Number of faces or face data
            confidence (float): ML confidence score
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert tags to string if provided
            tags_str = ",".join(tags) if tags else None
            
            # Handle faces parameter (could be count or actual face data)
            faces_str = None
            if isinstance(faces, int) and faces > 0:
                faces_str = ",".join([f"person_{i}" for i in range(faces)])
            elif isinstance(faces, list) and faces:
                faces_str = ",".join([f"person_{i}" for i in range(len(faces))])
            
            cursor.execute('''
                UPDATE images 
                SET tags = ?, extracted_text = ?, faces = ?, ml_confidence = ?, processed = TRUE
                WHERE id = ?
            ''', (tags_str, extracted_text, faces_str, confidence, image_id))
            
            self.conn.commit()
            logger.info(f"Updated processing for image ID: {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update image processing: {e}")
            return False
    
    def get_unprocessed_images(self, limit=None):
        """
        Get unprocessed images from the database.
        
        Args:
            limit (int): Maximum number of images to return
            
        Returns:
            list: List of unprocessed image records
        """
        try:
            cursor = self.conn.cursor()
            query = "SELECT id, path FROM images WHERE processed = FALSE"
            if limit:
                query += f" LIMIT {limit}"
                
            cursor.execute(query)
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Failed to get unprocessed images: {e}")
            return []
    
    def search_images(self, query, search_type="all"):
        """
        Search for images based on various criteria.
        
        Args:
            query (str): Search query
            search_type (str): Type of search ('all', 'tags', 'text', 'faces')
            
        Returns:
            list: List of matching image records
        """
        try:
            cursor = self.conn.cursor()
            results = []
            
            if search_type in ["tags", "all"]:
                cursor.execute('''
                    SELECT id, filename, path, tags 
                    FROM images 
                    WHERE tags LIKE ?
                ''', (f"%{query}%",))
                results.extend(cursor.fetchall())
            
            if search_type in ["text", "all"]:
                cursor.execute('''
                    SELECT id, filename, path, extracted_text 
                    FROM images 
                    WHERE extracted_text LIKE ?
                ''', (f"%{query}%",))
                results.extend(cursor.fetchall())
            
            if search_type in ["faces", "all"]:
                cursor.execute('''
                    SELECT id, filename, path, faces 
                    FROM images 
                    WHERE faces LIKE ?
                ''', (f"%{query}%",))
                results.extend(cursor.fetchall())
            
            # Remove duplicates by ID
            unique_results = []
            seen_ids = set()
            for result in results:
                if result[0] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result[0])
                    
            return unique_results
            
        except Exception as e:
            logger.error(f"Failed to search images: {e}")
            return []
    
    def get_all_images(self):
        """Get all images from the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, filename, path, size, tags, extracted_text, faces, creation_date
                FROM images
                ORDER BY creation_date DESC
            ''')
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get all images: {e}")
            return []
    
    def get_processed_images(self, limit=None):
        """
        Get processed images from the database.
        
        Args:
            limit (int): Maximum number of images to return
            
        Returns:
            list: List of processed image records
        """
        try:
            cursor = self.conn.cursor()
            query = '''
                SELECT id, filename, path, tags, extracted_text, faces 
                FROM images 
                WHERE processed = TRUE
            '''
            
            if limit:
                query += f" LIMIT {limit}"
                
            cursor.execute(query)
            
            # Convert to dictionary for easier access
            columns = [column[0] for column in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to get processed images: {e}")
            return []
    
    def get_image_stats(self):
        """Get database statistics."""
        try:
            cursor = self.conn.cursor()
            
            # Total images
            cursor.execute("SELECT COUNT(*) FROM images")
            total_images = cursor.fetchone()[0]
            
            # Processed images
            cursor.execute("SELECT COUNT(*) FROM images WHERE processed = TRUE")
            processed_images = cursor.fetchone()[0]
            
            # Images with tags
            cursor.execute("SELECT COUNT(*) FROM images WHERE tags IS NOT NULL AND tags != ''")
            tagged_images = cursor.fetchone()[0]
            
            # Images with text
            cursor.execute("SELECT COUNT(*) FROM images WHERE extracted_text IS NOT NULL AND extracted_text != ''")
            text_images = cursor.fetchone()[0]
            
            # Images with faces
            cursor.execute("SELECT COUNT(*) FROM images WHERE faces IS NOT NULL AND faces != ''")
            face_images = cursor.fetchone()[0]
            
            return {
                'total_images': total_images,
                'processed_images': processed_images,
                'tagged_images': tagged_images,
                'text_images': text_images,
                'face_images': face_images,
                'processing_rate': (processed_images / total_images * 100) if total_images > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get image stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
