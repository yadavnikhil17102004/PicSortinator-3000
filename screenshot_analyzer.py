"""
Screenshot Analyzer - Digital Investigation Tool
Enhanced Version: Google Cloud Vision AI + SQLite + FastAPI

This module processes screenshot images using Google Cloud Vision API for OCR, 
label detection, and SafeSearch. It categorizes images into chats, transactions,
threats, adult content, or uncategorized, and stores the results in a SQLite database.
"""

import os
import sys
import json
import uuid
import re
import cv2
import numpy as np
import datetime
import hashlib
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from PIL import Image
import pytesseract
import spacy
from google.cloud import vision
from google.oauth2 import service_account
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Base class for SQLAlchemy models
Base = declarative_base()

# Tesseract path setup (Windows users need to set this)
# For Windows: Replace with your Tesseract installation path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define database models
class Image(Base):
    __tablename__ = 'images'
    
    id = Column(String(36), primary_key=True)
    file_path = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    detected_text = Column(Text, nullable=True)
    category = Column(String(50), nullable=False, default='uncategorized', index=True)
    safe_search = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    tags = relationship("ImageTag", back_populates="image")
    
class Tag(Base):
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    tag_type = Column(String(50), nullable=False, index=True)
    frequency = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
class ImageTag(Base):
    __tablename__ = 'image_tags'
    
    image_id = Column(String(36), ForeignKey('images.id'), primary_key=True)
    tag_id = Column(Integer, ForeignKey('tags.id'), primary_key=True)
    confidence = Column(Float, nullable=True)
    image = relationship("Image", back_populates="tags")

# Categories enum
class Category(str, Enum):
    CHAT = "chat"
    TRANSACTION = "transaction"
    THREAT = "threat"
    ADULT = "adult_content"
    UNCATEGORIZED = "uncategorized"

# Tag types enum
class TagType(str, Enum):
    ENTITY = "entity"
    LABEL = "label"
    SAFESEARCH = "safesearch"
    KEYWORD = "keyword"


class ScreenshotAnalyzer:
    """Class for analyzing screenshot images for digital investigations."""
    
    def __init__(self, db_path='screenshot_db.sqlite', credentials_path=None):
        """
        Initialize the Screenshot Analyzer.
        
        Args:
            db_path (str): Path to SQLite database file.
            credentials_path (str): Path to Google Cloud Vision API credentials JSON.
        """
        # Initialize database
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize Google Cloud Vision client
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Use default credentials (environment variable GOOGLE_APPLICATION_CREDENTIALS)
            self.vision_client = vision.ImageAnnotatorClient()
        
        # Load spaCy NLP model for entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model not found, download it
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize threat word patterns
        self.threat_patterns = [
            r'threat', r'kill', r'die', r'death', r'murder', r'bomb', 
            r'attack', r'hurt', r'harm', r'danger', r'warning', r'expose',
            r'blackmail', r'extort', r'revenge', r'leak', r'publish',
            r'pay.*or else', r'deadline', r'last chance', r'consequences'
        ]
        
        # Initialize transaction patterns
        self.transaction_patterns = [
            r'(?:₹|\$|€|£|\bRS|\bINR\b)\s?\d[\d,]*\.?\d*',  # Currency
            r'(?:payment|transaction|transfer|credit|debit|deposit|withdraw)',
            r'(?:bank|account|wallet|UPI|NEFT|RTGS|IMPS)',
            r'(?:receipt|invoice|bill|paid|received|sent)',
            r'(?:balance|amount|total|sum|price|fee)',
        ]
        
        # Initialize chat patterns
        self.chat_patterns = [
            r'(?:message|chat|conversation|text|reply|said)',
            r'(?:whatsapp|telegram|signal|messenger|instagram|facebook|twitter)',
            r'(?:group|contact|friend|user)',
            r'(?:\bhi\b|\bhey\b|\bhello\b|\bthanks\b|\bty\b|\bok\b|\byes\b|\bno\b)',
        ]
        
    def process_image(self, image_path):
        """
        Process the input image using Google Cloud Vision API.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            dict: JSON-compatible dictionary with detection results.
        """
        # Check if the file exists and is an image
        if not os.path.exists(image_path):
            return self._create_error_response("File not found")
            
        if not self._is_valid_image(image_path):
            return self._create_error_response("Invalid image file")
        
        # Generate file hash
        file_hash = self._calculate_file_hash(image_path)
        
        # Check if this image has already been processed
        session = self.Session()
        existing_image = session.query(Image).filter_by(file_hash=file_hash).first()
        if existing_image:
            # Return existing analysis
            return self._create_response_from_db(existing_image)
        
        # Generate a unique ID for the image
        image_id = str(uuid.uuid4())
        
        try:
            # Read image file for Vision API
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            vision_image = vision.Image(content=content)
            
            # Features to extract from the image
            features = [
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=10),
                vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
            ]
            
            # Call Vision API
            response = self.vision_client.annotate_image({'image': vision_image, 'features': features})
            
            # Extract OCR text
            detected_text = ""
            if response.text_annotations:
                detected_text = response.text_annotations[0].description
            
            # Extract labels
            labels = []
            for label in response.label_annotations:
                labels.append({
                    'name': label.description,
                    'score': label.score
                })
            
            # Extract SafeSearch results
            safe_search = {
                'adult': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.adult),
                'violence': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.violence),
                'racy': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.racy)
            }
            
            # Extract entities using spaCy
            entities = self._extract_entities(detected_text)
            
            # Determine category
            category = self._classify_content(detected_text, labels, safe_search, entities)
            
            # Combine all tags
            all_tags = self._generate_tags(detected_text, labels, entities, category, safe_search)
            
            # Create DB record
            db_image = Image(
                id=image_id,
                file_path=image_path,
                file_hash=file_hash,
                detected_text=detected_text,
                category=category.value,
                safe_search=safe_search,
                created_at=datetime.datetime.utcnow()
            )
            
            # Add tags to DB
            for tag in all_tags:
                tag_obj = session.query(Tag).filter_by(name=tag['name']).first()
                if tag_obj:
                    # Update existing tag frequency
                    tag_obj.frequency += 1
                else:
                    # Create new tag
                    tag_obj = Tag(
                        name=tag['name'],
                        tag_type=tag['type'],
                        frequency=1
                    )
                    session.add(tag_obj)
                    session.flush()  # To get the tag ID
                
                # Link tag to image
                image_tag = ImageTag(
                    image_id=image_id,
                    tag_id=tag_obj.id,
                    confidence=tag.get('confidence')
                )
                db_image.tags.append(image_tag)
            
            # Save to database
            session.add(db_image)
            session.commit()
            
            # Create response
            response = {
                "image_id": image_id,
                "file_hash": file_hash,
                "detected_text": detected_text,
                "entities": [e['text'] for e in entities],
                "labels": [l['name'] for l in labels],
                "tags": [t['name'] for t in all_tags],
                "category": category.value,
                "safe_search": safe_search,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            
            return response
            
        except Exception as e:
            session.rollback()
            return self._create_error_response(f"Processing error: {str(e)}")
        finally:
            session.close()
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_entities(self, text):
        """Extract entities from text using spaCy."""
        entities = []
        
        if not text:
            return entities
        
        # Extract using spaCy
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'confidence': None  # spaCy doesn't provide confidence scores
            })
        
        # Extract phone numbers
        phone_patterns = [
            r'(?:\+?91[-\s]?)?[6-9]\d{9}',  # Indian mobile
            r'\+?1[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',  # US/Canada
            r'\b\d{10,12}\b'  # Generic phone numbers
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(0),
                    'type': 'PHONE_NUMBER',
                    'confidence': None
                })
        
        # Extract email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        for match in re.finditer(email_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'EMAIL',
                'confidence': None
            })
        
        # Extract amounts
        amount_pattern = r'(?:₹|\$|€|£|\bRS|\bINR\b)\s?\d[\d,]*\.?\d*'
        for match in re.finditer(amount_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'AMOUNT',
                'confidence': None
            })
        
        # Extract card numbers (last 4 digits only)
        card_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        for match in re.finditer(card_pattern, text):
            card_num = match.group(0)
            last4 = card_num[-4:]
            entities.append({
                'text': f"**** **** **** {last4}",
                'type': 'CARD',
                'confidence': None
            })
        
        return entities
    
    def _classify_content(self, text, labels, safe_search, entities):
        """Classify content into predefined categories."""
        # Check for adult content first based on SafeSearch
        if safe_search['adult'] in ['LIKELY', 'VERY_LIKELY'] or safe_search['racy'] in ['LIKELY', 'VERY_LIKELY']:
            return Category.ADULT
        
        # Check for threats
        for pattern in self.threat_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return Category.THREAT
        
        # Check for transactions
        transaction_score = 0
        for pattern in self.transaction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            transaction_score += len(matches)
        
        # Check for chat
        chat_score = 0
        for pattern in self.chat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            chat_score += len(matches)
        
        # Check labels for app interfaces
        app_labels = ['chat app', 'messenger', 'whatsapp', 'telegram', 
                      'social media', 'message', 'conversation']
        for label in labels:
            if any(app in label['name'].lower() for app in app_labels):
                chat_score += 3
        
        # Determine category based on scores
        if transaction_score > 3 and transaction_score > chat_score:
            return Category.TRANSACTION
        elif chat_score > 3:
            return Category.CHAT
        else:
            return Category.UNCATEGORIZED
    
    def _generate_tags(self, text, labels, entities, category, safe_search):
        """Generate tags from all available data."""
        tags = []
        
        # Add category as tag
        tags.append({
            'name': category.value,
            'type': TagType.KEYWORD,
            'confidence': 1.0
        })
        
        # Add entities as tags
        for entity in entities:
            tags.append({
                'name': f"{TagType.ENTITY.value}.{entity['type'].lower()}.{entity['text']}",
                'type': TagType.ENTITY,
                'confidence': entity.get('confidence')
            })
        
        # Add labels as tags
        for label in labels:
            tags.append({
                'name': f"{TagType.LABEL.value}.{label['name'].lower()}",
                'type': TagType.LABEL,
                'confidence': label.get('score')
            })
        
        # Add SafeSearch flags as tags
        for key, value in safe_search.items():
            if value in ['POSSIBLE', 'LIKELY', 'VERY_LIKELY']:
                tags.append({
                    'name': f"{TagType.SAFESEARCH.value}.{key}.{value.lower()}",
                    'type': TagType.SAFESEARCH,
                    'confidence': None
                })
        
        # Extract keywords from text
        if text:
            # Simple keyword extraction (can be improved)
            words = re.findall(r'\b\w{4,}\b', text.lower())
            word_freq = {}
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
            
            # Add top keywords as tags
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                tags.append({
                    'name': f"{TagType.KEYWORD.value}.{word}",
                    'type': TagType.KEYWORD,
                    'confidence': freq / len(words) if words else 0
                })
        
        return tags
    
    def _create_response_from_db(self, db_image):
        """Create response dictionary from database image record."""
        session = self.Session()
        
        try:
            # Get all tags for this image
            image_tags = session.query(ImageTag).filter_by(image_id=db_image.id).all()
            tag_ids = [t.tag_id for t in image_tags]
            tags = session.query(Tag).filter(Tag.id.in_(tag_ids)).all()
            
            # Separate tags by type
            entity_tags = [t.name for t in tags if t.tag_type == TagType.ENTITY]
            label_tags = [t.name for t in tags if t.tag_type == TagType.LABEL]
            all_tags = [t.name for t in tags]
            
            # Extract entities from entity tags
            entities = []
            for tag in entity_tags:
                parts = tag.split('.')
                if len(parts) >= 3:
                    entities.append('.'.join(parts[2:]))
            
            # Extract labels from label tags
            labels = []
            for tag in label_tags:
                parts = tag.split('.')
                if len(parts) >= 2:
                    labels.append('.'.join(parts[1:]))
            
            return {
                "image_id": db_image.id,
                "file_hash": db_image.file_hash,
                "detected_text": db_image.detected_text,
                "entities": entities,
                "labels": labels,
                "tags": all_tags,
                "category": db_image.category,
                "safe_search": db_image.safe_search,
                "timestamp": db_image.created_at.isoformat()
            }
        finally:
            session.close()
        
    def _is_valid_image(self, image_path):
        """
        Check if the file is a valid image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            bool: True if the file is a valid image, False otherwise.
        """
        try:
            img = Image.open(image_path)
            img.verify()  # Verify it's an image
            return True
        except:
            return False
    
    def _create_error_response(self, error_message):
        """
        Create an error response.
        
        Args:
            error_message (str): Error message.
            
        Returns:
            dict: Error response as a dictionary.
        """
        return {
            "image_id": "error",
            "processable": False,
            "detected_text": "",
            "entities": [],
            "labels": [],
            "tags": [],
            "category": "error",
            "safe_search": {},
            "error": error_message,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    
    def search_images(self, query=None, tags=None, category=None, limit=100, offset=0):
        """
        Search for images based on query, tags, or category.
        
        Args:
            query (str): Text to search for in detected_text.
            tags (list): List of tags to filter by.
            category (str): Category to filter by.
            limit (int): Maximum number of results to return.
            offset (int): Number of results to skip.
            
        Returns:
            list: List of matching images.
        """
        session = self.Session()
        try:
            # Start with a base query
            query_obj = session.query(Image)
            
            # Apply filters
            if category:
                query_obj = query_obj.filter(Image.category == category)
            
            if query:
                query_obj = query_obj.filter(Image.detected_text.like(f'%{query}%'))
            
            if tags:
                for tag in tags:
                    # Get tag IDs
                    tag_ids = session.query(Tag.id).filter(Tag.name.like(f'%{tag}%')).all()
                    tag_ids = [t[0] for t in tag_ids]
                    
                    # Get image IDs with these tags
                    if tag_ids:
                        image_ids = session.query(ImageTag.image_id).filter(ImageTag.tag_id.in_(tag_ids)).all()
                        image_ids = [i[0] for i in image_ids]
                        query_obj = query_obj.filter(Image.id.in_(image_ids))
            
            # Apply pagination
            query_obj = query_obj.order_by(Image.created_at.desc()).offset(offset).limit(limit)
            
            # Execute query
            results = query_obj.all()
            
            # Format results
            formatted_results = []
            for img in results:
                formatted_results.append(self._create_response_from_db(img))
            
            return formatted_results
        finally:
            session.close()
    
    def export_results(self, results, format_type='json', output_path=None):
        """
        Export search results to a file.
        
        Args:
            results (list): List of image results to export.
            format_type (str): Export format ('json', 'csv', or 'pdf').
            output_path (str): Path to save the exported file.
            
        Returns:
            str: Path to the exported file.
        """
        if not output_path:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'export_{timestamp}.{format_type}'
        
        if format_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        elif format_type == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['image_id', 'file_hash', 'category', 'detected_text', 
                                'entities', 'labels', 'tags', 'timestamp'])
                
                # Write data
                for result in results:
                    writer.writerow([
                        result['image_id'],
                        result['file_hash'],
                        result['category'],
                        result['detected_text'][:200] + '...' if len(result.get('detected_text', '')) > 200 else result.get('detected_text', ''),
                        ', '.join(result.get('entities', [])),
                        ', '.join(result.get('labels', [])),
                        ', '.join(result.get('tags', [])),
                        result.get('timestamp', '')
                    ])
        
        elif format_type == 'pdf':
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Add title
            elements.append(Paragraph(f"Screenshot Analysis Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                    styles['Title']))
            elements.append(Spacer(1, 12))
            
            # Add each image as a section
            for result in results:
                elements.append(Paragraph(f"Image ID: {result['image_id']}", styles['Heading2']))
                elements.append(Paragraph(f"Category: {result['category']}", styles['Normal']))
                elements.append(Paragraph(f"Timestamp: {result.get('timestamp', '')}", styles['Normal']))
                
                # Add tags table
                if result.get('tags'):
                    elements.append(Paragraph("Tags:", styles['Heading3']))
                    tag_data = [['Tag']]
                    for tag in result.get('tags', []):
                        tag_data.append([tag])
                    
                    tag_table = Table(tag_data, colWidths=[400])
                    tag_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(tag_table)
                    elements.append(Spacer(1, 12))
                
                # Add text excerpt
                if result.get('detected_text'):
                    elements.append(Paragraph("Detected Text:", styles['Heading3']))
                    elements.append(Paragraph(result.get('detected_text', '')[:500] + 
                                            ('...' if len(result.get('detected_text', '')) > 500 else ''), 
                                            styles['Normal']))
                
                elements.append(Spacer(1, 24))
            
            doc.build(elements)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        return output_path


# FastAPI application
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import shutil
import uvicorn
from pathlib import Path

app = FastAPI(
    title="Screenshot Analyzer API",
    description="API for analyzing screenshots using Google Cloud Vision",
    version="1.0.0",
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize analyzer
analyzer = ScreenshotAnalyzer()

@app.post("/upload", response_model=dict)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process an image.
    
    Returns JSON with OCR text, labels, tags, category, etc.
    """
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image
    result = analyzer.process_image(str(file_path))
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.get("/search", response_model=List[dict])
async def search_images(
    query: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Search for images by text, tags, or category.
    """
    results = analyzer.search_images(query, tags, category, limit, offset)
    return results

@app.post("/export")
async def export_results(
    query: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    category: Optional[str] = None,
    format_type: str = "json"
):
    """
    Export search results to a file.
    
    Returns the path to the exported file.
    """
    # Get search results
    results = analyzer.search_images(query, tags, category)
    
    # Export to file
    if format_type not in ["json", "csv", "pdf"]:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format_type}")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'export_{timestamp}.{format_type}'
    
    try:
        exported_file = analyzer.export_results(results, format_type, output_path)
        return FileResponse(
            path=exported_file,
            filename=Path(exported_file).name,
            media_type=f"application/{format_type}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main function to run the Screenshot Analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python screenshot_analyzer.py <image_path> [credentials_path]")
        print("  or use: python screenshot_analyzer.py --server [port] [credentials_path]")
        sys.exit(1)
    
    if sys.argv[1] == "--server":
        # Run as API server
        port = 8000
        credentials_path = None
        
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                credentials_path = sys.argv[2]
        
        if len(sys.argv) > 3:
            credentials_path = sys.argv[3]
        
        # Initialize analyzer with credentials if provided
        global analyzer
        analyzer = ScreenshotAnalyzer(credentials_path=credentials_path)
        
        # Start the server
        print(f"Starting API server on port {port}...")
        uvicorn.run("screenshot_analyzer:app", host="0.0.0.0", port=port, reload=False)
    else:
        # Run as command-line tool
        image_path = sys.argv[1]
        credentials_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Initialize analyzer with credentials if provided
        analyzer = ScreenshotAnalyzer(credentials_path=credentials_path)
        
        # Process the image
        result = analyzer.process_image(image_path)
        
        # Print the result as formatted JSON
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
