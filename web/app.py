#!/usr/bin/env python3
"""
🎆 PicSortinator 3000 - Web Interface 🎆
======================================

The most adorable photo organizer on the internet! ✨
Powered by AI, seasoned with love, served with a smile! 😊

🌈 Features:
- Drag & drop upload (so satisfying!)
- Real-time AI analysis (watch the magic happen!)
- Beautiful image galleries (eye candy galore!)
- Search that actually works (revolutionary!)
- Cute animations everywhere (because why not?)

💝 Made with love, coffee, and way too many emojis!

🥚 Hidden Feature: Check the console logs - our developers left Easter eggs for curious minds!
"""

import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, url_for
from werkzeug.utils import secure_filename
import threading
import time

# Add parent directory to path for our AI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules with graceful fallbacks
try:
    from modules.database import DatabaseManager
    from modules.tagging_lite import ImageTagger  # Use lightweight version instead
    from modules.faces import FaceDetector
    from modules.ocr import TextExtractor
    print("✅ All modules loaded successfully!")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("🔄 Trying to continue with basic functionality...")
    DatabaseManager = None
    ImageTagger = None
    FaceDetector = None
    TextExtractor = None

# 🎨 Initialize our adorable Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'picsortinator-3000-is-absolutely-adorable'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 🤖 Initialize our AI superstars (with fallbacks)
db = None
tagger = None
face_detector = None
ocr = None

if DatabaseManager:
    try:
        db = DatabaseManager()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")

if ImageTagger:
    try:
        tagger = ImageTagger()
        print("✅ Lightweight image tagger initialized")
    except Exception as e:
        print(f"⚠️ Image tagger initialization failed: {e}")

if FaceDetector:
    try:
        face_detector = FaceDetector()
        print("✅ Face detector initialized")
    except Exception as e:
        print(f"⚠️ Face detector initialization failed: {e}")

if TextExtractor:
    try:
        ocr = TextExtractor()
        print("✅ OCR text extractor initialized")
    except Exception as e:
        print(f"⚠️ OCR initialization failed: {e}")

# 📊 Global stats for our cute dashboard
processing_stats = {
    'total_processed': 0,
    'currently_processing': 0,
    'recent_uploads': []
}

# 🎭 Allowed file extensions (we're picky but cute about it!)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if file is an image (with style!)"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_cute_message():
    """Get a random cute message for the UI ✨"""
    messages = [
        "✨ Your photos are looking absolutely fabulous!",
        "🌈 AI magic is happening behind the scenes!",
        "🎭 Organizing your memories with love and precision!",
        "🚀 Transforming chaos into beautiful order!",
        "💝 Your digital life is about to get so much prettier!",
        "🎪 The AI circus is in town and it's spectacular!",
        "🌟 Making your photo collection sparkle!",
        "🎨 Painting your memories with intelligent tags!"
    ]
    import random
    return random.choice(messages)

def process_image_async(image_path, filename):
    """Process an image with all our AI goodness (in the background!)"""
    try:
        processing_stats['currently_processing'] += 1
        
        # Extract metadata first
        from modules.loader import ImageLoader
        loader = ImageLoader()
        metadata = loader.extract_metadata(Path(image_path))
        
        if not metadata:
            print(f"❌ Failed to extract metadata for {filename}")
            return False
        
        # 💾 Save to database first
        img_id = db.add_image(metadata)
        if not img_id:
            print(f"❌ Failed to add image to database: {filename}")
            return False
        
        # 🏷️ Get AI tags
        tags = tagger.tag_image(image_path) if tagger else []
        
        # 👥 Detect faces
        faces = face_detector.detect_faces(image_path) if face_detector else []
        
        # 📄 Extract text
        text = ocr.extract_text(image_path) if ocr else ""
        
        # Update database with processing results
        db.update_image_processing(img_id, tags, text, len(faces))
        
        # 📊 Update stats
        processing_stats['total_processed'] += 1
        processing_stats['recent_uploads'].append({
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'tags': tags[:3],  # Show first 3 tags
            'faces': len(faces),
            'has_text': bool(text and len(text.strip()) > 3)
        })
        
        # Keep only last 10 uploads
        if len(processing_stats['recent_uploads']) > 10:
            processing_stats['recent_uploads'] = processing_stats['recent_uploads'][-10:]
            
        return True
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return False
    finally:
        processing_stats['currently_processing'] -= 1

@app.route('/')
def index():
    """🏠 Our beautiful home page!"""
    try:
        if db:
            stats = db.get_image_stats()  # Use the correct method name
        else:
            stats = {'total_images': 0, 'processed_images': 0, 'tagged_images': 0, 'text_images': 0, 'face_images': 0}
    except Exception as e:
        print(f"Error getting stats: {e}")
        stats = {'total_images': 0, 'processed_images': 0, 'tagged_images': 0, 'text_images': 0, 'face_images': 0}
        
    return render_template('index.html', 
                         stats=stats, 
                         cute_message=get_cute_message(),
                         processing_stats=processing_stats)

@app.route('/gallery')
def gallery():
    """🖼️ Beautiful image gallery!"""
    page = request.args.get('page', 1, type=int)
    per_page = 24  # Show 24 images per page (perfect grid!)
    
    if not db:
        return render_template('gallery.html', images=[], page=page, error="Database not available")
    
    try:
        # Get recent images
        cursor = db.conn.cursor()
        offset = (page - 1) * per_page
        cursor.execute("""
            SELECT id, path, tags, face_count, extracted_text, scan_date
            FROM images 
            ORDER BY scan_date DESC 
            LIMIT ? OFFSET ?
        """, (per_page, offset))
        
        images = []
        for row in cursor.fetchall():
            img_id, file_path, tags, face_count, text, scan_date = row
            images.append({
                'id': img_id,
                'filename': os.path.basename(file_path) if file_path else 'Unknown',
                'path': file_path,
                'tags': tags.split(',') if tags else [],
                'faces': face_count or 0,
                'has_text': bool(text and len(text.strip()) > 3),
                'created_at': scan_date
            })
        
        return render_template('gallery.html', images=images, page=page)
        
    except Exception as e:
        print(f"Gallery error: {e}")
        return render_template('gallery.html', images=[], page=page, error=str(e))

@app.route('/search')
def search():
    """🔍 Magical search page!"""
    query = request.args.get('q', '').strip()
    results = []
    
    if query:
        # Search by tags
        tag_results = db.search_by_tags([query])
        
        # Search by text
        text_results = db.search_by_text(query)
        
        # Combine and deduplicate
        all_results = tag_results + text_results
        seen_ids = set()
        
        for row in all_results:
            img_id = row[0]
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                results.append({
                    'id': img_id,
                    'filename': os.path.basename(row[1]),
                    'path': row[1],
                    'tags': row[2].split(',') if row[2] else [],
                    'faces': row[4] or 0,
                    'has_text': bool(row[3] and len(row[3].strip()) > 3)
                })
    
    return render_template('search.html', query=query, results=results)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """📤 Adorable upload page!"""
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files selected! 😢'})
        
        files = request.files.getlist('files[]')
        uploaded_files = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                # 🔒 Secure the filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                
                # 💾 Save the file
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(upload_path)
                
                # 🚀 Process in background
                threading.Thread(
                    target=process_image_async, 
                    args=(upload_path, filename),
                    daemon=True
                ).start()
                
                uploaded_files.append({
                    'original_name': filename,
                    'status': 'processing',
                    'message': '🤖 AI is analyzing your beautiful photo!'
                })
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'message': f'🎉 Uploaded {len(uploaded_files)} files! AI magic in progress!'
        })
    
    return render_template('upload.html')

@app.route('/api/stats')
def api_stats():
    """📊 Real-time stats API (so we can show live updates!)"""
    try:
        if db:
            stats = db.get_image_stats()  # Use the correct method name
        else:
            stats = {'total_images': 0, 'processed_images': 0, 'tagged_images': 0, 'text_images': 0, 'face_images': 0}
    except Exception as e:
        print(f"Error getting API stats: {e}")
        stats = {'total_images': 0, 'processed_images': 0, 'tagged_images': 0, 'text_images': 0, 'face_images': 0}
        
    stats.update(processing_stats)
    return jsonify(stats)

@app.route('/api/processing-status')
def processing_status():
    """🔄 Check processing status (for real-time updates!)"""
    return jsonify(processing_stats)

@app.route('/api/search')
def api_search():
    """🔍 Advanced search API with filters!"""
    try:
        query = request.args.get('q', '').strip()
        sort_by = request.args.get('sort', 'date_desc')
        has_faces = request.args.get('has_faces', '')
        has_text = request.args.get('has_text', '')
        
        # Build the search conditions
        conditions = []
        params = []
        
        if query:
            # Search in tags, filename, and text content
            conditions.append("(tags LIKE ? OR filename LIKE ? OR extracted_text LIKE ?)")
            search_term = f"%{query}%"
            params.extend([search_term, search_term, search_term])
        
        if has_faces == 'yes':
            conditions.append("face_count > 0")
        elif has_faces == 'no':
            conditions.append("(face_count = 0 OR face_count IS NULL)")
            
        if has_text == 'yes':
            conditions.append("extracted_text IS NOT NULL AND extracted_text != ''")
        elif has_text == 'no':
            conditions.append("(extracted_text IS NULL OR extracted_text = '')")
        
        # Build WHERE clause
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        # Build ORDER BY clause
        order_map = {
            'date_desc': 'scan_date DESC',
            'date_asc': 'scan_date ASC',
            'name_asc': 'filename ASC',
            'name_desc': 'filename DESC',
            'size_desc': 'size DESC',
            'size_asc': 'size ASC'
        }
        order_clause = f"ORDER BY {order_map.get(sort_by, 'scan_date DESC')}"
        
        # Execute search
        cursor = db.conn.cursor()
        query_sql = f"""
            SELECT id, filename, path, size, scan_date, 
                   tags, face_count, extracted_text
            FROM images 
            {where_clause}
            {order_clause}
            LIMIT 500
        """
        
        cursor.execute(query_sql, params)
        results = []
        
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'filename': row[1] or os.path.basename(row[2]),
                'file_path': row[2],
                'file_size': row[3] or 0,
                'processed_date': row[4],
                'tags': row[5] or '',
                'face_count': row[6] or 0,
                'text_content': row[7] or ''
            })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': query,
            'filters': {
                'sort_by': sort_by,
                'has_faces': has_faces,
                'has_text': has_text
            }
        })
        
    except Exception as e:
        print(f"🔍 Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """🖼️ Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/image/<int:image_id>')
def serve_image(image_id):
    """🖼️ Serve images by ID (for gallery and search)"""
    try:
        cursor = db.conn.cursor()
        cursor.execute("SELECT path FROM images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        
        if not result:
            abort(404)
        
        file_path = result[0]
        if not os.path.exists(file_path):
            abort(404)
            
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        return send_from_directory(directory, filename)
        
    except Exception as e:
        print(f"🖼️ Error serving image {image_id}: {e}")
        abort(404)

@app.errorhandler(404)
def not_found(error):
    """😅 Cute 404 page!"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    """😰 Adorable error page!"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # 🌟 Make sure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("🎆 Starting PicSortinator 3000 Web Interface!")
    print("✨ Get ready for some serious photo organization magic!")
    print("🌈 Open your browser to: http://localhost:5000")
    print("💝 Made with love and way too much caffeine!")
    
    # 🚀 Launch our adorable app!
    app.run(debug=True, host='0.0.0.0', port=5000)
