#!/usr/bin/env python3
"""
🎉 PicSortinator 3000 Web Interface - Simplified Launch Version
🚀 A beautiful, fun, and cute photo organizer!
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory, abort
    from werkzeug.utils import secure_filename
    import threading
    import uuid
    from datetime import datetime
    
    print("✅ Flask and basic dependencies loaded successfully!")
    
except ImportError as e:
    print(f"❌ Error importing basic dependencies: {e}")
    print("Please install Flask: pip install flask")
    sys.exit(1)

# Try to import our AI modules with graceful fallbacks
AI_MODULES_AVAILABLE = True
db = None
tagger = None
face_detector = None
ocr = None

try:
    from modules.database import Database
    db = Database()
    print("✅ Database module loaded successfully!")
except ImportError as e:
    print(f"⚠️  Database module not available: {e}")
    AI_MODULES_AVAILABLE = False

# Skip TensorFlow-dependent modules for now to speed up startup
print("� Skipping heavy AI modules for faster startup...")
print("💡 AI processing will be loaded on-demand when needed!")

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'picsortinator-3000-is-awesome!'
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components (with fallbacks)
if AI_MODULES_AVAILABLE and db:
    print("🤖 Database ready for photo storage!")
else:
    print("📁 Running in file-only mode - uploads will be stored locally")

# Processing stats for real-time updates
processing_stats = {
    'total_processed': 0,
    'currently_processing': 0,
    'recent_uploads': [],
    'ai_available': AI_MODULES_AVAILABLE
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_cute_message():
    """Get a random cute message for the homepage"""
    messages = [
        "🌟 Your photos are looking fabulous today!",
        "📸 Ready to organize some memories?",
        "🎨 Let's make your photo collection sparkle!",
        "💝 PicSortinator is here to help!",
        "🚀 Time for some photo magic!",
        "🌈 Your pictures deserve the best organization!",
        "✨ Making photo management delightful since 2025!",
        "🎉 Let's sort some beautiful memories!"
    ]
    import random
    return random.choice(messages)

@app.route('/')
def index():
    """🏠 Our beautiful home page!"""
    if AI_MODULES_AVAILABLE and db:
        try:
            stats = db.get_statistics()
        except:
            stats = {'total_images': 0, 'total_faces': 0, 'total_tags': 0}
    else:
        stats = {'total_images': 0, 'total_faces': 0, 'total_tags': 0}
    
    return render_template('index.html', 
                         stats=stats, 
                         cute_message=get_cute_message(),
                         processing_stats=processing_stats)

@app.route('/gallery')
def gallery():
    """🖼️ Beautiful image gallery!"""
    if not AI_MODULES_AVAILABLE:
        return render_template('gallery.html', images=[], page=1, demo_mode=True)
    
    page = request.args.get('page', 1, type=int)
    per_page = 24
    
    try:
        # Get images from uploads folder for demo
        images = []
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        if upload_folder.exists():
            for i, img_file in enumerate(upload_folder.glob('*')):
                if img_file.is_file() and allowed_file(img_file.name):
                    images.append({
                        'id': i,
                        'filename': img_file.name,
                        'path': str(img_file),
                        'tags': ['demo', 'sample'],
                        'faces': 0,
                        'has_text': False,
                        'created_at': datetime.now().isoformat()
                    })
        
        return render_template('gallery.html', images=images, page=page)
    except Exception as e:
        print(f"Gallery error: {e}")
        return render_template('gallery.html', images=[], page=1)

@app.route('/search')
def search():
    """🔍 Magical search page!"""
    return render_template('search.html')

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
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(upload_path)
                
                uploaded_files.append({
                    'original_name': filename,
                    'status': 'uploaded' if not AI_MODULES_AVAILABLE else 'processing',
                    'message': '📁 File uploaded successfully!' if not AI_MODULES_AVAILABLE else '🤖 AI processing will be available soon!'
                })
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'message': f'🎉 Uploaded {len(uploaded_files)} files!'
        })
    
    return render_template('upload.html')

@app.route('/api/stats')
def api_stats():
    """📊 Real-time stats API"""
    if AI_MODULES_AVAILABLE and db:
        try:
            stats = db.get_statistics()
        except:
            stats = {'total_images': 0, 'total_faces': 0, 'total_tags': 0}
    else:
        stats = {'total_images': 0, 'total_faces': 0, 'total_tags': 0}
    
    stats.update(processing_stats)
    return jsonify(stats)

@app.route('/api/search')
def api_search():
    """🔍 Search API (demo mode)"""
    query = request.args.get('q', '').strip()
    
    if not AI_MODULES_AVAILABLE:
        return jsonify({
            'results': [],
            'count': 0,
            'query': query,
            'message': 'AI search will be available once all dependencies are installed!'
        })
    
    # Real search logic would go here
    return jsonify({
        'results': [],
        'count': 0,
        'query': query
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """🖼️ Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/image/<int:image_id>')
def serve_image(image_id):
    """🖼️ Serve images by ID"""
    # For demo, serve from uploads folder
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    image_files = list(upload_folder.glob('*'))
    
    if 0 <= image_id < len(image_files):
        img_file = image_files[image_id]
        return send_from_directory(str(img_file.parent), img_file.name)
    
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
    print("🎆 Starting PicSortinator 3000 Web Interface!")
    
    if AI_MODULES_AVAILABLE:
        print("✨ Full AI mode - All features available!")
    else:
        print("🎭 Demo mode - Upload and browse features available")
        print("💡 To enable AI features, install: tensorflow, mediapipe")
    
    print("🌈 Open your browser to: http://localhost:5000")
    print("💝 Made with love and way too much caffeine!")
    
    # Launch our adorable app!
    app.run(debug=True, host='0.0.0.0', port=5000)
