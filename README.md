# 🎆 PicSortinator 3000

**The ML-Powered Photo Organizer with Attitude**

Sort, tag, and laugh at your messy photo collection — all offline, with zero judgment (okay, maybe a little).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-green.svg)](https://github.com/yadavnikhil17102004/PicSortinator-3000)

## ✨ Features

PicSortinator 3000 brings military-grade organization to your photo chaos:

### 🤖 **AI-Powered Analysis**
- **Smart ML Tagging**: MobileNetV2-powered content recognition with 1000+ ImageNet classes
- **OCR Text Extraction**: Makes text in your screenshots and documents searchable
- **Face Detection**: Groups photos by people (no more faces in your curry photos! 🍛)
- **Confidence Scoring**: Know how sure the AI is about its classifications
- **Easter Egg Detection**: Hidden jokes for developers who read the code (you found one!) 🥚

### 📁 **Advanced Organization**
- **Multi-format Support**: JPG, PNG, BMP, GIF, TIFF, WebP, and more
- **Smart Database**: SQLite with full-text search and metadata indexing
- **Duplicate Detection**: Find identical images using perceptual hashing
- **Batch Processing**: Handle thousands of images efficiently

### 🔍 **Powerful Search**
- **Tag-based Search**: Find images by content (`python main.py search "car"`)
- **Text Search**: Search within extracted OCR text
- **Advanced Queries**: Boolean operators and complex filters
- **Export Options**: CSV, JSON, HTML reports

### 🛡️ **Privacy & Control**
- **100% Offline**: All processing happens locally - no cloud required
- **No Data Collection**: Your photos stay on your machine
- **Open Source**: Audit the code, contribute improvements

### 🎭 **Personality**
- **Sarcastic Interface**: Because organizing 10,000 photos should be fun
- **Progress Humor**: Entertaining messages during long operations
- **Achievement System**: Get roasted for your photography choices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended for large photo collections
- Tesseract OCR (installation instructions below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yadavnikhil17102004/PicSortinator-3000.git
   cd PicSortinator-3000
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR:**
   - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

### Basic Usage

```bash
# 1. Scan a directory for images
python main.py scan /path/to/your/photos

# 2. Process with AI analysis
python main.py process

# 3. Search your collection
python main.py search "sunset"
python main.py search "dog"

# 4. View detailed image info
python scripts/view_image.py id 123
python scripts/view_image.py file "vacation.jpg"

# 5. Query database
python scripts/query_tags.py

# 6. Organize by tags
python main.py organize

# 7. Export your data
python main.py export json
```

## 📂 Project Structure

```
PicSortinator-3000/
├── 📄 main.py                  # Main CLI application
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # You are here
├── 📄 LICENSE                 # MIT License
├── 📁 modules/                # Core functionality
│   ├── 🧠 tagging.py          # ML image classification
│   ├── 👁️ ocr.py               # Text extraction
│   ├── 👥 faces.py            # Face detection
│   ├── 🗄️ database.py         # Data management
│   ├── 📦 loader.py           # Image scanning
│   ├── 🤖 model_manager.py    # ML model handling
│   └── 🛠️ utils.py            # Helper functions
├── 📁 scripts/               # Utility scripts
│   ├── 🔍 view_image.py       # Image viewer/inspector
│   ├── 🔎 query_tags.py       # Database query tool
│   └── 🧪 test_pipeline.py    # Testing utilities
├── 📁 docs/                  # Documentation
│   ├── 📋 CONTRIBUTING.md     # Contribution guidelines
│   ├── 📝 CHANGELOG.md        # Version history
│   └── 👀 VIEWING_TAGS.md     # Tag viewing guide
├── 📁 models/                # ML models (auto-downloaded)
├── 📁 data/                  # Database storage
├── 📁 output/                # Organized photos & exports
├── 📁 tests/                 # Test files
└── 📁 examples/              # Usage examples
```

## 🎯 Use Cases

### 📸 **Photography Enthusiasts**
- Automatically tag nature, portrait, landscape photos
- Find photos by equipment used (extracted from EXIF)
- Organize by shooting location or date

### 💼 **Content Creators**
- Search screenshots by contained text
- Group photos by people for easy access
- Export metadata for portfolio organization

### 👨‍👩‍👧‍👦 **Family Archives**
- Face-based photo grouping for family albums
- Search birthday photos by text in images
- Organize events by automatically detected tags

### 🏢 **Business Use**
- Document management with OCR search
- Asset organization by visual content
- Batch processing for large image libraries

## 🛠️ Advanced Configuration

### Tesseract OCR Setup
Default Windows path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

For custom installations, modify in `modules/ocr.py`:
```python
tesseract_path = r'C:\Your\Custom\Path\tesseract.exe'
```

### Performance Tuning
- **Memory**: 8GB+ RAM recommended for 10k+ images
- **Storage**: SSD recommended for database performance
- **CPU**: Multi-core helps with batch processing

### Face Recognition Enhancement
For advanced face recognition with dlib:
```bash
# Requires Visual Studio Build Tools on Windows
pip install cmake
pip install dlib
pip install face_recognition
```

## 🤝 Contributing

We welcome contributions! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/yadavnikhil17102004/PicSortinator-3000.git
cd PicSortinator-3000
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run example
python scripts/test_pipeline.py
```

## 📊 Roadmap

### ✅ Phase 1: Core Infrastructure (Complete)
- [x] Project setup and database design
- [x] Basic image loading and metadata extraction
- [x] CLI interface foundation

### ✅ Phase 2: Analysis Engine (Complete)
- [x] ML-based image tagging (MobileNetV2)
- [x] OCR text extraction (Tesseract)
- [x] Face detection (OpenCV)
- [x] Comprehensive tag filtering

### 🚧 Phase 3: Smart Organization (In Progress)
- [ ] Intelligent folder sorting algorithms
- [ ] Custom tagging rules and filters
- [ ] Batch operations with undo
- [ ] Boolean search operators

### 🎨 Phase 4: Polish & Fun (Planned)
- [ ] Web interface for remote access
- [ ] Statistics dashboard
- [ ] Photo timeline visualization
- [ ] Plugin system for custom processors

### 💡 Future Ideas
- [ ] Mobile companion app
- [ ] Video file support
- [ ] Cloud service integrations
- [ ] Machine learning model fine-tuning

## 📝 License

MIT License - Sort responsibly! 📸

## 🙏 Acknowledgments

- **TensorFlow Team** - For MobileNetV2 and ImageNet
- **Tesseract OCR** - For open-source text recognition
- **OpenCV** - For computer vision capabilities
- **Python Community** - For amazing libraries

## 🐛 Issues & Support

Found a bug? Have a feature request? Want to add more sarcastic comments?

- 🐛 [Report Issues](https://github.com/yadavnikhil17102004/PicSortinator-3000/issues)
- 💡 [Feature Requests](https://github.com/yadavnikhil17102004/PicSortinator-3000/discussions)
- 📖 [Documentation](docs/)

---

**Made with ❤️ and a healthy dose of sarcasm by developers who understand photo organization pain.**
