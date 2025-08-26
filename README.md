# PicSortinator 3000

Sort, tag, and laugh at your messy photo collection — all offline.

## Features

ML tagging, OCR text search, face grouping, and sarcastic comments.

- **Multi-format image support**: Handles JPG, PNG, BMP, GIF, TIFF, and WebP files
- **Automatic ML tagging**: AI-powered content identification with confidence scoring
- **OCR text extraction**: Makes text in your images searchable
- **Face detection and grouping**: Automatically groups photos by people
- **Duplicate detection**: Find identical images using file hashing
- **Advanced database**: SQLite with full-text search and metadata indexing
- **Smart organization**: Sort images into folders by primary tags or custom rules
- **Multiple export formats**: CSV, JSON, or HTML reports of your collection
- **Offline operation**: All processing happens locally - no cloud required
- **Humorous interface**: Because organizing 10,000 photos should be fun

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

## Quick Start

```bash
# Scan a directory for images
python main.py scan /path/to/your/photos

# Process images with ML tagging, OCR, and face detection
python main.py process

# Search for images with specific tags
python main.py search "phone" 

# View detailed information about a specific image
python view_image.py id <image_id>
python view_image.py file <filename>

# Query database for images with specific tags
python query_tags.py

# Organize images by tags
python main.py organize

# Export your collection data
python main.py export json
```

## Setup Notes

### Tesseract OCR Configuration

The system is configured to use Tesseract OCR from this default location:
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If Tesseract is installed in a different location, you can modify the path in `modules/ocr.py`.

### ML Model Information

The application uses MobileNetV2 pre-trained on ImageNet for image classification. The model is automatically downloaded on first run and cached in the `models` directory.

### Face Detection

Basic face detection is implemented using OpenCV's Haar cascades. For advanced face recognition with dlib, additional setup may be required:

```bash
# On Windows with Visual Studio installed:
pip install cmake
pip install dlib
pip install face_recognition
```

## Project Structure

```
PicSortinator-3000/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md               # This file
├── view_image.py           # Tool to view detailed image info
├── query_tags.py           # Tool to query the database
├── modules/
│   ├── __init__.py         # Package definition
│   ├── loader.py           # Image scanning and metadata
│   ├── tagging.py          # ML-based auto tagging
│   ├── ocr.py              # Text recognition
│   ├── faces.py            # Face detection + grouping
│   ├── database.py         # SQLite handler
│   ├── model_manager.py    # ML model management
│   └── utils.py            # Shared helper functions
├── models/                 # ML models and data
│   ├── mobilenet_v2_imagenet.h5  # Cached ML model
│   └── imagenet_labels.txt       # Class labels
├── data/                   # Database storage
│   └── picsortinator.db    # SQLite database
├── output/                 # Sorted/exported results
└── tests/                  # Test scripts
```

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Project setup and database design
- [x] Basic image loading and metadata extraction
- [x] CLI interface foundation

### Phase 2: Analysis Engine ✅
- [x] ML-based image tagging with pre-trained models (MobileNetV2)
- [x] OCR text extraction and indexing (Tesseract)
- [x] Basic face detection (OpenCV)
- [x] Image orientation and format detection

### Phase 3: Smart Organization 📋
- [ ] Intelligent folder sorting algorithms
- [ ] Custom tagging rules and filters
- [ ] Batch operations and undo functionality
- [ ] Advanced search with boolean operators

### Phase 4: Polish & Fun 🎨
- [ ] Progress bars and humor during long operations
- [ ] Statistics dashboard and collection insights
- [ ] Export to photo management formats
- [ ] Plugin system for custom processors

### Future Ideas 💡
- [ ] Web interface for remote access
- [ ] Integration with cloud photo services
- [ ] Video file support
- [ ] Mobile companion app

## Contributing

Found a bug? Have a feature idea? Want to add more sarcastic comments?
Contributions welcome!

## License

MIT License - Sort responsibly! 📸
