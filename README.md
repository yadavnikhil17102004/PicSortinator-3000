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

# Search for images
python main.py search "beach sunset"

# Organize images by tags
python main.py organize

# Export your collection data
python main.py export json
```

## Project Structure

```
PicSortinator-3000/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md               # This file
├── modules/
│   ├── loader.py           # Image scanning and metadata
│   ├── tagging.py          # ML-based auto tagging
│   ├── ocr.py              # Text recognition
│   ├── faces.py            # Face detection + grouping
│   ├── database.py         # SQLite handler
│   └── utils.py            # Shared helper functions
├── data/                   # Input images for testing
├── output/                 # Sorted/exported results
└── tests/                  # Test scripts
```

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Project setup and database design
- [x] Basic image loading and metadata extraction
- [x] CLI interface foundation

### Phase 2: Analysis Engine 🚧
- [ ] ML-based image tagging with pre-trained models
- [ ] OCR text extraction and indexing
- [ ] Face detection and clustering
- [ ] Duplicate image detection

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
