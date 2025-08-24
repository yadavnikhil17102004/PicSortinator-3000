# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

ForenSnap Ultimate is an AI-powered digital forensics suite designed for analyzing screenshots in criminal investigations. It combines multiple AI technologies (BLIP, CLIP, BERT, Tesseract, EasyOCR) to automatically categorize, extract text, detect threats, identify platforms, and generate legal-compliant reports from screenshot evidence.

## Core Architecture

### Main Entry Points
- `forensnap_ultimate_complete.py` - Complete integrated system with all AI features
- `forensnap_ultimate.py` - Legacy version with core functionality  
- `run_forensnap_ultimate.py` - Smart launcher that selects the best version
- `run_forensnap.bat` - Windows batch launcher with dependency management
- `setup_forensnap.bat` - One-time setup script with desktop shortcuts

### Modular Structure
The system follows a plugin-based architecture with these key modules:

**Core Processing Pipeline**:
1. **OCR Layer** (`modules/enhanced_ocr.py`) - Hybrid Tesseract + EasyOCR with 40+ language support
2. **AI Analysis Layer** - BLIP for image captioning, CLIP for NSFW detection, BERT for threat analysis
3. **Platform Detection** - WhatsApp, Telegram, Instagram, Facebook, Twitter, Discord, Signal identification
4. **Content Categorization** - Chat, Transaction, Threat, Adult Content, Social Media classification
5. **Database Layer** - SQLite with forensic-grade metadata and chain of custody

**Specialized Modules**:
- `modules/nsfw_detector.py` - Local CLIP-based adult content detection
- `modules/face_object_detector.py` - Face recognition and evidence object detection
- `core/advanced_nlp_threat.py` - BERT-powered threat assessment with 14 categories
- `core/advanced_search.py` - Fuzzy search with TF-IDF similarity matching

## Common Development Commands

### Setup and Installation
```bash
# First-time setup (creates virtual environment and shortcuts)
setup_forensnap.bat

# Manual installation of dependencies
pip install -r requirements_core.txt
python -m spacy download en_core_web_sm

# Install enhanced dependencies for full features
pip install -r requirements.txt
```

### Running the Application
```bash
# Launch GUI (recommended)
run_forensnap.bat
# or
python run_forensnap_ultimate.py

# Command line usage
python run_forensnap_ultimate.py analyze "screenshot.png"
python run_forensnap_ultimate.py batch "C:\Evidence\Screenshots"
python run_forensnap_ultimate.py search "threatening message"
python run_forensnap_ultimate.py report
```

### Testing and Development
```bash
# Test OCR functionality
python run_forensnap_ultimate.py test-ocr "test_image.png"

# Test specific module
python test_analyzer.py
python -m pytest tests/ (if test directory exists)

# Check dependencies
python test_dependencies.py
```

### Database Management
```bash
# Database is automatically created at: forensnap_ultimate.db
# Backup database
copy forensnap_ultimate.db forensnap_backup.db

# View database schema (SQLAlchemy models in main files)
# Key tables: images, tags, image_tags, cases, investigators
```

## Key Technical Concepts

### AI Model Pipeline
The system loads multiple AI models in sequence:
1. **BLIP Model** - Image captioning for visual understanding
2. **CLIP Model** - Semantic analysis for NSFW detection  
3. **BERT/RoBERTa** - Contextual threat analysis
4. **spaCy Models** - Entity extraction and linguistic analysis

### Platform Detection Logic
Uses hybrid approach combining:
- **UI Color Analysis** - Platform-specific color schemes (WhatsApp green, Telegram blue)
- **Text Pattern Matching** - Platform-specific UI elements and terminology
- **Visual Element Detection** - Icons, buttons, layout patterns
- **Confidence Scoring** - Weighted combination of all detection methods

### Threat Detection Categories
Advanced NLP system classifies threats into 14 categories:
- Violence, Harassment, Blackmail, Self-harm, Hate Speech
- Stalking, Extortion, Weapon References, Deadline Threats
- Target Analysis (family, location, personal info)

### Database Schema Design
Forensic-grade schema with:
- **Chain of Custody** - SHA-256 file hashing for integrity verification
- **Audit Trails** - Complete processing history with timestamps
- **Metadata Preservation** - AI model versions, confidence scores, processing parameters
- **Investigation Context** - Case linking, investigator notes, evidence relationships

## Development Guidelines

### Adding New AI Models
1. Add model initialization in the `DependencyManager.REQUIRED_PACKAGES`
2. Create model loading logic in the main analyzer class
3. Implement analysis method following the pattern: `analyze_[feature]`
4. Add confidence scoring and error handling
5. Update database schema if storing new metadata

### Extending Platform Detection
1. Add platform definition to `PlatformDetector.PLATFORM_SIGNATURES`
2. Define color patterns, text patterns, and visual elements
3. Implement confidence scoring logic
4. Test with representative screenshots from the platform

### Adding New Content Categories
1. Update `Category` enum in database models
2. Add classification logic in `categorize_content` method
3. Create pattern matching rules for the new category
4. Update legal reporting templates to handle new category

### Performance Considerations
- **Memory Management** - Models are loaded once and cached
- **Batch Processing** - Uses threading for parallel image processing  
- **GPU Acceleration** - PyTorch models can utilize CUDA if available
- **Database Indexing** - Key fields (file_hash, category, timestamp) are indexed

## File Structure Navigation

### Configuration
- `config/settings.py` - Central configuration for all components
- Environment variables override defaults for production deployment

### Legacy Versions  
- `archive_old_versions/` - Previous versions for compatibility testing
- Version numbers indicate major feature additions

### Data Storage
- `data/` - SQLite database, exported reports, cached models
- `logs/` - Application logs with detailed debugging information
- `uploads/` - Temporary storage for batch processing

## Troubleshooting Common Issues

### Dependency Conflicts
- Python 3.13 compatibility issues with some AI packages
- Use `requirements_core.txt` for minimal working installation
- GPU/CPU versions of PyTorch may conflict

### OCR Performance
- Tesseract installation path may need manual configuration on Windows
- EasyOCR provides fallback when Tesseract unavailable
- Image preprocessing critical for accuracy

### AI Model Loading
- Models download automatically on first run (may take time)
- Disk space requirements: ~2GB for all models
- Network connectivity required for initial model downloads

### Memory Usage
- Batch processing may exceed system memory with large images
- Reduce batch size in `PERFORMANCE_SETTINGS` if needed
- Consider processing images sequentially for memory-constrained systems

## Integration Notes

This is a Windows-optimized forensics tool with:
- Batch file launchers for easy deployment
- Virtual environment management for dependency isolation
- Professional GUI designed for law enforcement workflows
- Legal compliance features for court admissibility

The modular architecture allows individual components to be extracted and reused in other forensic tools while maintaining the integrated workflow for comprehensive screenshot analysis.
