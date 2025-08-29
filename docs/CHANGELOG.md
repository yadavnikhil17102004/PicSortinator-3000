# Changelog

All notable changes to the PicSortinator-3000 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-27

### Added
- Initial project setup with modular architecture
- ML-based image tagging using MobileNetV2
- OCR text extraction with Tesseract
- Basic face detection using OpenCV
- SQLite database for image metadata
- Image scanning and metadata extraction
- Command-line interface for basic operations
- Tag querying and image viewing tools
- Comprehensive README with setup instructions

### Fixed
- OCR configuration to properly find Tesseract executable
- Missing methods in tagging and database modules:
  - ImageTagger.tag_image method
  - DatabaseManager.update_image_processing method
  - DatabaseManager.get_processed_images method

### Changed
- Updated project structure documentation
- Improved error handling in image processing pipeline
- Enhanced database schema for better organization

### Known Issues
- Advanced face recognition requires additional setup (dlib)
- Some ML tags may be generic (class IDs instead of descriptive labels)
- OCR text extraction quality varies depending on image clarity
