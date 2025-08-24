# 🔬 ForenSnap Ultimate - AI-Powered Digital Forensics Suite

**Version 2.0.0** | Advanced Screenshot Analysis for Digital Investigations

ForenSnap Ultimate is a comprehensive AI-powered digital forensics tool designed specifically for investigating digital evidence through automated screenshot analysis. Built for law enforcement, cybersecurity professionals, and digital forensics investigators.

## 🌟 Key Features

### 🤖 Advanced AI Analysis
- **Multi-language OCR**: Extracts text using both Tesseract and EasyOCR with automatic language detection
- **Social Media Detection**: Identifies platforms (WhatsApp, Telegram, Instagram, Facebook, Twitter, Discord, Signal)
- **Threat Analysis**: Advanced NLP-based threat detection with severity scoring
- **NSFW Content Detection**: Local AI-powered adult content identification using CLIP models
- **Image Captioning**: BLIP-based visual understanding and description generation

### 🔍 Investigation Capabilities
- **Comprehensive Entity Extraction**: Phone numbers, emails, URLs, currency amounts, dates
- **Category Classification**: Automatically sorts content into chats, transactions, threats, adult content
- **Threat Level Assessment**: Grades threats from low to critical with detailed analysis
- **Platform-Specific Analysis**: Tailored detection for different social media interfaces
- **Chain of Custody**: Maintains cryptographic integrity with SHA-256 file hashing

### 📊 Professional Reporting
- **Legal-Compliant PDF Reports**: Court-ready documentation with detailed analysis
- **Executive Summaries**: High-level overview for management and legal teams
- **Technical Details**: Comprehensive metadata for expert testimony
- **Evidence Authentication**: Digital signatures and integrity verification
- **Export Options**: JSON, CSV, and PDF formats for various use cases

### 🚀 User Experience
- **Modern GUI Interface**: Intuitive tabbed interface for easy navigation
- **Batch Processing**: Analyze hundreds of images automatically
- **Advanced Search**: Filter by category, platform, threat level, and content
- **Progress Tracking**: Real-time processing status and completion notifications
- **Database Management**: SQLite-based storage with advanced querying capabilities

## 🛠️ Installation & Setup

### Option 1: Quick Setup (Recommended)
1. **Run Setup**: Double-click `setup_forensnap.bat`
2. **Follow Prompts**: The setup will guide you through installation
3. **Launch**: Use the created desktop shortcut "ForenSnap Ultimate"

### Option 2: Manual Installation
1. **Install Python**: Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
   - ✅ **Important**: Check "Add Python to PATH" during installation
2. **Run Launcher**: Double-click `run_forensnap.bat`
   - Dependencies will be automatically installed on first run
3. **Start Analysis**: The GUI will launch automatically

### System Requirements
- **OS**: Windows 10/11 (64-bit recommended)
- **Python**: 3.8 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models and dependencies
- **Optional**: CUDA-compatible GPU for faster processing

## 🚀 Usage Guide

### GUI Mode (Recommended for Most Users)
1. **Launch**: Double-click desktop shortcut or run `run_forensnap.bat`
2. **Select Tab**: Choose from four main interfaces:
   - 🔍 **Single Image Analysis**: Analyze individual screenshots
   - ⚡ **Batch Processing**: Process multiple images automatically
   - 🔎 **Search Database**: Query and filter previous results
   - 📄 **Legal Reports**: Generate court-ready documentation

### Command Line Mode (Advanced Users)
```bash
# Analyze single image
run_forensnap.bat analyze "C:\evidence\screenshot.png"

# Batch process entire folder
run_forensnap.bat batch "C:\evidence\screenshots"

# Generate legal report
run_forensnap.bat report

# Search database
run_forensnap.bat search "threatening message"
```

## 📁 File Structure
```
ForenSnap With BLIP/
├── forensnap_ultimate.py      # Main application (comprehensive)
├── run_forensnap.bat          # Windows launcher with dependency management
├── setup_forensnap.bat        # One-time setup and shortcut creator
├── README_ULTIMATE.md         # This documentation
├── requirements.txt           # Python dependencies (legacy)
├── forensnap_ultimate.db      # SQLite database (created on first run)
├── forensnap.log             # Application logs
├── venv/                     # Virtual environment (created by setup)
└── uploads/                  # Temporary file storage
```

## 🔍 Analysis Capabilities

### Content Categories
- **Chat Messages**: WhatsApp, Telegram, SMS, social media conversations
- **Financial Transactions**: Banking, payment apps, cryptocurrency transfers
- **Threat Content**: Harassment, blackmail, violence, hate speech
- **Adult Content**: NSFW material detection with confidence scoring
- **Social Media**: Platform-specific content from various networks
- **Documents**: Screenshots of documents, forms, and official content

### Threat Detection Features
- **Violence Detection**: Weapons, physical threats, assault references
- **Harassment Analysis**: Stalking, intimidation, persistent unwanted contact
- **Blackmail Identification**: Extortion, coercion, threatening exposure
- **Hate Speech**: Discriminatory language, slurs, targeted harassment
- **Suicide Risk**: Self-harm indicators, suicidal ideation detection
- **Escalation Patterns**: Time-sensitive threats, deadlines, ultimatums

### Platform Detection
- **WhatsApp**: Green interface, typing indicators, voice messages
- **Telegram**: Blue interface, forwarded messages, group features
- **Instagram**: Stories, likes, follower counts, photo-centric interface
- **Facebook**: Like/comment/share buttons, timeline features
- **Twitter**: Tweets, retweets, hashtags, @ mentions
- **Discord**: Server channels, voice chat, gaming-focused interface
- **Signal**: Privacy-focused features, disappearing messages

## 📊 Output & Reporting

### Analysis Results Include:
```json
{
  "image_id": "unique-identifier",
  "category": "chat|transaction|threat|adult_content|social_media",
  "platform": "whatsapp|telegram|instagram|facebook|twitter|etc",
  "threat_level": "none|low|medium|high|critical",
  "confidence_score": 0.95,
  "detected_text": "Extracted text content",
  "detected_language": "en|es|fr|de|etc",
  "entities": ["phone_numbers", "emails", "currency"],
  "nsfw_classification": "UNLIKELY|POSSIBLE|LIKELY|VERY_LIKELY",
  "warnings": ["High threat level detected", "Adult content identified"],
  "processing_details": {
    "ocr_method": "hybrid",
    "ocr_confidence": 87.5,
    "platform_confidence": 0.92,
    "threat_score": 0.15,
    "skin_percentage": 0.05
  },
  "blip_caption": "AI-generated image description",
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### Legal Report Contents:
- **Executive Summary**: High-level findings and risk assessment
- **Methodology**: Technical approach and AI model specifications
- **Evidence Inventory**: Complete list of analyzed materials with hashes
- **Detailed Findings**: Category-by-category analysis results
- **Risk Assessment**: Threat level distribution and critical content identification
- **Technical Appendix**: Processing metadata, confidence scores, model versions
- **Chain of Custody**: Cryptographic integrity verification
- **Legal Disclaimers**: Compliance statements and expert validation requirements

## 🔧 Advanced Configuration

### OCR Optimization
- **Tesseract Installation**: For enhanced OCR accuracy
  - Download from: [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Install with multiple language packs for international investigations
- **EasyOCR Settings**: Automatic fallback with 40+ language support
- **Preprocessing**: Automatic image enhancement for better text recognition

### AI Model Settings
- **BLIP Captioning**: Visual-linguistic understanding for context
- **CLIP Analysis**: Semantic content analysis for NSFW detection
- **Language Detection**: Automatic identification of text language
- **Confidence Thresholds**: Adjustable sensitivity for different use cases

### Database Management
- **SQLite Backend**: Lightweight, file-based database for portability
- **Automatic Indexing**: Optimized queries for fast search operations
- **Backup Options**: Database export and recovery capabilities
- **Data Retention**: Configurable retention policies for compliance

## 🛡️ Security & Compliance

### Data Protection
- **Local Processing**: All AI analysis performed locally, no cloud dependencies
- **Encrypted Storage**: Optional database encryption for sensitive cases
- **Access Logging**: Comprehensive audit trails for all operations
- **User Authentication**: Multi-user support with role-based access control

### Legal Compliance
- **Digital Forensics Standards**: Follows NIST and ISO guidelines
- **Evidence Integrity**: SHA-256 hashing for tamper detection
- **Audit Trails**: Complete processing history for court proceedings
- **Expert Validation**: Results marked as requiring human verification
- **Privacy Protection**: GDPR/CCPA compliance features

### Chain of Custody
- **File Hashing**: Cryptographic integrity verification
- **Processing Logs**: Detailed metadata for each analysis step
- **Timestamp Accuracy**: UTC timestamps for international cases
- **Version Control**: Software version tracking for reproducibility

## 🚨 Troubleshooting

### Common Issues & Solutions

#### Installation Problems
- **Python Not Found**: Ensure Python is installed and added to PATH
- **Permission Errors**: Run as administrator if encountering access issues
- **Network Issues**: Check firewall settings for package downloads
- **Memory Errors**: Close other applications, ensure 4GB+ RAM available

#### Analysis Issues
- **OCR Failures**: Check image quality, try different image formats
- **Model Loading Errors**: Ensure sufficient disk space and memory
- **Performance Issues**: Consider GPU acceleration for large batches
- **Database Errors**: Check disk space and file permissions

#### GUI Problems
- **Interface Not Loading**: Update graphics drivers, check display scaling
- **Slow Performance**: Reduce batch sizes, close unnecessary programs
- **Export Failures**: Verify output directory permissions

### Performance Optimization
- **GPU Acceleration**: Install CUDA toolkit for faster AI processing
- **Memory Management**: Increase virtual memory for large datasets
- **Batch Size Tuning**: Adjust processing batches based on system capabilities
- **Database Optimization**: Regular database maintenance and indexing

## 📞 Support & Documentation

### Getting Help
- **Log Files**: Check `forensnap.log` for detailed error information
- **System Info**: Use built-in system diagnostics for technical support
- **Debug Mode**: Enable verbose logging for troubleshooting
- **Community**: Access online forums and user communities

### Additional Resources
- **Video Tutorials**: Step-by-step analysis demonstrations
- **Best Practices**: Investigation workflow recommendations
- **Case Studies**: Real-world usage examples and lessons learned
- **API Documentation**: Advanced integration and customization

## 📄 License & Legal

### Software License
This software is provided for legitimate digital forensics and investigation purposes only. Users are responsible for compliance with applicable laws and regulations in their jurisdiction.

### Disclaimer
ForenSnap Ultimate uses automated AI analysis that requires expert validation. Results should not be used as sole evidence but as investigative leads requiring human verification and additional corroboration.

### Privacy & Ethics
- Use only with proper legal authorization
- Respect privacy rights and applicable laws
- Follow professional ethics guidelines
- Ensure appropriate data handling and retention

---

## 🎯 Quick Start Summary

1. **Setup**: Run `setup_forensnap.bat` (one-time)
2. **Launch**: Double-click desktop shortcut
3. **Analyze**: Drop images or select folder
4. **Review**: Examine AI analysis results
5. **Report**: Generate legal documentation
6. **Export**: Save results in required format

**🔬 Ready to revolutionize your digital investigations with AI-powered analysis!**

---

*ForenSnap Ultimate v2.0.0 - Built with ❤️ for digital forensics professionals*
