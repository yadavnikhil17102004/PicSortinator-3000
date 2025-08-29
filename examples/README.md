# PicSortinator 3000 Examples

This directory contains example scripts demonstrating various features and capabilities of PicSortinator 3000.

## Examples Overview

### 🚀 [basic_workflow.py](basic_workflow.py)
**Complete workflow demonstration**
- Shows the full process from scanning to search
- Demonstrates all core components working together
- Good starting point for new users
- Includes error handling and progress feedback

### 🏷️ [custom_tagging.py](custom_tagging.py)
**Custom tagging configuration**
- How to add custom tag categories
- Adjusting confidence thresholds
- Domain-specific tagging setups
- Performance tuning tips

### 🔍 [search_examples.py](search_examples.py)
**Advanced search and filtering**
- Tag-based searches
- Text content searches
- Combined query examples
- Date and metadata filtering
- Similarity matching demonstrations

## Prerequisites

Before running examples, ensure you have:

1. **Installed dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test images directory:**
   Create a `test_images/` directory with sample photos

3. **Models downloaded:**
   The examples will automatically download required AI models on first run

## Running Examples

### Basic Usage
```bash
# Run from the project root directory
python examples/basic_workflow.py
```

### Custom Configuration
```bash
python examples/custom_tagging.py
```

### Advanced Search
```bash
python examples/search_examples.py
```

## Customization Tips

### Adding Your Own Categories
Edit the `custom_categories` in `custom_tagging.py`:
```python
custom_categories = {
    "my_category": ["tag1", "tag2", "tag3"],
    "another_category": ["more", "tags", "here"]
}
```

### Adjusting Performance
- **More tags:** Lower `confidence_threshold` (0.1-0.3)
- **Fewer, accurate tags:** Higher `confidence_threshold` (0.4-0.8)
- **Faster processing:** Reduce image size in preprocessing
- **Better accuracy:** Increase image quality and size

### Custom Search Queries
Use the database connection directly for complex queries:
```python
cursor = db.conn.cursor()
cursor.execute("YOUR CUSTOM SQL QUERY")
results = cursor.fetchall()
```

## Integration Examples

These examples show how to integrate PicSortinator components into your own applications:

- **Web app:** Use the search functions in a Flask/Django app
- **Desktop GUI:** Integrate with tkinter or PyQt
- **Batch processing:** Automate with the workflow examples
- **API service:** Build REST endpoints around the core functions

## Troubleshooting

**No images found:**
- Check that `test_images/` directory exists
- Verify image formats are supported (.jpg, .jpeg, .png, .bmp, .tiff)

**Models not downloading:**
- Check internet connection
- Verify write permissions in `models/` directory

**Poor tagging results:**
- Try different confidence thresholds
- Add domain-specific categories
- Ensure good image quality

**OCR not working:**
- Install Tesseract OCR system dependency
- Check image resolution and text clarity

## Contributing

Feel free to contribute more examples! Useful additions:
- Batch processing workflows
- Integration with cloud storage
- Custom AI model examples
- Performance optimization demos

## Support

For questions about these examples, see:
- Main [README.md](../README.md) for general setup
- [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for development guidelines
- Create an issue for bugs or feature requests
