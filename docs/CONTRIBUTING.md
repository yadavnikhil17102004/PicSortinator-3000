# Contributing to PicSortinator-3000

Thank you for considering contributing to PicSortinator-3000! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct, which is to be respectful, inclusive, and considerate of others.

## How Can I Contribute?

### Reporting Bugs

- **Check if the bug has already been reported** by searching on GitHub under [Issues](https://github.com/your-username/PicSortinator-3000/issues).
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/your-username/PicSortinator-3000/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

- Open a new issue with a clear title and detailed description of the enhancement.
- Provide examples of how the enhancement would be used and its benefits.

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the existing style.
6. Issue that pull request!

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Styleguide

* Follow PEP 8
* Use 4 spaces for indentation
* Include docstrings for all classes and functions
* Limit line length to 100 characters

### Documentation Styleguide

* Use Markdown for documentation
* Reference function and class names in backticks: `like_this()`
* Include code examples when useful

## Funny Contributions

PicSortinator-3000 embraces humor! Feel free to add:

- Sarcastic comments in user-facing messages
- Clever variable names (while keeping them professional)
- Easter eggs that surprise and delight users

Just make sure all humor is appropriate and doesn't distract from usability.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
