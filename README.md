Setup and Installation
Prerequisites:
Python 3.7+ (the code uses f-strings and type hints compatible with Python 3.7 and above).
Google Chrome (or Chromium) browser installed – the headless browser automation uses Chrome.
ChromeDriver matching your Chrome version, accessible in your PATH. (Alternatively, you can adjust the code to use Firefox + geckodriver or another browser. Selenium WebDriver will need the corresponding driver installed.)
Tesseract OCR engine installed on your system (optional, but required to extract text from images). On macOS you can use Homebrew (brew install tesseract), on Ubuntu sudo apt-get install tesseract-ocr. Ensure the tesseract command is in PATH so pytesseract can invoke it.
Python Dependencies: All required Python packages are listed in requirements.txt. This includes:
selenium – for controlling the web browser.
pytesseract and Pillow – for OCR (pytesseract is a wrapper for Tesseract, and Pillow is used to open image files).
transformers – Hugging Face Transformers library to load the QA model.
beautifulsoup4 – for parsing HTML and extracting links.
