# EzerScraper

A powerful web scraping backend service with OCR capabilities and AI-powered text analysis. This service is designed to work with the breeze project as its frontend interface.

## Features

- Web scraping using Selenium and BeautifulSoup4
- OCR text extraction from images using Tesseract
- AI-powered text analysis using transformers and Qwen
- Location-based data processing using geopy
- RESTful API endpoints for integration with breeze frontend

## Prerequisites

- Python 3.7 or higher
- Google Chrome or Chromium browser
- ChromeDriver (matching your Chrome version)
- Tesseract OCR engine
- Ollama (for running Qwen locally)

### Installing Tesseract

- **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### Setting up Ollama and Qwen

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Qwen model:
```bash
ollama pull qwen2.5
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ezerScraper-main
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required AI models:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the backend service:
```bash
python main.py
```

The service will start and listen for API requests from the breeze frontend application.

## API Endpoints

The following endpoints are available for integration with the breeze frontend:

- `POST /api/scrape` - Initiate web scraping
- `POST /api/analyze` - Perform text analysis
- `GET /api/status` - Check scraping status
(Add other relevant endpoints based on your actual implementation)

## Project Structure

- `main.py` - Main application entry point and API routes
- `scraper/` - Core scraping functionality
- `output/` - Scraped data output
- `screenshots/` - Captured screenshots

## Dependencies

Key packages and their versions:
- selenium >= 4.15.2
- beautifulsoup4 >= 4.12.2
- pytesseract >= 0.3.10
- torch >= 2.1.2
- transformers >= 4.36.2
- flask >= 3.0.0
- opencv-python >= 4.8.1.78

For a complete list, see `requirements.txt`.

## Integration with Breeze

This service is designed to work as the backend for the breeze project. The breeze frontend will make API calls to this service to perform web scraping and analysis tasks. For frontend setup and configuration, please refer to the breeze project documentation.
