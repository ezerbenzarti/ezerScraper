from PIL import Image
import pytesseract

def extract_text(image_path):
    """
    Uses Tesseract OCR to extract text from an image file.
    Returns the extracted text.
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        print(f"Error during OCR extraction on {image_path}: {e}")
        text = ""
    return text
