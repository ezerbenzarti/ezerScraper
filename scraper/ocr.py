import pytesseract
import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def do_ocr_screenshot(screenshot_path):
    """
    Perform OCR on a screenshot image.
    Returns the extracted text or empty string if OCR fails.
    """
    try:
        # Check if Tesseract is installed
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            logger.error("Tesseract not found. Please install Tesseract OCR and add it to your PATH")
            return ""

        # Read and preprocess image
        img = cv2.imread(screenshot_path)
        if img is None:
            logger.error(f"Could not read image: {screenshot_path}")
            return ""

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        # Perform OCR with improved configuration
        text = pytesseract.image_to_string(
            gray,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-_ '
        )
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""
