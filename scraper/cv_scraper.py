import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import requests
from bs4 import BeautifulSoup
from .crawler import (
    process_detail_page,
    interpret_prompt_with_llm,
    parse_prompt_for_fields,
    crawl_site
)
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
from datetime import datetime
import numpy as np
import pytesseract
import cv2
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_browser():
    """Setup headless browser for screenshots"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--start-maximized')
    return webdriver.Chrome(options=chrome_options)

def capture_full_page_screenshot(url):
    """Capture a full page screenshot of the webpage"""
    browser = None
    try:
        logger.info(f"Capturing full page screenshot of {url}")
        browser = setup_browser()
        browser.get(url)
        
        # Wait for page load
        wait = WebDriverWait(browser, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(3)
        
        # Get page height
        total_height = browser.execute_script("""
            return Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                document.body.offsetHeight,
                document.documentElement.offsetHeight
            );
        """)
        
        # Set viewport size
        browser.set_window_size(1920, total_height)
        
        # Scroll through page
        current_height = 0
        while current_height < total_height:
            browser.execute_script(f"window.scrollTo(0, {current_height});")
            time.sleep(0.5)
            current_height += 500
        
        browser.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Save screenshot
        os.makedirs('screenshots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = f'screenshots/screenshot_{timestamp}.png'
        browser.save_screenshot(screenshot_path)
        
        # Get card links if needed for deep crawling
        links = {}
        elements = browser.find_elements(By.TAG_NAME, 'a')
        for element in elements:
            try:
                link = element.get_attribute('href')
                text = element.text.strip()
                if link and text:
                    links[text] = link
            except Exception as e:
                continue
                
        return screenshot_path, links
        
    except Exception as e:
        logger.error(f"Error capturing screenshot: {str(e)}")
        return None, {}
    finally:
        if browser:
            browser.quit()

def extract_text_from_image_region(image_path, box):
    """Extract text from a specific region of an image using OCR"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Extract region with padding
        x1, y1, x2, y2 = map(int, box)
        # Add padding to capture full text
        padding = 10
        y1 = max(0, y1 - padding)
        y2 = min(img.shape[0], y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        
        card_region = img[y1:y2, x1:x2]
        
        # Image preprocessing
        # Convert to grayscale
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Convert back to RGB for Tesseract
        rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Configure Tesseract parameters
        custom_config = r'--oem 3 --psm 6 -l ara+fra+eng'
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        
        # Clean up the text
        lines = text.strip().split('\n')
        # Remove empty lines and clean each line
        lines = [line.strip() for line in lines if line.strip()]
        # Join with single spaces
        text = ' '.join(lines)
        
        if text:
            logger.info(f"Extracted text: {text[:100]}...")
            return text
        return None
            
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return None

def process_page_with_cv(url, prompt=None, qa_pipe=None, crawl_detail=False):
    """Process a single page using computer vision approach"""
    try:
        # Step 1: Get HTML content and capture screenshot
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        screenshot_path, _ = capture_full_page_screenshot(url)
        if not screenshot_path:
            return []
            
        # Step 2: Run YOLO detection
        model = YOLO('best.pt')
        results = model(screenshot_path)
        
        if not len(results) or not len(results[0].boxes):
            logger.warning("No cards detected")
            return []
            
        # Step 3: Process each detected card
        cards_data = []
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Save annotated image
        img = Image.open(screenshot_path)
        draw = ImageDraw.Draw(img)
        
        # Extract all links from the page
        page_links = {}
        base_url = '/'.join(url.split('/')[:3])  # Get base URL for resolving relative links
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Handle relative URLs
            if href.startswith('/'):
                href = base_url + href
            elif not href.startswith(('http://', 'https://')):
                href = base_url + '/' + href
                
            text = ' '.join(a.stripped_strings)
            if text:
                page_links[text.strip().lower()] = href
        
        for i, box in enumerate(boxes):
            # Draw detection box
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Extract text with OCR
            ocr_text = extract_text_from_image_region(screenshot_path, box)
            if not ocr_text:
                continue
                
            # Clean up OCR text
            clean_ocr = ' '.join(ocr_text.split())
            
            # Find matching link for this text
            detail_url = None
            ocr_lower = clean_ocr.lower()
            best_match = None
            best_match_ratio = 0
            
            for link_text, link_url in page_links.items():
                # Try exact match first
                if link_text == ocr_lower:
                    detail_url = link_url
                    break
                    
                # Try partial matching with word comparison
                ocr_words = set(ocr_lower.split())
                link_words = set(link_text.split())
                
                # Calculate word overlap ratio
                common_words = ocr_words & link_words
                if common_words:
                    ratio = len(common_words) / max(len(ocr_words), len(link_words))
                    if ratio > best_match_ratio:
                        best_match_ratio = ratio
                        best_match = link_url
            
            if not detail_url and best_match and best_match_ratio > 0.5:  # At least 50% word match
                detail_url = best_match
            
            # Create card data with just the essential information
            card_data = {'name': clean_ocr}
            if detail_url:
                card_data['detail_url'] = detail_url
                
            cards_data.append(card_data)
            logger.info(f"Processed card {i+1}: {clean_ocr[:100]}...")
        
        # Save annotated image
        annotated_path = screenshot_path.replace('.png', '_annotated.png')
        img.save(annotated_path)
        
        logger.info(f"Processed {len(cards_data)} cards")
        return cards_data
        
    except Exception as e:
        logger.error(f"Error processing page: {str(e)}")
        return []

def calculate_name_similarity(name1, name2):
    """Calculate similarity ratio between two names"""
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

def combine_cv_and_legacy_results(cv_results, legacy_results, similarity_threshold=0.6):
    """
    Use CV results to validate names from legacy results.
    The legacy method handles all the scraping and deep crawling,
    CV is only used to validate which names should be kept.
    """
    matched_results = []
    
    # Extract names detected by CV for validation
    cv_names = {item.get('name', '').strip().lower() for item in cv_results if item.get('name')}
    
    # Print header for visibility
    logger.info("\n" + "="*80)
    logger.info("VALIDATED RESULTS (USING CV FOR NAME VALIDATION):")
    logger.info("="*80)
    
    # Process each legacy result
    for legacy_item in legacy_results:
        legacy_name = legacy_item.get('name', '').strip()
        if not legacy_name:
            continue
            
        # Check if this name is validated by CV results
        best_ratio = 0
        for cv_name in cv_names:
            ratio = calculate_name_similarity(legacy_name.lower(), cv_name)
            if ratio > best_ratio:
                best_ratio = ratio
        
        # If name is validated by CV (similarity above threshold)
        if best_ratio > similarity_threshold:
            matched_results.append(legacy_item)  # Keep the complete legacy result
            
            # Print matched item with URL in a clear format
            logger.info("\nValidated Association:")
            logger.info("-" * 40)
            logger.info(f"Name: {legacy_name}")
            logger.info(f"URL:  {legacy_item.get('detail_url', 'No URL found')}")
            logger.info(f"Match confidence: {best_ratio:.2%}")
            logger.info("-" * 40)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info(f"Total validated results: {len(matched_results)} out of {len(legacy_results)}")
    logger.info("="*80 + "\n")
    
    return matched_results

def cv_crawl_site(start_url, prompt, qa_pipe, crawl_detail=False):
    """
    Enhanced crawl function that uses CV for name validation only.
    The actual scraping and deep crawling is handled by the legacy method.
    """
    try:
        logger.info(f"Starting CV-validated crawl: {start_url}")
        
        # Run CV method first to get validated names
        logger.info("Running CV method for name detection...")
        cv_results = process_page_with_cv(start_url, prompt, qa_pipe, False)
        
        # Run legacy method with deep crawling if requested
        logger.info("Running legacy method for scraping and deep crawling...")
        legacy_results = crawl_site(
            start_url=start_url,
            prompt=prompt,
            depth=1,
            max_pages=None,
            qa_pipe=qa_pipe,
            crawl_detail=crawl_detail  # Pass through the deep crawling flag
        )
        
        logger.info(f"\nFound {len(cv_results)} names from CV and {len(legacy_results)} from legacy method")
        
        # Use CV results to validate legacy results
        validated_results = combine_cv_and_legacy_results(cv_results, legacy_results)
        
        return validated_results
        
    except Exception as e:
        logger.error(f"Error in CV-validated crawl: {str(e)}")
        return [] 