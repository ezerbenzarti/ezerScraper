import re
import time
import cv2
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langdetect import detect
import spacy
import logging
from transformers import pipeline as hf_pipeline
from . import browser, qa_model
from .utils import is_internal
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################
# 1) spaCy Model Caching   #
############################
_MODELS = {}

def get_model(lang_code):
    global _MODELS
    try:
        if lang_code.startswith("fr"):
            if "fr" not in _MODELS:
                try:
                    _MODELS["fr"] = spacy.load("fr_core_news_lg")
                except OSError:
                    logger.info("Downloading French language model...")
                    spacy.cli.download("fr_core_news_lg")
                    _MODELS["fr"] = spacy.load("fr_core_news_lg")
            return _MODELS["fr"]
        elif lang_code.startswith("ar"):
            if "ar" not in _MODELS:
                try:
                    _MODELS["ar"] = spacy.load("xx_ent_wiki_sm")  # Using multilingual model for Arabic
                except OSError:
                    logger.info("Downloading multilingual model for Arabic...")
                    spacy.cli.download("xx_ent_wiki_sm")
                    _MODELS["ar"] = spacy.load("xx_ent_wiki_sm")
            return _MODELS["ar"]
        elif lang_code.startswith("en"):
            if "en" not in _MODELS:
                try:
                    _MODELS["en"] = spacy.load("en_core_web_lg")
                except OSError:
                    logger.info("Downloading English language model...")
                    spacy.cli.download("en_core_web_lg")
                    _MODELS["en"] = spacy.load("en_core_web_lg")
            return _MODELS["en"]
        else:
            if "multi" not in _MODELS:
                try:
                    _MODELS["multi"] = spacy.load("xx_ent_wiki_sm")
                except OSError:
                    logger.info("Downloading multilingual model...")
                    spacy.cli.download("xx_ent_wiki_sm")
                    _MODELS["multi"] = spacy.load("xx_ent_wiki_sm")
            return _MODELS["multi"]
    except Exception as e:
        logger.error(f"Error loading language model: {e}")
        # Fallback to multilingual model
        if "multi" not in _MODELS:
            try:
                _MODELS["multi"] = spacy.load("xx_ent_wiki_sm")
            except OSError:
                spacy.cli.download("xx_ent_wiki_sm")
                _MODELS["multi"] = spacy.load("xx_ent_wiki_sm")
        return _MODELS["multi"]

############################
# 2) Prompt Field Parsing  #
############################
_field_extractor = hf_pipeline("text2text-generation", model="t5-small")

def interpret_prompt_with_llm(prompt):
    """
    Use local Qwen2.5 model through Ollama to interpret the user's prompt
    and extract relevant fields in a structured way.
    """
    try:
        # Prepare the prompt for the LLM
        system_prompt = """You are a helpful assistant that extracts contact information fields from user prompts.
        Identify which fields the user wants to extract (name, phone, email, address, domain, poste).
        Return a JSON object with the fields as keys and boolean values indicating if they should be extracted.
        Example output: {"name": true, "phone": true, "email": true, "address": false, "domain": false, "poste": false}"""
        
        logger.info(f"Sending prompt to LLM: {prompt}")
        
        # Call Ollama API
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "qwen2.5",
                "prompt": f"{system_prompt}\n\nUser prompt: {prompt}",
                "stream": False
            }
        )
        
        logger.info(f"LLM Response status: {response.status_code}")
        logger.info(f"LLM Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            # Extract the JSON response from the LLM's output
            try:
                # The response might be in the 'response' field
                llm_output = result.get('response', '')
                logger.info(f"LLM raw output: {llm_output}")
                
                # Find JSON in the output
                json_start = llm_output.find('{')
                json_end = llm_output.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = llm_output[json_start:json_end]
                    logger.info(f"Extracted JSON: {json_str}")
                    fields = json.loads(json_str)
                    # Convert to list of field names
                    extracted_fields = [field for field, include in fields.items() if include]
                    logger.info(f"Extracted fields: {extracted_fields}")
                    return extracted_fields
                else:
                    logger.warning("No JSON found in LLM output")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Error parsing LLM response: {e}")
                logger.error(f"Failed to parse: {llm_output}")
        
        # Fallback to basic field detection if LLM fails
        logger.warning("Falling back to basic field detection")
        return parse_prompt_for_fields(prompt)
    except Exception as e:
        logger.error(f"Error in LLM interpretation: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        # Fallback to basic field detection
        return parse_prompt_for_fields(prompt)

def parse_prompt_for_fields(prompt):
    """
    Parse the prompt to determine which fields to extract.
    First tries LLM interpretation, falls back to basic keyword matching.
    """
    # Try LLM interpretation first
    fields = interpret_prompt_with_llm(prompt)
    if fields:
        return fields
    
    # Fallback to basic keyword matching
    plower = prompt.lower()
    fields = []
    if "name" in plower or "nom" in plower or "association" in plower:
        fields.append("name")
    if "phone" in plower or "telephone" in plower or "tel" in plower or "numéro" in plower:
        fields.append("phone")
    if "email" in plower or "mail" in plower or "courriel" in plower:
        fields.append("email")
    if "address" in plower or "adresse" in plower or "location" in plower:
        fields.append("address")
    if "domain" in plower or "website" in plower or "site" in plower:
        fields.append("domain")
    if "poste" in plower or "job" in plower or "title" in plower or "fonction" in plower:
        fields.append("poste")
    
    # Always include name if no fields were detected
    if not fields:
        fields.append("name")
    
    return fields

############################
# 3) OCR Functionality     #
############################
def do_ocr_screenshot(screenshot_path):
    try:
        img = cv2.imread(screenshot_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

############################
# 4) Regex & Helper Funcs  #
############################
def extract_phone(text):
    """
    Extract phone numbers with specific handling for Tunisian numbers.
    """
    # First try to find numbers in a Téléphone/Tel/Phone labeled field
    phone_labels = ['téléphone', 'telephone', 'tel', 'phone', 'contact', 'numéro', 'numero']
    lines = text.lower().split('\n')
    for line in lines:
        line = line.strip()
        if any(label in line for label in phone_labels):
            # Extract digits from this line
            digits = ''.join(c for c in line if c.isdigit())
            if len(digits) >= 8:
                # Try to find a valid phone number in these digits
                for i in range(len(digits)-7):
                    candidate = digits[i:i+8]
                    if candidate[0] in ['2', '5', '9']:
                        return candidate

    # Look for 8-digit numbers with valid prefixes
    number_matches = re.findall(r'\b\d{8}\b', text)
    for num in number_matches:
        if num[0] in ['2', '5', '9']:
            return num

    # Look for numbers with common separators
    patterns = [
        r'(?:[\s.]?\d{2}){4}',  # 12 34 56 78 or 12.34.56.78
        r'\d{2}[\s.-]\d{2}[\s.-]\d{2}[\s.-]\d{2}',  # 12-34-56-78
        r'(?:\+216|00216)?[2|5|9]\d{7}',  # +216/00216 followed by 8 digits
        r'(?:\+216|00216)?[2|5|9]\d{2}[\s.-]\d{2}[\s.-]\d{2}[\s.-]\d{2}'  # Same with separators
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            # Clean up the number
            digits = ''.join(c for c in match.group(0) if c.isdigit())
            # Handle country code if present
            if len(digits) > 8:
                digits = digits[-8:]
            if len(digits) == 8 and digits[0] in ['2', '5', '9']:
                return digits

    # Look for any 8+ digit sequences and try to extract valid numbers
    digit_sequences = re.findall(r'\d{8,}', text)
    for seq in digit_sequences:
        # Try to find a valid 8-digit sequence
        for i in range(len(seq)-7):
            candidate = seq[i:i+8]
            if candidate[0] in ['2', '5', '9']:
                return candidate

    # If nothing found, try one last aggressive search for digit sequences
    all_digits = ''.join(c for c in text if c.isdigit())
    for i in range(len(all_digits)-7):
        candidate = all_digits[i:i+8]
        if candidate[0] in ['2', '5', '9']:
            return candidate

    return ""

def is_valid_tunisian_number(number):
    """Helper function to validate Tunisian phone numbers"""
    if not number:
        return False
    digits = ''.join(c for c in number if c.isdigit())
    return len(digits) == 8 and digits[0] in ['2', '5', '9']

def extract_email(text):
    """
    Enhanced email extraction that handles various formats and common email patterns.
    """
    # Common email labels in French and English
    email_labels = [
        'email', 'e-mail', 'mail', 'courriel', 'contact',
        'adresse email', 'adresse e-mail', 'adresse mail',
        'adresse courriel', 'adresse de contact'
    ]
    
    # First try to find emails near labels
    lines = text.lower().split('\n')
    for line in lines:
        line = line.strip()
        if any(label in line for label in email_labels):
            # Look for email pattern in this line
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            match = re.search(email_pattern, line)
            if match and not is_placeholder_email(match.group(0)):
                return match.group(0)
    
    # If no labeled email found, try to find any email in the text
    # More flexible pattern that handles various formats
    patterns = [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Standard email
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\.[a-zA-Z]{2,}',  # Multi-level domains
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # With word boundary
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\b'  # Optional second-level domain
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            email = match.group(0)
            if not is_placeholder_email(email):
                return email
    
    # If still no email found, try to find sequences that look like emails
    words = text.split()
    for word in words:
        if '@' in word and '.' in word:
            # Clean up the word to only keep email-relevant characters
            cleaned = ''.join(c for c in word if c.isalnum() or c in '@._-')
            if '@' in cleaned and '.' in cleaned:
                parts = cleaned.split('@')
                if len(parts) == 2:
                    local, domain = parts
                    if local and domain and '.' in domain:
                        email = f"{local}@{domain}"
                        if not is_placeholder_email(email):
                            return email
    
    return ""

def is_placeholder_email(email):
    """
    Check if the email is a placeholder or example.
    """
    if not email:
        return True
        
    email = email.lower()
    
    # Common placeholder patterns
    placeholders = [
        "example", "exemple", "sample", "test", "demo",
        "your.email", "your-email", "your_email",
        "email@", "mail@", "contact@",
        "info@", "support@", "admin@",
        "user@", "username@", "name@",
        "someone@", "someone@example.com",
        "ton-email@", "votre-mail@", "votre-email@",
        "votre.email@", "votre_mail@", "votre_email@"
    ]
    
    # Check if email contains any placeholder pattern
    return any(p in email for p in placeholders)

def extract_website(text):
    """Extract website URL from text."""
    pattern = re.compile(r'(https?://[^\s]+|www\.[^\s]+)')
    match = pattern.search(text)
    return match.group(0).strip() if match else ""

def extract_industry(text, qa_pipe=None, context=None):
    """Extract industry/field of activity from text."""
    # Common industry keywords in French and English
    industry_keywords = {
        'en': ['industry', 'sector', 'field', 'business', 'specialization', 'expertise'],
        'fr': ['industrie', 'secteur', 'domaine', 'activité', 'spécialité', 'métier']
    }
    
    # Try to find industry in labeled fields first
    lines = text.lower().split('\n')
    for line in lines:
        line = line.strip()
        for lang in ['en', 'fr']:
            for keyword in industry_keywords[lang]:
                if keyword in line:
                    # Extract text after the keyword
                    parts = line.split(keyword + ':', 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                    parts = line.split(keyword + ' :', 1)
                    if len(parts) > 1:
                        return parts[1].strip()
    
    # If no labeled field found and QA pipeline is available, try QA
    if qa_pipe and context:
        industry = extract_field_by_qa("What is the industry, sector, or field of activity of this organization?", context, qa_pipe)
        if industry:
            return industry
    
    return ""

def is_address_line(line):
    address_keywords = [
        "rue", "avenue", "bp", "quartier", "route", "lot", "zone", "imm",
        "street", "road", "blvd", "boulevard", "zip", "postal", "cedex"
    ]
    low = line.lower()
    return any(kw in low for kw in address_keywords) and re.search(r'\d', line)

def find_address_in_lines(lines):
    for line in lines:
        if is_address_line(line):
            return line.strip()
    return ""

def extract_field_by_qa(question, context, qa_pipe):
    try:
        qa_result = qa_pipe(question=question, context=context)
        if qa_result["score"] > 0.3:
            return qa_result["answer"].strip()
    except Exception as e:
        print(f"QA extraction error: {e}")
    return ""

############################
# 5) Table-based Extraction #
############################
def _extract_table_data(html, fields):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None
    rows = table.find_all("tr")
    if not rows:
        return None

    results = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        item = {}
        for idx, field in enumerate(fields):
            if idx < len(cells):
                val = cells[idx].get_text(" ", strip=True)
                item[field] = val
            else:
                item[field] = ""
        if item.get("name"):
            results.append(item)
    return results if results else None

############################
# 6) Detail Page Processing#
############################
def process_detail_page(detail_url, qa_pipe, fields):
    """
    Processes a detail page independently.
    Loads the detail page and extracts its HTML, visible text, and screenshot,
    then performs line-by-line scanning plus QA extraction for each requested field.
    """
    try:
        dhtml, dvis, dscreenshot = browser.render_page(detail_url)
    except Exception as e:
        print(f"Error loading detail page {detail_url}: {e}")
        return {}
    if not dhtml:
        return {}
    
    try:
        soup = BeautifulSoup(dhtml, "html.parser")
        context = dvis
        ocr_text = do_ocr_screenshot(dscreenshot)
        combined_context = context + "\n" + ocr_text
        lines = [l.strip() for l in combined_context.splitlines() if l.strip()]
        address_tag = soup.find("address")
        found_address = address_tag.get_text(" ", strip=True) if address_tag else ""
    
        detail = {}
        for f in fields:
            if f == "name":
                continue
            elif f == "phone":
                phone_found = ""
                for line in lines:
                    ph = extract_phone(line)
                    if ph:
                        phone_found = ph
                        break
                detail["phone"] = phone_found
            elif f == "email":
                email_found = ""
                for line in lines:
                    em = extract_email(line)
                    if em and not is_placeholder_email(em):
                        email_found = em
                        break
                detail["email"] = email_found
            elif f == "address":
                if found_address:
                    detail["address"] = found_address
                else:
                    line_addr = find_address_in_lines(lines)
                    if line_addr:
                        detail["address"] = line_addr
                    else:
                        ans = extract_field_by_qa("What is the address of this association?", combined_context, qa_pipe)
                        detail["address"] = ans
            elif f == "domain":
                # Extract both industry and website
                industry = extract_industry(combined_context, qa_pipe, combined_context)
                website = ""
                for line in lines:
                    website = extract_website(line)
                    if website:
                        break
                if not website:
                    website = extract_field_by_qa("What is the website or URL of this organization?", combined_context, qa_pipe)
                detail["domain"] = industry  # Store industry in domain field
                detail["website"] = website  # Add website as a new field
            elif f == "poste":
                poste_found = ""
                for line in lines:
                    low = line.lower()
                    if "poste" in low or "role" in low or "fonction" in low:
                        poste_found = line
                        break
                if not poste_found:
                    poste_found = extract_field_by_qa("What is the job title or poste of the contact person?", combined_context, qa_pipe)
                detail["poste"] = poste_found
            else:
                question = f"What is the {f}?"
                detail[f] = extract_field_by_qa(question, combined_context, qa_pipe)
        return detail
    except Exception as e:
        print(f"Error processing detail page {detail_url}: {e}")
        return {}

############################
# 7) Main Crawl Logic      #
############################
def crawl_site(start_url, prompt, depth, max_pages, qa_pipe, crawl_detail=False):
    """
    1) Parse fields from the prompt.
    2) Load the main page; if table-based, extract data and return.
    3) Otherwise, gather candidate anchors from <main> or div#main.
    4) For each candidate anchor, create one item. If crawl_detail is enabled,
       process its detail page independently via process_detail_page().
    5) Return a list of dictionaries with the extracted fields.
    """
    fields = parse_prompt_for_fields(prompt)
    print("Parsed fields from prompt:", fields)

    print(f"Loading main page: {start_url}")
    try:
        html, visible_text, screenshot_path = browser.render_page(start_url)
    except Exception as e:
        print(f"Error loading main page: {e}")
        return []
    if not html:
        return []

    table_data = _extract_table_data(html, fields)
    if table_data:
        print("Table detected; using table extraction.")
        return table_data

    soup = BeautifulSoup(html, "html.parser")
    if soup.find("main"):
        anchors = soup.find("main").find_all("a")
    elif soup.find("div", id="main"):
        anchors = soup.find("div", id="main").find_all("a")
    else:
        anchors = [a for a in soup.find_all("a") if not a.find_parent(["header", "nav", "footer"])]

    results = []
    visited = set()
    try:
        lang_code = soup.find("html")["lang"].lower()
    except Exception:
        try:
            lang_code = detect(visible_text)
        except Exception:
            lang_code = "en"
    model = get_model(lang_code)

    for a in anchors:
        try:
            candidate_text = a.get_text(strip=True)
            if not candidate_text or len(candidate_text) < 3:
                continue
            lower_text = candidate_text.lower()
            if lower_text in ["login", "signup", "home", "about", "contact", "connexion", "inscription", "accueil", "à propos"]:
                continue

            doc = model(candidate_text)
            is_org = any(ent.label_ == "ORG" for ent in doc.ents)
            tokens = candidate_text.split()
            capital_count = sum(1 for t in tokens if t and t[0].isupper())
            is_capitalized = (len(tokens) >= 2 and capital_count >= len(tokens)/2)
            if not (is_org or is_capitalized):
                continue

            item = {"name": candidate_text}
            detail_href = a.get("href")
            if crawl_detail and detail_href:
                detail_url = urljoin(start_url, detail_href)
                try:
                    if detail_url not in visited and (max_pages is None or len(visited) < max_pages):
                        visited.add(detail_url)
                        print(f"Processing detail page for '{candidate_text}': {detail_url}")
                        detail_info = process_detail_page(detail_url, qa_pipe, fields)
                        if detail_info:
                            item.update(detail_info)
                        time.sleep(1)  # slight delay to avoid overloading
                except Exception as de:
                    print(f"Error processing detail page {detail_url}: {de}")
            else:
                parent = a.find_parent()
                parent_text = parent.get_text(" ", strip=True) if parent else ""
                main_ocr = do_ocr_screenshot(screenshot_path)
                combined_text = parent_text + "\n" + main_ocr
                lines = [l.strip() for l in combined_text.splitlines() if l.strip()]
                if "phone" in fields:
                    phone_found = ""
                    for line in lines:
                        ph = extract_phone(line)
                        if ph:
                            phone_found = ph
                            break
                    item["phone"] = phone_found
                if "email" in fields:
                    email_found = ""
                    for line in lines:
                        em = extract_email(line)
                        if em and not is_placeholder_email(em):
                            email_found = em
                            break
                    item["email"] = email_found
                if "address" in fields:
                    addr = ""
                    for line in lines:
                        if is_address_line(line):
                            addr = line
                            break
                    if not addr:
                        addr = extract_field_by_qa("What is the address of this association?", combined_text, qa_pipe)
                    item["address"] = addr
                if "domain" in fields:
                    # Extract both industry and website
                    industry = extract_industry(combined_text, qa_pipe, combined_text)
                    website = ""
                    for line in lines:
                        website = extract_website(line)
                        if website:
                            break
                    if not website:
                        website = extract_field_by_qa("What is the website or URL of this organization?", combined_text, qa_pipe)
                    item["domain"] = industry  # Store industry in domain field
                    item["website"] = website  # Add website as a new field
                if "poste" in fields:
                    poste_found = ""
                    for line in lines:
                        low = line.lower()
                        if "poste" in low or "role" in low or "fonction" in low:
                            poste_found = line
                            break
                    if not poste_found:
                        poste_found = extract_field_by_qa("What is the job title or poste of the contact person?", combined_text, qa_pipe)
                    item["poste"] = poste_found
            results.append(item)
        except Exception as e:
            print(f"Error processing anchor: {e}")
            continue

    return results
 