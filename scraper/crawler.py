import re
import time
import cv2
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langdetect import detect
import spacy
from transformers import pipeline as hf_pipeline
from . import browser, qa_model
from .utils import is_internal

############################
# 1) spaCy Model Caching   #
############################
_MODELS = {}

def get_model(lang_code):
    global _MODELS
    if lang_code.startswith("fr"):
        if "fr" not in _MODELS:
            _MODELS["fr"] = spacy.load("fr_core_news_sm")
        return _MODELS["fr"]
    elif lang_code.startswith("ar"):
        if "ar" not in _MODELS:
            try:
                _MODELS["ar"] = spacy.load("ar_core_news_sm")
            except Exception:
                print("Arabic model not found; fallback to xx_ent_wiki_sm.")
                _MODELS["ar"] = spacy.load("xx_ent_wiki_sm")
        return _MODELS["ar"]
    elif lang_code.startswith("en"):
        if "en" not in _MODELS:
            _MODELS["en"] = spacy.load("en_core_web_sm")
        return _MODELS["en"]
    else:
        if "multi" not in _MODELS:
            _MODELS["multi"] = spacy.load("xx_ent_wiki_sm")
        return _MODELS["multi"]

############################
# 2) Prompt Field Parsing  #
############################
_field_extractor = hf_pipeline("text2text-generation", model="t5-small")

def parse_prompt_for_fields(prompt):
    plower = prompt.lower()
    fields = []
    if "name" in plower:
        fields.append("name")
    if "phone" in plower or "telephone" in plower or "tel" in plower:
        fields.append("phone")
    if "email" in plower or "mail" in plower:
        fields.append("email")
    if "address" in plower or "adresse" in plower:
        fields.append("address")
    if "domain" in plower or "website" in plower or "site" in plower:
        fields.append("domain")
    if "poste" in plower or "job" in plower or "title" in plower:
        fields.append("poste")
    if fields:
        return fields
    input_text = f"Extract comma separated field names from: {prompt}"
    output = _field_extractor(input_text, max_length=50, do_sample=False)
    fields_text = output[0]["generated_text"].strip()
    guess = [f.strip() for f in fields_text.split(",") if f.strip()]
    if "name" not in guess:
        guess.insert(0, "name")
    return guess

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
    pattern = re.compile(r'(?:\+?\d{1,3}[\s.\-()]*)?\d{2,}[\s.\-()]*\d{2,}[\s.\-()]*\d{2,}[\s.\-()]*\d{2,}')
    match = pattern.search(text)
    return match.group(0).strip() if match else ""

def extract_email(text):
    pattern = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
    match = pattern.search(text)
    return match.group(0).strip() if match else ""

def is_placeholder_email(email):
    em = email.lower()
    placeholders = [
        "ton-email@",
        "votre-mail@",
        "example@",
        "exemple@",
        "test@",
        "support@o2switch.fr"
    ]
    for p in placeholders:
        if p in em:
            return True
    return False

def extract_domain(text):
    pattern = re.compile(r'(https?://[^\s]+|www\.[^\s]+)')
    match = pattern.search(text)
    return match.group(0).strip() if match else ""

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
                dom = ""
                for line in lines:
                    dom = extract_domain(line)
                    if dom:
                        break
                if not dom:
                    dom = extract_field_by_qa("What is the website or domain of this association?", combined_context, qa_pipe)
                detail["domain"] = dom
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
                    if detail_url not in visited and len(visited) < max_pages:
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
                    dom = ""
                    for line in lines:
                        dom = extract_domain(line)
                        if dom:
                            break
                    if not dom:
                        dom = extract_field_by_qa("What is the website or domain of this association?", combined_text, qa_pipe)
                    item["domain"] = dom
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
        except Exception as ce:
            print(f"Error processing candidate anchor: {ce}")

    dedup = {}
    for r in results:
        nm = r.get("name", "")
        if nm not in dedup or any(r[k] for k in r if k != "name"):
            dedup[nm] = r
    return list(dedup.values())
