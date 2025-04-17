from urllib.parse import urlparse, urljoin
import os
import json
from datetime import datetime

def get_domain(url):
    """
    Returns the domain (netloc) of the URL.
    """
    parsed = urlparse(url)
    return parsed.netloc

def is_internal(start_url, new_url):
    """
    Checks whether new_url is in the same domain as start_url.
    """
    return get_domain(start_url) == get_domain(new_url)

def ensure_output_dir():
    """Ensure the output directory exists"""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_results(results):
    """Save results to output.json in the output directory"""
    output_dir = ensure_output_dir()
    output_file = os.path.join(output_dir, "output.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False
